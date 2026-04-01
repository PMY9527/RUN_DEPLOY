// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/algorithms/algorithms.h"
#include "cmg_viz_shm.h"
#include <cmath>
#include <numeric>

class State_RLResidual : public FSMState
{
public:
    State_RLResidual(int state_mode, std::string state_string);

    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // Cache default joint positions for gated offset (standing pose)
        auto& djp = env->robot->data.default_joint_pos;
        default_joint_pos_.resize(djp.size());
        for (int i = 0; i < djp.size(); ++i)
            default_joint_pos_[i] = djp[i];

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->episode_length += 1;
                env->robot->update();

                auto& jp = env->robot->data.joint_pos;
                auto& jv = env->robot->data.joint_vel;
                auto cmd = isaaclab::observations_map()["keyboard_velocity_commands"](env.get(), {}); // velocity_commands if using joystick

                if (cmd[0] < 0.5f) cmd[0] = 0.0f;

                cmg->forward_ar(
                    {jp.data(), jp.data() + jp.size()},
                    {jv.data(), jv.data() + jv.size()},
                    {cmd.data(), cmd.data() + cmd.size()}
                );
                auto motion_ref = cmg->get_motion_ref();  // 58 dims (pos + vel, USD order)
                auto qr = cmg->get_qref();                // 29 dims (positions only)

                auto obs = env->observation_manager->compute();

                obs["motion"] = motion_ref;
                obs["cmg_input"] = cmd;

                auto residual = env->alg->act(obs);

                float cmd_mag = std::sqrt(cmd[0]*cmd[0] + cmd[1]*cmd[1] + cmd[2]*cmd[2]);
                float gate = std::min(cmd_mag / 0.1f, 1.0f);

                auto& joint_action = static_cast<isaaclab::JointAction&>(*env->action_manager->_terms[0]);
                joint_action._offset.resize(29);
                for (size_t i = 0; i < 29; ++i)
                    joint_action._offset[i] = gate * qr[i] + (1.0f - gate) * default_joint_pos_[i];

                env->action_manager->process_action(residual);

                if (cmg_viz.ok()) {
                    auto qref_vel = std::vector<float>(motion_ref.begin() + 29, motion_ref.end());
                    cmg_viz.write(
                        qr,                                                    // qref positions
                        qref_vel,                                              // qref velocities
                        {jp.data(), jp.data() + jp.size()},                    // actual positions
                        {jv.data(), jv.data() + jv.size()},                    // actual velocities
                        {cmd.data(), cmd.data() + cmd.size()},                 // velocity commands
                        residual,                                              // policy residual
                        env->action_manager->processed_actions()               // final combined
                    );
                }

                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<isaaclab::CMGRunner> cmg;
    CMGVizWriter cmg_viz;

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::vector<float> default_joint_pos_;
};

REGISTER_FSM(State_RLResidual)
