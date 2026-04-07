#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
// Untouched.
namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    static auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
    static std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    static std::string prev_key = "";
    constexpr float step = 0.5f;

    std::string key = FSMState::keyboard->key();

    // Increment on new key press (edge-triggered)
    if (!key.empty() && key != prev_key) {
        if      (key == "w") cmd[0] += step;
        else if (key == "s") cmd[0] -= step;
        else if (key == "a") cmd[1] += step;
        else if (key == "d") cmd[1] -= step;
        else if (key == "q") cmd[2] += step/5;
        else if (key == "e") cmd[2] -= step/5;
        else if (key == "x") { cmd[0] = 0; cmd[1] = 0; cmd[2] = 0; }

        cmd[0] = std::clamp(cmd[0], ranges["lin_vel_x"][0].as<float>(), ranges["lin_vel_x"][1].as<float>());
        cmd[1] = std::clamp(cmd[1], ranges["lin_vel_y"][0].as<float>(), ranges["lin_vel_y"][1].as<float>());
        cmd[2] = std::clamp(cmd[2], ranges["ang_vel_z"][0].as<float>(), ranges["ang_vel_z"][1].as<float>());

        printf("\r[CMD] vx:%+.1f vy:%+.1f wz:%+.1f", cmd[0], cmd[1], cmd[2]);
        fflush(stdout);
    }
    prev_key = key;

    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["amp_policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    auto policy_path = policy_dir / "exported" / "policy1.onnx";
    printf("[RLBase] policy: %s\n", policy_path.c_str());
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_path);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

// void State_RLBase::run()
// {
//     auto action = env->action_manager->processed_actions();
//     for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
//         lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
//     }
// }

void State_RLBase::run()
{
    static auto prev_action = env->action_manager->processed_actions();
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        if (std::abs(action[i] - prev_action[i]) < 0.02f) {
            action[i] = prev_action[i];
        }
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
    prev_action = action;
}