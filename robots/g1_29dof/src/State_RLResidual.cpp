#include "FSM/State_RLResidual.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLResidual::State_RLResidual(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto cmg_dir = param::parser_policy_dir(cfg["cmg_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    auto policy_path = policy_dir / "exported" / "residual.onnx";
    auto cmg_path = cmg_dir / "exported" / "cmg_exported.onnx";
    printf("[RLResidual] policy: %s\n", policy_path.filename().c_str());
    printf("[RLResidual] cmg:    %s\n", cmg_path.filename().c_str());
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_path);
    cmg = std::make_unique<isaaclab::CMGRunner>(cmg_path, cmg_dir / "stats" / "cmg_stats.yaml");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLResidual::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}





// 2026-03-23_00-26-09 -> policy_new_cmg_base -> new CMG, scale 0.5 --> policy_new_cmg_post2,  --> better turning, 2026-03-23_21-43-35
// policy_non_ar_new -> old CMG, scale 0.25
//forward_ar() <-> forward() @State_RLResidual.h ar/non-ar
// min-max      <-> z-score  @algorithm.h old/new CMG
// 2026-03-25_01-52-20 model_11000.pt -> AMP Base