# RUN_DEPLOY

C++ / ONNX Runtime deployment for the RuN residual locomotion controller on the **Unitree G1 (29 DoF)**. A single process loads two frozen ONNX graphs — a Mixture-of-Experts Conditional Motion Generator (CMG) and an LSTM residual policy — and drives the robot at 50 Hz over Unitree DDS. The CMG and policy are trained in the companion repos [`cmg_workspace`](https://github.com/PMY9527/cmg_workspace) and [`unitree_rl_lab`](https://github.com/PMY9527/unitree_rl_lab); this repo only runs them on hardware/simulation.

## Build

Prerequisites: Ubuntu 20.04/22.04, CMake ≥ 3.12, Unitree SDK2 installed system-wide, Eigen 3, Boost (`program_options`), `yaml-cpp`, `fmt`, `spdlog`. ONNX Runtime is vendored under `thirdparty/`.

```bash
cd robots/g1_29dof
mkdir -p build && cd build
cmake ..
make -j4
```

Produces the `g1_ctrl` binary.

## Run

### Simulation (MuJoCo)

For sim-to-sim validation, start the MuJoCo G1 simulator on the loopback interface, then run the same binary against `lo`:

```bash
./g1_ctrl --network lo
```

DDS traffic stays on the local machine, so nothing needs to change on the robot side — MuJoCo publishes `LowState_t` and subscribes to `LowCmd_t` exactly like the real robot.

Control is via the terminal keyboard (focus the terminal running `g1_ctrl`). FSM modes are switched with:

| Key | Mode                |
| --- | ------------------- |
| `1` | Passive (safe stop) |
| `2` | FixStand            |
| `3` | Velocity            |

These global hotkeys are registered in every state, so you can jump directly between modes without chaining through intermediate ones.

### Hardware

On your wired PC, with the robot hung from a safety rig:

```bash
./g1_ctrl --network <iface>
```

where `<iface>` is the network interface attached to the robot (e.g. `eth0`). The controller boots into the `Passive` state.

Control is via a PlayStation-style joystick:

1. `LT + Up` → enter **FixStand**. The robot ramps into the default stand pose.
2. `RB + X` → enter **Velocity**. CMG + residual policy start running at 50 Hz.
3. Left stick commands linear velocity, right stick commands yaw rate. Command ranges are set in `robots/g1_29dof/config/policy/velocity/residual/params/deploy.yaml`.
4. `LT + B` → **Passive** (safe stop) from any state.
