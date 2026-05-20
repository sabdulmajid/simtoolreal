# Isaac Lab Migration Plan

This repo is not currently Isaac Lab-compatible. The migration should be a new task implementation with parity tests, not an in-place rewrite of the Isaac Gym task.

Current bridge evidence: `scripts/validate_isaaclab_toolpose_assets.py` and
`scripts/run_isaaclab_toolpose_asset_probe.sh` now load the real SHARPA robot
URDF, table URDF, and a procedurally generated handle-head tool in Isaac Lab on
both Blackwell GPUs. The probe exposes the 29-joint action tensor, palm and
fingertip body tensors, tool root state tensor, and a finite 140-dimensional
policy-observation candidate. It is not a full environment or training port.

## Source Files To Port

- `isaacgymenvs/tasks/simtoolreal/env.py`: main environment, reset logic, state buffers, rewards, observations, actions, asset creation, random forces, video/debug utilities.
- `isaacgymenvs/tasks/simtoolreal/generate_objects.py`: procedural handle/head URDF generation.
- `isaacgymenvs/tasks/simtoolreal/object_size_distributions.py`: object size and density sampling.
- `isaacgymenvs/tasks/simtoolreal/adjacent_links.py`: collision-filter/friction helper data.
- `isaacgymenvs/tasks/simtoolreal/utils.py`: DOF property setup and tolerance curriculum helpers.
- `isaacgymenvs/utils/observation_action_utils_sharpa.py`: deployment-compatible observation and action math.
- `isaacgymenvs/utils/torch_jit_utils.py`: quaternion, scaling, tensor helper functions.
- `isaacgymenvs/cfg/task/SimToolReal*.yaml`: environment configuration.
- `isaacgymenvs/cfg/train/SimToolReal*PPO.yaml`: training and asymmetric critic configuration.
- `assets/urdf/**`: robot, table, procedural object, and DexToolBench assets.
- `dextoolbench/eval.py` and `dextoolbench/eval_interactive.py`: evaluation overrides that must target the new Lab env.
- `deployment/isaac/*.py`: deployment/sim2sim wrappers that will need a Lab backend.

## Target Structure

```text
source/simtoolreal_lab/
  pyproject.toml
  simtoolreal_lab/
    __init__.py
    tasks/
      direct/
        simtoolreal/
          __init__.py
          simtoolreal_env.py
          simtoolreal_env_cfg.py
          assets.py
          object_generation.py
          observations.py
          rewards.py
          resets.py
          actions.py
          curriculum.py
          dextoolbench_eval_cfg.py
    tests/
      test_object_generation.py
      test_observation_parity.py
      test_reward_parity.py
      test_reset_parity.py
```

Keep the Isaac Gym implementation as the reference until parity is demonstrated.

## Components To Port

- Asset loading: Kuka+Sharpa URDF, table variants, DexToolBench object URDFs, goal object visualization/collision settings, friction and mass properties.
- Procedural tool generation: handle/head URDF generation, object size distributions, generated-asset caching, VHACD/collision choices.
- Robot/articulation state access: joint positions, joint velocities, rigid body states, palm frame, fingertip bodies, DOF force sensors if later enabled.
- Object pose state access: root pose, linear/angular velocities, object scale multiplier, goal pose, fixed goal trajectory state.
- Reset logic: table/object pose noise, random object rotation, fixed object pose, reset buffers, success counters, good-state reset buffers.
- Reward logic: lifting reward, fingertip distance delta reward, keypoint reward, success bonus, action penalties, object velocity penalties, reset reasons.
- Observation construction: clean critic state, noisy/delayed policy observations, observation/action queues, object-state delay, joint velocity noise, dropout/curriculum scales.
- Action application: arm relative target integration, hand target scaling, moving averages, target clamping, privileged torque actions.
- Asymmetric critic state: `stateList` versus `obsList`, central-value config, dict observation wiring.
- Domain randomization: Isaac Gym DR config equivalents for observations/actions, gravity, DOF properties, rigid body mass, rigid shape friction/restitution, random forces and velocity impulses.

## Known Pitfalls

- Quaternion convention: Isaac Gym tensors here use `xyzw`; Isaac Lab/Isaac Sim APIs often expose `wxyz` in some interfaces. Add explicit conversion tests at every boundary.
- Joint order: current policy assumes a 29-action order with 7 arm DOFs followed by Sharpa hand DOFs. The asset probe found that Isaac Lab imports all 29 expected joint names, but the raw articulation order does not match the Isaac Gym policy order after the first eight joints. The Lab task must build explicit action/observation gather-scatter maps.
- GPU physics buffers: Isaac Gym exposes wrapped root/DOF/rigid-body tensors directly; Isaac Lab uses different scene/articulation buffers and update timing.
- Collision geometry: Isaac Gym URDF loading, `replace_cylinder_with_capsule`, VHACD, convex decomposition, and collision filters may not match Isaac Sim import behavior.
- Contact stability: fingertip/table/tool contacts are central to reward and reset behavior; solver iterations, contact offsets, rest offsets, and material combine modes must be tuned and tested.
- Procedural assets: temporary URDF generation currently happens during env creation. In Lab, prefer deterministic generated assets under a cache directory with hashes.
- Camera/video behavior: current camera sensors are optional and tied to Gym viewer behavior; Lab rendering should be isolated from training throughput.
- Checkpoint compatibility: existing checkpoints are rl_games/PyTorch policies and env-state dictionaries from Isaac Gym. Do not assume env-state restore compatibility in Lab.

## Phased Implementation

1. Asset/state probe: done for the real SHARPA robot, table, and generated tool on GPU 0 and GPU 1.
2. Package skeleton: create `source/simtoolreal_lab` with installable Isaac Lab extension metadata.
3. Minimal `DirectRLEnv`: reuse the probe's asset configs, state access, and 140-dimensional observation candidate in a proper Isaac Lab environment class.
4. Joint mapping: add explicit Isaac Gym policy-order to Isaac Lab articulation-order maps and tests.
5. Reset parity: reproduce deterministic reset with all noise disabled and compare object, table, robot, and goal states.
6. Observation parity: compare `compute_observation` outputs for a saved deterministic state.
7. Reward parity: compare individual reward components for fixed states and actions.
8. Action parity: compare one-step joint target updates from identical observations/actions.
9. Evaluation parity: run one DexToolBench task with a pretrained policy and compare success trajectory qualitatively and numerically.
10. Training bring-up: train at small `num_envs`, then scale.
11. Distributed training: only after single-GPU Lab training is stable.

## Tests Needed

- Import test: Lab extension imports without starting simulation.
- Asset test: all robot/table/object assets load; expected joint/body names exist.
- Quaternion test: round-trip conversions preserve poses.
- Joint-order test: policy action index maps to the intended joint.
- Reset test: deterministic reset matches a saved Isaac Gym reference within tolerance.
- Observation test: policy observation and critic state dimensions and values match reference cases.
- Reward test: each reward component matches reference cases.
- Step test: one action produces expected target tensor and no NaNs.
- DexToolBench smoke test: one task, one episode, one GPU.
- Scaling smoke test: increasing env count does not change observation/reward shapes or crash.
