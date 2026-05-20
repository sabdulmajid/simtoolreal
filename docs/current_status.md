# Current Status

Last validated from this workspace on 2026-05-20T10:41Z.

## What Works Now

- `scripts/download_isaacgym.sh` downloads and extracts Isaac Gym Preview 4 outside the Git repo at `/pub7/neel2/external/isaacgym_preview4/isaacgym`.
- `scripts/create_compat_env.sh` creates or reuses a repo-compatible Python 3.8 environment at `/pub7/neel2/conda_envs/simtoolreal-py38`.
- The compatibility environment uses Python 3.8.20 and torch 2.4.1+cu124.
- Isaac Gym Preview 4 installs into the compatibility environment and imports successfully.
- The repo and repo-local `rl_games` fork install editable into that environment.
- `rl_games.torch_runner` imports from `rl_games/rl_games/torch_runner.py`.
- `tyro`, Hydra, OmegaConf, gym, W&B, TensorBoard, OpenCV, tqdm, requests, `pytorch3d`, `viser`, `trimesh`, `transforms3d`, `pyvirtualdisplay`, `pytorch-kinematics`, and the smoke wrapper dependencies are installed.
- `scripts/check_system.py` runs inside the compatibility environment and writes `reports/system_check.json`.
- `scripts/download_assets.sh` successfully downloaded the pretrained policy.
- `pretrained_policy/config.yaml` and `pretrained_policy/model.pth` exist locally.
- The pretrained policy directory is ignored by Git, so the 395 MB checkpoint should not be committed.
- The experiment ledger and aggregator record setup, Isaac Gym download, asset download, dry-runs, and failures without dropping failed runs.
- The aggregate summary now refuses to report a "best run" until a measured evaluation, training, or profile run succeeds.
- Pretrained evaluation, DexToolBench evaluation, scratch smoke, and finetune smoke now reach CUDA execution rather than stopping at missing imports.
- A bounded Isaac Lab Cartpole smoke test runs on both Blackwell GPUs through `scripts/run_blackwell_isaaclab_smoke.sh`.
- `scripts/create_isaaclab_blackwell_env.sh` creates/reuses a clean Blackwell-targeted Isaac Lab environment at `/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell`.
- The clean Isaac Lab path uses Python 3.11.15, Isaac Sim 5.1, and torch 2.7.0+cu128, whose CUDA arch list includes `sm_120` and `compute_120`.
- GPU 0 and GPU 1 each completed a 16-env, 16-step Isaac Lab Cartpole smoke test with structured reports.
- GPU 0 also completed a bounded 12,288-env, 32-step Isaac Lab Cartpole scale smoke.
- `scripts/run_isaaclab_toolpose_asset_probe.sh` launches Isaac Lab and loads the actual SimToolReal SHARPA robot URDF, table URDF, and a procedurally generated handle-head tool URDF.
- The ToolPose asset/state probe completed on GPU 0 and GPU 1. It found all 29 expected SHARPA/KUKA joints, the palm body, all five fingertip bodies, the tool root state tensor, and a finite 140-dimensional policy-observation candidate tensor.

## What Is Still Blocked

- Torch 2.4.1+cu124 in the Python 3.8 environment reports CUDA arch support only through `sm_90`; both GPUs are Blackwell `sm_120`.
- A minimal torch CUDA kernel fails on both visible GPUs with `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
- Pretrained evaluation, DexToolBench evaluation, scratch smoke, and finetune smoke all fail at CUDA execution with the same Blackwell/PyTorch runtime incompatibility.
- Isaac Lab does not import in this compatibility environment. That is expected and is not the current target.
- DexToolBench data is still missing under `dextoolbench/data/`.
- No real SimToolReal training, pretrained evaluation, DexToolBench evaluation, reward, or checkpoint compatibility has been demonstrated on Blackwell yet.
- The clean Isaac Lab environment still has dependency conflicts under `pip check`. The current probe succeeds, but reports are recorded as `success_with_dependency_conflicts`.
- The `/pub7/neel2/isaaclab_ws/IsaacLab` checkout fails before simulation because its app launcher expects a missing `apps/isaacsim_5` layout. The working smoke path currently uses `/pub7/neel/vlm/IsaacLab`.
- SimToolReal has not been fully ported to Isaac Lab yet. The passing ToolPose probe is asset/state-port evidence, not a complete environment, trained policy, or original SimToolReal reproduction.
- Isaac Lab imports the same 29 joint names as the Isaac Gym reference, but the joint order differs after the first eight joints. The Lab port must use an explicit action/observation index map rather than assuming raw articulation order matches Isaac Gym.

## Current Evidence

- System report: `reports/system_check.json`
- Isaac Gym download report: `reports/download_isaacgym.json`
- Compatibility env creation report: `reports/create_compat_env.json`
- Compatibility env validation report: `reports/validate_compat_env.json`
- Asset download report: `reports/download_assets.json`
- Pretrained eval report: `reports/run_pretrained_eval.json`
- DexToolBench eval report: `reports/run_dextoolbench_eval.json`
- Scratch smoke report: `reports/train_scratch_smoke.json`
- Finetune smoke report: `reports/finetune_smoke.json`
- Experiment ledger: `reports/experiment_runs.jsonl`
- Experiment CSV: `reports/experiment_results.csv`
- Experiment JSON: `reports/experiment_results.json`
- Experiment summary: `reports/experiment_summary.md`
- Blackwell Isaac Lab GPU0 report: `reports/blackwell_isaaclab_smoke.json`
- Blackwell Isaac Lab GPU0 metrics: `reports/blackwell_isaaclab_validation.json`
- Blackwell Isaac Lab GPU1 report: `reports/blackwell_isaaclab_smoke_gpu1.json`
- Blackwell Isaac Lab GPU1 metrics: `reports/blackwell_isaaclab_validation_gpu1.json`
- Clean Isaac Lab env report: `reports/create_isaaclab_blackwell_env.json`
- Clean Isaac Lab GPU0 report: `reports/blackwell_isaaclab_smoke_clean_gpu0.json`
- Clean Isaac Lab GPU0 metrics: `reports/blackwell_isaaclab_validation_clean_gpu0.json`
- Clean Isaac Lab GPU1 report: `reports/blackwell_isaaclab_smoke_clean_gpu1.json`
- Clean Isaac Lab GPU1 metrics: `reports/blackwell_isaaclab_validation_clean_gpu1.json`
- Clean Isaac Lab 12,288-env GPU0 report: `reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json`
- Clean Isaac Lab 12,288-env GPU0 metrics: `reports/blackwell_isaaclab_validation_clean_gpu0_12288.json`
- ToolPose asset/state GPU0 report: `reports/isaaclab_toolpose_asset_probe_gpu0.json`
- ToolPose asset/state GPU0 metrics: `reports/isaaclab_toolpose_asset_probe_metrics_gpu0.json`
- ToolPose asset/state GPU1 report: `reports/isaaclab_toolpose_asset_probe_gpu1.json`
- ToolPose asset/state GPU1 metrics: `reports/isaaclab_toolpose_asset_probe_metrics_gpu1.json`
- Blackwell Isaac Lab path notes: `docs/blackwell_isaaclab_path.md`

Latest relevant logs:

- `logs/download_isaacgym_20260520T063819Z.log`
- `logs/create_compat_env_20260520T063132Z.log`
- `logs/validate_compat_env_20260520T063825Z.log`
- `logs/download_assets_20260520T054540Z.log`
- `logs/run_pretrained_eval_20260520T063847Z.log`
- `logs/run_dextoolbench_eval_20260520T063906Z.log`
- `logs/train_scratch_smoke_20260520T063928Z.log`
- `logs/finetune_smoke_20260520T063948Z.log`
- `logs/blackwell_isaaclab_smoke_20260520T092605Z.log`
- `logs/blackwell_isaaclab_smoke_20260520T092646Z.log`
- `logs/create_isaaclab_blackwell_env_20260520T101502Z.log`
- `logs/blackwell_isaaclab_smoke_clean_gpu0_20260520T101612Z.log`
- `logs/blackwell_isaaclab_smoke_clean_gpu1_20260520T101643Z.log`
- `logs/blackwell_isaaclab_smoke_clean_gpu0_12288_20260520T101730Z.log`
- `logs/isaaclab_toolpose_asset_probe_20260520T103520Z.log`
- `logs/isaaclab_toolpose_asset_probe_20260520T104129Z.log`

## Current Best Original-Stack Command

Use the compatibility environment wrapper:

```bash
bash scripts/run_in_compat_env.sh python scripts/check_system.py
```

Expected current result: Python 3.8, torch 2.4.1+cu124, Isaac Gym import OK, repo-local `rl_games` import OK, pretrained policy files present, and torch CUDA kernel smoke failed on Blackwell `sm_120`.

## Current Best Blackwell Runtime Commands

```bash
bash scripts/create_isaaclab_blackwell_env.sh
ACCEPT_EULA=Y ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  bash scripts/run_blackwell_isaaclab_smoke.sh
ACCEPT_EULA=Y GPU_ID=1 ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/blackwell_isaaclab_smoke_clean_gpu1.json \
  METRICS_PATH=reports/blackwell_isaaclab_validation_clean_gpu1.json \
  bash scripts/run_blackwell_isaaclab_smoke.sh
ACCEPT_EULA=Y NUM_ENVS=12288 STEPS=32 ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json \
  METRICS_PATH=reports/blackwell_isaaclab_validation_clean_gpu0_12288.json \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

Latest clean-env results:

| report | status | num_envs | throughput_fps | peak_vram_mib | peak_gpu_util_percent |
| --- | --- | ---: | ---: | ---: | ---: |
| `reports/blackwell_isaaclab_smoke_clean_gpu0.json` | `success_with_dependency_conflicts` | 16 | 1133.56 | 33254 | 52 |
| `reports/blackwell_isaaclab_smoke_clean_gpu1.json` | `success_with_dependency_conflicts` | 16 | 841.08 | 31425 | 58 |
| `reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json` | `success_with_dependency_conflicts` | 12288 | 798156.94 | 34858 | 39 |

Latest ToolPose asset/state probe results:

| report | status | num_envs | observation_shape | robot_joints | throughput_env_steps_per_sec | peak_vram_mib | peak_gpu_util_percent |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `reports/isaaclab_toolpose_asset_probe_gpu0.json` | `success_with_dependency_conflicts` | 2 | `[2, 140]` | 29 | 87.93 | 38256 | 37 |
| `reports/isaaclab_toolpose_asset_probe_gpu1.json` | `success_with_dependency_conflicts` | 1 | `[1, 140]` | 29 | 24.90 | 31627 | 56 |

## Next Required External Action

The Isaac Gym dependency is now installed. The current external blocker is the Python 3.8 PyTorch/Blackwell conflict:

```text
Isaac Gym Preview 4 requires Python <3.9.
Python 3.8 PyTorch wheels available here stop at torch 2.4.1.
torch 2.4.1+cu121 and torch 2.4.1+cu124 do not advertise sm_120 and fail CUDA kernels on the installed Blackwell GPUs.
```

The clean next milestone is no longer more original-stack scaling. Original Isaac Gym on Blackwell remains blocked, while the clean Isaac Lab Blackwell runtime path now loads the real SimToolReal ToolPose assets and state tensors. Proceed by turning the asset/state probe into a minimal Isaac Lab ToolPose `DirectRLEnv`, with explicit joint-order mapping and parity tests against the Isaac Gym implementation.
