# Blackwell Isaac Lab Path

Last validated from this workspace on 2026-05-20T10:41Z.

## Decision

The original SimToolReal Isaac Gym Preview 4 stack is blocked on the installed
Blackwell GPUs because its Python 3.8-compatible PyTorch build does not contain
`sm_120` CUDA kernels. This is not a dead end for the research project. The
machine has a modern Isaac Lab / PyTorch path that can run CUDA kernels and a
bounded Isaac Lab simulator task on both GPUs.

This document is intentionally narrow. It does not claim that SimToolReal
training or evaluation has been ported to Isaac Lab yet.

## Evidence

Original Isaac Gym compatibility path:

- Python: `3.8.20`
- torch: `2.4.1+cu124`
- torch CUDA arch list: through `sm_90`
- installed GPUs: NVIDIA RTX PRO 6000 Blackwell, `sm_120`
- failing reports:
  - `reports/system_check.json`
  - `reports/validate_compat_env.json`
  - `reports/run_pretrained_eval.json`
  - `reports/run_dextoolbench_eval.json`
  - `reports/train_scratch_smoke.json`
  - `reports/finetune_smoke.json`

Clean Isaac Lab smoke path:

- environment script: `scripts/create_isaaclab_blackwell_env.sh`
- environment path: `/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell`
- wrapper: `scripts/run_blackwell_isaaclab_smoke.sh`
- validator: `scripts/validate_blackwell_isaaclab.py`
- Python: `3.11.15`
- torch: `2.7.0+cu128`
- torch CUDA arch list includes `sm_120` and `compute_120`
- Isaac Lab import path: `/pub7/neel/vlm/IsaacLab/source/isaaclab`
- Isaac Sim import path: `/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell/lib/python3.11/site-packages/isaacsim`
- task: `Isaac-Cartpole-Direct-v0`
- smoke size: 16 envs, 16 steps

GPU 0 report:

- report: `reports/blackwell_isaaclab_smoke_clean_gpu0.json`
- metrics: `reports/blackwell_isaaclab_validation_clean_gpu0.json`
- status: `success_with_dependency_conflicts`
- throughput: about 1134 FPS
- peak VRAM: 33254 MiB
- peak GPU utilization: 52%

GPU 1 report:

- report: `reports/blackwell_isaaclab_smoke_clean_gpu1.json`
- metrics: `reports/blackwell_isaaclab_validation_clean_gpu1.json`
- status: `success_with_dependency_conflicts`
- throughput: about 841 FPS
- peak VRAM: 31425 MiB
- peak GPU utilization: 58%

Bounded scale smoke:

- report: `reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json`
- metrics: `reports/blackwell_isaaclab_validation_clean_gpu0_12288.json`
- GPU: 0
- smoke size: 12,288 envs, 32 steps
- status: `success_with_dependency_conflicts`
- throughput: about 798,157 env-steps/sec
- peak VRAM: 34858 MiB
- peak GPU utilization: 39%

SimToolReal ToolPose asset/state probe:

- wrapper: `scripts/run_isaaclab_toolpose_asset_probe.sh`
- validator: `scripts/validate_isaaclab_toolpose_assets.py`
- assets loaded:
  - `assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`
  - `assets/urdf/table_narrow.urdf`
  - generated handle-head URDF under `/tmp/simtoolreal_isaaclab_assets`
- state checks:
  - 29 expected KUKA+SHARPA joints present
  - `iiwa14_link_7` palm body present
  - five fingertip bodies present
  - tool root state tensor shape is `[num_envs, 13]`
  - action target tensor shape is `[num_envs, 29]`
  - finite policy-observation candidate tensor shape is `[num_envs, 140]`
- GPU 0 report: `reports/isaaclab_toolpose_asset_probe_gpu0.json`
- GPU 0 metrics: `reports/isaaclab_toolpose_asset_probe_metrics_gpu0.json`
- GPU 0 result: 2 envs, 4 steps, 87.93 env-steps/sec, 38256 MiB peak VRAM, 37% peak GPU utilization
- GPU 1 report: `reports/isaaclab_toolpose_asset_probe_gpu1.json`
- GPU 1 metrics: `reports/isaaclab_toolpose_asset_probe_metrics_gpu1.json`
- GPU 1 result: 1 env, 2 steps, 24.90 env-steps/sec, 31627 MiB peak VRAM, 56% peak GPU utilization

## Important Caveats

The clean environment is usable for bounded Isaac Lab runtime validation, but it
is not fully dependency-clean. Earlier clean-environment validation saw the
FastAPI/Starlette conflict:

```text
fastapi 0.115.7 has requirement starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1.
```

This appears to be an upstream packaging conflict between Isaac Sim 5.1's
FastAPI pin and this Isaac Lab checkout's Starlette pin. The bounded smoke tests
therefore report `success_with_dependency_conflicts`, not plain `success`.

The ToolPose asset probe also reports `success_with_dependency_conflicts`. On
this machine, `pip check` additionally sees the editable `simtoolreal` package
and reports missing legacy Isaac Gym dependencies in the clean Isaac Lab env.
Those missing legacy dependencies are intentionally not required for the scoped
asset/state probe, which imports the procedural generator by file path and does
not import the Isaac Gym task package.

The ToolPose probe found an important porting detail: Isaac Lab imports all 29
expected joint names, but its raw articulation order does not match the
Isaac Gym policy order after the first eight joints. The Isaac Lab environment
must build an explicit policy-index-to-articulation-index mapping before any
checkpoint, action, or observation parity claim.

The `/pub7/neel2/isaaclab_ws/IsaacLab` checkout is not currently the working
launcher path. Its app launcher expects an `apps/isaacsim_5` experience layout
that is missing in that checkout, and the smoke test fails before simulation
launch with:

```text
ModuleNotFoundError: No module named 'omni.kit.usd'
```

The wrapper currently auto-selects the working `/pub7/neel/vlm/IsaacLab`
checkout unless `ISAACLAB_ROOT` is provided.

## How To Run

GPU 0:

```bash
bash scripts/create_isaaclab_blackwell_env.sh
ACCEPT_EULA=Y ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

GPU 1:

```bash
ACCEPT_EULA=Y GPU_ID=1 ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/blackwell_isaaclab_smoke_clean_gpu1.json \
  METRICS_PATH=reports/blackwell_isaaclab_validation_clean_gpu1.json \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

Bounded scale smoke:

```bash
ACCEPT_EULA=Y NUM_ENVS=12288 STEPS=32 \
  ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json \
  METRICS_PATH=reports/blackwell_isaaclab_validation_clean_gpu0_12288.json \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

ToolPose asset/state probe on GPU 0:

```bash
ACCEPT_EULA=Y ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/isaaclab_toolpose_asset_probe_gpu0.json \
  METRICS_PATH=reports/isaaclab_toolpose_asset_probe_metrics_gpu0.json \
  NUM_ENVS=2 STEPS=4 \
  bash scripts/run_isaaclab_toolpose_asset_probe.sh
```

ToolPose asset/state probe on GPU 1:

```bash
ACCEPT_EULA=Y GPU_ID=1 ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/isaaclab_toolpose_asset_probe_gpu1.json \
  METRICS_PATH=reports/isaaclab_toolpose_asset_probe_metrics_gpu1.json \
  NUM_ENVS=1 STEPS=2 \
  bash scripts/run_isaaclab_toolpose_asset_probe.sh
```

Override paths when validating a clean environment:

```bash
ACCEPT_EULA=Y \
ISAACLAB_CONDA_ENV=/path/to/conda-env \
ISAACLAB_ROOT=/path/to/IsaacLab \
GPU_ID=0 \
bash scripts/run_blackwell_isaaclab_smoke.sh
```

The script maps `ACCEPT_EULA=Y` to Isaac Sim's required
`OMNI_KIT_ACCEPT_EULA=Y`. Users are responsible for accepting NVIDIA's EULA.

## Recommended Fix

Proceed. Isaac Gym Preview 4 should not be the Blackwell scaling backend for
this machine. The clean next step is:

1. Keep the clean Isaac Lab environment and the dependency-conflict report as
   the baseline runtime evidence.
2. Turn the passing ToolPose asset/state probe into a minimal Isaac Lab
   `DirectRLEnv`.
3. Validate parity against Isaac Gym observations, rewards, reset behavior,
   action application, and robot/object state tensors before training.

Do not claim SimToolReal reproduction or scaling until the ToolPose environment
itself has been ported and matched against the Isaac Gym behavior.
