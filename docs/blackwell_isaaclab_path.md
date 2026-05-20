# Blackwell Isaac Lab Path

Last validated from this workspace on 2026-05-20T10:17Z.

## Decision

The original SimToolReal Isaac Gym Preview 4 stack is blocked on the installed
Blackwell GPUs because its Python 3.8-compatible PyTorch build does not contain
`sm_120` CUDA kernels. This is not a dead end for the research project. The
machine has a modern Isaac Lab / PyTorch path that can run CUDA kernels and a
bounded Isaac Lab simulator task on both GPUs.

This document is intentionally narrow. It does not claim that SimToolReal has
been ported to Isaac Lab yet.

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

## Important Caveats

The clean environment is usable for bounded Isaac Lab runtime validation, but it
is not fully dependency-clean. `pip check` reports:

```text
fastapi 0.115.7 has requirement starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1.
```

This appears to be an upstream packaging conflict between Isaac Sim 5.1's
FastAPI pin and this Isaac Lab checkout's Starlette pin. The bounded smoke tests
therefore report `success_with_dependency_conflicts`, not plain `success`.

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
2. Start a scoped Isaac Lab port of the minimal ToolPose tracking environment.
3. Validate parity against Isaac Gym observations, rewards, reset behavior,
   action application, and robot/object state tensors before training.

Do not claim SimToolReal reproduction or scaling until the ToolPose environment
itself has been ported and matched against the Isaac Gym behavior.
