# Blackwell Isaac Lab Path

Last validated from this workspace on 2026-05-20T09:27Z.

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

Modern Isaac Lab smoke path:

- wrapper: `scripts/run_blackwell_isaaclab_smoke.sh`
- validator: `scripts/validate_blackwell_isaaclab.py`
- Python: `3.11.14`
- torch: `2.10.0+cu128`
- torch CUDA arch list includes `sm_120`
- Isaac Lab import path: `/pub7/neel/vlm/IsaacLab/source/isaaclab`
- Isaac Sim import path: `/pub7/neel/miniconda3/envs/isaaclab2/lib/python3.11/site-packages/isaacsim`
- task: `Isaac-Cartpole-Direct-v0`
- smoke size: 16 envs, 16 steps

GPU 0 report:

- report: `reports/blackwell_isaaclab_smoke.json`
- metrics: `reports/blackwell_isaaclab_validation.json`
- status: `success_with_dependency_conflicts`
- throughput: about 1031 FPS
- peak VRAM: 25392 MiB
- peak GPU utilization: 48%

GPU 1 report:

- report: `reports/blackwell_isaaclab_smoke_gpu1.json`
- metrics: `reports/blackwell_isaaclab_validation_gpu1.json`
- status: `success_with_dependency_conflicts`
- throughput: about 1056 FPS
- peak VRAM: 24293 MiB
- peak GPU utilization: 47%

## Important Caveats

The current `isaaclab2` environment is not clean. `pip check` reports dependency
conflicts, including Isaac Sim packages expecting torch `2.7.0` while the active
environment has torch `2.10.0+cu128`. The bounded smoke test succeeds, but this
is not yet a production-quality reproducibility environment.

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
bash scripts/run_blackwell_isaaclab_smoke.sh
```

GPU 1:

```bash
GPU_ID=1 \
REPORT_PATH=reports/blackwell_isaaclab_smoke_gpu1.json \
METRICS_PATH=reports/blackwell_isaaclab_validation_gpu1.json \
bash scripts/run_blackwell_isaaclab_smoke.sh
```

Override paths when validating a clean environment:

```bash
ISAACLAB_CONDA_ENV=isaaclab-clean \
ISAACLAB_ROOT=/path/to/IsaacLab \
GPU_ID=0 \
bash scripts/run_blackwell_isaaclab_smoke.sh
```

## Recommended Fix

Proceed, but stop trying to make Isaac Gym Preview 4 the Blackwell scaling
backend. The clean next step is:

1. Create a clean Isaac Lab environment with a Blackwell-capable torch build.
2. Make `pip check` pass or explicitly document the remaining Isaac Sim package
   pins.
3. Rerun `scripts/run_blackwell_isaaclab_smoke.sh` on GPU 0 and GPU 1.
4. Start a scoped Isaac Lab port of the minimal ToolPose tracking environment.

Do not claim SimToolReal reproduction or scaling until the ToolPose environment
itself has been ported and matched against the Isaac Gym behavior.
