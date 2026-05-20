# GPU Scaling Plan

The safe path for this repo is independent single-GPU runs, one process per GPU, controlled with `CUDA_VISIBLE_DEVICES`. Do not assume working distributed multi-GPU training.

Independent jobs come first because this repository's SimToolReal launcher explicitly sets `multi_gpu=False`; distributed code exists in the vendored `rl_games`, but the SimToolReal integration path is not verified and contains `cuda:0` assumptions.

## Verified Code Path

- `isaacgymenvs/launch_training.py` launches `python -m isaacgymenvs.train`.
- Default `num_envs` is 24576.
- Default `num_blocks` is 6.
- `num_envs % num_blocks == 0` is enforced before launch.
- `sapg_block_size = num_envs // num_blocks`.
- The launcher hard-codes `multi_gpu=False`.
- Base config defaults are `sim_device=cuda:0`, `rl_device=cuda:0`, and `graphics_device_id=0`.
- W&B defaults to active in the launcher and uses `wandb_entity="tylerlum"` unless overridden.

The vendored `rl_games` has distributed code, but the SimToolReal path is not verified for it. The fork also contains `cuda:0` debug assumptions in distributed branches, so multi-process training should not be treated as supported.

## Sweep Matrix

With `num_blocks=6`:

| num_envs | SAPG block size | Divisible by 6 |
| ---: | ---: | --- |
| 12288 | 2048 | yes |
| 24576 | 4096 | yes |
| 49152 | 8192 | yes |

## Helper Scripts

GPU 0:

```bash
WANDB_ACTIVATE=False NUM_ENVS=24576 scripts/launch_train_gpu0.sh
```

GPU 1:

```bash
WANDB_ACTIVATE=False NUM_ENVS=24576 scripts/launch_train_gpu1.sh
```

Run simultaneous independent jobs:

```bash
WANDB_ACTIVATE=False NUM_ENVS=24576 EXPERIMENT_NAME=1_gpu0_run scripts/launch_train_gpu0.sh
WANDB_ACTIVATE=False NUM_ENVS=24576 EXPERIMENT_NAME=1_gpu1_run scripts/launch_train_gpu1.sh
```

Dry-run the sweep:

```bash
DRY_RUN=1 GPU_ID=0 scripts/sweep_num_envs.sh
DRY_RUN=1 GPU_ID=1 scripts/sweep_num_envs.sh
```

Execute the sweep after smoke tests pass:

```bash
DRY_RUN=0 GPU_ID=0 WANDB_ACTIVATE=False scripts/sweep_num_envs.sh
```

Dry-run profile commands:

```bash
python scripts/profile_gpu_scaling.py --gpu-id 0
```

Run short profiles only after `bash scripts/validate_compat_env.sh` passes and a 128-env smoke test runs successfully:

```bash
python scripts/profile_gpu_scaling.py --gpu-id 0 --max-epochs 3 --run
```

Dry-run/profile reports:

```text
reports/gpu_scaling_results.csv
reports/gpu_scaling_results.json
reports/gpu_scaling_summary.md
reports/experiment_results.csv
reports/experiment_results.json
reports/experiment_summary.md
```

Scaling rows are also appended to `reports/experiment_runs.jsonl` so failed,
dry-run, and successful scaling attempts are retained in the common experiment
table.

## Metrics To Capture

- Whether Isaac Gym starts successfully on the visible GPU.
- Time to create environments.
- Epoch wall time.
- Frames per second from rl_games logs.
- Peak `nvidia-smi` memory.
- GPU utilization during rollout and update.
- Any PhysX contact-buffer, CUDA illegal-instruction, PTX, or out-of-memory failures.
- Checkpoint path produced by rl_games.

Filling VRAM is not the goal. The useful optimization target is stable environment steps per second and task progress per wall-clock hour. A larger `num_envs` setting is worse if it causes lower throughput, unstable physics, NaNs, slower policy updates, or less frequent useful checkpoints.

## Interpreting Results

- OOM: lower `num_envs`, confirm no other processes are using the GPU, and retry the next lower sweep point.
- NaNs: reduce learning pressure first; capture exact logs and do not treat the env count as stable.
- Low GPU utilization: check CPU-side Isaac Gym setup, rendering/camera settings, W&B overhead, and Python subprocess bottlenecks.
- Physics bottlenecks: inspect contact buffer errors, contact pair limits, collision geometry, and PhysX warnings.
- Slow wall-clock despite high utilization: compare env steps/sec and reward/task progress, not VRAM use.

## Failure Handling

- If 49152 envs fails with OOM, rerun 24576 and 12288 to establish a working ceiling.
- If all env counts fail before environment creation on Blackwell, stop Isaac Gym scaling work and resolve the Python 3.8 PyTorch/Blackwell compatibility blocker first.
- If only evaluation works but training fails, capture the exact Isaac Gym/PhysX error and freeze the evaluation environment separately from the training environment.

## Current Status

Do not run real scaling yet. Isaac Gym Preview 4 now imports, but torch 2.4.1+cu124 cannot run CUDA kernels on the installed Blackwell `sm_120` GPUs. The latest validation report is `reports/validate_compat_env.json` with `status: gpu_runtime_error`.
