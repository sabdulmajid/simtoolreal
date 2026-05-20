# Next Steps

## Immediate Commands

The Python 3.8 compatibility environment now exists and can be recreated:

```bash
bash scripts/download_isaacgym.sh
bash scripts/create_compat_env.sh
bash scripts/run_in_compat_env.sh python scripts/check_system.py
```

The next real blocker is not missing Isaac Gym anymore. Isaac Gym Preview 4 is installed and importable in the compatibility environment. The failing validation command is:

```bash
bash scripts/validate_compat_env.sh
```

Current result:

```text
status: gpu_runtime_error
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

This happens before any meaningful training can run. Do not launch larger scaling jobs until this command passes.

The smoke path has been rerun after Isaac Gym installation:

```bash
bash scripts/run_in_compat_env.sh bash scripts/run_pretrained_eval.sh
bash scripts/run_in_compat_env.sh bash scripts/run_dextoolbench_eval.sh
bash scripts/run_in_compat_env.sh bash scripts/train_scratch_smoke.sh
bash scripts/run_in_compat_env.sh bash scripts/finetune_smoke.sh
```

All four reach CUDA execution and fail with the same Blackwell/PyTorch kernel incompatibility.

Download one DexToolBench task only after the CUDA validation passes:

```bash
bash scripts/run_in_compat_env.sh python download_dextoolbench_data.py \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down
```

## Compatibility Decision

The current stack conflict is:

```text
Isaac Gym Preview 4 requires Python <3.9.
Python 3.8 PyTorch wheels available here stop at torch 2.4.1.
torch 2.4.1+cu121 and torch 2.4.1+cu124 do not advertise sm_120 and fail CUDA kernels on Blackwell.
```

The chosen path is Isaac Lab for Blackwell. The evidence is now:

- original Isaac Gym smoke commands reach CUDA and fail on `sm_120`;
- the clean Isaac Lab environment uses Python 3.11.15 and torch 2.7.0+cu128;
- torch advertises `sm_120` and runs a CUDA kernel on Blackwell;
- Isaac Lab Cartpole smoke runs on GPU 0 and GPU 1;
- a bounded 12,288-env Isaac Lab Cartpole smoke runs on GPU 0;
- the clean Isaac Lab environment still has one FastAPI/Starlette dependency conflict.

Do not spend more time scaling the original Isaac Gym stack on this Blackwell
machine.

## Immediate Isaac Lab Commands

Validate GPU 0:

```bash
bash scripts/create_isaaclab_blackwell_env.sh
ACCEPT_EULA=Y ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

Validate GPU 1:

```bash
ACCEPT_EULA=Y GPU_ID=1 ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/blackwell_isaaclab_smoke_clean_gpu1.json \
  METRICS_PATH=reports/blackwell_isaaclab_validation_clean_gpu1.json \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

Validate one bounded scale point:

```bash
ACCEPT_EULA=Y NUM_ENVS=12288 STEPS=32 \
  ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json \
  METRICS_PATH=reports/blackwell_isaaclab_validation_clean_gpu0_12288.json \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

Expected current result: `success_with_dependency_conflicts`. This means the
simulator and CUDA smoke worked, but the environment still has the documented
FastAPI/Starlette dependency conflict.

## Next Milestone

Begin a scoped port of the minimal ToolPose tracking environment to Isaac Lab.
Do not port the full repo yet. The first port milestone should only cover asset
loading, reset, observation/action plumbing, reward parity, and a deterministic
one-step smoke test.

## If A Blackwell-Capable Python 3.8 Torch Build Is Installed

Rerun in this exact order:

```bash
bash scripts/validate_compat_env.sh
bash scripts/run_in_compat_env.sh python scripts/check_system.py
GPU_ID=0 NUM_ENVS=128 NUM_BLOCKS=1 MAX_EPOCHS=1 \
  bash scripts/run_in_compat_env.sh bash scripts/train_scratch_smoke.sh
```

Only then rerun evaluation and finetuning smoke tests. This remains a fallback
path for original Isaac Gym reproduction on compatible hardware, not the
recommended Blackwell path.

## When To Scale

Run scaling only after one smoke training or evaluation path succeeds:

```bash
python scripts/profile_gpu_scaling.py --gpu-id 0 --num-envs 12288 --smoke-test
DRY_RUN=1 GPU_ID=1 bash scripts/sweep_num_envs.sh
```

Execute real sweeps only after the dry-run commands and small smoke test are clean.
