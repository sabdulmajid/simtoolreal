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

Choose one path before further scaling:

- Original Isaac Gym on non-Blackwell GPU: quickest way to reproduce the old stack if another GPU is available.
- Original Isaac Gym on Blackwell with a custom Python 3.8 torch build: preserves the repo stack, but may require source builds and careful ABI testing.
- Scoped Isaac Lab migration: likely the clean Blackwell path, but it is a simulator migration and should not be described as original Isaac Gym reproduction.

## If A Blackwell-Capable Python 3.8 Torch Build Is Installed

Rerun in this exact order:

```bash
bash scripts/validate_compat_env.sh
bash scripts/run_in_compat_env.sh python scripts/check_system.py
GPU_ID=0 NUM_ENVS=128 NUM_BLOCKS=1 MAX_EPOCHS=1 \
  bash scripts/run_in_compat_env.sh bash scripts/train_scratch_smoke.sh
```

Only then rerun evaluation and finetuning smoke tests.

## When To Scale

Run scaling only after one smoke training or evaluation path succeeds:

```bash
python scripts/profile_gpu_scaling.py --gpu-id 0 --num-envs 12288 --smoke-test
DRY_RUN=1 GPU_ID=1 bash scripts/sweep_num_envs.sh
```

Execute real sweeps only after the dry-run commands and small smoke test are clean.
