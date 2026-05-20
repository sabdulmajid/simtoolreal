# Next Steps

## Immediate Commands

The Python 3.8 compatibility environment now exists and can be recreated:

```bash
bash scripts/create_compat_env.sh
bash scripts/run_in_compat_env.sh python scripts/check_system.py
```

The next real blocker is Isaac Gym Preview 4. Install it only from an extracted local package:

```bash
export ISAAC_GYM_ROOT=/path/to/extracted/isaacgym
bash scripts/create_compat_env.sh
bash scripts/validate_compat_env.sh
```

Pass criterion:

```text
reports/system_check.json -> imports -> Isaac Gym -> import_ok: true
```

After Isaac Gym imports, rerun the smoke path in order:

```bash
bash scripts/run_in_compat_env.sh bash scripts/run_pretrained_eval.sh
bash scripts/run_in_compat_env.sh bash scripts/run_dextoolbench_eval.sh
bash scripts/run_in_compat_env.sh bash scripts/train_scratch_smoke.sh
bash scripts/run_in_compat_env.sh bash scripts/finetune_smoke.sh
```

Download one DexToolBench task only after the Isaac Gym import passes:

```bash
bash scripts/run_in_compat_env.sh python download_dextoolbench_data.py \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down
```

## If Isaac Gym Imports But Runtime Fails

Capture the exact runtime failure:

```bash
GPU_ID=0 NUM_ENVS=128 NUM_BLOCKS=1 MAX_EPOCHS=1 \
  bash scripts/run_in_compat_env.sh bash scripts/train_scratch_smoke.sh
```

Then update:

- `docs/known_blockers.md`
- `docs/current_status.md`
- `reports/experiment_summary.md`

Do not run larger `num_envs` until the 128/768 env smoke path works.

## If PyTorch Fails On Blackwell

The current Python 3.8 torch wheel is 2.4.1+cu121 and does not advertise `sm_120`. If runtime logs show an unsupported architecture or kernel image failure, the next milestone is a clean compatibility container/conda path with a Blackwell-capable torch build, not algorithm changes.

## When To Scale

Run scaling only after one smoke training or evaluation path succeeds:

```bash
python scripts/profile_gpu_scaling.py --gpu-id 0 --num-envs 12288 --smoke-test
DRY_RUN=1 GPU_ID=1 bash scripts/sweep_num_envs.sh
```

Execute real sweeps only after the dry-run commands and small smoke test are clean.
