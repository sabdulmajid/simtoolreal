# Experiment Discipline

This repository now treats every wrapper launch as a recorded run. The goal is
not to claim performance before measurement; it is to make every success,
failure, and dry-run auditable.

## Result Files

The wrapper scripts append normalized rows to:

```bash
reports/experiment_runs.jsonl
```

The aggregate reports are regenerated automatically when a wrapper writes a
status report:

```bash
reports/experiment_results.csv
reports/experiment_results.json
reports/experiment_summary.md
```

You can regenerate the aggregate files manually:

```bash
python scripts/aggregate_results.py
```

## Recorded Fields

Each row records the run command, git commit, dirty git status, environment
name, Python version, torch/CUDA versions, GPU id/name, `num_envs`, seed, mode,
checkpoint path, start/end times, exit code, log path, peak VRAM, GPU
utilization, throughput/FPS if present in logs, and final metric/reward/loss if
present in logs.

Failed runs are retained with their failure status. Dependency failures,
missing assets, config errors, and dry-runs are not silently dropped.

## Smoke Tests

Use smoke mode before launching expensive jobs:

```bash
SMOKE_TEST=1 bash scripts/download_assets.sh
SMOKE_TEST=1 bash scripts/run_pretrained_eval.sh
SMOKE_TEST=1 NUM_ENVS=768 NUM_BLOCKS=6 MAX_EPOCHS=1 bash scripts/train_scratch_smoke.sh
SMOKE_TEST=1 NUM_ENVS=768 NUM_BLOCKS=6 MAX_EPOCHS=1 bash scripts/finetune_smoke.sh
python scripts/profile_gpu_scaling.py --gpu-id 0 --num-envs 12288 --smoke-test
```

In the current shell, these commands are expected to fail before Isaac Gym
runtime because the active environment does not import `isaacgym`. That failure
is still a valid recorded result.

## Reproducing The Best Run

Regenerate the summary:

```bash
python scripts/aggregate_results.py
```

Open:

```bash
reports/experiment_summary.md
```

If a successful run exists, the `Best Recorded Run` section includes the exact
command to rerun plus the git commit, dirty status, environment name, Python
version, torch/CUDA versions, GPU id/name, log path, and metrics. Reproduce
that run from the same git commit and compatible environment before comparing
new results.

If no successful run exists, do not select a best run from dry-run or failed
rows. Fix the environment blocker first, then rerun the smoke tests.
