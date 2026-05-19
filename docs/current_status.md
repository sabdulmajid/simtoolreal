# Current Status

Last validated from this workspace on 2026-05-19T00:31Z.

## What Works

- Repository inspection completed without changing RL or environment logic.
- `scripts/check_system.py` runs and writes `reports/system_check.json`.
- GPU discovery works through PyTorch and `nvidia-smi`.
- Both GPUs are visible as NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, about 97887 MiB each, compute capability `sm_120`.
- Safe wrapper scripts exist for asset download, pretrained eval, DexToolBench eval, scratch smoke, finetune smoke, independent GPU launch, env-count sweep, and profile dry-runs.
- Experiment wrappers append structured rows to `reports/experiment_runs.jsonl` and regenerate `reports/experiment_results.csv`, `reports/experiment_results.json`, and `reports/experiment_summary.md`.
- GPU scaling dry-run command generation works for 12288 envs through `scripts/profile_gpu_scaling.py`.
- GPU1 sweep dry-run command generation prints 12288, 24576, and 49152 env commands with `num_blocks=6`.

## What Fails In The Current Shell

- `isaacgym` does not import.
- `isaaclab` and `omni.isaac.lab` do not import in this shell.
- `rl_games.torch_runner` does not import because the repo-local `rl_games` fork is not installed into the active environment.
- `tyro` is missing, so asset download wrappers cannot run in this shell.
- `uv` and `python3.8` are not on `PATH` in this shell.
- Pretrained policy files are missing: `pretrained_policy/config.yaml` and `pretrained_policy/model.pth`.
- DexToolBench data is missing: `dextoolbench/data/`.
- Pretrained eval, DexToolBench eval, scratch training, and finetuning have not started successfully in this shell.

## Evidence

- System report: `reports/system_check.json`
- Asset download report: `reports/download_assets.json`
- Pretrained eval report: `reports/run_pretrained_eval.json`
- DexToolBench eval report: `reports/run_dextoolbench_eval.json`
- Scratch smoke report: `reports/train_scratch_smoke.json`
- Finetune smoke report: `reports/finetune_smoke.json`
- GPU scaling dry-run JSON: `reports/gpu_scaling_results.json`
- GPU scaling dry-run CSV: `reports/gpu_scaling_results.csv`
- GPU scaling dry-run summary: `reports/gpu_scaling_summary.md`
- Experiment result CSV: `reports/experiment_results.csv`
- Experiment result JSON: `reports/experiment_results.json`
- Experiment summary: `reports/experiment_summary.md`
- Experiment discipline docs: `docs/experiment_discipline.md`

Logs from the latest validation run:

- `logs/download_assets_20260519T002947Z.log`
- `logs/run_pretrained_eval_20260519T002953Z.log`
- `logs/run_dextoolbench_eval_20260519T003001Z.log`
- `logs/train_scratch_smoke_20260519T003016Z.log`
- `logs/finetune_smoke_20260519T003028Z.log`
- `logs/sweep_num_envs_12288_gpu1_20260519T003048Z.log`
- `logs/sweep_num_envs_24576_gpu1_20260519T003050Z.log`
- `logs/sweep_num_envs_49152_gpu1_20260519T003052Z.log`

## Current Best Command

The only validated command in the current shell is:

```bash
python scripts/check_system.py
```

The next productive command is to create and activate the documented Python 3.8 Isaac Gym environment, then rerun the system check.

Focused debug update: the first hard failure is `python -c 'import isaacgym'`, which fails with `ModuleNotFoundError`. This is an external environment/install blocker, not an RL algorithm, Hydra config, headless rendering, asset path, or training runtime failure.

Verification command after environment repair:

```bash
python -c 'import isaacgym; print(isaacgym.__file__)'
```

## Current Largest Stable `num_envs`

No training smoke test has started in this shell. Largest stable `num_envs` is therefore unknown. The scaling harness has only dry-run evidence: `scripts/profile_gpu_scaling.py` wrote a 12288-env dry-run report, and `scripts/sweep_num_envs.sh` printed GPU1 dry-run commands for 12288, 24576, and 49152 envs.

## Evaluation And Training Status

- Pretrained eval works: no, blocked by missing pretrained files and missing Isaac Gym environment.
- DexToolBench eval works: no, blocked by missing Isaac Gym environment and assets.
- Scratch training starts: no, blocked by missing Isaac Gym environment.
- Finetuning starts: no, blocked by missing checkpoint and missing Isaac Gym environment.
- True multi-GPU distributed training: unverified and not claimed.
- Isaac Lab support: unverified and not claimed.
