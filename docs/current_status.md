# Current Status

Last validated from this workspace on 2026-05-20T05:53Z.

## What Works Now

- `scripts/create_compat_env.sh` creates or reuses a repo-compatible Python 3.8 environment at `/pub7/neel2/conda_envs/simtoolreal-py38`.
- The compatibility environment uses Python 3.8.20 and torch 2.4.1+cu121.
- The repo and repo-local `rl_games` fork install editable into that environment.
- `rl_games.torch_runner` imports from `rl_games/rl_games/torch_runner.py`.
- `tyro`, Hydra, OmegaConf, gym, W&B, TensorBoard, OpenCV, tqdm, requests, and the smoke wrapper dependencies are installed.
- `scripts/check_system.py` runs inside the compatibility environment and writes `reports/system_check.json`.
- `scripts/download_assets.sh` successfully downloaded the pretrained policy.
- `pretrained_policy/config.yaml` and `pretrained_policy/model.pth` exist locally.
- The pretrained policy directory is ignored by Git, so the 395 MB checkpoint should not be committed.
- The experiment ledger and aggregator record setup, asset download, dry-runs, and failures without dropping failed runs.
- The aggregate summary now refuses to report a "best run" until a measured evaluation, training, or profile run succeeds.

## What Is Still Blocked

- `isaacgym` does not import in the Python 3.8 compatibility environment.
- Isaac Gym Preview 4 was not found under the searched local paths and was not installed because `ISAAC_GYM_ROOT` is unset.
- Torch 2.4.1+cu121 in the Python 3.8 environment reports CUDA arch support through `sm_90`; the GPUs are Blackwell `sm_120`.
- Isaac Lab does not import in this compatibility environment. That is expected and is not the current target.
- DexToolBench data is still missing under `dextoolbench/data/`.
- Pretrained evaluation, DexToolBench evaluation, scratch smoke, and finetune smoke all stop at the missing `isaacgym` dependency before training or simulation starts.
- No real SimToolReal throughput, reward, FPS, peak run VRAM, or largest stable `num_envs` has been measured yet.

## Current Evidence

- System report: `reports/system_check.json`
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

Latest relevant logs:

- `logs/create_compat_env_20260520T055316Z.log`
- `logs/validate_compat_env_20260520T055343Z.log`
- `logs/download_assets_20260520T054540Z.log`
- `logs/run_pretrained_eval_20260520T055638Z.log`
- `logs/run_dextoolbench_eval_20260520T055638Z.log`
- `logs/train_scratch_smoke_20260520T055638Z.log`
- `logs/finetune_smoke_20260520T055638Z.log`

## Current Best Command

Use the compatibility environment wrapper:

```bash
bash scripts/run_in_compat_env.sh python scripts/check_system.py
```

Expected current result: Python 3.8, torch 2.4.1+cu121, two Blackwell GPUs visible, repo-local `rl_games` import OK, pretrained policy files present, `isaacgym` import failed.

## Next Required External Action

Install Isaac Gym Preview 4 into the compatibility environment, using a local extracted Isaac Gym package:

```bash
export ISAAC_GYM_ROOT=/path/to/isaacgym
bash scripts/create_compat_env.sh
bash scripts/validate_compat_env.sh
```

The blocker is fixed only when `reports/system_check.json` shows `Isaac Gym.import_ok == true`.
