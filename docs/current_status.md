# Current Status

Last validated from this workspace on 2026-05-20T06:41Z.

## What Works Now

- `scripts/download_isaacgym.sh` downloads and extracts Isaac Gym Preview 4 outside the Git repo at `/pub7/neel2/external/isaacgym_preview4/isaacgym`.
- `scripts/create_compat_env.sh` creates or reuses a repo-compatible Python 3.8 environment at `/pub7/neel2/conda_envs/simtoolreal-py38`.
- The compatibility environment uses Python 3.8.20 and torch 2.4.1+cu124.
- Isaac Gym Preview 4 installs into the compatibility environment and imports successfully.
- The repo and repo-local `rl_games` fork install editable into that environment.
- `rl_games.torch_runner` imports from `rl_games/rl_games/torch_runner.py`.
- `tyro`, Hydra, OmegaConf, gym, W&B, TensorBoard, OpenCV, tqdm, requests, `pytorch3d`, `viser`, `trimesh`, `transforms3d`, `pyvirtualdisplay`, `pytorch-kinematics`, and the smoke wrapper dependencies are installed.
- `scripts/check_system.py` runs inside the compatibility environment and writes `reports/system_check.json`.
- `scripts/download_assets.sh` successfully downloaded the pretrained policy.
- `pretrained_policy/config.yaml` and `pretrained_policy/model.pth` exist locally.
- The pretrained policy directory is ignored by Git, so the 395 MB checkpoint should not be committed.
- The experiment ledger and aggregator record setup, Isaac Gym download, asset download, dry-runs, and failures without dropping failed runs.
- The aggregate summary now refuses to report a "best run" until a measured evaluation, training, or profile run succeeds.
- Pretrained evaluation, DexToolBench evaluation, scratch smoke, and finetune smoke now reach CUDA execution rather than stopping at missing imports.

## What Is Still Blocked

- Torch 2.4.1+cu124 in the Python 3.8 environment reports CUDA arch support only through `sm_90`; both GPUs are Blackwell `sm_120`.
- A minimal torch CUDA kernel fails on both visible GPUs with `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
- Pretrained evaluation, DexToolBench evaluation, scratch smoke, and finetune smoke all fail at CUDA execution with the same Blackwell/PyTorch runtime incompatibility.
- Isaac Lab does not import in this compatibility environment. That is expected and is not the current target.
- DexToolBench data is still missing under `dextoolbench/data/`.
- No real SimToolReal throughput, reward, FPS, peak run VRAM, or largest stable `num_envs` has been measured yet.

## Current Evidence

- System report: `reports/system_check.json`
- Isaac Gym download report: `reports/download_isaacgym.json`
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

- `logs/download_isaacgym_20260520T063819Z.log`
- `logs/create_compat_env_20260520T063132Z.log`
- `logs/validate_compat_env_20260520T063825Z.log`
- `logs/download_assets_20260520T054540Z.log`
- `logs/run_pretrained_eval_20260520T063847Z.log`
- `logs/run_dextoolbench_eval_20260520T063906Z.log`
- `logs/train_scratch_smoke_20260520T063928Z.log`
- `logs/finetune_smoke_20260520T063948Z.log`

## Current Best Command

Use the compatibility environment wrapper:

```bash
bash scripts/run_in_compat_env.sh python scripts/check_system.py
```

Expected current result: Python 3.8, torch 2.4.1+cu124, Isaac Gym import OK, repo-local `rl_games` import OK, pretrained policy files present, and torch CUDA kernel smoke failed on Blackwell `sm_120`.

## Next Required External Action

The Isaac Gym dependency is now installed. The current external blocker is the Python 3.8 PyTorch/Blackwell conflict:

```text
Isaac Gym Preview 4 requires Python <3.9.
Python 3.8 PyTorch wheels available here stop at torch 2.4.1.
torch 2.4.1+cu121 and torch 2.4.1+cu124 do not advertise sm_120 and fail CUDA kernels on the installed Blackwell GPUs.
```

The clean next milestone is a compatibility decision, not more scaling: either run the original Isaac Gym stack on a non-Blackwell GPU, build or locate a Python 3.8 torch package with Blackwell kernel support, or move to an Isaac Lab port under a modern Python/PyTorch stack.
