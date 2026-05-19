# Completion Audit

Objective: execute `simtoolreal_reproduction_gpu_goal.md` for the original SimToolReal stack only, validate imports, attempt pretrained/DexToolBench evaluation if possible, create safe smoke scripts, build an independent `CUDA_VISIBLE_DEVICES` GPU scaling harness, document evidence, and avoid false success claims.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Read and execute root goal file | `simtoolreal_reproduction_gpu_goal.md` was inspected; this audit maps its requirements | complete |
| Do not delete existing files | No destructive git commands used; new files added only | complete |
| Do not rewrite RL logic | No existing training/environment logic changed | complete |
| Do not port to Isaac Lab | No Isaac Lab source/project files created | complete |
| Validate Isaac Gym import | `reports/system_check.json` shows `isaacgym` import failure in current shell | blocker documented |
| Validate Isaac Lab import | `reports/system_check.json` shows `isaaclab` and `omni.isaac.lab` import failures in current shell | blocker documented |
| Validate repo-local `rl_games` import | `reports/system_check.json` shows `rl_games.torch_runner` import failure | blocker documented |
| `scripts/check_system.py` exists | `scripts/check_system.py` | complete |
| `scripts/check_system.py` prints human-readable summary | Command `python scripts/check_system.py` completed and printed diagnostics | complete |
| `reports/system_check.json` generated | `reports/system_check.json` | complete |
| System report includes timestamp, OS/platform, Python, repo, git, torch/CUDA, GPUs/VRAM, nvidia-smi, imports, package versions, required dirs, pretrained files, DexToolBench data | `reports/system_check.json` | complete |
| Check script does not require training | It only imports/probes modules and calls `nvidia-smi`; no env construction | complete |
| `scripts/download_assets.sh` exists | `scripts/download_assets.sh` | complete |
| `scripts/run_pretrained_eval.sh` exists | `scripts/run_pretrained_eval.sh` | complete |
| `scripts/run_dextoolbench_eval.sh` exists | `scripts/run_dextoolbench_eval.sh` | complete |
| `scripts/train_scratch_smoke.sh` exists | `scripts/train_scratch_smoke.sh` | complete |
| `scripts/finetune_smoke.sh` exists | `scripts/finetune_smoke.sh` | complete |
| Reproduction scripts locate repo root robustly | They source `scripts/_common.sh` and use `simtoolreal_repo_root` | complete |
| Reproduction scripts create logs/reports | `logs/*.log`, `reports/*.json` from smoke runs | complete |
| Reproduction scripts fail clearly | Reports/logs show missing `tyro`, `isaacgym`, and checkpoint/pretrained files | complete |
| Reproduction scripts avoid GUI by default | `run_pretrained_eval.sh` defaults to headless `dextoolbench/eval.py`; UI requires `INTERACTIVE=1` | complete |
| Smoke training conservative by default | `NUM_ENVS=768`, `MAX_EPOCHS=1`, W&B disabled in smoke scripts | complete |
| `scripts/launch_train_gpu0.sh` exists | `scripts/launch_train_gpu0.sh` | complete |
| `scripts/launch_train_gpu1.sh` exists | `scripts/launch_train_gpu1.sh` | complete |
| `scripts/sweep_num_envs.sh` exists | `scripts/sweep_num_envs.sh` | complete |
| `scripts/profile_gpu_scaling.py` exists | `scripts/profile_gpu_scaling.py` | complete |
| Independent GPU jobs use `CUDA_VISIBLE_DEVICES` | launch, sweep, smoke, and profile scripts set or print `CUDA_VISIBLE_DEVICES` | complete |
| Sweep covers 12288, 24576, 49152 envs | `scripts/sweep_num_envs.sh`, `reports/gpu_scaling_results.json` dry-run entries | complete |
| `num_envs` divisibility respected | scripts check divisibility; dry-run SAPG block sizes 2048/4096/8192 | complete |
| Profiler produces CSV/JSON/Markdown reports | `reports/gpu_scaling_results.csv`, `reports/gpu_scaling_results.json`, `reports/gpu_scaling_summary.md` | complete |
| Profiler captures run metrics if available | Implementation records exit, timestamps, VRAM/utilization samples, parsed FPS/reward/loss when executed | complete, unexecuted due blocker |
| `docs/reproduction_plan.md` created/updated | `docs/reproduction_plan.md` | complete |
| `docs/gpu_scaling_plan.md` created/updated | `docs/gpu_scaling_plan.md` | complete |
| `docs/current_status.md` created | `docs/current_status.md` | complete |
| `docs/known_blockers.md` created | `docs/known_blockers.md` | complete |
| `docs/next_steps.md` created | `docs/next_steps.md` | complete |
| Run smallest safe commands first | `python scripts/check_system.py`; smoke wrappers run and failed safely | complete |
| Run pretrained eval if possible | Not possible in current shell; `reports/run_pretrained_eval.json` documents missing pretrained files | blocker documented |
| Run DexToolBench eval if possible | Not possible in current shell; `reports/run_dextoolbench_eval.json` documents missing Isaac Gym | blocker documented |
| Training smoke starts if possible | Not possible in current shell; `reports/train_scratch_smoke.json` documents missing Isaac Gym | blocker documented |
| Finetune smoke starts if possible | Not possible in current shell; `reports/finetune_smoke.json` documents missing checkpoint | blocker documented |
| Do not claim Isaac Lab support | Docs explicitly say Isaac Lab is unverified/not importable in this shell | complete |
| Do not claim true multi-GPU support | Docs explicitly limit to independent GPU jobs; true distributed remains unverified | complete |
| Final evidence paths exist | `reports/` and `logs/` files listed in `docs/current_status.md` | complete |

## Acceptance Path

The success path is not satisfied because no Isaac Gym environment or training smoke test starts in this shell.

The blocker path is satisfied:

- `scripts/check_system.py` runs successfully.
- `reports/system_check.json` is generated.
- The current blocker is reproduced with exact logs/reports.
- `docs/known_blockers.md` explains root cause as far as currently observable.
- `docs/next_steps.md` gives the concrete workaround plan.
- No success is claimed for Isaac Gym execution, Isaac Lab compatibility, pretrained evaluation, DexToolBench evaluation, scratch training, finetuning, or true multi-GPU distributed training.
