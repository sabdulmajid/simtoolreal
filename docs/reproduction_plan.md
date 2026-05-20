# SimToolReal Reproduction Plan

Run commands from the repository root.

## 0. Diagnose The Current Shell

```bash
python scripts/check_system.py
```

This writes `reports/system_check.json`.

## 1. Create The Isaac Gym Environment

Use the checked-in compatibility script:

```bash
bash scripts/download_isaacgym.sh
bash scripts/create_compat_env.sh
bash scripts/run_in_compat_env.sh python scripts/check_system.py
```

Default environment path:

```text
/pub7/neel2/conda_envs/simtoolreal-py38
```

The download script stores the NVIDIA Isaac Gym package outside the repository under `/pub7/neel2/external` and writes `reports/download_isaacgym.json`. To use a different extracted package:

```bash
export ISAAC_GYM_ROOT=/path/to/extracted/isaacgym
bash scripts/create_compat_env.sh
bash scripts/validate_compat_env.sh
```

The script installs the repo and repo-local `rl_games` fork editable. It installs Isaac Gym only when `${ISAAC_GYM_ROOT}/python` exists.

Current validation in this workspace:

```text
Python: 3.8.20
torch: 2.4.1+cu124
rl_games.torch_runner: import OK
isaacgym: import OK
pretrained_policy/config.yaml: present
pretrained_policy/model.pth: present
torch CUDA kernel smoke: failed on sm_120 with "no kernel image"
```

Expected pass criteria: `isaacgym` imports, `rl_games.torch_runner` imports, CUDA is available, torch reports an architecture list compatible with the visible GPU, and `scripts/validate_compat_env.sh` can run its CUDA kernel smoke. On this Blackwell machine, the import criteria now pass and the CUDA kernel criterion fails.

## 2. Download The Pretrained Policy

```bash
python download_pretrained_policy.py
```

Safe wrapper with logs and a report:

```bash
bash scripts/run_in_compat_env.sh bash scripts/download_assets.sh
```

Expected files:

```text
pretrained_policy/config.yaml
pretrained_policy/model.pth
```

## 3. Run Interactive Pretrained Evaluation

Use one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python dextoolbench/eval_interactive.py \
  --config-path pretrained_policy/config.yaml \
  --checkpoint-path pretrained_policy/model.pth \
  --port 8080
```

Open `http://localhost:8080`.

Headless one-task wrapper:

```bash
bash scripts/run_in_compat_env.sh bash scripts/run_pretrained_eval.sh
```

Set `INTERACTIVE=1` to use the viser UI wrapper path.

## 4. Download DexToolBench Data

List available datasets:

```bash
python download_dextoolbench_data.py --list
```

Download one task first:

```bash
python download_dextoolbench_data.py \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down
```

Download all data after the one-task path works:

```bash
python download_dextoolbench_data.py
```

## 5. Run Numerical DexToolBench Evaluation

One task:

```bash
CUDA_VISIBLE_DEVICES=0 python dextoolbench/eval.py \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down \
  --checkpoint-path pretrained_policy/model.pth \
  --config-path pretrained_policy/config.yaml \
  --output-dir evals/smoke/hammer/claw_hammer/swing_down/pretrained_policy \
  --num-episodes 1 \
  --policy-name pretrained_policy
```

All tasks:

```bash
CUDA_VISIBLE_DEVICES=0 python dextoolbench/run_all_evals.py
```

Safe wrapper:

```bash
bash scripts/run_in_compat_env.sh bash scripts/run_dextoolbench_eval.sh
```

Set `RUN_ALL=1` only after the one-task command succeeds.

Note: `deployment/rl_player.py` currently sets `CUDA_VISIBLE_DEVICES="0"` internally when creating a player. Treat GPU selection for evaluation as GPU-0-only until that is fixed in a later implementation pass.

## 6. Train From Scratch

Disable W&B unless the entity/project is configured:

```bash
CUDA_VISIBLE_DEVICES=0 python isaacgymenvs/launch_training.py \
  --custom-experiment-name scratch_24576 \
  --num-envs 24576 \
  --num-blocks 6 \
  --no-wandb-activate
```

Equivalent helper:

```bash
WANDB_ACTIVATE=False NUM_ENVS=24576 scripts/launch_train_gpu0.sh
```

Short smoke wrapper:

```bash
bash scripts/run_in_compat_env.sh bash scripts/train_scratch_smoke.sh
```

## 7. Finetune From Pretrained

```bash
CUDA_VISIBLE_DEVICES=0 python isaacgymenvs/launch_training.py \
  --custom-experiment-name finetune_pretrained_24576 \
  --checkpoint pretrained_policy/model.pth \
  --num-envs 24576 \
  --num-blocks 6 \
  --no-wandb-activate
```

Equivalent helper:

```bash
WANDB_ACTIVATE=False NUM_ENVS=24576 scripts/launch_train_gpu0.sh \
  --checkpoint pretrained_policy/model.pth
```

Short smoke wrapper:

```bash
bash scripts/run_in_compat_env.sh bash scripts/finetune_smoke.sh
```

## 8. Record Reproducibility Metadata

After the first successful run:

```bash
python scripts/check_system.py | tee train_dir/system_check.txt
uv pip freeze | tee train_dir/requirements.lock.txt
git rev-parse HEAD | tee train_dir/git_commit.txt
git status --short | tee train_dir/git_status.txt
```

## 9. Blackwell Isaac Lab Compatibility Smoke

The original Isaac Gym path above is blocked on this Blackwell machine until a
Python 3.8 torch build with `sm_120` kernels exists. The current forward path is
to validate Isaac Lab separately before porting the minimal environment:

```bash
bash scripts/create_isaaclab_blackwell_env.sh
ACCEPT_EULA=Y ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

GPU 1:

```bash
ACCEPT_EULA=Y GPU_ID=1 ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/blackwell_isaaclab_smoke_clean_gpu1.json \
  METRICS_PATH=reports/blackwell_isaaclab_validation_clean_gpu1.json \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

Bounded 12,288-env runtime smoke:

```bash
ACCEPT_EULA=Y NUM_ENVS=12288 STEPS=32 \
  ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json \
  METRICS_PATH=reports/blackwell_isaaclab_validation_clean_gpu0_12288.json \
  bash scripts/run_blackwell_isaaclab_smoke.sh
```

Current result in this workspace: the clean Isaac Lab environment uses Python
3.11.15 and torch 2.7.0+cu128, advertises `sm_120`, runs a CUDA kernel, runs
Cartpole on both GPUs, and completes a 12,288-env GPU0 smoke. The result is
recorded as `success_with_dependency_conflicts` because `pip check` reports the
known FastAPI/Starlette conflict.

## 10. Blackwell ToolPose Asset/State Probe

After the Isaac Lab smoke passes, validate that the actual SimToolReal ToolPose
assets and state tensors load under the Blackwell-compatible runtime:

GPU 0:

```bash
ACCEPT_EULA=Y ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/isaaclab_toolpose_asset_probe_gpu0.json \
  METRICS_PATH=reports/isaaclab_toolpose_asset_probe_metrics_gpu0.json \
  NUM_ENVS=2 STEPS=4 \
  bash scripts/run_isaaclab_toolpose_asset_probe.sh
```

GPU 1:

```bash
ACCEPT_EULA=Y GPU_ID=1 ISAACLAB_CONDA_ENV=/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell \
  REPORT_PATH=reports/isaaclab_toolpose_asset_probe_gpu1.json \
  METRICS_PATH=reports/isaaclab_toolpose_asset_probe_metrics_gpu1.json \
  NUM_ENVS=1 STEPS=2 \
  bash scripts/run_isaaclab_toolpose_asset_probe.sh
```

Current result in this workspace: both probes complete with
`success_with_dependency_conflicts`. They load the real KUKA+SHARPA robot URDF,
the table URDF, and a generated handle-head tool URDF. The probe verifies all
29 expected joints, the palm body, all five fingertip bodies, an action target
shape of `[num_envs, 29]`, a tool root-state shape of `[num_envs, 13]`, and a
finite policy-observation candidate shape of `[num_envs, 140]`.

This is not a trained policy result. It is the reproducible handoff point for
the scoped Isaac Lab ToolPose environment port.

## Current Evidence From This Shell

- System report: `reports/system_check.json`
- Isaac Gym download report: `reports/download_isaacgym.json`
- Compatibility validation report: `reports/validate_compat_env.json`
- Asset download wrapper report: `reports/download_assets.json`
- Pretrained eval wrapper report: `reports/run_pretrained_eval.json`
- DexToolBench eval wrapper report: `reports/run_dextoolbench_eval.json`
- Scratch smoke report: `reports/train_scratch_smoke.json`
- Finetune smoke report: `reports/finetune_smoke.json`
- Experiment ledger: `reports/experiment_runs.jsonl`
- Aggregated result table: `reports/experiment_results.csv`
- Aggregated result JSON: `reports/experiment_results.json`
- Best-run summary: `reports/experiment_summary.md`
- Blackwell Isaac Lab GPU0 report: `reports/blackwell_isaaclab_smoke.json`
- Blackwell Isaac Lab GPU1 report: `reports/blackwell_isaaclab_smoke_gpu1.json`
- Clean Isaac Lab environment report: `reports/create_isaaclab_blackwell_env.json`
- Clean Isaac Lab GPU0 report: `reports/blackwell_isaaclab_smoke_clean_gpu0.json`
- Clean Isaac Lab GPU1 report: `reports/blackwell_isaaclab_smoke_clean_gpu1.json`
- Clean Isaac Lab 12,288-env report: `reports/blackwell_isaaclab_smoke_clean_gpu0_12288.json`
- ToolPose asset/state GPU0 report: `reports/isaaclab_toolpose_asset_probe_gpu0.json`
- ToolPose asset/state GPU0 metrics: `reports/isaaclab_toolpose_asset_probe_metrics_gpu0.json`
- ToolPose asset/state GPU1 report: `reports/isaaclab_toolpose_asset_probe_gpu1.json`
- ToolPose asset/state GPU1 metrics: `reports/isaaclab_toolpose_asset_probe_metrics_gpu1.json`

These reports currently show that the Python 3.8 dependency stack, Isaac Gym Preview 4, repo-local `rl_games`, and pretrained checkpoint are in place. Original SimToolReal execution remains blocked because torch 2.4.1+cu124 does not run CUDA kernels on the installed Blackwell `sm_120` GPUs. The Isaac Lab path is now stronger than a generic runtime smoke: it loads the real ToolPose robot/table/tool assets and exposes the required state tensor shapes on both GPUs.

See `docs/experiment_discipline.md` for the exact recorded fields and the
workflow for reproducing the best recorded run.
