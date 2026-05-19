# SimToolReal Reproduction Plan

Run commands from the repository root.

## 0. Diagnose The Current Shell

```bash
python scripts/check_system.py
```

This writes `reports/system_check.json`. The current shell audited during this pass is not ready for this repo's Isaac Gym path: Python is 3.12.3, Isaac Gym is not importable, Isaac Lab is not importable, and the repo-local `rl_games` package is not installed.

## 1. Create The Isaac Gym Environment

The official repo documentation requires Python 3.8 for Isaac Gym Preview 4:

```bash
uv venv --python 3.8
echo 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH' >> .venv/bin/activate
source .venv/bin/activate
uv pip install -e .
```

Install Isaac Gym Preview 4 outside this repo:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4 -O IsaacGym_Preview_4_Package.tar.gz
tar -xzf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
uv pip install -e .
cd -
```

Install the repo-local `rl_games` fork last so it overrides any PyPI `rl-games` package:

```bash
cd rl_games
uv pip install -e .
cd -
```

Then re-run diagnostics:

```bash
python scripts/check_system.py
```

Expected pass criteria: `isaacgym` imports, `rl_games.torch_runner` imports, CUDA is available, and torch reports an architecture list compatible with the visible GPU. On Blackwell, this is the first likely failure point.

## 2. Download The Pretrained Policy

```bash
python download_pretrained_policy.py
```

Safe wrapper with logs and a report:

```bash
bash scripts/download_assets.sh
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
bash scripts/run_pretrained_eval.sh
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
bash scripts/run_dextoolbench_eval.sh
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
bash scripts/train_scratch_smoke.sh
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
bash scripts/finetune_smoke.sh
```

## 8. Record Reproducibility Metadata

After the first successful run:

```bash
python scripts/check_system.py | tee train_dir/system_check.txt
uv pip freeze | tee train_dir/requirements.lock.txt
git rev-parse HEAD | tee train_dir/git_commit.txt
git status --short | tee train_dir/git_status.txt
```

## Current Evidence From This Shell

- System report: `reports/system_check.json`
- Asset download wrapper report: `reports/download_assets.json`
- Pretrained eval wrapper report: `reports/run_pretrained_eval.json`
- DexToolBench eval wrapper report: `reports/run_dextoolbench_eval.json`
- Scratch smoke report: `reports/train_scratch_smoke.json`
- Finetune smoke report: `reports/finetune_smoke.json`
- Experiment ledger: `reports/experiment_runs.jsonl`
- Aggregated result table: `reports/experiment_results.csv`
- Aggregated result JSON: `reports/experiment_results.json`
- Best-run summary: `reports/experiment_summary.md`

These reports currently show the blocker path, not a successful reproduction run.

See `docs/experiment_discipline.md` for the exact recorded fields and the
workflow for reproducing the best recorded run.
