# Next Steps

## Immediate Commands

Create the documented Isaac Gym environment. `uv` was not on `PATH` during validation; install it first or use an existing Python 3.8 environment manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.8
echo 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH' >> .venv/bin/activate
source .venv/bin/activate
uv pip install -e .
cd /path/to/extracted/isaacgym/python
uv pip install -e .
cd /pub7/neel2/simtoolreal/rl_games
uv pip install -e .
cd /pub7/neel2/simtoolreal
python scripts/check_system.py
```

If the imports pass, download assets:

```bash
bash scripts/download_assets.sh
```

Then run the smallest evaluation and training smoke tests:

```bash
bash scripts/run_pretrained_eval.sh
bash scripts/run_dextoolbench_eval.sh
bash scripts/train_scratch_smoke.sh
bash scripts/finetune_smoke.sh
```

Only after smoke tests pass, dry-run and execute scaling:

```bash
python scripts/profile_gpu_scaling.py --gpu-id 0
DRY_RUN=1 GPU_ID=1 scripts/sweep_num_envs.sh
python scripts/profile_gpu_scaling.py --gpu-id 0 --max-epochs 3 --run
```

## Next Engineering Milestone

- Make the Python 3.8 Isaac Gym environment import cleanly.
- If Isaac Gym fails on Blackwell, capture the exact PhysX/CUDA error in `logs/` and update `docs/known_blockers.md`.
- If Isaac Gym imports, run a one-env environment creation smoke test before any large `num_envs` sweep.
- Fix `deployment/rl_player.py` GPU pinning in a small behavior-preserving patch after pretrained evaluation is otherwise ready.

## Next Research Milestone

- Establish a baseline pretrained DexToolBench result for one task.
- Establish whether scratch training can start at the conservative smoke settings.
- Measure stable throughput at 12288, 24576, and 49152 envs on a single GPU.
- Compare independent GPU0 and GPU1 jobs only after one-GPU stability is known.

## When To Attempt Isaac Lab

Attempt the Isaac Lab port only after one of these is true:

- Original Isaac Gym reproduction is working and has baseline evaluation/training evidence.
- Original Isaac Gym is blocked by a confirmed Blackwell/PhysX binary incompatibility with exact logs.

Do not start the Isaac Lab port during this milestone.
