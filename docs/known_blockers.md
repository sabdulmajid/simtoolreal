# Known Blockers

## Current Environment Is Not The SimToolReal Isaac Gym Environment

Evidence: `reports/system_check.json`

- Current Python: `/pub3/neel/pyenv/bin/python`, version 3.12.3.
- Documented SimToolReal training environment requires Python 3.8 for Isaac Gym Preview 4.
- `isaacgym` import fails with `ModuleNotFoundError`.
- `rl_games.torch_runner` import fails with `ModuleNotFoundError`.
- `tyro` is missing, blocking repo CLI wrappers such as asset download and `launch_training.py`.

Focused debug classification:

- Failing command: `python -c 'import isaacgym'`
- Error summary: `ModuleNotFoundError: No module named 'isaacgym'`
- Relevant log paths: `reports/system_check.json`, `logs/train_scratch_smoke_20260519T003016Z.log`, `logs/run_dextoolbench_eval_20260519T003001Z.log`
- Likely root cause: active shell is not the documented SimToolReal Python 3.8 + Isaac Gym Preview 4 environment.
- Category: missing dependency, Python version mismatch, Isaac Gym install issue.
- Not yet proven: PyTorch/CUDA mismatch, Blackwell GPU compatibility issue, Hydra/config issue, headless/rendering issue, or training/runtime issue. Execution does not get far enough to test those.

Local checks performed:

```bash
python --version
which python
python -c 'import isaacgym'
python -c 'import tyro'
command -v uv || true
command -v python3.8 || true
find /pub3/neel /pub7/neel2 -maxdepth 6 \( -type d -name 'isaacgym' -o -type f -name 'libPhysXGpu_64.so' -o -type d -name 'IsaacGym*' \) 2>/dev/null | head -100
```

Result:

- Active Python is 3.12.3.
- `uv` is not on `PATH`.
- `python3.8` is not on `PATH`.
- No local Isaac Gym package or `libPhysXGpu_64.so` was found in the searched repo/user paths.

Regression verification after fixing the external environment:

```bash
python -c 'import isaacgym; print(isaacgym.__file__)'
python -c 'import tyro, rl_games.torch_runner; print("deps ok")'
python scripts/check_system.py
```

The blocker is fixed only when `reports/system_check.json` shows `Isaac Gym.import_ok == true` and `rl_games package.import_ok == true`.

## Isaac Lab Is Not Importable In This Shell

Evidence: `reports/system_check.json`

- `isaaclab` import fails.
- `omni.isaac.lab` import fails.
- This does not prove Isaac Lab is absent from the machine; it proves the active shell does not have it on `PYTHONPATH` or in the active environment.

## Missing Assets

Evidence:

- `reports/system_check.json`
- `reports/run_pretrained_eval.json`
- `reports/finetune_smoke.json`

Missing files:

- `pretrained_policy/config.yaml`
- `pretrained_policy/model.pth`
- `dextoolbench/data/`

## Isaac Gym Preview 4 On Blackwell Is High Risk

The machine GPUs are Blackwell `sm_120`. Current PyTorch in this shell supports `sm_120`, but Isaac Gym Preview 4 is an older binary stack. The risk is that PhysX GPU binaries may not contain Blackwell-compatible cubins/PTX. This must be tested in the real Python 3.8 Isaac Gym environment.

Relevant references are listed in `docs/project_plan.md`.

## GUI And Headless Risks

- Interactive evaluation uses a viser web UI.
- Isaac Gym viewer and camera settings can affect performance and headless behavior.
- Default wrappers prefer headless one-task evaluation; use `INTERACTIVE=1` only after the headless path works.

## Hydra And Config Risks

- `launch_training.py` uses tyro for CLI parsing and Hydra overrides internally.
- The profile/smoke scripts bypass `launch_training.py` only to set short-run `max_epochs`; they preserve the same SimToolReal task and SAPG/PPO override surface.
- `num_envs` must remain divisible by `num_blocks` for SAPG block sizing.

## Unverified Assumptions

- No successful Isaac Gym environment creation has occurred in this shell.
- No pretrained policy checkpoint has been loaded.
- No DexToolBench episode has run.
- No scratch or finetune training run has entered the first epoch.
- No largest stable `num_envs` is known.
- No true distributed multi-GPU codepath is verified.
