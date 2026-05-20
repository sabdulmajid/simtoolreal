# Known Blockers

## 1. Isaac Gym Preview 4 Is Not Installed

Evidence:

- `reports/system_check.json`
- `reports/validate_compat_env.json`
- `logs/validate_compat_env_20260520T055343Z.log`
- `logs/run_pretrained_eval_20260520T055638Z.log`
- `logs/train_scratch_smoke_20260520T055638Z.log`

Failing command:

```bash
bash scripts/validate_compat_env.sh
```

Error summary:

```text
ModuleNotFoundError: No module named 'isaacgym'
```

Current classification:

- Missing dependency: yes.
- Python version mismatch: mostly fixed. The compatibility env is Python 3.8.20.
- PyTorch/CUDA mismatch: possible remaining blocker, see below.
- Isaac Gym install issue: yes.
- Blackwell GPU compatibility issue: not yet proven for Isaac Gym because the package is absent.
- Asset/path issue: pretrained policy fixed; DexToolBench data still missing.
- Hydra/config issue: not reached.
- rl_games issue: fixed in the compatibility env.
- Headless/rendering issue: not reached.
- Training/runtime issue: not reached.

Clean install path:

```bash
export ISAAC_GYM_ROOT=/path/to/extracted/isaacgym
bash scripts/create_compat_env.sh
bash scripts/validate_compat_env.sh
```

`scripts/create_compat_env.sh` installs Isaac Gym only when `${ISAAC_GYM_ROOT}/python` exists.

## 2. Python 3.8 PyTorch Wheel Does Not Advertise Blackwell `sm_120`

Evidence: `reports/system_check.json`

The Python 3.8 compatibility environment currently has:

```text
torch: 2.4.1+cu121
torch.version.cuda: 12.1
torch CUDA arch list: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90
GPU capability: sm_120
```

PyTorch emits this warning:

```text
NVIDIA RTX PRO 6000 Blackwell ... sm_120 is not compatible with the current PyTorch installation.
```

This means the environment is good enough to expose imports and dependency state, but may still fail when PyTorch kernels or Isaac Gym GPU physics execute on the Blackwell GPUs. The clean fix is a Python 3.8-compatible torch build with Blackwell support if one is available, or an isolated container/host path with a simulator stack verified against the installed driver.

Do not claim training compatibility until a smoke environment actually launches and runs.

## 3. DexToolBench Data Is Missing

Evidence: `reports/system_check.json`

Current state:

```text
dextoolbench/data/: missing
```

Download one task only after Isaac Gym imports:

```bash
bash scripts/run_in_compat_env.sh python download_dextoolbench_data.py \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down
```

## 4. Isaac Lab Is Out Of Scope For The Current Fix

Isaac Lab does not import in the Python 3.8 compatibility environment. That does not matter for the current original-SimToolReal reproduction path. Do not start the Isaac Lab migration until the original Isaac Gym path is either working or proven impossible with exact logs.
