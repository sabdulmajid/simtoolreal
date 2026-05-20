# Compatibility Environment

This repo now has a reproducible Python 3.8 environment path for the original Isaac Gym SimToolReal stack.

## Create Or Reuse The Environment

```bash
bash scripts/create_compat_env.sh
```

Default location:

```text
/pub7/neel2/conda_envs/simtoolreal-py38
```

Override it with:

```bash
SIMTOOLREAL_ENV_PATH=/some/env/path bash scripts/create_compat_env.sh
```

The script installs:

- Python 3.8
- pip, setuptools, wheel
- torch 2.4.1+cu124 and torchvision 0.19.1+cu124 by default
- SimToolReal editable
- repo-local `rl_games` editable
- the lightweight wrapper dependencies used by smoke tests and asset download

Isaac Gym is not bundled in Git. Download and extract it outside the repo:

```bash
bash scripts/download_isaacgym.sh
```

Default extracted location:

```text
/pub7/neel2/external/isaacgym_preview4/isaacgym
```

If you have another extracted Isaac Gym Preview 4 package, install it through the same script:

```bash
export ISAAC_GYM_ROOT=/path/to/extracted/isaacgym
bash scripts/create_compat_env.sh
```

The script expects:

```text
${ISAAC_GYM_ROOT}/python
```

## Run Commands In The Environment

```bash
bash scripts/run_in_compat_env.sh python scripts/check_system.py
```

Examples:

```bash
bash scripts/run_in_compat_env.sh bash scripts/download_assets.sh
bash scripts/run_in_compat_env.sh bash scripts/run_pretrained_eval.sh
bash scripts/run_in_compat_env.sh bash scripts/train_scratch_smoke.sh
```

## Validate Without Training

```bash
bash scripts/validate_compat_env.sh
```

Current expected status on this Blackwell workstation:

```text
status: gpu_runtime_error
reason: torch imports, Isaac Gym imports, but a minimal CUDA kernel fails on the visible GPU(s)
```

Current evidence:

- `reports/download_isaacgym.json`
- `reports/create_compat_env.json`
- `reports/validate_compat_env.json`
- `reports/system_check.json`
- `logs/download_isaacgym_20260520T063819Z.log`
- `logs/create_compat_env_20260520T063132Z.log`
- `logs/validate_compat_env_20260520T063825Z.log`

## Current Blackwell Warning

The validated Python 3.8 environment sees both GPUs, but torch 2.4.1+cu124 does not advertise `sm_120` support:

```text
torch CUDA arch list: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90
GPU capability: sm_120
```

This is no longer hypothetical. `scripts/validate_compat_env.sh` runs a tiny CUDA operation and it fails with:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Original Isaac Gym Preview 4 also requires Python `<3.9`, so simply moving to a modern Python/PyTorch wheel breaks the Isaac Gym import path. Treat this as the active blocker until a Python 3.8 torch build with `sm_120` support is found or the project moves to a different simulator stack.
