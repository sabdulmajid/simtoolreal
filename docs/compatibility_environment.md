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
- torch 2.4.1+cu121 and torchvision 0.19.1+cu121
- SimToolReal editable
- repo-local `rl_games` editable
- the lightweight wrapper dependencies used by smoke tests and asset download

Isaac Gym is intentionally not bundled. If you have an extracted Isaac Gym Preview 4 package, install it through the same script:

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

Current expected status before Isaac Gym installation:

```text
status: dependency_error
reason: ModuleNotFoundError: No module named 'isaacgym'
```

Current evidence:

- `reports/create_compat_env.json`
- `reports/validate_compat_env.json`
- `reports/system_check.json`
- `logs/create_compat_env_20260520T055316Z.log`
- `logs/validate_compat_env_20260520T055343Z.log`

## Current Blackwell Warning

The validated Python 3.8 environment sees both GPUs, but torch 2.4.1+cu121 does not advertise `sm_120` support:

```text
torch CUDA arch list: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90
GPU capability: sm_120
```

This may require a newer torch/CUDA build or a container with a verified Blackwell-compatible stack. Treat it as an unresolved runtime risk until a real Isaac Gym smoke test runs.
