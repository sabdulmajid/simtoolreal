# Known Blockers

## 1. Python 3.8 PyTorch Does Not Run CUDA Kernels On Blackwell

Evidence:

- `reports/system_check.json`
- `reports/validate_compat_env.json`
- `logs/validate_compat_env_20260520T063825Z.log`
- `logs/run_pretrained_eval_20260520T063847Z.log`
- `logs/run_dextoolbench_eval_20260520T063906Z.log`
- `logs/train_scratch_smoke_20260520T063928Z.log`
- `logs/finetune_smoke_20260520T063948Z.log`

Failing command:

```bash
bash scripts/validate_compat_env.sh
```

Error summary:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Current classification:

- Missing dependency: no for the smoke path that reaches CUDA; Isaac Gym and the local runtime imports now work.
- Python version mismatch: structural risk. Isaac Gym Preview 4 requires Python `<3.9`.
- PyTorch/CUDA mismatch: yes.
- Isaac Gym install issue: fixed. `isaacgym` imports from `/pub7/neel2/external/isaacgym_preview4/isaacgym/python`.
- Blackwell GPU compatibility issue: yes. The installed GPUs report `sm_120`; torch 2.4.1+cu121 and torch 2.4.1+cu124 advertise only through `sm_90`.
- Asset/path issue: pretrained policy fixed; DexToolBench data still missing.
- Hydra/config issue: one smoke wrapper issue was fixed by making experiment names start with a numeric prefix required by `isaacgymenvs/train.py`.
- rl_games issue: fixed in the compatibility env.
- Headless/rendering issue: not the current failure.
- Training/runtime issue: yes, but caused by torch CUDA kernel incompatibility before training can proceed.

The current Python 3.8 compatibility environment has:

```text
Python: 3.8.20
torch: 2.4.1+cu124
torch.version.cuda: 12.4
torch CUDA arch list: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90
GPU capability: sm_120
```

Before/after status:

- Before this pass: missing Isaac Gym stopped all evaluation/training commands before simulator launch.
- After this pass: Isaac Gym imports, the environment starts loading assets, and all smoke commands fail at CUDA execution with `no kernel image`.

Clean options:

- Use a non-Blackwell GPU for the original Isaac Gym Preview 4 stack.
- Build or locate a Python 3.8-compatible PyTorch package with Blackwell `sm_120` kernel support, then rerun `bash scripts/validate_compat_env.sh`.
- Treat original Isaac Gym on Blackwell as blocked and start a scoped Isaac Lab migration under a modern Python/PyTorch stack. Do not claim this as original SimToolReal reproduction.

## 2. Isaac Gym Preview 4 Is Installed, But Only In The Python 3.8 Compatibility Path

Evidence:

- `reports/download_isaacgym.json`
- `reports/system_check.json`
- `logs/download_isaacgym_20260520T063819Z.log`
- `logs/create_compat_env_20260520T063132Z.log`

Current state:

```text
archive_path: /pub7/neel2/external/IsaacGym_Preview_4_Package.tar.gz
isaac_gym_root: /pub7/neel2/external/isaacgym_preview4/isaacgym
isaacgym import: OK
binding: gym_38.so
```

Important constraint:

```text
Isaac Gym Preview 4 setup.py declares python_requires='>=3.6,<3.9'.
The package contains gym_38.so and rlgpu_38.so bindings for Python 3.8.
```

A modern Python/PyTorch environment can support Blackwell more easily, but Isaac Gym Preview 4 does not import there. The blocker is now a stack compatibility conflict, not just a missing package.

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

Isaac Lab does not import in the Python 3.8 compatibility environment. That does not matter for the original Isaac Gym reproduction path.

The Blackwell path is now partially validated under a modern Python/PyTorch environment:

- `reports/blackwell_isaaclab_smoke.json`
- `reports/blackwell_isaaclab_validation.json`
- `reports/blackwell_isaaclab_smoke_gpu1.json`
- `reports/blackwell_isaaclab_validation_gpu1.json`

Current status:

- Isaac Lab Cartpole smoke runs on both GPUs.
- torch 2.10.0+cu128 advertises `sm_120`.
- a minimal torch CUDA kernel succeeds.
- `pip check` fails, so the active `isaaclab2` environment is not a clean reproducibility environment.
- `/pub7/neel2/isaaclab_ws/IsaacLab` fails due a missing `apps/isaacsim_5` layout; the working launcher path is `/pub7/neel/vlm/IsaacLab`.

This means the project is not a dead end, but the fix is a clean Isaac Lab path plus a scoped environment port. It is not more Isaac Gym scaling on Blackwell.
