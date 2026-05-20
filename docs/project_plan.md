# SimToolReal Reproducible Research Plan

This pass audited the repository without changing training logic. The project is currently an Isaac Gym Preview 4 project with a vendored `rl_games` fork, not an Isaac Lab project.

## Audit Findings

- The documented training environment is Python 3.8 because Isaac Gym Preview 4 is required (`docs/installation.md`).
- `pyproject.toml` allows Python `>=3.8,<3.12`, leaves `torch` and `torchvision` unpinned, and points uv at PyTorch CUDA 11.8 wheels.
- `setup.py` is older and less complete than `pyproject.toml`; treat `pyproject.toml` as the primary dependency source.
- The root package depends on `rl-games>=1.6.0`, but the intended algorithm code is the repo-local `rl_games/` fork. Install the local fork after installing the root package.
- Training enters through `isaacgymenvs/launch_training.py`, which launches `python -m isaacgymenvs.train`.
- Default training uses `num_envs=24576`, `num_blocks=6`, SAPG-style exploration overrides, `multi_gpu=False`, and W&B enabled with a hard-coded default entity.
- The config defaults use `sim_device=cuda:0`, `rl_device=cuda:0`, `graphics_device_id=0`, `pipeline=gpu`, and PhysX.
- Checkpoint/config output is split between Hydra `train_dir/...` and rl_games `runs/<experiment>/...`.
- Current shell environment is not the documented training environment: it is Python 3.12.3 with torch 2.9.0+cu128; Isaac Gym, Isaac Lab, and `rl_games.torch_runner` are not importable there.
- The host has two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs, each reported as sm_120 with about 95 GiB usable VRAM.

## Compatibility Blockers

- Isaac Gym Preview 4 now installs and imports in the Python 3.8 compatibility environment. The remaining local blocker is not an import error.
- The validated Python 3.8 environment uses torch 2.4.1+cu124. It advertises CUDA kernels only through `sm_90`, while both installed GPUs are Blackwell `sm_120`.
- `scripts/validate_compat_env.sh` proves the blocker with a minimal CUDA operation: `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
- Pretrained evaluation, DexToolBench evaluation, scratch smoke, and finetune smoke all reach CUDA execution and fail with the same kernel-image error. Do not start GPU scaling until this validation passes.
- Isaac Gym Preview 4's Python package declares `python_requires='>=3.6,<3.9'` and ships Python 3.8 bindings. Modern Blackwell-capable PyTorch wheels are not available for this Python 3.8 path in the tested package indexes, so the conflict is structural.
- A separate clean Isaac Lab environment now runs bounded Cartpole smoke tasks on both Blackwell GPUs with torch 2.7.0+cu128, and a 12,288-env GPU0 smoke point has been measured. Treat this as a validated runtime direction, not a completed SimToolReal port.
- The Isaac Lab environment still reports one FastAPI/Starlette dependency conflict, so runtime reports use `success_with_dependency_conflicts`.
- True multi-GPU training is not verified. The SimToolReal launcher forces `multi_gpu=False`, and the vendored distributed path contains `cuda:0` assumptions.

References: [NVIDIA CUDA 12.8 SM_120 support](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html), [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-compatibility-guide/index.html), [PyTorch 2.7 Blackwell/CUDA 12.8 release](https://pytorch.org/blog/pytorch-2-7/), [NVIDIA Isaac Gym Preview 4 Blackwell forum report](https://forums.developer.nvidia.com/t/isaac-gym-preview-4-incompatible-with-rtx-5080-blackwell-sm-120-libphysxgpu-64-so-missing-sm-120-kernels/367941).

## Roadmap

### Phase 0: System Check

- Run `scripts/check_system.py` in the current shell and in every candidate training environment.
- Record `python --version`, `torch.__version__`, `torch.version.cuda`, `torch.cuda.get_arch_list()`, GPU names, VRAM, and package import status.
- Require `bash scripts/validate_compat_env.sh` to pass before any expensive run. On this machine it currently fails with `gpu_runtime_error`.

### Phase 1: Pretrained Evaluation

- Create the documented Isaac Gym environment.
- Download pretrained policy.
- Verify torch can run CUDA kernels on the visible GPU.
- Run one interactive DexToolBench task with `CUDA_VISIBLE_DEVICES=0`.
- Run one numerical `dextoolbench/eval.py` task before running all tasks.

### Phase 2: Finetuning And GPU Scaling

- Use one process per GPU with `CUDA_VISIBLE_DEVICES`.
- Keep W&B disabled until the first local run is stable, or set a real entity/project explicitly.
- Sweep `num_envs` at 12288, 24576, and 49152 with `num_blocks=6`.
- Track startup success, epoch time, GPU memory, GPU utilization, and failures.

### Phase 3: DexToolBench Evaluation Automation

- Make `dextoolbench/run_all_evals.py` parameterized by policy path, GPU id, output dir, number of episodes, and selected task subsets.
- Ensure results include config/checkpoint hashes and system diagnostics.

### Phase 4: Isaac Lab Port

- Use the clean Isaac Lab compatibility environment created by `scripts/create_isaaclab_blackwell_env.sh`.
- Port only the minimal ToolPose tracking environment as a new Isaac Lab task rather than editing the Isaac Gym task in place.
- Reuse assets and pure math helpers where possible.
- Build parity tests before training.

### Phase 5: Dual-GPU Distributed Isaac Lab Training

- Only after a single-GPU Isaac Lab task is stable, add distributed training using Isaac Lab-supported workflows.
- Validate per-rank seeding, environment partitioning, checkpointing, and W&B/TensorBoard behavior.

### Phase 6: Research Extension

- Add controlled experiments after reproduction is solid: object sets, domain randomization schedules, observation ablations, policy architecture variants, or sim-to-real robustness studies.
