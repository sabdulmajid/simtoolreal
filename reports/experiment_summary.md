# Experiment Summary

Generated: 2026-05-20T09:29:24Z

Total rows: 49

## Status Counts

- `dependency_error`: 14
- `dry_run`: 4
- `failed`: 7
- `gpu_runtime_error`: 12
- `missing_asset`: 2
- `success`: 8
- `success_with_dependency_conflicts`: 2

## Mode Counts

- `asset_download`: 2
- `blackwell_isaaclab_validation`: 4
- `dextoolbench_eval`: 5
- `environment_setup`: 5
- `environment_validation`: 7
- `finetune`: 5
- `isaacgym_download`: 2
- `pretrained_eval`: 7
- `scratch`: 12

## Best Recorded Run

No successful measured evaluation, training, or profile run has been recorded yet.

## Latest Failures

| time | source | status | message | log |
| --- | --- | --- | --- | --- |
| 2026-05-20T09:24:50Z | blackwell_isaaclab_smoke_vlm | failed | Blackwell Isaac Lab smoke failed. See log. | logs/blackwell_isaaclab_smoke_vlm_20260520T092450Z.log |
| 2026-05-20T09:24:05Z | blackwell_isaaclab_smoke | failed | Isaac Lab bounded GPU simulation or torch CUDA smoke failed. | logs/blackwell_isaaclab_smoke_20260520T092403Z.log |
| 2026-05-20T06:39:51Z | finetune_smoke | gpu_runtime_error | Finetune smoke reached CUDA execution, but torch cannot run kernels on the visible GPU. See log. | logs/finetune_smoke_20260520T063948Z.log |
| 2026-05-20T06:39:31Z | train_scratch_smoke | gpu_runtime_error | Scratch smoke reached CUDA execution, but torch cannot run kernels on the visible GPU. See log. | logs/train_scratch_smoke_20260520T063928Z.log |
| 2026-05-20T06:39:10Z | run_dextoolbench_eval | gpu_runtime_error | DexToolBench evaluation reached CUDA execution, but torch cannot run kernels on the visible GPU. See log. | logs/run_dextoolbench_eval_20260520T063906Z.log |
