# Experiment Summary

Generated: 2026-05-20T06:42:58Z

Total rows: 45

## Status Counts

- `dependency_error`: 14
- `dry_run`: 4
- `failed`: 5
- `gpu_runtime_error`: 12
- `missing_asset`: 2
- `success`: 8

## Mode Counts

- `asset_download`: 2
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
| 2026-05-20T06:39:51Z | finetune_smoke | gpu_runtime_error | Finetune smoke reached CUDA execution, but torch cannot run kernels on the visible GPU. See log. | logs/finetune_smoke_20260520T063948Z.log |
| 2026-05-20T06:39:31Z | train_scratch_smoke | gpu_runtime_error | Scratch smoke reached CUDA execution, but torch cannot run kernels on the visible GPU. See log. | logs/train_scratch_smoke_20260520T063928Z.log |
| 2026-05-20T06:39:10Z | run_dextoolbench_eval | gpu_runtime_error | DexToolBench evaluation reached CUDA execution, but torch cannot run kernels on the visible GPU. See log. | logs/run_dextoolbench_eval_20260520T063906Z.log |
| 2026-05-20T06:38:50Z | run_pretrained_eval | gpu_runtime_error | Pretrained evaluation reached CUDA execution, but torch cannot run kernels on the visible GPU. See log. | logs/run_pretrained_eval_20260520T063847Z.log |
| 2026-05-20T06:38:25Z | validate_compat_env | gpu_runtime_error | Torch imports, but a minimal CUDA kernel fails on the visible GPU(s). | logs/validate_compat_env_20260520T063825Z.log |
