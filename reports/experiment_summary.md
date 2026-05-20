# Experiment Summary

Generated: 2026-05-20T05:58:12Z

Total rows: 22

## Status Counts

- `dependency_error`: 13
- `dry_run`: 4
- `missing_asset`: 2
- `success`: 3

## Mode Counts

- `asset_download`: 2
- `dextoolbench_eval`: 3
- `environment_setup`: 2
- `environment_validation`: 2
- `finetune`: 3
- `pretrained_eval`: 3
- `scratch`: 7

## Best Recorded Run

No successful measured evaluation, training, or profile run has been recorded yet.

## Latest Failures

| time | source | status | message | log |
| --- | --- | --- | --- | --- |
| 2026-05-20T05:56:38Z | finetune_smoke | dependency_error | Missing dependency: isaacgym. Run bash scripts/create_compat_env.sh with ISAAC_GYM_ROOT pointing at Isaac Gym Preview 4, then rerun through scripts/run_in_compat_env.sh. | logs/finetune_smoke_20260520T055638Z.log |
| 2026-05-20T05:56:38Z | train_scratch_smoke | dependency_error | Missing dependency: isaacgym. Run bash scripts/create_compat_env.sh with ISAAC_GYM_ROOT pointing at Isaac Gym Preview 4, then rerun through scripts/run_in_compat_env.sh. | logs/train_scratch_smoke_20260520T055638Z.log |
| 2026-05-20T05:56:38Z | run_dextoolbench_eval | dependency_error | Missing dependency: isaacgym. Run bash scripts/create_compat_env.sh with ISAAC_GYM_ROOT pointing at Isaac Gym Preview 4, then rerun through scripts/run_in_compat_env.sh. | logs/run_dextoolbench_eval_20260520T055638Z.log |
| 2026-05-20T05:56:38Z | run_pretrained_eval | dependency_error | Missing dependency: isaacgym. Run bash scripts/create_compat_env.sh with ISAAC_GYM_ROOT pointing at Isaac Gym Preview 4, then rerun through scripts/run_in_compat_env.sh. | logs/run_pretrained_eval_20260520T055638Z.log |
| 2026-05-20T05:53:43Z | validate_compat_env | dependency_error | Compatibility env exists, but at least one required import failed. See log. | logs/validate_compat_env_20260520T055343Z.log |
