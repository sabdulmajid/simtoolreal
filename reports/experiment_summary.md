# Experiment Summary

Generated: 2026-05-19T00:31:02Z

Total rows: 9

## Status Counts

- `dependency_error`: 3
- `dry_run`: 4
- `missing_asset`: 2

## Mode Counts

- `asset_download`: 1
- `dextoolbench_eval`: 1
- `finetune`: 1
- `pretrained_eval`: 1
- `scratch`: 5

## Best Recorded Run

No successful run has been recorded yet.

## Latest Failures

| time | source | status | message | log |
| --- | --- | --- | --- | --- |
| 2026-05-19T00:30:28Z | finetune_smoke | missing_asset | Missing checkpoint: pretrained_policy/model.pth. Run bash scripts/download_assets.sh first. | logs/finetune_smoke_20260519T003028Z.log |
| 2026-05-19T00:30:16Z | train_scratch_smoke | dependency_error | Missing dependency: isaacgym. Activate the Isaac Gym SimToolReal environment first. | logs/train_scratch_smoke_20260519T003016Z.log |
| 2026-05-19T00:30:01Z | run_dextoolbench_eval | dependency_error | Missing dependency: isaacgym. Activate the Isaac Gym SimToolReal environment first. | logs/run_dextoolbench_eval_20260519T003001Z.log |
| 2026-05-19T00:29:53Z | run_pretrained_eval | missing_asset | Missing pretrained policy file: pretrained_policy/config.yaml. Run bash scripts/download_assets.sh first. | logs/run_pretrained_eval_20260519T002953Z.log |
| 2026-05-19T00:29:47Z | download_assets | dependency_error | Missing dependency: tyro. Activate the SimToolReal environment and run 'uv pip install -e .' first. | logs/download_assets_20260519T002947Z.log |
