#!/usr/bin/env python3
"""Aggregate SimToolReal experiment ledger rows into JSON, CSV, and Markdown."""

from __future__ import annotations

import csv
import datetime as _datetime
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FIELDS = [
    "run_id",
    "recorded_at_utc",
    "source",
    "status",
    "exit_code",
    "mode",
    "smoke_test",
    "git_commit",
    "git_dirty",
    "environment_name",
    "python_executable",
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "cuda_available",
    "gpu_id",
    "gpu_name",
    "num_envs",
    "num_blocks",
    "sapg_block_size",
    "seed",
    "checkpoint_path",
    "started_at_utc",
    "ended_at_utc",
    "elapsed_sec",
    "log_path",
    "metrics_path",
    "peak_vram_mib",
    "avg_gpu_util_percent",
    "peak_gpu_util_percent",
    "throughput_fps",
    "final_metric",
    "final_reward",
    "final_loss",
    "command",
    "message",
    "notes",
    "source_report_path",
    "git_status_short",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_ledger(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(repair_row(json.loads(line)))
        except Exception as exc:
            rows.append(
                {
                    "run_id": "parse_error_{}".format(len(rows)),
                    "status": "ledger_parse_error",
                    "message": "{}: {}".format(type(exc).__name__, exc),
                }
            )
    return rows


def parse_first(patterns: Iterable[str], text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def as_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def repair_row(row: Dict[str, Any]) -> Dict[str, Any]:
    command = str(row.get("command") or "")
    source = str(row.get("source") or "")
    if not row.get("mode"):
        if "finetune" in source or "checkpoint=" in command:
            row["mode"] = "finetune"
        elif "train_scratch" in source or "scratch" in source or "sweep_num_envs" in source:
            row["mode"] = "scratch"
        elif "create_compat_env" in source:
            row["mode"] = "environment_setup"
        elif "validate_compat_env" in source:
            row["mode"] = "environment_validation"
        elif "pretrained" in source:
            row["mode"] = "pretrained_eval"
        elif "dextoolbench" in source:
            row["mode"] = "dextoolbench_eval"
    if row.get("num_blocks") in (None, ""):
        row["num_blocks"] = as_int(parse_first([r"--num-blocks[= ]([0-9]+)"], command))
    if row.get("sapg_block_size") in (None, ""):
        num_envs = as_int(row.get("num_envs"))
        num_blocks = as_int(row.get("num_blocks"))
        if num_envs and num_blocks:
            row["sapg_block_size"] = num_envs // num_blocks
    return row


def sort_key(row: Dict[str, Any]) -> str:
    return str(row.get("started_at_utc") or row.get("recorded_at_utc") or "")


def numeric(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def best_row(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    measured_modes = {"scratch", "finetune", "pretrained_eval", "dextoolbench_eval", "profile"}
    candidates = [
        row
        for row in rows
        if row.get("status") == "success" and row.get("mode") in measured_modes
    ]
    metric_candidates = [
        row for row in candidates if numeric(row, "final_metric") is not None
    ]
    if metric_candidates:
        return max(metric_candidates, key=lambda row: numeric(row, "final_metric") or float("-inf"))
    reward_candidates = [
        row for row in candidates if numeric(row, "final_reward") is not None
    ]
    if reward_candidates:
        return max(reward_candidates, key=lambda row: numeric(row, "final_reward") or float("-inf"))
    throughput_candidates = [
        row for row in candidates if numeric(row, "throughput_fps") is not None
    ]
    if throughput_candidates:
        return max(
            throughput_candidates,
            key=lambda row: numeric(row, "throughput_fps") or float("-inf"),
        )
    return candidates[-1] if candidates else None


def format_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ")


def csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ")
    return value


def write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    mode_counts = Counter(str(row.get("mode") or "unknown") for row in rows)
    best = best_row(rows)
    latest_failures = [
        row
        for row in reversed(rows)
        if row.get("status") not in ("success", "dry_run", "skipped")
    ][:5]

    lines = [
        "# Experiment Summary",
        "",
        "Generated: {}".format(utc_now()),
        "",
        "Total rows: {}".format(len(rows)),
        "",
        "## Status Counts",
        "",
    ]
    if status_counts:
        for status, count in sorted(status_counts.items()):
            lines.append("- `{}`: {}".format(status, count))
    else:
        lines.append("- No experiment rows recorded yet.")

    lines.extend(["", "## Mode Counts", ""])
    if mode_counts:
        for mode, count in sorted(mode_counts.items()):
            lines.append("- `{}`: {}".format(mode, count))
    else:
        lines.append("- No modes recorded yet.")

    lines.extend(["", "## Best Recorded Run", ""])
    if best is None:
        lines.append("No successful measured evaluation, training, or profile run has been recorded yet.")
    else:
        lines.append("- run_id: `{}`".format(best.get("run_id")))
        lines.append("- status: `{}`".format(best.get("status")))
        lines.append("- mode: `{}`".format(best.get("mode")))
        lines.append("- final_metric: `{}`".format(format_value(best.get("final_metric"))))
        lines.append("- final_reward: `{}`".format(format_value(best.get("final_reward"))))
        lines.append("- throughput_fps: `{}`".format(format_value(best.get("throughput_fps"))))
        lines.append("- log_path: `{}`".format(format_value(best.get("log_path"))))
        lines.append("")
        lines.append("Reproduce with:")
        lines.append("")
        lines.append("```bash")
        lines.append(format_value(best.get("command")))
        lines.append("```")
        lines.append("")
        lines.append("Environment evidence:")
        lines.append("- git_commit: `{}`".format(format_value(best.get("git_commit"))))
        lines.append("- git_dirty: `{}`".format(format_value(best.get("git_dirty"))))
        lines.append("- environment_name: `{}`".format(format_value(best.get("environment_name"))))
        lines.append("- python_version: `{}`".format(format_value(best.get("python_version"))))
        lines.append("- torch_version: `{}`".format(format_value(best.get("torch_version"))))
        lines.append("- torch_cuda_version: `{}`".format(format_value(best.get("torch_cuda_version"))))
        lines.append("- gpu: `{}` `{}`".format(format_value(best.get("gpu_id")), format_value(best.get("gpu_name"))))

    lines.extend(["", "## Latest Failures", ""])
    if latest_failures:
        lines.append("| time | source | status | message | log |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in latest_failures:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    format_value(row.get("started_at_utc") or row.get("recorded_at_utc")),
                    format_value(row.get("source")),
                    format_value(row.get("status")),
                    format_value(row.get("message")),
                    format_value(row.get("log_path")),
                )
            )
    else:
        lines.append("No failed runs recorded.")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    root = repo_root()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = load_ledger(reports / "experiment_runs.jsonl")
    rows = sorted(rows, key=sort_key)

    json_path = reports / "experiment_results.json"
    csv_path = reports / "experiment_results.csv"
    summary_path = reports / "experiment_summary.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in FIELDS})

    write_summary(rows, summary_path)
    print("Wrote {}".format(json_path))
    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
