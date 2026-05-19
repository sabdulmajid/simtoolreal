#!/usr/bin/env python3
"""Append normalized experiment rows to the SimToolReal run ledger."""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LEDGER_PATH = Path("reports/experiment_runs.jsonl")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_command(cmd: List[str], timeout: int = 15) -> Tuple[Optional[int], str, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError as exc:
        return None, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
        return None, (stdout or "").strip(), (stderr or "").strip()


def relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"status": "report_parse_error", "message": "{}: {}".format(type(exc).__name__, exc)}


def get_git_info() -> Dict[str, Any]:
    rc, stdout, stderr = run_command(["git", "rev-parse", "HEAD"])
    commit = stdout if rc == 0 else None
    rc, stdout, stderr = run_command(["git", "status", "--short"])
    status_short = stdout if rc == 0 else ""
    return {
        "git_commit": commit,
        "git_dirty": bool(status_short),
        "git_status_short": status_short,
    }


def get_environment_name() -> str:
    conda = os.environ.get("CONDA_DEFAULT_ENV")
    if conda:
        return conda
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv).name
    return "system"


def get_torch_info() -> Dict[str, Any]:
    code = (
        "import json, torch; "
        "print(json.dumps({'torch_version': torch.__version__, "
        "'torch_cuda_version': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available()}))"
    )
    rc, stdout, stderr = run_command([sys.executable, "-c", code], timeout=20)
    if rc != 0:
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": None,
            "torch_error": stderr or stdout,
        }
    try:
        return json.loads(stdout)
    except Exception:
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": None,
            "torch_error": stdout,
        }


def query_gpu_name(gpu_id: Any) -> Optional[str]:
    if gpu_id in (None, ""):
        return None
    rc, stdout, stderr = run_command(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
    )
    if rc == 0 and stdout:
        return stdout.splitlines()[0].strip()
    return None


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


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def infer_mode(report: Dict[str, Any], source: str, command: str) -> Optional[str]:
    if report.get("mode"):
        return str(report["mode"])
    if "finetune" in source or "checkpoint=" in command:
        return "finetune"
    if "train_scratch" in source or "scratch" in source or "sweep_num_envs" in source:
        return "scratch"
    if "pretrained" in source:
        return "pretrained_eval"
    if "dextoolbench" in source:
        return "dextoolbench_eval"
    if "download" in source:
        return "asset_download"
    if "gpu_scaling" in source or "profile" in source:
        return "profile"
    return None


def load_metrics(report: Dict[str, Any], root: Path) -> Dict[str, Any]:
    metrics_path = report.get("metrics_path")
    if not metrics_path:
        return {}
    path = Path(str(metrics_path))
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return {"metrics_path": str(metrics_path)}
    payload = load_json(path)
    payload["metrics_path"] = str(metrics_path)
    return payload


def normalize_common(report: Dict[str, Any], source_report_path: Optional[Path]) -> Dict[str, Any]:
    root = repo_root()
    source = source_report_path.stem if source_report_path is not None else "gpu_scaling"
    command = str(report.get("command") or "")
    metrics = load_metrics(report, root)
    merged: Dict[str, Any] = {}
    merged.update(report)
    merged.update({key: value for key, value in metrics.items() if value is not None})

    gpu_id = merged.get("gpu_id")
    if gpu_id in (None, ""):
        gpu_id = parse_first([r"CUDA_VISIBLE_DEVICES=([^ ]+)"], command)
    num_envs = merged.get("num_envs")
    if num_envs in (None, ""):
        num_envs = parse_first([r"task\.env\.numEnvs=([0-9]+)", r"--num-envs[= ]([0-9]+)"], command)
    num_blocks = merged.get("num_blocks")
    if num_blocks in (None, ""):
        num_blocks = parse_first([r"--num-blocks[= ]([0-9]+)"], command)
    sapg_block_size = merged.get("sapg_block_size")
    if sapg_block_size in (None, "") and as_int(num_envs) and as_int(num_blocks):
        sapg_block_size = as_int(num_envs) // as_int(num_blocks)
    seed = merged.get("seed")
    if seed in (None, ""):
        seed = parse_first([r"(?:^| )seed=([0-9]+)", r"--seed[= ]([0-9]+)"], command)
    checkpoint = merged.get("checkpoint_path") or merged.get("checkpoint")
    if not checkpoint:
        checkpoint = parse_first(
            [r"checkpoint=([^ ]+)", r"--checkpoint-path[= ]([^ ]+)"],
            command,
        )

    row: Dict[str, Any] = {
        "recorded_at_utc": utc_now(),
        "source": source,
        "source_report_path": relative_or_str(source_report_path, root) if source_report_path else None,
        "status": merged.get("status"),
        "exit_code": as_int(merged.get("exit_code")),
        "message": merged.get("message"),
        "command": command,
        "environment_name": get_environment_name(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "gpu_id": str(gpu_id) if gpu_id not in (None, "") else None,
        "gpu_name": query_gpu_name(gpu_id),
        "num_envs": as_int(num_envs),
        "num_blocks": as_int(num_blocks),
        "sapg_block_size": as_int(sapg_block_size),
        "seed": as_int(seed),
        "mode": infer_mode(merged, source, command),
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "started_at_utc": merged.get("started_at_utc"),
        "ended_at_utc": merged.get("ended_at_utc"),
        "elapsed_sec": as_float(merged.get("elapsed_sec")),
        "log_path": merged.get("log_path"),
        "metrics_path": merged.get("metrics_path"),
        "peak_vram_mib": as_float(merged.get("peak_vram_mib")),
        "avg_gpu_util_percent": as_float(merged.get("avg_gpu_util_percent")),
        "peak_gpu_util_percent": as_float(merged.get("peak_gpu_util_percent")),
        "throughput_fps": as_float(merged.get("throughput_fps") or merged.get("training_fps")),
        "final_metric": as_float(merged.get("final_metric")),
        "final_reward": as_float(merged.get("final_reward")),
        "final_loss": as_float(merged.get("final_loss")),
        "smoke_test": bool(str(merged.get("smoke_test", "")).lower() in ("1", "true", "yes")),
        "notes": merged.get("notes"),
    }
    row.update(get_git_info())
    row.update(get_torch_info())
    run_id_basis = json.dumps(
        {
            "source": row["source"],
            "started_at_utc": row["started_at_utc"],
            "ended_at_utc": row["ended_at_utc"],
            "command": row["command"],
            "log_path": row["log_path"],
            "status": row["status"],
        },
        sort_keys=True,
    )
    row["run_id"] = hashlib.sha1(run_id_basis.encode("utf-8")).hexdigest()[:16]
    return row


def existing_run_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if payload.get("run_id"):
            ids.add(payload["run_id"])
    return ids


def append_rows(rows: List[Dict[str, Any]], aggregate: bool) -> None:
    root = repo_root()
    ledger = root / LEDGER_PATH
    ledger.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_run_ids(ledger)
    new_rows = [row for row in rows if row.get("run_id") not in seen]
    if new_rows:
        with ledger.open("a") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    if aggregate:
        subprocess.run([sys.executable, str(root / "scripts/aggregate_results.py")], check=False)


def rows_from_gpu_scaling(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = []
    for result in payload:
        report = dict(result)
        report.setdefault("message", result.get("notes"))
        rows.append(normalize_common(report, path))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", help="Single wrapper JSON report to record.")
    parser.add_argument("--gpu-scaling-path", help="gpu_scaling_results.json to record.")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.report_path and not args.gpu_scaling_path:
        print("ERROR: --report-path or --gpu-scaling-path is required", file=sys.stderr)
        return 2
    root = repo_root()
    rows: List[Dict[str, Any]] = []
    if args.report_path:
        report_path = Path(args.report_path)
        if not report_path.is_absolute():
            report_path = root / report_path
        report = load_json(report_path)
        rows.append(normalize_common(report, report_path))
    if args.gpu_scaling_path:
        scaling_path = Path(args.gpu_scaling_path)
        if not scaling_path.is_absolute():
            scaling_path = root / scaling_path
        rows.extend(rows_from_gpu_scaling(scaling_path))
    append_rows(rows, aggregate=not args.no_aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
