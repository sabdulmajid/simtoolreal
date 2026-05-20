#!/usr/bin/env python3
"""Run a command while sampling lightweight GPU and log metrics.

This helper is intentionally generic: it does not import SimToolReal, Isaac Gym,
or torch. It is safe to use around smoke tests that may fail during dependency
checks, and it exits with the wrapped command's exit code.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def query_gpu(gpu_id: str) -> Dict[str, Optional[float]]:
    if gpu_id == "":
        return {"memory_used_mib": None, "gpu_util_percent": None}
    command = [
        "nvidia-smi",
        "-i",
        gpu_id,
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return {"memory_used_mib": None, "gpu_util_percent": None}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"memory_used_mib": None, "gpu_util_percent": None}
    parts = [part.strip() for part in proc.stdout.splitlines()[0].split(",")]
    try:
        memory_used_mib = float(parts[0])
    except Exception:
        memory_used_mib = None
    try:
        gpu_util_percent = float(parts[1])
    except Exception:
        gpu_util_percent = None
    return {"memory_used_mib": memory_used_mib, "gpu_util_percent": gpu_util_percent}


def last_float(patterns: List[str], text: str) -> Optional[float]:
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def parse_log_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    if not log_path.exists():
        return {
            "throughput_fps": None,
            "final_metric": None,
            "final_reward": None,
            "final_loss": None,
        }
    text = log_path.read_text(errors="replace")
    throughput_fps = last_float(
        [
            r"\b(?:fps|FPS)\b[^0-9]*([0-9]+(?:\.[0-9]+)?)",
            r"\b(?:steps/s|steps_per_second)\b[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ],
        text,
    )
    final_reward = last_float(
        [
            r"\b(?:mean_reward|average_reward|reward|rewards)\b[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)",
        ],
        text,
    )
    final_loss = last_float(
        [r"\b(?:loss|losses)\b[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)"],
        text,
    )
    success_rate = last_float(
        [
            r"\b(?:success_rate|success rate|success)\b[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)",
        ],
        text,
    )
    final_metric = success_rate if success_rate is not None else final_reward
    return {
        "throughput_fps": throughput_fps,
        "final_metric": final_metric,
        "final_reward": final_reward,
        "final_loss": final_loss,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", default="", help="Host GPU index to sample.")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=0.0, help="0 disables timeout")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")
    return args


def main() -> int:
    args = parse_args()
    root = repo_root()
    log_path = (root / args.log_path).resolve() if not Path(args.log_path).is_absolute() else Path(args.log_path)
    metrics_path = (
        (root / args.metrics_path).resolve()
        if not Path(args.metrics_path).is_absolute()
        else Path(args.metrics_path)
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    start_monotonic = time.time()
    samples: List[Dict[str, Optional[float]]] = []
    env = os.environ.copy()
    if args.gpu_id != "":
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    timed_out = False
    exit_code = 1
    command_string = " ".join(shlex.quote(part) for part in args.command)
    with log_path.open("w") as log_file:
        log_file.write("Started UTC: {}\n".format(started_at))
        log_file.write("CUDA_VISIBLE_DEVICES={}\n".format(env.get("CUDA_VISIBLE_DEVICES", "")))
        log_file.write("Command: {}\n\n".format(command_string))
        log_file.flush()

        proc = subprocess.Popen(
            args.command,
            cwd=root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            samples.append(query_gpu(args.gpu_id))
            if args.timeout_s > 0 and time.time() - start_monotonic > args.timeout_s:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(max(args.sample_interval, 0.1))
        samples.append(query_gpu(args.gpu_id))
        exit_code = 124 if timed_out else int(proc.returncode or 0)
        ended_at = utc_now()
        log_file.write("\nEnded UTC: {}\n".format(ended_at))
        log_file.write("Exit code: {}\n".format(exit_code))

    elapsed = time.time() - start_monotonic
    mem_values = [
        sample["memory_used_mib"]
        for sample in samples
        if sample.get("memory_used_mib") is not None
    ]
    util_values = [
        sample["gpu_util_percent"]
        for sample in samples
        if sample.get("gpu_util_percent") is not None
    ]
    parsed = parse_log_metrics(log_path)
    if exit_code != 0:
        parsed["final_metric"] = None
        parsed["final_reward"] = None
        parsed["final_loss"] = None
    payload: Dict[str, Any] = {
        "status": "timeout" if timed_out else ("success" if exit_code == 0 else "failed"),
        "exit_code": exit_code,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_sec": round(elapsed, 3),
        "gpu_id": args.gpu_id,
        "log_path": relative_or_str(log_path, root),
        "command": command_string,
        "peak_vram_mib": max(mem_values) if mem_values else None,
        "avg_gpu_util_percent": sum(util_values) / len(util_values) if util_values else None,
        "peak_gpu_util_percent": max(util_values) if util_values else None,
        "sample_count": len(samples),
    }
    payload.update(parsed)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
