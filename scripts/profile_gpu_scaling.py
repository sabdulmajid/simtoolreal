#!/usr/bin/env python3
"""Dry-run or launch short SimToolReal GPU scaling profiles.

The default mode only prints commands. Use --run to execute. When executing,
the script sets train.params.config.max_epochs to a small value so profiling
does not silently become a full training run.
"""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_ENV_COUNTS = (12288, 24576, 49152)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_command(
    num_envs: int,
    num_blocks: int,
    max_epochs: int,
    seed: int,
    experiment: str,
    checkpoint: Optional[str],
) -> List[str]:
    sapg_block_size = num_envs // num_blocks
    command = [
        sys.executable,
        "-m",
        "isaacgymenvs.train",
        "++task.env.useSparseReward=False",
        "headless=True",
        f"task.env.numEnvs={num_envs}",
        "train.params.config.minibatch_size=98304",
        "multi_gpu=False",
        "train.params.config.good_reset_boundary=0",
        "task.env.goodResetBoundary=0",
        "train.params.config.use_others_experience=lf",
        "train.params.config.off_policy_ratio=1.0",
        "train.params.config.expl_type=mixed_expl_learn_param",
        "train.params.config.expl_reward_type=entropy",
        f"train.params.config.expl_coef_block_size={sapg_block_size}",
        "train.params.config.expl_reward_coef_scale=0.005",
        "train.params.network.space.continuous.fixed_sigma=coef_cond",
        "wandb_activate=False",
        f"experiment=profile_{experiment}",
        f"hydra.run.dir=./train_dir/profile/{experiment}",
        "task=SimToolRealLSTMAsymmetric",
        "task.env.objectScaleNoiseMultiplierRange=[0.9,1.1]",
        "task.env.forceConsecutiveNearGoalSteps=True",
        "task.env.forceScale=20",
        "task.env.torqueScale=2.0",
        "task.env.objectAngVelPenaltyScale=0.0",
        f"train.params.config.max_epochs={max_epochs}",
        "train.params.config.save_frequency=0",
        f"seed={seed}",
    ]
    if checkpoint:
        command.append("checkpoint={}".format(checkpoint))
    return command


def check_imports() -> List[str]:
    missing = []
    for module in ("hydra", "omegaconf", "torch", "isaacgym", "rl_games.torch_runner"):
        code = f"import {module}"
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            missing.append(module)
    return missing


def query_gpu(gpu_id: int) -> Dict[str, Optional[float]]:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
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


def parse_training_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "training_fps": None,
        "final_reward": None,
        "final_loss": None,
    }
    if not log_path.exists():
        return metrics
    text = log_path.read_text(errors="replace")
    fps_matches = re.findall(r"(?:fps|FPS)[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    if fps_matches:
        metrics["training_fps"] = float(fps_matches[-1])
    reward_matches = re.findall(r"(?:reward|rewards)[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)", text)
    if reward_matches:
        metrics["final_reward"] = float(reward_matches[-1])
    loss_matches = re.findall(r"(?:loss|losses)[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)", text)
    if loss_matches:
        metrics["final_loss"] = float(loss_matches[-1])
    return metrics


def run_profile(
    cmd: List[str],
    gpu_id: int,
    log_path: Path,
    timeout_s: Optional[int],
) -> Dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_monotonic = time.time()
    started_at = utc_now()
    samples: List[Dict[str, Optional[float]]] = []
    with log_path.open("w") as log_file:
        log_file.write("Command:\n")
        log_file.write(" ".join(shlex.quote(part) for part in cmd))
        log_file.write("\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        timed_out = False
        while proc.poll() is None:
            samples.append(query_gpu(gpu_id))
            if timeout_s is not None and time.time() - start_monotonic > timeout_s:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(2.0)
        samples.append(query_gpu(gpu_id))
    elapsed = time.time() - start_monotonic
    ended_at = utc_now()
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
    parsed = parse_training_metrics(log_path)
    status = "timeout" if timed_out else ("success" if proc.returncode == 0 else "failed")
    return {
        "status": status,
        "exit_code": proc.returncode if proc.returncode is not None else -1,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_sec": round(elapsed, 3),
        "peak_vram_mib": max(mem_values) if mem_values else None,
        "avg_gpu_util_percent": sum(util_values) / len(util_values) if util_values else None,
        "peak_gpu_util_percent": max(util_values) if util_values else None,
        "training_fps": parsed["training_fps"],
        "final_reward": parsed["final_reward"],
        "final_loss": parsed["final_loss"],
        "log_path": str(log_path),
        "notes": "Timed out" if timed_out else "",
    }


def write_reports(results: List[Dict[str, Any]]) -> None:
    reports_dir = repo_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "gpu_scaling_results.json"
    csv_path = reports_dir / "gpu_scaling_results.csv"
    summary_path = reports_dir / "gpu_scaling_summary.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    fields = [
        "timestamp_utc",
        "status",
        "gpu_id",
        "num_envs",
        "num_blocks",
        "sapg_block_size",
        "seed",
        "smoke_test",
        "mode",
        "checkpoint",
        "started_at_utc",
        "ended_at_utc",
        "elapsed_sec",
        "exit_code",
        "peak_vram_mib",
        "avg_gpu_util_percent",
        "peak_gpu_util_percent",
        "training_fps",
        "final_reward",
        "final_loss",
        "log_path",
        "command",
        "notes",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})

    lines = [
        "# GPU Scaling Summary",
        "",
        "Generated: {}".format(utc_now()),
        "",
        "| status | gpu | num_envs | mode | exit | peak_vram_mib | avg_util | fps | log |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            "| {status} | {gpu_id} | {num_envs} | {mode} | {exit_code} | {peak_vram_mib} | {avg_gpu_util_percent} | {training_fps} | {log_path} |".format(
                **{key: result.get(key, "") for key in fields}
            )
        )
    summary_path.write_text("\n".join(lines) + "\n")
    print("Wrote {}".format(json_path))
    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(summary_path))
    recorder = repo_root() / "scripts" / "record_experiment.py"
    if recorder.exists():
        subprocess.run(
            [sys.executable, str(recorder), "--gpu-scaling-path", str(json_path)],
            check=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--num-envs", type=int, nargs="*", default=list(DEFAULT_ENV_COUNTS))
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Tag rows as smoke tests")
    parser.add_argument("--timeout-s", type=int, default=0, help="0 disables timeout")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint for finetune profiling")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of printing them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = utc_now()
    mode = "finetune" if args.checkpoint else "scratch"
    results: List[Dict[str, Any]] = []
    missing = check_imports() if args.run else []
    if missing:
        message = (
            "Missing Python dependencies: "
            + ", ".join(missing)
            + ". Activate the SimToolReal environment and install the repo first."
        )
        print("ERROR: {}".format(message), file=sys.stderr)
        for num_envs in args.num_envs:
            results.append(
                {
                    "timestamp_utc": timestamp,
                    "status": "dependency_error",
                    "gpu_id": args.gpu_id,
                    "num_envs": num_envs,
                    "num_blocks": args.num_blocks,
                    "sapg_block_size": num_envs // args.num_blocks
                    if num_envs % args.num_blocks == 0
                    else None,
                    "seed": args.seed,
                    "smoke_test": args.smoke_test,
                    "mode": mode,
                    "checkpoint": args.checkpoint or None,
                    "started_at_utc": timestamp,
                    "ended_at_utc": utc_now(),
                    "elapsed_sec": 0,
                    "exit_code": 1,
                    "peak_vram_mib": None,
                    "avg_gpu_util_percent": None,
                    "peak_gpu_util_percent": None,
                    "training_fps": None,
                    "final_reward": None,
                    "final_loss": None,
                    "log_path": None,
                    "command": None,
                    "notes": message,
                }
            )
        write_reports(results)
        return 1

    for num_envs in args.num_envs:
        if num_envs % args.num_blocks != 0:
            print(
                "ERROR: num_envs={} is not divisible by num_blocks={}".format(
                    num_envs, args.num_blocks
                ),
                file=sys.stderr,
            )
            return 2
        sapg_block_size = num_envs // args.num_blocks
        experiment = "envs_{}_gpu{}".format(num_envs, args.gpu_id)
        cmd = build_command(
            num_envs,
            args.num_blocks,
            args.max_epochs,
            args.seed,
            experiment,
            args.checkpoint or None,
        )
        printable = " ".join(shlex.quote(part) for part in cmd)
        if not args.run:
            print("CUDA_VISIBLE_DEVICES={} {}".format(args.gpu_id, printable))
            results.append(
                {
                    "timestamp_utc": timestamp,
                    "status": "dry_run",
                    "gpu_id": args.gpu_id,
                    "num_envs": num_envs,
                    "num_blocks": args.num_blocks,
                    "sapg_block_size": sapg_block_size,
                    "seed": args.seed,
                    "smoke_test": args.smoke_test,
                    "mode": mode,
                    "checkpoint": args.checkpoint or None,
                    "started_at_utc": None,
                    "ended_at_utc": None,
                    "elapsed_sec": 0,
                    "exit_code": 0,
                    "peak_vram_mib": None,
                    "avg_gpu_util_percent": None,
                    "peak_gpu_util_percent": None,
                    "training_fps": None,
                    "final_reward": None,
                    "final_loss": None,
                    "log_path": None,
                    "command": "CUDA_VISIBLE_DEVICES={} {}".format(
                        args.gpu_id, printable
                    ),
                    "notes": "Command was not executed.",
                }
            )
            continue
        log_path = repo_root() / "train_dir" / "profile" / f"{experiment}.log"
        timeout_s = None if args.timeout_s == 0 else args.timeout_s
        result = run_profile(cmd, args.gpu_id, log_path, timeout_s)
        result.update(
            {
                "timestamp_utc": timestamp,
                "gpu_id": args.gpu_id,
                "num_envs": num_envs,
                "num_blocks": args.num_blocks,
                "sapg_block_size": sapg_block_size,
                "seed": args.seed,
                "smoke_test": args.smoke_test,
                "mode": mode,
                "checkpoint": args.checkpoint or None,
                "command": "CUDA_VISIBLE_DEVICES={} {}".format(
                    args.gpu_id, printable
                ),
            }
        )
        print(
            "status={status} exit={exit_code} elapsed_sec={elapsed_sec} log={log_path}".format(
                **result
            )
        )
        results.append(result)
        if result["exit_code"] != 0:
            write_reports(results)
            return int(result["exit_code"])
    write_reports(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
