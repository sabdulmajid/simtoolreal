#!/usr/bin/env python3
"""Bounded Isaac Lab smoke validation for Blackwell GPUs.

This is not a SimToolReal port. It proves whether the local modern Isaac Lab /
PyTorch stack can launch a GPU simulator task and execute CUDA kernels on the
installed Blackwell GPU. The run is intentionally tiny and writes a structured
JSON report for auditability.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_command(cmd: List[str], timeout: int = 30) -> Tuple[Optional[int], str, str]:
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


def git_info() -> Dict[str, Any]:
    rc, stdout, stderr = run_command(["git", "rev-parse", "HEAD"])
    commit = stdout if rc == 0 else None
    rc, stdout, stderr = run_command(["git", "status", "--short"])
    status = stdout if rc == 0 else stderr
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status,
    }


def import_path(module_name: str) -> Dict[str, Any]:
    try:
        module = __import__(module_name)
        return {
            "import_ok": True,
            "file": getattr(module, "__file__", "<namespace>"),
        }
    except Exception as exc:
        return {
            "import_ok": False,
            "error": "{}: {}".format(type(exc).__name__, exc),
        }


def torch_probe(device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"import_ok": False, "kernel_smoke_ok": False}
    try:
        import torch
    except Exception as exc:
        info["error"] = "{}: {}".format(type(exc).__name__, exc)
        return info

    info.update(
        {
            "import_ok": True,
            "version": torch.__version__,
            "file": getattr(torch, "__file__", None),
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_arch_list": [],
            "device": device,
            "devices": [],
        }
    )
    try:
        info["cuda_arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as exc:
        info["cuda_arch_list_error"] = "{}: {}".format(type(exc).__name__, exc)

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(index)
                info["devices"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "compute_capability": "sm_{}{}".format(
                            props.major, props.minor
                        ),
                        "total_vram_mib": int(props.total_memory / (1024**2)),
                    }
                )
            except Exception as exc:
                info["devices"].append(
                    {
                        "index": index,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
    try:
        tensor = (torch.ones((1024,), device=device) + 1.0).sum()
        torch.cuda.synchronize()
        info["kernel_smoke_ok"] = True
        info["kernel_smoke_result"] = float(tensor.detach().cpu())
    except Exception as exc:
        info["kernel_smoke_error"] = "{}: {}".format(type(exc).__name__, exc)
    return info


def pip_check() -> Dict[str, Any]:
    rc, stdout, stderr = run_command([sys.executable, "-m", "pip", "check"], timeout=60)
    return {
        "returncode": rc,
        "ok": rc == 0,
        "stdout": stdout,
        "stderr": stderr,
    }


def nvidia_smi_gpu(gpu_id: str) -> Dict[str, Any]:
    rc, stdout, stderr = run_command(
        [
            "nvidia-smi",
            "-i",
            gpu_id,
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    info: Dict[str, Any] = {
        "returncode": rc,
        "stdout": stdout,
        "stderr": stderr,
    }
    if rc == 0 and stdout:
        parts = [part.strip() for part in stdout.splitlines()[0].split(",")]
        if len(parts) >= 6:
            info.update(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": int(float(parts[2])),
                    "memory_used_mib": int(float(parts[3])),
                    "utilization_gpu_percent": int(float(parts[4])),
                    "driver_version": parts[5],
                }
            )
    return info


class SmiSampler:
    def __init__(self, gpu_id: str, interval_sec: float = 1.0) -> None:
        self.gpu_id = gpu_id
        self.interval_sec = interval_sec
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = nvidia_smi_gpu(self.gpu_id)
            sample["timestamp_utc"] = utc_now()
            self.samples.append(sample)
            self._stop.wait(self.interval_sec)

    def summary(self) -> Dict[str, Any]:
        used = [
            sample.get("memory_used_mib")
            for sample in self.samples
            if isinstance(sample.get("memory_used_mib"), int)
        ]
        util = [
            sample.get("utilization_gpu_percent")
            for sample in self.samples
            if isinstance(sample.get("utilization_gpu_percent"), int)
        ]
        return {
            "sample_count": len(self.samples),
            "peak_vram_mib": max(used) if used else None,
            "avg_gpu_util_percent": (sum(util) / len(util)) if util else None,
            "peak_gpu_util_percent": max(util) if util else None,
            "samples": self.samples,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-path",
        default="reports/blackwell_isaaclab_validation.json",
        help="Path for the structured validation report.",
    )
    return parser


def run_isaaclab_cartpole(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "success": False,
    }
    try:
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)
        try:
            try:
                env.reset(seed=args.seed)
            except TypeError:
                env.reset()
            start = time.perf_counter()
            reward_mean = None
            for _ in range(args.steps):
                actions = (
                    2.0
                    * torch.rand(env.action_space.shape, device=env.unwrapped.device)
                    - 1.0
                )
                _, rewards, _, _, _ = env.step(actions)
                reward_mean = float(rewards.detach().mean().cpu())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            result.update(
                {
                    "success": True,
                    "elapsed_sec": elapsed,
                    "throughput_fps": (args.num_envs * args.steps / elapsed)
                    if elapsed > 0
                    else None,
                    "final_reward": reward_mean,
                    "env_device": str(env.unwrapped.device),
                }
            )
        finally:
            env.close()
    except Exception as exc:
        result.update(
            {
                "success": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    root = repo_root()
    os.chdir(str(root))

    parser = build_parser()
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        args, _ = parser.parse_known_args()
        report = {
            "status": "failed",
            "message": "Isaac Lab AppLauncher import failed.",
            "error": "{}: {}".format(type(exc).__name__, exc),
            "started_at_utc": utc_now(),
            "ended_at_utc": utc_now(),
            "system": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
            },
            "imports": {"isaaclab": import_path("isaaclab")},
        }
        write_report(root / args.report_path, report)
        print(json.dumps(report, sort_keys=True))
        return 1

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    started = utc_now()
    simulation_app = None
    sim_result: Dict[str, Any] = {}
    app_error: Optional[str] = None
    sampler = SmiSampler(str(args.gpu_id))
    sampler.start()
    try:
        launcher = AppLauncher(args)
        simulation_app = launcher.app
        sim_result = run_isaaclab_cartpole(args)
    except Exception as exc:
        app_error = "{}: {}".format(type(exc).__name__, exc)
        sim_result = {"success": False, "error": app_error, "traceback": traceback.format_exc()}
    finally:
        sampler.stop()

    torch_info = torch_probe(args.device)
    imports = {
        "isaaclab": import_path("isaaclab"),
        "isaaclab_tasks": import_path("isaaclab_tasks"),
        "isaacsim": import_path("isaacsim"),
        "gymnasium": import_path("gymnasium"),
    }
    pip = pip_check()
    smi = sampler.summary()
    sim_ok = bool(sim_result.get("success"))
    torch_ok = bool(torch_info.get("kernel_smoke_ok"))
    status = "success"
    message = "Isaac Lab bounded GPU simulation succeeded."
    exit_code = 0
    if not sim_ok or not torch_ok:
        status = "failed"
        exit_code = 1
        message = "Isaac Lab bounded GPU simulation or torch CUDA smoke failed."
    elif not pip.get("ok"):
        status = "success_with_dependency_conflicts"
        message = "Isaac Lab simulation succeeded, but pip check reports dependency conflicts."

    payload: Dict[str, Any] = {
        "status": status,
        "exit_code": exit_code,
        "message": message,
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "command": " ".join([sys.executable] + sys.argv),
        "mode": "blackwell_isaaclab_validation",
        "smoke_test": True,
        "gpu_id": str(args.gpu_id),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "task": args.task,
        "steps": args.steps,
        "device": args.device,
        "system": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "environment_name": os.environ.get("CONDA_DEFAULT_ENV")
            or Path(os.environ.get("VIRTUAL_ENV", "")).name
            or "system",
            "git": git_info(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "torch": torch_info,
        "imports": imports,
        "pip_check": pip,
        "nvidia_smi_before_after": {
            "after": nvidia_smi_gpu(str(args.gpu_id)),
        },
        "simulation": sim_result,
        "app_error": app_error,
        "metrics": {
            "peak_vram_mib": smi.get("peak_vram_mib"),
            "avg_gpu_util_percent": smi.get("avg_gpu_util_percent"),
            "peak_gpu_util_percent": smi.get("peak_gpu_util_percent"),
            "throughput_fps": sim_result.get("throughput_fps"),
            "final_reward": sim_result.get("final_reward"),
        },
        "gpu_samples": smi,
    }
    report_path = root / args.report_path
    write_report(report_path, payload)
    print(json.dumps(payload, sort_keys=True))
    if simulation_app is not None:
        try:
            simulation_app.close()
        except Exception:
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
