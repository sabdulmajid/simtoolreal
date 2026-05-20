#!/usr/bin/env python3
"""Print and save local system diagnostics for SimToolReal.

This script is intentionally read-only. It imports packages only for discovery
and never creates an Isaac Gym environment or starts training.
"""

from __future__ import annotations

import datetime as _datetime
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


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
        return (
            None,
            (stdout or "").strip(),
            "Timed out after {}s. {}".format(timeout, (stderr or "").strip()),
        )


def python_probe(code: str, timeout: int = 15) -> Tuple[bool, str]:
    rc, stdout, stderr = run_command([sys.executable, "-c", code], timeout=timeout)
    output = "\n".join(part for part in (stdout, stderr) if part)
    return rc == 0, output


def read_os_release() -> str:
    path = Path("/etc/os-release")
    if not path.exists():
        return "Unavailable"
    values: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME", "Unavailable")


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    rc, stdout, stderr = run_command(["git", "rev-parse", "HEAD"])
    commit = stdout if rc == 0 else None
    rc, stdout, stderr = run_command(["git", "status", "--short"])
    dirty_status = stdout if rc == 0 else None
    return {
        "commit": commit,
        "dirty": bool(dirty_status),
        "status_short": dirty_status,
        "error": None if rc == 0 else (stderr or stdout),
    }


def collect_basic_system(repo_root: Path) -> Dict[str, Any]:
    return {
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "os": read_os_release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace(os.linesep, " "),
        "repo_root": str(repo_root),
        "git": get_git_info(repo_root),
    }


def print_basic_system(info: Dict[str, Any]) -> None:
    print_section("OS / Python")
    print("Timestamp UTC: {}".format(info["timestamp_utc"]))
    print("OS: {}".format(info["os"]))
    print("Kernel: {}".format(info["platform"]))
    print("Machine: {}".format(info["machine"]))
    print("Python executable: {}".format(info["python_executable"]))
    print("Python version: {}".format(info["python_version"]))
    print("Repo root: {}".format(info["repo_root"]))
    print("Git commit: {}".format(info["git"].get("commit")))
    print("Git dirty: {}".format(info["git"].get("dirty")))


def collect_torch_status() -> Dict[str, Any]:
    info: Dict[str, Any] = {"import_ok": False}
    try:
        import torch
    except Exception as exc:
        info["error"] = "{}: {}".format(type(exc).__name__, exc)
        return info

    info.update(
        {
            "import_ok": True,
            "version": torch.__version__,
            "file": getattr(torch, "__file__", "unknown"),
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_arch_list": [],
            "gpus": [],
        }
    )
    try:
        info["cuda_arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as exc:
        info["cuda_arch_list_error"] = "{}: {}".format(type(exc).__name__, exc)

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(idx)
                total_gib = props.total_memory / (1024**3)
                info["gpus"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "compute_capability": "sm_{}{}".format(
                            props.major, props.minor
                        ),
                        "total_vram_gib": total_gib,
                    }
                )
            except Exception as exc:
                info["gpus"].append(
                    {
                        "index": idx,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
    return info


def print_torch_status(info: Dict[str, Any]) -> None:
    print_section("PyTorch / CUDA")
    if not info.get("import_ok"):
        print("torch import: FAILED ({})".format(info.get("error", "unknown")))
        return
    print("torch version: {}".format(info["version"]))
    print("torch file: {}".format(info["file"]))
    print("torch.version.cuda: {}".format(info["torch_cuda_version"]))
    print("CUDA available: {}".format(info["cuda_available"]))
    print("CUDA device count: {}".format(info["cuda_device_count"]))
    if "cuda_arch_list_error" in info:
        print("torch CUDA arch list: FAILED ({})".format(info["cuda_arch_list_error"]))
    else:
        print("torch CUDA arch list: {}".format(info["cuda_arch_list"]))
    for gpu in info.get("gpus", []):
        if "error" in gpu:
            print("GPU {}: FAILED ({})".format(gpu.get("index"), gpu["error"]))
        else:
            print(
                "GPU {index}: {name}, capability {compute_capability}, "
                "VRAM {total_vram_gib:.2f} GiB".format(**gpu)
            )


def collect_nvidia_smi() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": shutil.which("nvidia-smi") is not None}
    if not info["available"]:
        info["error"] = "nvidia-smi not found on PATH"
        return info
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ]
    rc, stdout, stderr = run_command(query)
    info["gpu_query_returncode"] = rc
    info["gpu_query_stdout"] = stdout
    info["gpu_query_stderr"] = stderr
    info["gpus"] = []
    if rc == 0:
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                info["gpus"].append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_total_mib": parts[2],
                        "memory_used_mib": parts[3],
                        "driver_version": parts[4],
                    }
                )

    rc, stdout, stderr = run_command(["nvidia-smi"])
    info["summary_returncode"] = rc
    info["summary_stdout"] = stdout
    info["summary_stderr"] = stderr
    return info


def print_nvidia_smi(info: Dict[str, Any]) -> None:
    print_section("nvidia-smi")
    if not info.get("available"):
        print(info.get("error", "nvidia-smi unavailable"))
        return
    if info.get("gpu_query_returncode") == 0:
        print("GPU query columns: index, name, memory.total MiB, memory.used MiB, driver")
        print(info.get("gpu_query_stdout") or "(no output)")
    else:
        print(
            "GPU query failed: {}".format(
                info.get("gpu_query_stderr") or info.get("gpu_query_stdout")
            )
        )
    print()
    if info.get("summary_returncode") == 0:
        print(info.get("summary_stdout", ""))
    else:
        print(
            "nvidia-smi summary failed: {}".format(
                info.get("summary_stderr") or info.get("summary_stdout")
            )
        )


def module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def collect_import_status() -> Dict[str, Any]:
    probes = [
        (
            "Isaac Gym",
            "isaacgym",
            "import isaacgym; print(getattr(isaacgym, '__file__', '<namespace>'))",
        ),
        (
            "Isaac Lab (isaaclab)",
            "isaaclab",
            "import isaaclab; print(getattr(isaaclab, '__file__', '<namespace>'))",
        ),
        (
            "Isaac Lab (omni.isaac.lab)",
            "omni.isaac.lab",
            "import omni.isaac.lab as lab; print(getattr(lab, '__file__', '<namespace>'))",
        ),
        (
            "rl_games package",
            "rl_games.torch_runner",
            "import rl_games.torch_runner as tr; print(getattr(tr, '__file__', '<namespace>'))",
        ),
    ]
    results: Dict[str, Any] = {}
    for label, spec_name, code in probes:
        ok, output = python_probe(code)
        results[label] = {
            "spec_name": spec_name,
            "find_spec": module_exists(spec_name),
            "import_ok": ok,
            "output": output,
        }
    return results


def print_import_status(info: Dict[str, Any]) -> None:
    print_section("Package Imports")
    for label, result in info.items():
        print("{}:".format(label))
        print("  find_spec({!r}): {}".format(result["spec_name"], result["find_spec"]))
        status = "OK" if result["import_ok"] else "FAILED"
        print("  import: {}".format(status))
        output = result.get("output", "")
        if output:
            for line in output.splitlines():
                print("    {}".format(line))


def collect_package_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {}
    for name, module_name in [
        ("numpy", "numpy"),
        ("hydra", "hydra"),
        ("omegaconf", "omegaconf"),
    ]:
        code = (
            "import {module}; "
            "print(getattr({module}, '__version__', '<no __version__>'))"
        ).format(module=module_name)
        ok, output = python_probe(code)
        versions[name] = {
            "import_ok": ok,
            "version": output.splitlines()[-1] if ok and output else None,
            "output": output,
        }
    return versions


def collect_repo_paths(repo_root: Path) -> Dict[str, Any]:
    required_dirs = [
        "assets",
        "deployment",
        "dextoolbench",
        "docs",
        "isaacgymenvs",
        "recorded_data",
        "rl_games",
    ]
    pretrained_files = ["pretrained_policy/config.yaml", "pretrained_policy/model.pth"]
    return {
        "required_dirs": {
            item: (repo_root / item).exists() for item in required_dirs
        },
        "pretrained_policy_files": {
            item: (repo_root / item).exists() for item in pretrained_files
        },
        "dextoolbench_data_exists": (repo_root / "dextoolbench" / "data").exists(),
    }


def collect_environment_hints() -> Dict[str, str]:
    return {
        key: os.environ.get(key, "")
        for key in (
        "CUDA_VISIBLE_DEVICES",
        "ISAAC_GYM_ROOT",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        )
    }


def print_package_versions(info: Dict[str, Any]) -> None:
    print_section("Package Versions")
    for name, result in info.items():
        if result["import_ok"]:
            print("{}: {}".format(name, result["version"]))
        else:
            print("{}: FAILED ({})".format(name, result.get("output", "")))


def print_repo_paths(info: Dict[str, Any]) -> None:
    print_section("Repo Artifacts")
    print("Required dirs:")
    for path, exists in info["required_dirs"].items():
        print("  {}: {}".format(path, exists))
    print("Pretrained policy files:")
    for path, exists in info["pretrained_policy_files"].items():
        print("  {}: {}".format(path, exists))
    print("DexToolBench data exists: {}".format(info["dextoolbench_data_exists"]))


def print_environment_hints(info: Dict[str, str]) -> None:
    print_section("Environment")
    for key, value in info.items():
        print("{}: {}".format(key, value if value else "<unset>"))


def collect_compatibility_warnings(report: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    python_version = sys.version_info
    if python_version >= (3, 12):
        warnings.append(
            "Active Python is {}.{}. Isaac Gym Preview 4 workflows in this repo "
            "expect Python 3.8.".format(python_version.major, python_version.minor)
        )

    torch_info = report.get("torch", {})
    arch_list = set(torch_info.get("cuda_arch_list") or [])
    for gpu in torch_info.get("gpus") or []:
        capability = gpu.get("compute_capability")
        name = gpu.get("name", "unknown GPU")
        if capability and arch_list and capability not in arch_list:
            warnings.append(
                "{} reports {}, but this torch build advertises {}. "
                "A newer PyTorch/CUDA stack may be required before GPU kernels "
                "run on this device.".format(name, capability, sorted(arch_list))
            )

    imports = report.get("imports", {})
    isaacgym = imports.get("Isaac Gym", {})
    if not isaacgym.get("import_ok"):
        warnings.append(
            "isaacgym does not import. Install Isaac Gym Preview 4 into the active "
            "Python 3.8 environment before running evaluation or training."
        )

    rl_games = imports.get("rl_games package", {})
    if not rl_games.get("import_ok"):
        warnings.append(
            "rl_games.torch_runner does not import. Install the repo-local rl_games "
            "fork with 'pip install -e rl_games --no-deps'."
        )

    artifacts = report.get("repo_artifacts", {})
    pretrained = artifacts.get("pretrained_policy_files", {})
    missing_policy = [path for path, exists in pretrained.items() if not exists]
    if missing_policy:
        warnings.append(
            "Missing pretrained policy files: {}. Run 'bash scripts/download_assets.sh' "
            "inside the compatibility environment.".format(", ".join(missing_policy))
        )
    return warnings


def print_compatibility_warnings(warnings: List[str]) -> None:
    print_section("Compatibility Warnings")
    if not warnings:
        print("No compatibility warnings detected.")
        return
    for warning in warnings:
        print("- {}".format(warning))


def collect_report(repo_root: Path) -> Dict[str, Any]:
    report = {
        "system": collect_basic_system(repo_root),
        "torch": collect_torch_status(),
        "nvidia_smi": collect_nvidia_smi(),
        "imports": collect_import_status(),
        "package_versions": collect_package_versions(),
        "repo_artifacts": collect_repo_paths(repo_root),
        "environment": collect_environment_hints(),
    }
    report["compatibility_warnings"] = collect_compatibility_warnings(report)
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(str(repo_root))
    report = collect_report(repo_root)

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "system_check.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print_basic_system(report["system"])
    print_torch_status(report["torch"])
    print_nvidia_smi(report["nvidia_smi"])
    print_import_status(report["imports"])
    print_package_versions(report["package_versions"])
    print_repo_paths(report["repo_artifacts"])
    print_environment_hints(report["environment"])
    print_compatibility_warnings(report["compatibility_warnings"])
    print_section("Report")
    print("Wrote {}".format(report_path))


if __name__ == "__main__":
    main()
