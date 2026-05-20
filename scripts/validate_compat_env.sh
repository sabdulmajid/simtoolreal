#!/usr/bin/env bash
set -euo pipefail

# Validate the compatibility env without starting training.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROOT="$(simtoolreal_repo_root)"
cd "${ROOT}"
mkdir -p logs reports

started_at="$(simtoolreal_timestamp)"
stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_path="logs/validate_compat_env_${stamp}.log"
report_path="reports/validate_compat_env.json"
probe_path="$(mktemp)"
ENV_PATH="${SIMTOOLREAL_ENV_PATH:-/pub7/neel2/conda_envs/simtoolreal-py38}"

{
  echo "Started: ${started_at}"
  echo "Repo: ${ROOT}"
  echo "Env path: ${ENV_PATH}"
} >"${log_path}"

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  message="Compatibility env not found at ${ENV_PATH}. Run bash scripts/create_compat_env.sh first."
  echo "${message}" | tee -a "${log_path}" >&2
  write_json_report "${report_path}" "missing_environment" 1 "${message}" "bash scripts/validate_compat_env.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "{\"env_path\":\"${ENV_PATH}\"}"
  exit 1
fi

export VIRTUAL_ENV="${ENV_PATH}"
export CONDA_PREFIX="${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

set +e
bash scripts/run_in_compat_env.sh python - "${probe_path}" <<'PY' >>"${log_path}" 2>&1
import importlib
import json
import sys

probe_path = sys.argv[1]
checks = {}
checks["python"] = {"ok": sys.version_info[:2] == (3, 8), "version": sys.version}

# Isaac Gym must be imported before torch in processes that use both.
for name in ["isaacgym", "tyro", "torch", "rl_games.torch_runner"]:
    try:
        module = importlib.import_module(name)
        checks[name] = {"ok": True, "file": getattr(module, "__file__", "<namespace>")}
    except Exception as exc:
        checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

try:
    import torch
    checks["torch_cuda"] = {
        "ok": bool(torch.cuda.is_available()),
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "arch_list": list(torch.cuda.get_arch_list()),
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": idx,
                "name": torch.cuda.get_device_properties(idx).name,
                "capability": f"sm_{torch.cuda.get_device_properties(idx).major}{torch.cuda.get_device_properties(idx).minor}",
            }
            for idx in range(torch.cuda.device_count())
        ],
    }
    kernel_results = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(idx):
                    value = (torch.ones((4,), device="cuda") + 1).sum()
                    torch.cuda.synchronize()
                    kernel_results.append({"index": idx, "ok": True, "result": float(value.cpu())})
            except Exception as exc:
                kernel_results.append({"index": idx, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    checks["torch_cuda_kernel_smoke"] = {
        "ok": bool(kernel_results) and all(item.get("ok") for item in kernel_results),
        "results": kernel_results,
    }
except Exception as exc:
    checks["torch_cuda"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    checks["torch_cuda_kernel_smoke"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

print(json.dumps(checks, indent=2, sort_keys=True))
with open(probe_path, "w") as handle:
    json.dump(checks, handle, indent=2, sort_keys=True)
    handle.write("\n")
failed = [key for key, value in checks.items() if isinstance(value, dict) and not value.get("ok", False)]
raise SystemExit(0 if not failed else 1)
PY
probe_exit=$?

bash scripts/run_in_compat_env.sh python scripts/check_system.py >>"${log_path}" 2>&1
check_exit=$?
set -e

ended_at="$(simtoolreal_timestamp)"
classification="$(python - "${probe_path}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    checks = json.load(open(path))
except Exception as exc:
    print("failed|Could not parse compatibility probe output: {}: {}".format(type(exc).__name__, exc))
    raise SystemExit(0)

import_failures = [
    name
    for name in ("python", "isaacgym", "tyro", "torch", "rl_games.torch_runner")
    if not checks.get(name, {}).get("ok", False)
]
kernel = checks.get("torch_cuda_kernel_smoke", {})
if import_failures:
    print("dependency_error|Compatibility env exists, but required import(s) failed: {}.".format(", ".join(import_failures)))
elif not checks.get("torch_cuda", {}).get("ok", False):
    print("gpu_runtime_error|CUDA is not available to torch in the compatibility env.")
elif not kernel.get("ok", False):
    print("gpu_runtime_error|Torch imports, but a minimal CUDA kernel fails on the visible GPU(s).")
else:
    print("success|Compatibility env validation passed.")
PY
)"
status="${classification%%|*}"
message="${classification#*|}"

if [[ "${probe_exit}" -eq 0 && "${check_exit}" -eq 0 ]]; then
  status="success"
  exit_code=0
else
  exit_code=1
fi

write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "bash scripts/validate_compat_env.sh" "${log_path}" "${started_at}" "${ended_at}" "{\"env_path\":\"${ENV_PATH}\",\"probe_exit_code\":${probe_exit},\"system_check_exit_code\":${check_exit}}"
rm -f "${probe_path}"
exit "${exit_code}"
