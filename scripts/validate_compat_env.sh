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
bash scripts/run_in_compat_env.sh python - <<'PY' >>"${log_path}" 2>&1
import importlib
import json
import platform
import sys

checks = {}
checks["python"] = {"ok": sys.version_info[:2] == (3, 8), "version": sys.version}
for name in ["tyro", "torch", "rl_games.torch_runner", "isaacgym"]:
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
except Exception as exc:
    checks["torch_cuda"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(checks, indent=2, sort_keys=True))
missing = [key for key, value in checks.items() if isinstance(value, dict) and not value.get("ok", False)]
raise SystemExit(0 if not missing else 1)
PY
probe_exit=$?

bash scripts/run_in_compat_env.sh python scripts/check_system.py >>"${log_path}" 2>&1
check_exit=$?
set -e

ended_at="$(simtoolreal_timestamp)"
if [[ "${probe_exit}" -eq 0 && "${check_exit}" -eq 0 ]]; then
  status="success"
  exit_code=0
  message="Compatibility env validation passed."
else
  status="dependency_error"
  exit_code=1
  message="Compatibility env exists, but at least one required import failed. See log."
fi

write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "bash scripts/validate_compat_env.sh" "${log_path}" "${started_at}" "${ended_at}" "{\"env_path\":\"${ENV_PATH}\",\"probe_exit_code\":${probe_exit},\"system_check_exit_code\":${check_exit}}"
exit "${exit_code}"
