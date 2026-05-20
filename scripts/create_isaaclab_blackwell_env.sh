#!/usr/bin/env bash

set -u
set -o pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

root="$(simtoolreal_repo_root)"
cd "${root}"

mkdir -p logs reports

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
report_path="${REPORT_PATH:-reports/create_isaaclab_blackwell_env.json}"
log_path="${LOG_PATH:-logs/create_isaaclab_blackwell_env_${timestamp}.log}"

conda_sh="${CONDA_SH:-/pub7/neel/miniconda3/etc/profile.d/conda.sh}"
env_path="${ISAACLAB_ENV_PATH:-/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell}"
isaaclab_root="${ISAACLAB_ROOT:-/pub7/neel/vlm/IsaacLab}"
python_version="${ISAACLAB_PYTHON_VERSION:-3.11}"
force_recreate="${FORCE_RECREATE:-0}"

started_at="$(simtoolreal_timestamp)"
command_text="ISAACLAB_ENV_PATH=${env_path} ISAACLAB_ROOT=${isaaclab_root} bash scripts/create_isaaclab_blackwell_env.sh"

finish_report() {
  local status="$1"
  local exit_code="$2"
  local message="$3"
  local extra_json="${4:-}"
  if [[ -z "${extra_json}" ]]; then
    extra_json="{}"
  fi
  local ended_at
  ended_at="$(simtoolreal_timestamp)"
  write_json_report \
    "${report_path}" \
    "${status}" \
    "${exit_code}" \
    "${message}" \
    "${command_text}" \
    "${log_path}" \
    "${started_at}" \
    "${ended_at}" \
    "${extra_json}"
}

if [[ ! -f "${conda_sh}" ]]; then
  finish_report "failed" "1" "Missing conda activation script: ${conda_sh}" "{}"
  exit 1
fi

if [[ ! -d "${isaaclab_root}/source/isaaclab" ]]; then
  finish_report "failed" "1" "Missing Isaac Lab source tree: ${isaaclab_root}" "{}"
  exit 1
fi

{
  echo "started_at_utc=${started_at}"
  echo "env_path=${env_path}"
  echo "isaaclab_root=${isaaclab_root}"
  echo "python_version=${python_version}"
  echo "force_recreate=${force_recreate}"
  echo "command=${command_text}"
  echo
} >"${log_path}"

set +e
{
  unset VIRTUAL_ENV
  source "${conda_sh}"

  if [[ "${force_recreate}" == "1" && -d "${env_path}" ]]; then
    conda env remove -y -p "${env_path}"
  fi

  if [[ ! -d "${env_path}" ]]; then
    conda create -y -p "${env_path}" "python=${python_version}" importlib_metadata
  fi

  conda activate "${env_path}"
  python -m pip install --upgrade pip setuptools "wheel==0.45.1"

  # Isaac Sim 5.1 pins torch==2.7.0/torchvision==0.22.0/torchaudio==2.7.0.
  # The cu128 wheels advertise sm_120 and run on Blackwell.
  python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    "torchaudio==2.7.0"

  # RL + app + extscache is sufficient for headless RL-style Isaac Lab tasks
  # without pulling ROS, GUI examples, or all optional stacks.
  python -m pip install "isaacsim[rl,extscache]==5.1.0.0"

  python -m pip install -e "${isaaclab_root}/source/isaaclab"
  python -m pip install -e "${isaaclab_root}/source/isaaclab_assets"
  python -m pip install -e "${isaaclab_root}/source/isaaclab_tasks"

  # Isaac Lab imports its HDF5 dataset handler through isaaclab_tasks,
  # but this checkout does not declare h5py in its setup dependencies.
  python -m pip install "h5py>=3.10,<4"

  python -m pip check

  python - <<'PY'
import importlib.util
import json
import gymnasium
import isaaclab
import torch

payload = {
    "python_ok": True,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_arch_list": torch.cuda.get_arch_list(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "isaaclab": getattr(isaaclab, "__file__", None),
    "isaaclab_assets_find_spec": importlib.util.find_spec("isaaclab_assets") is not None,
    "isaaclab_tasks_find_spec": importlib.util.find_spec("isaaclab_tasks") is not None,
    "isaacsim_find_spec": importlib.util.find_spec("isaacsim") is not None,
    "gymnasium": getattr(gymnasium, "__file__", None),
}
if torch.cuda.is_available():
    value = (torch.ones((128,), device="cuda:0") + 1.0).sum()
    torch.cuda.synchronize()
    payload["cuda_kernel_smoke"] = float(value.cpu())
print(json.dumps(payload, sort_keys=True))
PY
} >>"${log_path}" 2>&1
exit_code=$?
set -e

extra_json="$(
  ENV_PATH="${env_path}" ISAACLAB_ROOT_PATH="${isaaclab_root}" LOG_PATH="${log_path}" python - <<'PY'
import json
import os
import re
from pathlib import Path

log_path = Path(os.environ["LOG_PATH"])
text = log_path.read_text(errors="replace") if log_path.exists() else ""
probe = None
for line in reversed(text.splitlines()):
    line = line.strip()
    if line.startswith("{") and "torch_version" in line:
        try:
            probe = json.loads(line)
        except Exception:
            pass
        break
payload = {
    "mode": "isaaclab_environment_setup",
    "environment_path": os.environ["ENV_PATH"],
    "isaaclab_root": os.environ["ISAACLAB_ROOT_PATH"],
    "probe": probe,
}
print(json.dumps(payload, sort_keys=True))
PY
)"

if ! EXTRA_JSON="${extra_json}" python - <<'PY' >/dev/null 2>&1
import json
import os

json.loads(os.environ["EXTRA_JSON"])
PY
then
  extra_json="$(
    ENV_PATH="${env_path}" ISAACLAB_ROOT_PATH="${isaaclab_root}" python - <<'PY'
import json
import os

print(json.dumps({
    "mode": "isaaclab_environment_setup",
    "environment_path": os.environ["ENV_PATH"],
    "isaaclab_root": os.environ["ISAACLAB_ROOT_PATH"],
    "probe": None,
    "probe_parse_error": True,
}, sort_keys=True))
PY
  )"
fi

if [[ "${exit_code}" -eq 0 ]] && grep -Eqi "has requirement|requires .* but you have" "${log_path}"; then
  finish_report "success_with_dependency_conflicts" "0" "Isaac Lab Blackwell environment imports and CUDA smoke pass, but pip check reports dependency conflicts." "${extra_json}"
  exit 0
fi

if [[ "${exit_code}" -eq 0 ]]; then
  finish_report "success" "0" "Clean Isaac Lab Blackwell environment created and pip check passed." "${extra_json}"
  exit 0
fi

if grep -qi "pip check" "${log_path}" && grep -qi "requires" "${log_path}"; then
  finish_report "failed" "${exit_code}" "Isaac Lab environment setup failed dependency validation. See log." "${extra_json}"
else
  finish_report "failed" "${exit_code}" "Isaac Lab environment setup failed. See log." "${extra_json}"
fi
exit "${exit_code}"
