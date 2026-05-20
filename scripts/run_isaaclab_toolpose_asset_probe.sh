#!/usr/bin/env bash

set -u
set -o pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

root="$(simtoolreal_repo_root)"
cd "${root}"

mkdir -p logs reports

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
report_path="${REPORT_PATH:-reports/isaaclab_toolpose_asset_probe.json}"
metrics_path="${METRICS_PATH:-reports/isaaclab_toolpose_asset_probe_metrics.json}"
log_path="${LOG_PATH:-logs/isaaclab_toolpose_asset_probe_${timestamp}.log}"

conda_sh="${CONDA_SH:-/pub7/neel/miniconda3/etc/profile.d/conda.sh}"
default_clean_env="/pub7/neel2/conda_envs/simtoolreal-isaaclab-blackwell"
if [[ -z "${ISAACLAB_CONDA_ENV:-}" && -d "${default_clean_env}" ]]; then
  conda_env="${default_clean_env}"
else
  conda_env="${ISAACLAB_CONDA_ENV:-isaaclab2}"
fi

select_isaaclab_root() {
  if [[ -n "${ISAACLAB_ROOT:-}" ]]; then
    echo "${ISAACLAB_ROOT}"
    return
  fi
  local candidate
  for candidate in /pub7/neel/vlm/IsaacLab /pub7/neel2/isaaclab_ws/IsaacLab; do
    [[ -x "${candidate}/isaaclab.sh" ]] || continue
    if grep -q "isaacsim_5" "${candidate}/source/isaaclab/isaaclab/app/app_launcher.py" 2>/dev/null \
      && [[ ! -d "${candidate}/apps/isaacsim_5" ]]; then
      continue
    fi
    echo "${candidate}"
    return
  done
  echo "/pub7/neel2/isaaclab_ws/IsaacLab"
}

isaaclab_root="$(select_isaaclab_root)"
gpu_id="${GPU_ID:-0}"
visible_device="${VISIBLE_DEVICE:-0}"
num_envs="${NUM_ENVS:-4}"
steps="${STEPS:-8}"
seed="${SEED:-42}"
generated_asset_dir="${GENERATED_ASSET_DIR:-/tmp/simtoolreal_isaaclab_assets}"

started_at="$(simtoolreal_timestamp)"

if [[ -n "${ACCEPT_EULA:-}" && -z "${OMNI_KIT_ACCEPT_EULA:-}" ]]; then
  export OMNI_KIT_ACCEPT_EULA="${ACCEPT_EULA}"
fi

if [[ ! -f "${conda_sh}" ]]; then
  ended_at="$(simtoolreal_timestamp)"
  write_json_report \
    "${report_path}" \
    "failed" \
    "1" \
    "Missing conda activation script: ${conda_sh}" \
    "source ${conda_sh} && conda activate ${conda_env}" \
    "${log_path}" \
    "${started_at}" \
    "${ended_at}" \
    "{\"mode\":\"isaaclab_toolpose_asset_probe\",\"smoke_test\":true,\"metrics_path\":\"${metrics_path}\"}"
  exit 1
fi

if [[ ! -x "${isaaclab_root}/isaaclab.sh" ]]; then
  ended_at="$(simtoolreal_timestamp)"
  write_json_report \
    "${report_path}" \
    "failed" \
    "1" \
    "Missing Isaac Lab launcher: ${isaaclab_root}/isaaclab.sh" \
    "${isaaclab_root}/isaaclab.sh -p scripts/validate_isaaclab_toolpose_assets.py" \
    "${log_path}" \
    "${started_at}" \
    "${ended_at}" \
    "{\"mode\":\"isaaclab_toolpose_asset_probe\",\"smoke_test\":true,\"metrics_path\":\"${metrics_path}\"}"
  exit 1
fi

unset VIRTUAL_ENV
source "${conda_sh}"
conda activate "${conda_env}"

cmd=(
  "${isaaclab_root}/isaaclab.sh"
  -p
  "${root}/scripts/validate_isaaclab_toolpose_assets.py"
  --headless
  --device
  "cuda:${visible_device}"
  --gpu-id
  "${gpu_id}"
  --num-envs
  "${num_envs}"
  --steps
  "${steps}"
  --seed
  "${seed}"
  --generated-asset-dir
  "${generated_asset_dir}"
  --report-path
  "${metrics_path}"
)

{
  echo "started_at_utc=${started_at}"
  echo "conda_env=${conda_env}"
  echo "isaaclab_root=${isaaclab_root}"
  echo "CUDA_VISIBLE_DEVICES=${gpu_id}"
  echo "OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-unset}"
  echo "generated_asset_dir=${generated_asset_dir}"
  echo "command=$(quote_command "${cmd[@]}")"
  echo
} >"${log_path}"

CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${root}:${PYTHONPATH:-}" "${cmd[@]}" >>"${log_path}" 2>&1
exit_code=$?
ended_at="$(simtoolreal_timestamp)"

status="failed"
message="SimToolReal ToolPose Isaac Lab asset/state probe failed. See log."
extra_json="{\"mode\":\"isaaclab_toolpose_asset_probe\",\"smoke_test\":true,\"metrics_path\":\"${metrics_path}\",\"gpu_id\":\"${gpu_id}\",\"num_envs\":${num_envs},\"seed\":${seed}}"

if [[ -f "${metrics_path}" ]]; then
  parsed="$(
    METRICS_PATH="${metrics_path}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["METRICS_PATH"])
payload = json.loads(path.read_text())
metrics = payload.get("metrics", {})
probe = payload.get("asset_probe", {})
robot = probe.get("robot", {})
state_tensors = probe.get("state_tensors", {})
out = {
    "status": payload.get("status", "failed"),
    "message": payload.get("message", ""),
    "extra": {
        "mode": payload.get("mode", "isaaclab_toolpose_asset_probe"),
        "smoke_test": True,
        "metrics_path": str(path),
        "gpu_id": payload.get("gpu_id"),
        "num_envs": payload.get("num_envs"),
        "seed": payload.get("seed"),
        "throughput_fps": metrics.get("throughput_fps"),
        "peak_vram_mib": metrics.get("peak_vram_mib"),
        "avg_gpu_util_percent": metrics.get("avg_gpu_util_percent"),
        "peak_gpu_util_percent": metrics.get("peak_gpu_util_percent"),
        "robot_num_joints": robot.get("num_joints"),
        "robot_num_bodies": robot.get("num_bodies"),
        "joint_order_matches_isaacgym": robot.get("joint_order_matches_isaacgym"),
        "policy_observation_shape": state_tensors.get("policy_observation_shape"),
        "notes": "SimToolReal ToolPose asset/state probe, not a trained policy run.",
    },
}
print(json.dumps(out, sort_keys=True))
PY
  )"
  status="$(PARSED="${parsed}" python - <<'PY'
import json, os
print(json.loads(os.environ["PARSED"]).get("status", "failed"))
PY
)"
  message="$(PARSED="${parsed}" python - <<'PY'
import json, os
print(json.loads(os.environ["PARSED"]).get("message", ""))
PY
)"
  extra_json="$(PARSED="${parsed}" python - <<'PY'
import json, os
print(json.dumps(json.loads(os.environ["PARSED"]).get("extra", {}), sort_keys=True))
PY
)"
fi

write_json_report \
  "${report_path}" \
  "${status}" \
  "${exit_code}" \
  "${message}" \
  "CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=${root} $(quote_command "${cmd[@]}")" \
  "${log_path}" \
  "${started_at}" \
  "${ended_at}" \
  "${extra_json}"

exit "${exit_code}"
