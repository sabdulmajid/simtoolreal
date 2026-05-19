#!/usr/bin/env bash
set -euo pipefail

# Environment variables:
#   SMOKE_TEST=0                  Set to 1 to validate preflight only.
#   DOWNLOAD_PRETRAINED=1          Download pretrained_policy.zip.
#   DOWNLOAD_DEXTOOLBENCH=0        Download one DexToolBench task by default.
#   DOWNLOAD_ALL_DEXTOOLBENCH=0    Download the full DexToolBench dataset.
#   OBJECT_CATEGORY=hammer         DexToolBench category for one-task download.
#   OBJECT_NAME=claw_hammer        DexToolBench object for one-task download.
#   TASK_NAME=swing_down           DexToolBench task for one-task download.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROOT="$(simtoolreal_repo_root)"
cd "${ROOT}"
mkdir -p logs reports

started_at="$(simtoolreal_timestamp)"
stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_path="logs/download_assets_${stamp}.log"
report_path="reports/download_assets.json"

SMOKE_TEST="${SMOKE_TEST:-0}"
DOWNLOAD_PRETRAINED="${DOWNLOAD_PRETRAINED:-1}"
DOWNLOAD_DEXTOOLBENCH="${DOWNLOAD_DEXTOOLBENCH:-0}"
DOWNLOAD_ALL_DEXTOOLBENCH="${DOWNLOAD_ALL_DEXTOOLBENCH:-0}"
OBJECT_CATEGORY="${OBJECT_CATEGORY:-hammer}"
OBJECT_NAME="${OBJECT_NAME:-claw_hammer}"
TASK_NAME="${TASK_NAME:-swing_down}"

for module in requests tyro tqdm; do
  if ! require_python_module "${module}"; then
    message="Missing dependency: ${module}. Activate the SimToolReal environment and run 'uv pip install -e .' first."
    echo "${message}" | tee "${log_path}" >&2
    extra_json="{\"mode\":\"asset_download\",\"smoke_test\":\"${SMOKE_TEST}\"}"
    write_json_report "${report_path}" "dependency_error" 1 "${message}" "bash scripts/download_assets.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "${extra_json}"
    exit 1
  fi
done

{
  echo "Started: ${started_at}"
  echo "Repo: ${ROOT}"
  echo "SMOKE_TEST=${SMOKE_TEST}"
  echo "DOWNLOAD_PRETRAINED=${DOWNLOAD_PRETRAINED}"
  echo "DOWNLOAD_DEXTOOLBENCH=${DOWNLOAD_DEXTOOLBENCH}"
  echo "DOWNLOAD_ALL_DEXTOOLBENCH=${DOWNLOAD_ALL_DEXTOOLBENCH}"
} >"${log_path}"

commands=()
if [[ "${SMOKE_TEST}" == "1" ]]; then
  message="Smoke test completed dependency checks; no downloads requested."
  echo "${message}" | tee -a "${log_path}"
  extra_json="{\"mode\":\"asset_download\",\"smoke_test\":\"${SMOKE_TEST}\",\"download_pretrained\":\"${DOWNLOAD_PRETRAINED}\",\"download_dextoolbench\":\"${DOWNLOAD_DEXTOOLBENCH}\",\"download_all_dextoolbench\":\"${DOWNLOAD_ALL_DEXTOOLBENCH}\"}"
  write_json_report "${report_path}" "skipped" 0 "${message}" "SMOKE_TEST=1 bash scripts/download_assets.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "${extra_json}"
  exit 0
fi

if [[ "${DOWNLOAD_PRETRAINED}" == "1" ]]; then
  commands+=("python download_pretrained_policy.py")
fi

if [[ "${DOWNLOAD_DEXTOOLBENCH}" == "1" ]]; then
  if [[ "${DOWNLOAD_ALL_DEXTOOLBENCH}" == "1" ]]; then
    commands+=("python download_dextoolbench_data.py")
  else
    commands+=("python download_dextoolbench_data.py --object-category ${OBJECT_CATEGORY} --object-name ${OBJECT_NAME} --task-name ${TASK_NAME}")
  fi
fi

if [[ "${#commands[@]}" -eq 0 ]]; then
  message="Nothing requested. Set DOWNLOAD_PRETRAINED=1 or DOWNLOAD_DEXTOOLBENCH=1."
  echo "${message}" | tee -a "${log_path}" >&2
  extra_json="{\"mode\":\"asset_download\",\"smoke_test\":\"${SMOKE_TEST}\"}"
  write_json_report "${report_path}" "skipped" 0 "${message}" "bash scripts/download_assets.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "${extra_json}"
  exit 0
fi

exit_code=0
for cmd in "${commands[@]}"; do
  echo "Running: ${cmd}" | tee -a "${log_path}"
  set +e
  bash -lc "${cmd}" >>"${log_path}" 2>&1
  step_exit_code=$?
  set -e
  if [[ "${step_exit_code}" -ne 0 ]]; then
    exit_code="${step_exit_code}"
    break
  fi
done

ended_at="$(simtoolreal_timestamp)"
if [[ "${exit_code}" -eq 0 ]]; then
  status="success"
  message="Asset command(s) completed."
else
  status="failed"
  message="Asset command failed. See log."
fi

extra_json="{\"mode\":\"asset_download\",\"smoke_test\":\"${SMOKE_TEST}\",\"download_pretrained\":\"${DOWNLOAD_PRETRAINED}\",\"download_dextoolbench\":\"${DOWNLOAD_DEXTOOLBENCH}\",\"download_all_dextoolbench\":\"${DOWNLOAD_ALL_DEXTOOLBENCH}\"}"
write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "${commands[*]}" "${log_path}" "${started_at}" "${ended_at}" "${extra_json}"
exit "${exit_code}"
