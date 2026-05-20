#!/usr/bin/env bash
set -euo pipefail

# Download and extract Isaac Gym Preview 4 outside this repository.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROOT="$(simtoolreal_repo_root)"
cd "${ROOT}"
mkdir -p logs reports

started_at="$(simtoolreal_timestamp)"
stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_path="logs/download_isaacgym_${stamp}.log"
report_path="reports/download_isaacgym.json"

ISAAC_GYM_URL="${ISAAC_GYM_URL:-https://developer.nvidia.com/isaac-gym-preview-4}"
EXTERNAL_DIR="${SIMTOOLREAL_EXTERNAL_DIR:-$(dirname "${ROOT}")/external}"
ARCHIVE_PATH="${EXTERNAL_DIR}/IsaacGym_Preview_4_Package.tar.gz"
EXTRACT_DIR="${EXTERNAL_DIR}/isaacgym_preview4"
ISAAC_GYM_ROOT="${EXTRACT_DIR}/isaacgym"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

mkdir -p "${EXTERNAL_DIR}"

{
  echo "Started: ${started_at}"
  echo "URL: ${ISAAC_GYM_URL}"
  echo "Archive: ${ARCHIVE_PATH}"
  echo "Extract dir: ${EXTRACT_DIR}"
} >"${log_path}"

if [[ ! -f "${ARCHIVE_PATH}" || "${FORCE_DOWNLOAD}" == "1" ]]; then
  echo "Downloading Isaac Gym Preview 4..." | tee -a "${log_path}"
  curl -L --fail -o "${ARCHIVE_PATH}" "${ISAAC_GYM_URL}" >>"${log_path}" 2>&1
else
  echo "Archive already exists; reusing ${ARCHIVE_PATH}" | tee -a "${log_path}"
fi

if [[ ! -d "${ISAAC_GYM_ROOT}/python" || "${FORCE_EXTRACT}" == "1" ]]; then
  echo "Extracting Isaac Gym Preview 4..." | tee -a "${log_path}"
  rm -rf "${EXTRACT_DIR}"
  mkdir -p "${EXTRACT_DIR}"
  tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}" >>"${log_path}" 2>&1
else
  echo "Extracted Isaac Gym already exists; reusing ${ISAAC_GYM_ROOT}" | tee -a "${log_path}"
fi

ended_at="$(simtoolreal_timestamp)"
if [[ -d "${ISAAC_GYM_ROOT}/python" ]]; then
  status="success"
  exit_code=0
  message="Isaac Gym Preview 4 is available at ${ISAAC_GYM_ROOT}."
else
  status="failed"
  exit_code=1
  message="Isaac Gym extraction did not produce ${ISAAC_GYM_ROOT}/python."
fi

write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "bash scripts/download_isaacgym.sh" "${log_path}" "${started_at}" "${ended_at}" "{\"mode\":\"isaacgym_download\",\"isaac_gym_root\":\"${ISAAC_GYM_ROOT}\",\"archive_path\":\"${ARCHIVE_PATH}\"}"

if [[ "${exit_code}" -eq 0 ]]; then
  echo "Use with:"
  echo "  export ISAAC_GYM_ROOT=${ISAAC_GYM_ROOT}"
  echo "  bash scripts/create_compat_env.sh"
fi

exit "${exit_code}"
