#!/usr/bin/env bash
set -euo pipefail

# Build the Python 3.8 compatibility environment used by the original
# Isaac Gym Preview 4 SimToolReal stack. This does not install Isaac Gym unless
# ISAAC_GYM_ROOT points at an extracted Isaac Gym package.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROOT="$(simtoolreal_repo_root)"
cd "${ROOT}"
mkdir -p logs reports

started_at="$(simtoolreal_timestamp)"
stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_path="logs/create_compat_env_${stamp}.log"
report_path="reports/create_compat_env.json"

ENV_PATH="${SIMTOOLREAL_ENV_PATH:-/pub7/neel2/conda_envs/simtoolreal-py38}"
TORCH_VERSION="${SIMTOOLREAL_TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${SIMTOOLREAL_TORCHVISION_VERSION:-0.19.1}"
TORCH_INDEX_URL="${SIMTOOLREAL_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
FORCE_TORCH_REINSTALL="${SIMTOOLREAL_FORCE_TORCH_REINSTALL:-0}"
CONDA_EXE="${CONDA_EXE:-}"
EXTERNAL_DIR="${SIMTOOLREAL_EXTERNAL_DIR:-$(dirname "${ROOT}")/external}"
DEFAULT_ISAAC_GYM_ROOT="${EXTERNAL_DIR}/isaacgym_preview4/isaacgym"

find_conda() {
  if [[ -n "${CONDA_EXE}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  for candidate in /pub7/neel/miniconda3/bin/conda /pub3/neel/miniconda3/bin/conda "${HOME}/miniconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

status="failed"
message="Compatibility environment creation failed. See log."
exit_code=1
trap 'exit_code=$?; if [[ "${status}" != "success" ]]; then write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "bash scripts/create_compat_env.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "{\"env_path\":\"${ENV_PATH}\"}"; fi' EXIT

exec > >(tee "${log_path}") 2>&1

echo "Started: ${started_at}"
echo "Repo: ${ROOT}"
echo "Env path: ${ENV_PATH}"

if ! conda_bin="$(find_conda)"; then
  status="missing_dependency"
  message="Could not find conda. Set CONDA_EXE or install Miniconda/Mambaforge."
  echo "${message}" >&2
  exit 1
fi
echo "Conda: ${conda_bin}"

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  "${conda_bin}" create -y -p "${ENV_PATH}" python=3.8 pip
else
  echo "Environment already exists; reusing ${ENV_PATH}"
fi

PY="${ENV_PATH}/bin/python"
"${PY}" -m pip install -U pip setuptools wheel
torch_install_args=(--index-url "${TORCH_INDEX_URL}")
case "${FORCE_TORCH_REINSTALL}" in
  1|true|True|TRUE|yes|Yes|YES) torch_install_args+=(--force-reinstall) ;;
esac
"${PY}" -m pip install "${torch_install_args[@]}" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
"${PY}" -m pip install \
  tyro \
  hydra-core==1.3.2 \
  omegaconf==2.3.0 \
  gym==0.23.1 \
  termcolor \
  requests \
  tqdm \
  pyyaml \
  psutil \
  setproctitle \
  tensorboard \
  tensorboardx \
  wandb==0.12.21 \
  opencv-python==4.10.0.84

"${PY}" -m pip install \
  pytorch3d==0.3.0 \
  viser \
  trimesh==3.23.5 \
  transforms3d \
  matplotlib \
  pyopenssl \
  pyvirtualdisplay \
  pytorch-kinematics \
  yourdfpy

"${PY}" -m pip install -e "${ROOT}" --no-deps
"${PY}" -m pip install -e "${ROOT}/rl_games" --no-deps

isaacgym_status="not_requested"
if [[ -z "${ISAAC_GYM_ROOT:-}" && -d "${DEFAULT_ISAAC_GYM_ROOT}/python" ]]; then
  ISAAC_GYM_ROOT="${DEFAULT_ISAAC_GYM_ROOT}"
fi

if [[ -n "${ISAAC_GYM_ROOT:-}" ]]; then
  if [[ -d "${ISAAC_GYM_ROOT}/python" ]]; then
    "${PY}" -m pip install -e "${ISAAC_GYM_ROOT}/python"
    isaacgym_status="installed_from_${ISAAC_GYM_ROOT}/python"
  else
    isaacgym_status="invalid_ISAAC_GYM_ROOT_${ISAAC_GYM_ROOT}"
    status="dependency_error"
    message="ISAAC_GYM_ROOT is set, but ${ISAAC_GYM_ROOT}/python does not exist."
    echo "${message}" >&2
    exit 1
  fi
else
  echo "ISAAC_GYM_ROOT is unset; Isaac Gym Preview 4 was not installed." >&2
fi

bash scripts/run_in_compat_env.sh python scripts/check_system.py

export VIRTUAL_ENV="${ENV_PATH}"
export CONDA_PREFIX="${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

status="success"
message="Compatibility environment is ready; Isaac Gym status: ${isaacgym_status}."
write_json_report "${report_path}" "${status}" 0 "${message}" "bash scripts/create_compat_env.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "{\"env_path\":\"${ENV_PATH}\",\"torch_version\":\"${TORCH_VERSION}\",\"torch_index_url\":\"${TORCH_INDEX_URL}\",\"force_torch_reinstall\":\"${FORCE_TORCH_REINSTALL}\",\"isaacgym_status\":\"${isaacgym_status}\"}"
trap - EXIT

echo
echo "Use this environment with:"
echo "  bash scripts/run_in_compat_env.sh python scripts/check_system.py"
