#!/usr/bin/env bash
set -euo pipefail

# Run a command inside the local Python 3.8 SimToolReal compatibility env.
# The env is intentionally outside the repo by default so it is never committed.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROOT="$(simtoolreal_repo_root)"
ENV_PATH="${SIMTOOLREAL_ENV_PATH:-/pub7/neel2/conda_envs/simtoolreal-py38}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: SIMTOOLREAL_ENV_PATH=${ENV_PATH} bash scripts/run_in_compat_env.sh <command> [args...]" >&2
  exit 2
fi

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  echo "Compatibility env not found at ${ENV_PATH}." >&2
  echo "Create it with: bash scripts/create_compat_env.sh" >&2
  exit 1
fi

export VIRTUAL_ENV="${ENV_PATH}"
export CONDA_PREFIX="${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
cd "${ROOT}"
exec "$@"
