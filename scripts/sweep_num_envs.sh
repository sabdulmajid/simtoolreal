#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh

GPU_ID="${GPU_ID:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
SEED="${SEED:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-False}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-simtoolreal}"
DRY_RUN="${DRY_RUN:-1}"

require_python_module() {
  local module="$1"
  python -c "import ${module}" >/dev/null 2>&1 || {
    echo "Missing dependency: ${module}. Activate the SimToolReal environment and install the repo, Isaac Gym, and local rl_games first." >&2
    exit 1
  }
}

if [[ "${DRY_RUN}" != "1" ]]; then
  require_python_module tyro
  require_python_module isaacgym
  require_python_module rl_games.torch_runner
fi

env_counts=(12288 24576 49152)

for num_envs in "${env_counts[@]}"; do
  if (( num_envs % NUM_BLOCKS != 0 )); then
    echo "Skipping ${num_envs}: not divisible by NUM_BLOCKS=${NUM_BLOCKS}" >&2
    continue
  fi

  cmd=(
    env "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    python isaacgymenvs/launch_training.py
    --custom-experiment-name "sweep_envs_${num_envs}_gpu${GPU_ID}"
    --num-envs "${num_envs}"
    --num-blocks "${NUM_BLOCKS}"
    --seed "${SEED}"
    --wandb-project "${WANDB_PROJECT}"
  )

  case "${WANDB_ACTIVATE}" in
    1|true|True|TRUE|yes|Yes|YES) cmd+=(--wandb-activate) ;;
    *) cmd+=(--no-wandb-activate) ;;
  esac

  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=(--wandb-entity "${WANDB_ENTITY}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${cmd[@]}" "$@"
    printf '\n'
    started_at="$(simtoolreal_timestamp)"
    ended_at="$(simtoolreal_timestamp)"
    stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
    report_path="reports/sweep_num_envs_${num_envs}_gpu${GPU_ID}_${stamp}.json"
    log_path="logs/sweep_num_envs_${num_envs}_gpu${GPU_ID}_${stamp}.log"
    mkdir -p logs reports
    printf 'DRY RUN:' >"${log_path}"
    printf ' %q' "${cmd[@]}" "$@" >>"${log_path}"
    printf '\n' >>"${log_path}"
    extra_json="{\"gpu_id\":${GPU_ID},\"num_envs\":${num_envs},\"num_blocks\":${NUM_BLOCKS},\"sapg_block_size\":$((num_envs / NUM_BLOCKS)),\"seed\":${SEED},\"mode\":\"scratch\",\"smoke_test\":\"${SMOKE_TEST}\"}"
    write_json_report "${report_path}" "dry_run" 0 "Sweep command was not executed." "$(printf '%q ' "${cmd[@]}" "$@")" "${log_path}" "${started_at}" "${ended_at}" "${extra_json}"
  else
    started_at="$(simtoolreal_timestamp)"
    stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
    log_path="logs/sweep_num_envs_${num_envs}_gpu${GPU_ID}_${stamp}.log"
    metrics_path="reports/metrics/sweep_num_envs_${num_envs}_gpu${GPU_ID}_${stamp}.json"
    report_path="reports/sweep_num_envs_${num_envs}_gpu${GPU_ID}_${stamp}.json"
    mkdir -p logs reports reports/metrics
    run_cmd=("${cmd[@]:2}" "$@")
    set +e
    python scripts/run_with_metrics.py --gpu-id "${GPU_ID}" --log-path "${log_path}" --metrics-path "${metrics_path}" -- "${run_cmd[@]}"
    exit_code=$?
    set -e
    ended_at="$(simtoolreal_timestamp)"
    if [[ "${exit_code}" -eq 0 ]]; then
      status="success"
      message="Sweep command completed."
    else
      status="failed"
      message="Sweep command failed. See log."
    fi
    extra_json="{\"gpu_id\":${GPU_ID},\"num_envs\":${num_envs},\"num_blocks\":${NUM_BLOCKS},\"sapg_block_size\":$((num_envs / NUM_BLOCKS)),\"seed\":${SEED},\"mode\":\"scratch\",\"smoke_test\":\"${SMOKE_TEST}\",\"metrics_path\":\"${metrics_path}\"}"
    write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "CUDA_VISIBLE_DEVICES=${GPU_ID} $(quote_command "${run_cmd[@]}")" "${log_path}" "${started_at}" "${ended_at}" "${extra_json}"
    if [[ "${exit_code}" -ne 0 ]]; then
      exit "${exit_code}"
    fi
  fi
done
