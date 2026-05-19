#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh

started_at="$(simtoolreal_timestamp)"
stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_path="logs/launch_train_gpu1_${stamp}.log"
metrics_path="reports/metrics/launch_train_gpu1_${stamp}.json"
report_path="reports/launch_train_gpu1_${stamp}.json"
mkdir -p logs reports reports/metrics

GPU_ID=1
for module in tyro isaacgym rl_games.torch_runner; do
  if ! require_python_module "${module}"; then
    message="Missing dependency: ${module}. Activate the SimToolReal environment and install the repo, Isaac Gym, and local rl_games first."
    echo "${message}" | tee "${log_path}" >&2
    extra_json="{\"gpu_id\":${GPU_ID},\"mode\":\"scratch\",\"smoke_test\":\"${SMOKE_TEST:-0}\"}"
    write_json_report "${report_path}" "dependency_error" 1 "${message}" "bash scripts/launch_train_gpu1.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "${extra_json}"
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES=1

NUM_ENVS="${NUM_ENVS:-24576}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
SEED="${SEED:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu1_train}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-False}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-simtoolreal}"

args=(
  --custom-experiment-name "${EXPERIMENT_NAME}"
  --num-envs "${NUM_ENVS}"
  --num-blocks "${NUM_BLOCKS}"
  --seed "${SEED}"
  --wandb-project "${WANDB_PROJECT}"
)

case "${WANDB_ACTIVATE}" in
  1|true|True|TRUE|yes|Yes|YES) args+=(--wandb-activate) ;;
  *) args+=(--no-wandb-activate) ;;
esac

if [[ -n "${WANDB_ENTITY}" ]]; then
  args+=(--wandb-entity "${WANDB_ENTITY}")
fi

cmd=(python isaacgymenvs/launch_training.py "${args[@]}" "$@")
set +e
python scripts/run_with_metrics.py --gpu-id "${GPU_ID}" --log-path "${log_path}" --metrics-path "${metrics_path}" -- "${cmd[@]}"
exit_code=$?
set -e
ended_at="$(simtoolreal_timestamp)"
if [[ "${exit_code}" -eq 0 ]]; then
  status="success"
  message="GPU1 training command completed."
else
  status="failed"
  message="GPU1 training command failed. See log."
fi
extra_json="{\"gpu_id\":${GPU_ID},\"num_envs\":${NUM_ENVS},\"num_blocks\":${NUM_BLOCKS},\"sapg_block_size\":$((NUM_ENVS / NUM_BLOCKS)),\"seed\":${SEED},\"mode\":\"scratch\",\"smoke_test\":\"${SMOKE_TEST}\",\"metrics_path\":\"${metrics_path}\"}"
write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "CUDA_VISIBLE_DEVICES=${GPU_ID} $(quote_command "${cmd[@]}")" "${log_path}" "${started_at}" "${ended_at}" "${extra_json}"
exit "${exit_code}"
