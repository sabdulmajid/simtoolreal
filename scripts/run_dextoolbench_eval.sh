#!/usr/bin/env bash
set -euo pipefail

# Environment variables:
#   GPU_ID=0
#   SMOKE_TEST=1
#   SEED=
#   RUN_ALL=0                  Set to 1 to call dextoolbench/run_all_evals.py.
#   OBJECT_CATEGORY=hammer
#   OBJECT_NAME=claw_hammer
#   TASK_NAME=swing_down
#   NUM_EPISODES=1

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROOT="$(simtoolreal_repo_root)"
cd "${ROOT}"
mkdir -p logs reports

started_at="$(simtoolreal_timestamp)"
stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_path="logs/run_dextoolbench_eval_${stamp}.log"
metrics_path="reports/metrics/run_dextoolbench_eval_${stamp}.json"
report_path="reports/run_dextoolbench_eval.json"

GPU_ID="${GPU_ID:-0}"
SMOKE_TEST="${SMOKE_TEST:-1}"
SEED="${SEED:-}"
RUN_ALL="${RUN_ALL:-0}"
OBJECT_CATEGORY="${OBJECT_CATEGORY:-hammer}"
OBJECT_NAME="${OBJECT_NAME:-claw_hammer}"
TASK_NAME="${TASK_NAME:-swing_down}"
NUM_EPISODES="${NUM_EPISODES:-1}"
CONFIG_PATH="${CONFIG_PATH:-pretrained_policy/config.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-pretrained_policy/model.pth}"

for module in isaacgym torch rl_games.torch_runner tyro; do
  if ! require_python_module "${module}"; then
    message="Missing dependency: ${module}. Activate the Isaac Gym SimToolReal environment first."
    echo "${message}" | tee "${log_path}" >&2
    extra_json="{\"gpu_id\":${GPU_ID},\"mode\":\"dextoolbench_eval\",\"checkpoint_path\":\"${CHECKPOINT_PATH}\",\"seed\":\"${SEED}\",\"smoke_test\":\"${SMOKE_TEST}\"}"
    write_json_report "${report_path}" "dependency_error" 1 "${message}" "bash scripts/run_dextoolbench_eval.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "${extra_json}"
    exit 1
  fi
done

if [[ "${RUN_ALL}" == "1" ]]; then
  cmd=(python dextoolbench/run_all_evals.py)
else
  output_dir="evals/smoke/${OBJECT_CATEGORY}/${OBJECT_NAME}/${TASK_NAME}/pretrained_policy"
  cmd=(python dextoolbench/eval.py --object-category "${OBJECT_CATEGORY}" --object-name "${OBJECT_NAME}" --task-name "${TASK_NAME}" --checkpoint-path "${CHECKPOINT_PATH}" --config-path "${CONFIG_PATH}" --output-dir "${output_dir}" --num-episodes "${NUM_EPISODES}" --policy-name pretrained_policy)
fi

set +e
python scripts/run_with_metrics.py --gpu-id "${GPU_ID}" --log-path "${log_path}" --metrics-path "${metrics_path}" -- "${cmd[@]}"
exit_code=$?
set -e

ended_at="$(simtoolreal_timestamp)"
if [[ "${exit_code}" -eq 0 ]]; then
  status="success"
  message="DexToolBench evaluation command completed."
else
  status="failed"
  message="DexToolBench evaluation command failed. See log."
fi
extra_json="{\"gpu_id\":${GPU_ID},\"mode\":\"dextoolbench_eval\",\"run_all\":\"${RUN_ALL}\",\"object_category\":\"${OBJECT_CATEGORY}\",\"object_name\":\"${OBJECT_NAME}\",\"task_name\":\"${TASK_NAME}\",\"checkpoint_path\":\"${CHECKPOINT_PATH}\",\"seed\":\"${SEED}\",\"smoke_test\":\"${SMOKE_TEST}\",\"metrics_path\":\"${metrics_path}\"}"
write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "CUDA_VISIBLE_DEVICES=${GPU_ID} $(quote_command "${cmd[@]}")" "${log_path}" "${started_at}" "${ended_at}" "${extra_json}"
exit "${exit_code}"
