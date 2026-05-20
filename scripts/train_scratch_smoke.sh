#!/usr/bin/env bash
set -euo pipefail

# Environment variables:
#   GPU_ID=0
#   SMOKE_TEST=1
#   NUM_ENVS=768       Must be divisible by NUM_BLOCKS.
#   NUM_BLOCKS=6
#   MAX_EPOCHS=1
#   SEED=42

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROOT="$(simtoolreal_repo_root)"
cd "${ROOT}"
mkdir -p logs reports

started_at="$(simtoolreal_timestamp)"
stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_path="logs/train_scratch_smoke_${stamp}.log"
metrics_path="reports/metrics/train_scratch_smoke_${stamp}.json"
report_path="reports/train_scratch_smoke.json"

GPU_ID="${GPU_ID:-0}"
SMOKE_TEST="${SMOKE_TEST:-1}"
NUM_ENVS="${NUM_ENVS:-768}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
SEED="${SEED:-42}"

if (( NUM_ENVS % NUM_BLOCKS != 0 )); then
  message="NUM_ENVS=${NUM_ENVS} must be divisible by NUM_BLOCKS=${NUM_BLOCKS}."
  echo "${message}" | tee "${log_path}" >&2
  extra_json="{\"gpu_id\":${GPU_ID},\"num_envs\":${NUM_ENVS},\"num_blocks\":${NUM_BLOCKS},\"seed\":${SEED},\"mode\":\"scratch\",\"smoke_test\":\"${SMOKE_TEST}\"}"
  write_json_report "${report_path}" "config_error" 2 "${message}" "bash scripts/train_scratch_smoke.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "${extra_json}"
  exit 2
fi

for module in isaacgym torch rl_games.torch_runner hydra omegaconf; do
  if ! require_python_module "${module}"; then
    message="Missing dependency: ${module}. Run bash scripts/create_compat_env.sh with ISAAC_GYM_ROOT pointing at Isaac Gym Preview 4, then rerun through scripts/run_in_compat_env.sh."
    echo "${message}" | tee "${log_path}" >&2
    extra_json="{\"gpu_id\":${GPU_ID},\"num_envs\":${NUM_ENVS},\"num_blocks\":${NUM_BLOCKS},\"sapg_block_size\":$((NUM_ENVS / NUM_BLOCKS)),\"max_epochs\":${MAX_EPOCHS},\"seed\":${SEED},\"mode\":\"scratch\",\"smoke_test\":\"${SMOKE_TEST}\"}"
    write_json_report "${report_path}" "dependency_error" 1 "${message}" "bash scripts/train_scratch_smoke.sh" "${log_path}" "${started_at}" "$(simtoolreal_timestamp)" "${extra_json}"
    exit 1
  fi
done

SAPG_BLOCK_SIZE=$((NUM_ENVS / NUM_BLOCKS))
cmd=(python -m isaacgymenvs.train ++task.env.useSparseReward=False headless=True "task.env.numEnvs=${NUM_ENVS}" train.params.config.minibatch_size=98304 multi_gpu=False train.params.config.good_reset_boundary=0 task.env.goodResetBoundary=0 train.params.config.use_others_experience=lf train.params.config.off_policy_ratio=1.0 train.params.config.expl_type=mixed_expl_learn_param train.params.config.expl_reward_type=entropy "train.params.config.expl_coef_block_size=${SAPG_BLOCK_SIZE}" train.params.config.expl_reward_coef_scale=0.005 train.params.network.space.continuous.fixed_sigma=coef_cond wandb_activate=False "experiment=1_smoke_scratch_${NUM_ENVS}" "hydra.run.dir=./train_dir/smoke/scratch_${NUM_ENVS}_${stamp}" task=SimToolRealLSTMAsymmetric 'task.env.objectScaleNoiseMultiplierRange=[0.9,1.1]' task.env.forceConsecutiveNearGoalSteps=True task.env.forceScale=20 task.env.torqueScale=2.0 task.env.objectAngVelPenaltyScale=0.0 "train.params.config.max_epochs=${MAX_EPOCHS}" train.params.config.save_frequency=0 "seed=${SEED}")

set +e
python scripts/run_with_metrics.py --gpu-id "${GPU_ID}" --log-path "${log_path}" --metrics-path "${metrics_path}" -- "${cmd[@]}"
exit_code=$?
set -e

ended_at="$(simtoolreal_timestamp)"
if [[ "${exit_code}" -eq 0 ]]; then
  status="success"
  message="Scratch smoke training command completed."
elif grep -Eqi "CUDA error: no kernel image|no kernel image is available for execution" "${log_path}"; then
  status="gpu_runtime_error"
  message="Scratch smoke reached CUDA execution, but torch cannot run kernels on the visible GPU. See log."
elif grep -Eqi "ModuleNotFoundError|No module named" "${log_path}"; then
  status="dependency_error"
  message="Scratch smoke failed on a missing Python dependency. See log."
else
  status="failed"
  message="Scratch smoke training command failed. See log."
fi
extra_json="{\"gpu_id\":${GPU_ID},\"num_envs\":${NUM_ENVS},\"num_blocks\":${NUM_BLOCKS},\"sapg_block_size\":${SAPG_BLOCK_SIZE},\"max_epochs\":${MAX_EPOCHS},\"seed\":${SEED},\"mode\":\"scratch\",\"smoke_test\":\"${SMOKE_TEST}\",\"metrics_path\":\"${metrics_path}\"}"
write_json_report "${report_path}" "${status}" "${exit_code}" "${message}" "CUDA_VISIBLE_DEVICES=${GPU_ID} $(quote_command "${cmd[@]}")" "${log_path}" "${started_at}" "${ended_at}" "${extra_json}"
exit "${exit_code}"
