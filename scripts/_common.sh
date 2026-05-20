#!/usr/bin/env bash

simtoolreal_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

simtoolreal_timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

require_python_module() {
  local module="$1"
  python -c "import ${module}" >/dev/null 2>&1
}

write_json_report() {
  local report_path="$1"
  local status="$2"
  local exit_code="$3"
  local message="$4"
  local command="$5"
  local log_path="$6"
  local started_at="$7"
  local ended_at="$8"
  local extra_json="${9:-}"
  if [[ -z "${extra_json}" ]]; then
    extra_json="{}"
  fi

  REPORT_PATH="${report_path}" \
  STATUS="${status}" \
  EXIT_CODE="${exit_code}" \
  MESSAGE="${message}" \
  COMMAND="${command}" \
  LOG_PATH="${log_path}" \
  STARTED_AT="${started_at}" \
  ENDED_AT="${ended_at}" \
  EXTRA_JSON="${extra_json}" \
  python - <<'PY'
import json
import os
from pathlib import Path

try:
    extra = json.loads(os.environ.get("EXTRA_JSON") or "{}")
except json.JSONDecodeError as exc:
    extra = {"extra_json_error": str(exc), "raw_extra_json": os.environ.get("EXTRA_JSON")}

payload = {
    "status": os.environ["STATUS"],
    "exit_code": int(os.environ["EXIT_CODE"]),
    "message": os.environ["MESSAGE"],
    "command": os.environ["COMMAND"],
    "log_path": os.environ["LOG_PATH"],
    "started_at_utc": os.environ["STARTED_AT"],
    "ended_at_utc": os.environ["ENDED_AT"],
}
payload.update(extra)
path = Path(os.environ["REPORT_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

  local root
  root="$(simtoolreal_repo_root)"
  if [[ -f "${root}/scripts/record_experiment.py" ]]; then
    python "${root}/scripts/record_experiment.py" --report-path "${report_path}" || {
      echo "WARNING: failed to append experiment result for ${report_path}" >&2
    }
  fi
}

quote_command() {
  printf '%q ' "$@"
}
