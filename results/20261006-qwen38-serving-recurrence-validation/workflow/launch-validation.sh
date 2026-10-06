#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261006-qwen38-serving-recurrence-validation
HOST=qwen38-mtp-h100-recurrence-check-20261006
python3 "$ROOT/check-ready.py"
BUDGET=$(python3 - "$ROOT" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]); request=json.loads((root/'provider-create-request.json').read_text())
end=datetime.datetime.fromisoformat(request['auto_delete']['date_threshold'].replace('Z','+00:00'))
remaining=int((end-datetime.datetime.now(datetime.timezone.utc)).total_seconds())
assert remaining>3600,'Insufficient time for validation plus archive/cleanup reserve'
print(min(5400,remaining-1800))
PY
)
finish() {
  local code=$?
  trap - EXIT
  set +e
  printf '%s\n' "$code" > "$ROOT/validation-workflow-exit.txt"
  timeout 120 scp -q "$HOST:qwen-validation.log" "$ROOT/remote-validation.log"
  ssh -o ConnectTimeout=15 "$HOST" 'if test -f ~/qwen-mtp-run/failure.json; then cat ~/qwen-mtp-run/failure.json; fi' > "$ROOT/failure-if-present.json"
  exit "$code"
}
trap finish EXIT
printf 'OWNED_VALIDATION_BUDGET_SECONDS=%s RESERVE1800 ZERO_OPTIMIZER_UPDATES\n' "$BUDGET"
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" "bash -lc 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\${LD_LIBRARY_PATH:-}; exec nohup timeout --signal=TERM --kill-after=120s ${BUDGET}s ~/qwen-mtp-env/bin/python ~/qwen-mtp-code/validate-recurrence.py > ~/qwen-validation.log 2>&1'"
printf 'REAL_CORRECTNESS_PROFILING_WORKFLOW_EXIT0_ZERO_UPDATES\n'
