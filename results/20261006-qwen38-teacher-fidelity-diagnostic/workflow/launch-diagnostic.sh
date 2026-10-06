#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic
HOST=qwen38-teacher-fidelity-check-20261006
finish() {
  local status=$?
  trap - EXIT
  set +e
  printf '%s\n' "$status" > "$ROOT/diagnostic-workflow-exit.txt"
  timeout 120 scp -q "$HOST:qwen-fidelity.log" "$ROOT/remote-fidelity.log"
  mkdir -p "$ROOT/observed"
  timeout 120 rsync -a --include='*/' --include='*.json' --include='*.jsonl' --include='*.log' --include='*.txt' --exclude='*.pt' --exclude='*private*' --exclude='*' "$HOST:qwen-mtp-run/" "$ROOT/observed/"
  exit "$status"
}
trap finish EXIT
python3 "$ROOT/check-ready.py"
LIMIT=$(python3 - "$ROOT" <<'PY'
from pathlib import Path
import json,sys,datetime,math
r=json.loads((Path(sys.argv[1])/'provider-create-request.json').read_text())
assert r['auto_delete']['spend_threshold']=='6.60'
deadline=datetime.datetime.fromisoformat(r['auto_delete']['date_threshold'].replace('Z','+00:00'))
remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
assert remaining>=2400, 'Insufficient time for diagnostic and 20-minute cleanup reserve'
print(min(3000,math.floor(remaining)-1200))
PY
)
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" "bash -lc 'export USE_HUB_KERNELS=NO; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\${LD_LIBRARY_PATH:-}; exec nohup timeout --signal=TERM --kill-after=120s ${LIMIT}s ~/qwen-mtp-env/bin/python ~/qwen-mtp-code/diagnose-fidelity.py > ~/qwen-fidelity.log 2>&1'"
