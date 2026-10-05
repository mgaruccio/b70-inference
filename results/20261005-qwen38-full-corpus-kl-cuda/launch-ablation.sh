#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261005-qwen38-full-corpus-kl-cuda
HOST=qwen38-mtp-h100-kl-20261005
REMOTE_STARTED=0
finish() {
  local result=$?
  trap - EXIT
  set +e
  printf '%s\n' "$result" > "$ROOT/training-workflow-exit.txt"
  if (( result == 255 )); then
    printf 'SSH transport lost; remote terminal state unknown. Preserve capped owned lease for inspection, not a failed-training claim.\n'
    exit "$result"
  fi
  if (( REMOTE_STARTED == 0 )); then
    timeout 900 ssh "$HOST" 'python3 -c '\''import pathlib,runpy; runpy.run_path(str(pathlib.Path.home()/"qwen-mtp-code/run-pilot.py"),run_name="archive_only")["archive"]("final")'\'''
  fi
  timeout 120 scp -q "$HOST:qwen-pilot.log" "$ROOT/remote-pilot.log"
  for name in comparison.json stock-baseline.json state-drift.json live-gate.json cache-restore.json failure.json; do
    timeout 120 scp -q "$HOST:qwen-mtp-run/$name" "$ROOT/$name"
  done
  timeout 120 scp -q "$HOST:qwen-mtp-run/training/tuned-mtp.json" "$ROOT/training-report.json"
  if (( result == 0 )); then
    python3 "$ROOT/verify-final.py"
  else
    python3 "$ROOT/verify-final.py" --allow-incomplete
  fi
  backup=$?
  if (( backup == 0 )); then
    timeout 120 python3 "$ROOT/provider.py" --delete
    deleted=$?
    if (( deleted != 0 )); then result=1; fi
  else
    printf 'Final archive verification failed; named capped lease retained for immediate rescue.\n'
    result=1
  fi
  exit "$result"
}
trap finish EXIT
python3 "$ROOT/check-ready.py"
ssh -o ConnectTimeout=20 "$HOST" 'test "$(cat ~/qwen-mtp-run/bootstrap-exit.txt)" = 0; test -f ~/qwen-mtp-run/cache-restore.json'
# Keep 15 minutes before the provider cap for archive/upload/readback and cleanup.
BUDGET=$(python3 - "$ROOT" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]);request=json.loads((root/'provider-create-request.json').read_text())
deadline=datetime.datetime.fromisoformat(request['auto_delete']['date_threshold'].replace('Z','+00:00'))
remaining=int((deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds())-900
assert remaining>=9000,'Insufficient capped lease time for the full-data experiment; no short-update fallback'
print(remaining)
PY
)
printf 'LIVE_PARITY_THEN_FULL_CACHED374_64_KL_TEN_EPOCHS3740_UPDATES_NO_RECAPTURE budget_seconds=%s\n' "$BUDGET"
REMOTE_STARTED=1
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" "bash -lc 'exec nohup timeout --signal=TERM --kill-after=120s $BUDGET env LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} OMP_NUM_THREADS=8 ~/qwen-mtp-env/bin/python -u ~/qwen-mtp-code/run-pilot.py > ~/qwen-pilot.log 2>&1'"
printf 'FULL_CACHED_KL_WORKFLOW_FINISHED_BACKUP_BEFORE_GPU_DELETION\n'
