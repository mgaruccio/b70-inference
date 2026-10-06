#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261006-qwen38-kl-dev-checkpoint-sweep
HOST=qwen38-mtp-h100-dev-sweep-20261006
REMOTE_STARTED=0
finish() {
  local result=$?
  trap - EXIT
  set +e
  printf '%s\n' "$result" > "$ROOT/sweep-workflow-exit.txt"
  if (( result == 255 )); then
    printf 'SSH transport lost; remote terminal state unknown. Preserve capped owned lease for inspection, not a failed-training claim.\n'
    exit "$result"
  fi
  if (( REMOTE_STARTED == 0 )); then
    timeout 900 ssh "$HOST" 'python3 -c '\''import pathlib,runpy; runpy.run_path(str(pathlib.Path.home()/"qwen-mtp-code/run-pilot.py"),run_name="archive_only")["archive"]("final")'\'''
  fi
  timeout 120 scp -q "$HOST:qwen-pilot.log" "$ROOT/remote-pilot.log"
  for name in sweep.json sweep-progress.json cache-restore.json failure.json; do
    timeout 120 scp -q "$HOST:qwen-mtp-run/$name" "$ROOT/$name"
  done
  # Saved training report is historical; this sweep performs zero optimizer updates.
  if (( result == 0 )); then
    python3 "$ROOT/verify-final.py"
  else
    python3 "$ROOT/verify-final.py" --allow-incomplete
  fi
  backup=$?
  if (( backup == 0 )); then
    if (( result == 0 )); then
      timeout 120 python3 "$ROOT/provider.py" --delete
      deleted=$?
      if (( deleted != 0 )); then result=1; fi
    else
      printf 'Failure archive verified; retain the SAME capped owned lease for immediate correction. Never recreate.\n'
    fi
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
assert remaining>=9000,'Insufficient capped lease time for all ten checkpoints and dev64 confirmation; no partial-sweep fallback'
print(remaining)
PY
)
printf 'DEV64_D8_ALL_TEN_SAVED_EPOCHS_ZERO_TRAINING budget_seconds=%s\n' "$BUDGET"
REMOTE_STARTED=1
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" "bash -lc 'exec nohup timeout --signal=TERM --kill-after=120s $BUDGET env LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} OMP_NUM_THREADS=8 ~/qwen-mtp-env/bin/python -u ~/qwen-mtp-code/sweep.py > ~/qwen-pilot.log 2>&1'"
printf 'FULL_DEV_CHECKPOINT_SWEEP_FINISHED_BACKUP_BEFORE_GPU_DELETION\n'
