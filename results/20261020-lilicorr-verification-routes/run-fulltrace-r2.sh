#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?capture or after required}
case "$MODE" in capture|after) ;; *) exit 2;; esac
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no inference-host "bash -s -- $MODE" <<'BASH'
set -euo pipefail
cd /home/mike/b70-evals/20261020-lilicorr-verification-routes
if test "$1" = capture; then
    python3 -B - <<'PY'
import shutil
from pathlib import Path
root=Path('/home/mike/b70-evals/20261019-lilicorr-q8-drafter')
assert not (root/'verification-routes-20261020-r2').exists()
free=shutil.disk_usage(root).free
assert free >= 20*1024**3, ('fulltrace storage headroom',free)
print('FULLTRACE_STORAGE_PREFLIGHT_AVAILABLE_BYTES',free,flush=True)
PY
    python3 -B run-performance.py --lead-run-timing
else
    python3 -B run-performance.py --lead-after-timing-restore
fi
BASH
