#!/usr/bin/env bash
# Bounded REMOTE CPU ONLY. No Docker/API/native/build/hardware commands or cleanup.
set -euo pipefail
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity
[[ $(hostname) == inference-host ]]
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) == "$R" ]]
mkdir -p "$R/worker-scratch" "$R/preparation"
attempt=$(mktemp -d "$R/worker-scratch/cpu-XXXXXXXX")
export BOLT_LONG_CPU_SCRATCH="$attempt" TMPDIR="$attempt" PYTHONDONTWRITEBYTECODE=1
cd "$attempt"
printf 'CPU_ATTEMPT=%s\n' "$attempt"
{
    printf 'tier=development; synthetic CPU regressions, NOT live 8K E2E\n'
    printf 'command=timeout --signal=TERM --kill-after=10s 180s python3 -B %s/test-long.py\n' "$R"
    date -u +%FT%TZ
    hostname
    uname -srmo
    python3 --version
    bash -n "$R/cpu-checks.sh"
} > environment.txt 2>&1
set +e
timeout --signal=TERM --kill-after=10s 180s python3 -B "$R/test-long.py" > tests.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > exit-status.txt
if (( status != 0 )); then
    tail -80 tests.log
    printf 'CPU_FAILED_PRESERVED=%s\n' "$attempt"
    exit "$status"
fi
# Freeze exactly tested controller/analyzer/helpers before any lead launch. A changed
# file invalidates this gate. Every previous attempt (including failures) remains.
python3 -B - "$R" "$attempt" <<'PY'
import json, runpy, sys
from pathlib import Path
root, attempt = map(Path, sys.argv[1:])
c = runpy.run_path(str(root / 'contracts.py'))
record = dict(tests_passed=True, scope='CPU-only; synthetic 8K, no new E2E',
              attempt=str(attempt), hashes=c['code_hashes']())
c['save'](attempt / 'tested-code.json', record)
c['save'](root / 'preparation/tested-code.json', record)
for name in ('environment.txt', 'tests.log', 'exit-status.txt', 'hash-checks.json', 'disk-budget.json'):
    (root / 'preparation' / name).write_bytes((attempt / name).read_bytes())
print('CPU_PASSED_CODE_FROZEN=' + str(attempt))
PY
