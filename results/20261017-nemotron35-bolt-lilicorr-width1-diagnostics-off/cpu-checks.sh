#!/usr/bin/env bash
# REMOTE bounded CPU-only fixtures. No native/GPU/Docker/API/hardware execution.
set -euo pipefail
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off
[[ $(hostname) == inference-host ]]
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) == "$R" ]]
mkdir -p "$R/worker-scratch" "$R/preparation"
attempt=$(mktemp -d "$R/worker-scratch/cpu-XXXXXXXX")
export BOLT_OFF_CPU_SCRATCH="$attempt" TMPDIR="$attempt" PYTHONDONTWRITEBYTECODE=1
cd "$attempt"
printf 'CPU_ATTEMPT=%s\n' "$attempt"
{
    printf 'tier=development; CPU recorded/synthetic fixtures, NOT live diagnostics-off E2E\n'
    printf 'command=timeout --signal=TERM --kill-after=10s 180s python3 -B [inline wrapper in cpu-checks.sh]\n'
    date -u +%FT%TZ
    hostname
    uname -srmo
    python3 --version
    bash -n "$R/cpu-checks.sh"
} > environment.txt 2>&1
set +e
timeout --signal=TERM --kill-after=10s 180s python3 -B - "$R" "$attempt" > tests.log 2>&1 <<'PY'
import json, runpy, sys
from pathlib import Path
root, attempt = map(Path, sys.argv[1:])
c = runpy.run_path(str(root / 'contracts.py'))
before = dict(hashes=c['code_hashes'](), references=c['reference_hashes'](c['frozen_jobs']()))
c['save'](attempt / 'tested-before.json', before)
sys.argv = [str(root / 'test-off.py')]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
assert before['hashes'] == c['code_hashes'](), 'code changed during CPU tests'
assert before['references'] == c['reference_hashes'](c['frozen_jobs']()), 'reference drift during CPU tests'
record = dict(before, tests_passed=True, scope='CPU-only recorded/synthetic fixtures; real pair pending', attempt=str(attempt))
c['save'](attempt / 'tested-code.json', record)
c['save'](root / 'preparation/tested-code.json', record)
print('CPU_PASSED_EXACT_CODE_FROZEN=' + str(attempt))
PY
status=$?
set -e
printf '%s\n' "$status" > exit-status.txt
if (( status != 0 )); then
    tail -100 tests.log
    printf 'CPU_FAILED_PRESERVED=%s\n' "$attempt"
    exit "$status"
fi
for name in environment.txt tests.log exit-status.txt hash-checks.json source-contracts.json disk-budget.json; do
    cp -- "$attempt/$name" "$R/preparation/$name"
done
printf 'CPU_PASSED_PRESERVED=%s\n' "$attempt"
