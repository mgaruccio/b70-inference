#!/usr/bin/env bash
# REMOTE bounded CPU fixtures only. Real run belongs to the lead.
set -euo pipefail
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing
[[ $(hostname) == inference-host ]]
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) == "$R" ]]
mkdir -p "$R/worker-scratch" "$R/preparation"
attempt=$(mktemp -d "$R/worker-scratch/cpu-XXXXXXXX")
export BOLT_TIMING_CPU_SCRATCH="$attempt" TMPDIR="$attempt" PYTHONDONTWRITEBYTECODE=1
cd "$attempt"
printf 'CPU_ATTEMPT=%s\n' "$attempt"
{
    printf 'tier=development; recorded/synthetic CPU fixtures, NOT real benchmark\n'
    printf 'command=timeout --signal=TERM --kill-after=10s 180s python3 -B [inline wrapper in cpu-checks.sh]\n'
    date -u +%FT%TZ
    hostname
    uname -srmo
    python3 --version
    bash -n "$R/cpu-checks.sh"
} > environment.txt 2>&1
set +e
timeout --signal=TERM --kill-after=10s 180s python3 -B - "$R" "$attempt" > tests.log 2>&1 <<'PY'
import ast, runpy, sys
from pathlib import Path
root, attempt = map(Path, sys.argv[1:])
for path in root.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
c = runpy.run_path(str(root / 'contracts.py'))
h = c['helpers']()
before = dict(hashes=c['code_hashes'](h), references=c['reference_hashes'](h))
c['save'](attempt / 'tested-before.json', before)
sys.argv = [str(root / 'test-timing.py')]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
assert before['hashes'] == c['code_hashes'](h), 'code changed during CPU checks'
assert before['references'] == c['reference_hashes'](h), 'reference drift during CPU checks'
record = dict(before, tests_passed=True, scope='CPU fixtures only; lead real E2E pending', attempt=str(attempt))
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
for name in environment.txt tests.log exit-status.txt hash-checks.json source-contracts.json; do
    cp -- "$attempt/$name" "$R/preparation/$name"
done
printf 'CPU_PASSED_PRESERVED=%s\n' "$attempt"
