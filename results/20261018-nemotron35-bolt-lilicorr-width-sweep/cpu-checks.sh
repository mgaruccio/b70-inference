#!/usr/bin/env bash
# REMOTE ONLY: CPU fixtures; the lead owns all live qualification/timing.
set -euo pipefail
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width-sweep
[[ $(hostname) == inference-host ]]
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) == "$R" ]]
mkdir -p "$R/worker-scratch" "$R/preparation"
attempt=$(mktemp -d "$R/worker-scratch/cpu-XXXXXXXX")
export BOLT_WIDTH_SWEEP_CPU_SCRATCH="$attempt" TMPDIR="$attempt" PYTHONDONTWRITEBYTECODE=1
cd "$attempt"
printf 'CPU_ATTEMPT=%s\n' "$attempt"
{
    printf 'tier=development; recorded/synthetic CPU fixtures only; NOT real benchmark\n'
    printf 'command=timeout --signal=TERM --kill-after=10s 180s python3 -B test-width-sweep.py\n'
    date -u +%FT%TZ
    hostname
    uname -srmo
    python3 --version
    bash -n "$R/cpu-checks.sh"
} > environment.txt 2>&1
set +e
timeout --signal=TERM --kill-after=10s 180s python3 -B - "$R" "$attempt" > tests.log 2>&1 <<'PY'
import ast
import runpy
import sys
from pathlib import Path

root, attempt = map(Path, sys.argv[1:])
for path in root.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
c = runpy.run_path(str(root / 'contracts.py'))
h = c['helpers']()
before = dict(hashes=c['code_hashes'](h), references=c['reference_hashes'](h))
c['save'](attempt / 'hash-checks-before.json', before)
sys.argv = [str(root / 'test-width-sweep.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
sys.argv = [str(root / 'test-qualification-analysis.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
after = dict(hashes=c['code_hashes'](h), references=c['reference_hashes'](h))
assert before == after, 'source/helper/reference drift during CPU fixtures'
c['save'](attempt / 'hash-checks.json', after['hashes'])
c['save'](attempt / 'source-contracts.json', c['source_evidence'](h))
record = dict(tests_passed=True,
              scope='CPU fixtures only; lead real qualification/timing pending',
              attempt=str(attempt), hashes=after['hashes'], references=after['references'])
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
