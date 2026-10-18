#!/usr/bin/env bash
# REMOTE ONLY; lead runs this before the separately authorized live control.
set -euo pipefail
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width3-suppression
[[ $(hostname) == inference-host ]]
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) == "$R" ]]
mkdir -p "$R/worker-scratch" "$R/preparation"
attempt=$(mktemp -d "$R/worker-scratch/cpu-XXXXXXXX")
export BOLT_SUPPRESSION_CPU_SCRATCH="$attempt" TMPDIR="$attempt" PYTHONDONTWRITEBYTECODE=1
{
    printf 'tier=development preparation; CPU fixtures only; no live suppression proof\n'
    printf 'command=bash %s/cpu-checks.sh\n' "$R"
    printf 'python=timeout --signal=TERM --kill-after=10s 180s python3 -B - %s %s (inline runner below)\n' "$R" "$attempt"
    date -u +%FT%TZ
    hostname
    uname -srmo
    python3 --version
    bash -n "$R/cpu-checks.sh"
} > "$attempt/environment.txt" 2>&1
printf 'CPU_ATTEMPT=%s\n' "$attempt"
set +e
timeout --signal=TERM --kill-after=10s 180s python3 -B - "$R" "$attempt" > "$attempt/tests.log" 2>&1 <<'PY'
import ast
import runpy
import sys
from pathlib import Path

root, attempt = map(Path, sys.argv[1:])
for path in root.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
p = runpy.run_path(str(root / 'probe.py'))
c, h = p['C'], p['C']['helpers']()
before = p['freeze'](h)
c['save'](attempt / 'freeze-before.json', before)
sys.argv = [str(root / 'test-suppression.py')]
runpy.run_path(sys.argv[0], run_name='__main__')  # unittest exit=False; failed tests raise.
after = p['freeze'](h)
assert before == after, 'code/reference/source/binary drift during CPU fixtures'
c['save'](attempt / 'freeze-after.json', after)
c['save'](attempt / 'source-contracts.json', c['source_evidence'](h))
record = dict(tests_passed=True, scope='CPU fixtures only; live capture and independent analysis pending',
              attempt=str(attempt), freeze=after)
c['save'](attempt / 'tested-code.json', record)
c['save'](root / 'preparation/tested-code.json', record)
print('CPU_PASSED_EXACT_CODE_FROZEN=' + str(attempt))
PY
status=$?
set -e
printf '%s\n' "$status" > "$attempt/exit-status.txt"
if (( status != 0 )); then
    tail -100 "$attempt/tests.log"
    printf 'CPU_FAILED_PRESERVED=%s\n' "$attempt"
    exit "$status"
fi
printf 'CPU_PASSED_PRESERVED=%s\n' "$attempt"
