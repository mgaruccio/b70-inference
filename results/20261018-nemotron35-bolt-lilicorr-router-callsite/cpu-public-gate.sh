#!/usr/bin/env bash
# LEAD/REMOTE ONLY, after successful candidate build and fresh N9. No GPU/container/API calls.
set -euo pipefail
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
[[ $(hostname) == inference-host ]]
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) == "$R" ]]
[[ $(cat "$R/build-attempt-r1/exit-code.txt") == 0 ]]
grep -Fq BOLT_VERIFY_CALLSITE_BUILD_PASSED "$R/post-build-verification.log"
grep -Fq BOLT_VERIFY_CALLSITE_DEPENDENCIES_PASSED "$R/runtime-precheck.log"
test -f "$R/n9/run-r1/independent-restoration.json"
mkdir -p "$R/worker-scratch" "$R/preparation"
attempt=$(mktemp -d "$R/worker-scratch/public-cpu-XXXXXXXX")
export BOLT_ROUTER_CALLSITE_CPU_SCRATCH="$attempt" TMPDIR="$attempt" PYTHONDONTWRITEBYTECODE=1
{
    printf 'tier=development preparation; CPU fixtures only; no live qualification proof\n'
    printf 'command=bash %s/cpu-public-gate.sh\n' "$R"
    printf 'python=timeout --signal=TERM --kill-after=10s 180s python3 -B - %s %s (inline runner below)\n' "$R" "$attempt"
    date -u +%FT%TZ
    hostname
    uname -srmo
    python3 --version
    bash -n "$R/cpu-public-gate.sh"
} > "$attempt/environment.txt" 2>&1
printf 'CPU_ATTEMPT=%s\n' "$attempt"
set +e
timeout --signal=TERM --kill-after=10s 180s python3 -B - "$R" "$attempt" > "$attempt/tests.log" 2>&1 <<'PY'
import ast
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

root, attempt = map(Path, sys.argv[1:])
for path in root.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
p = runpy.run_path(str(root / 'public-gate.py'))
c, h = p['C'], p['C']['helpers']()
before = p['freeze'](h)  # Requires the real new build and freshly restored N9 proof.
c['save'](attempt / 'freeze-before.json', before)
for cell in p['CELLS']:
    argv = p['make_create'](cell, h)
    c['save'](attempt / (cell + '-create-argv.json'), argv)
    subprocess.run(['bash', '-n'], input=argv[-1], text=True, check=True)
# Native test fixtures use CASE-relative scratch: never let them write in the old case.
native = attempt / 'native-fixtures'
native.mkdir()
for file in ('test-trace.py', 'analyze-trace.py', 'run-trace.py', 'build-candidate.sh', 'router-verify-phase.patch'):
    shutil.copyfile(p['PHASE'] / file, native / file)
sys.argv = [str(native / 'test-trace.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
# Both suites install no-process/no-network/no-DRM audit hooks; exit=False returns here.
for file in ('test-first-frame.py', 'test-public-gate.py'):
    sys.argv = [str(root / file)]
    runpy.run_path(sys.argv[0], run_name='__main__')
after = p['freeze'](h)
assert before == after, 'code/reference/source/binary drift during CPU fixtures'
c['save'](attempt / 'freeze-after.json', after)
c['save'](attempt / 'source-contracts.json', c['source_evidence'](h))
record = dict(tests_passed=True,
              scope='CPU native/public/frame fixtures only; same-role trace fixtures require new public capture',
              attempt=str(attempt), freeze=after)
c['save'](attempt / 'public-tested-code.json', record)
c['save'](root / 'preparation/public-tested-code.json', record)
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
