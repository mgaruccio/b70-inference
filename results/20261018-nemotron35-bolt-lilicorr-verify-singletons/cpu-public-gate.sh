#!/usr/bin/env bash
# LEAD/REMOTE ONLY.  Run isolated CPU fixtures only; native/public evidence is pending.
# No container, model, API, GPU, install, or remote-write action is performed here.
set -euo pipefail
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
[[ $(hostname) == inference-host ]]
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) == "$R" ]]
mkdir -p "$R/worker-scratch"
attempt=$(mktemp -d "$R/worker-scratch/public-cpu-XXXXXXXX")
export BOLT_VERIFY_SINGLETONS_CPU_SCRATCH="$attempt" TMPDIR="$attempt" PYTHONDONTWRITEBYTECODE=1
{
    printf 'tier=development isolated CPU fixtures; native/public qualification pending\n'
    printf 'command=bash %s/cpu-public-gate.sh\n' "$R"
    printf 'python=timeout --signal=TERM --kill-after=10s 180s python3 -B - %s %s\n' "$R" "$attempt"
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
import json
import runpy
import sys
from pathlib import Path

root, attempt = map(Path, sys.argv[1:])
files = sorted(root.glob('*.py')) + sorted((root / 'n9').glob('*.py'))
for path in files:
    ast.parse(path.read_text(), filename=str(path))

public = runpy.run_path(str(root / 'public-gate.py'))
pending = 'LEAD_NATIVE_INTEGRATION pending: singleton source/build proof'
try:
    public['candidate_proof']()
except RuntimeError as exc:
    assert str(exc) == pending, str(exc)
else:
    raise AssertionError('candidate proof did not fail closed')

control = runpy.run_path(str(root / 'n9' / 'run-control.py'))
assert control['BACKENDS'] == {
    'VSN9-PARENT': control['PARENT'],
    'VSN9-CANDIDATE': control['CANDIDATE'],
}
try:
    control['candidate_proof']()
except RuntimeError as exc:
    assert str(exc) == pending, str(exc)
else:
    raise AssertionError('N9 candidate proof did not fail closed')

# These suites exercise the inherited lifecycle/container/parser fixtures only;
# their audit hooks reject real process, network, and DRM access.
for path in (root / 'test-public-gate.py', root / 'n9' / 'test-control.py'):
    sys.argv = [str(path)]
    try:
        runpy.run_path(str(path), run_name='__main__')
    except SystemExit as exc:
        assert exc.code in (None, 0), (path, exc.code)

record = {
    'tests_passed': True,
    'scope': 'isolated public/N9 lifecycle and inherited F32 CPU fixtures only',
    'native_contract': pending,
    'dependent_not_run': ['native build', 'fresh paired N9', 'public qualification', 'trace capture'],
    'attempt': str(attempt),
}
(attempt / 'isolated-tests.json').write_text(json.dumps(record, indent=2) + '\n')
print('CPU_PASSED_ISOLATED_PENDING_NATIVE=' + str(attempt))
PY
status=$?
set -e
printf '%s\n' "$status" > "$attempt/exit-status.txt"
if (( status != 0 )); then
    tail -100 "$attempt/tests.log"
    printf 'CPU_FAILED_PRESERVED=%s\n' "$attempt"
    exit "$status"
fi
printf 'CPU_PASSED_ISOLATED_PENDING_NATIVE=%s\n' "$attempt"
