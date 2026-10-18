#!/usr/bin/env bash
# LEAD/REMOTE ONLY.  Run isolated CPU fixtures only; native/public evidence is pending.
# No container, model, API, GPU, install, or remote-write action is performed here.
set -euo pipefail
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
[[ $(hostname) == inference-host ]]
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) == "$R" ]]
mkdir -p "$R/worker-scratch" "$R/preparation"
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
import subprocess
import sys
from pathlib import Path

root, attempt = map(Path, sys.argv[1:])
files = sorted(root.glob('*.py')) + sorted((root / 'n9').glob('*.py'))
for path in files:
    ast.parse(path.read_text(), filename=str(path))

public = runpy.run_path(str(root / 'public-gate.py'))
proof = public['candidate_proof']()
control = runpy.run_path(str(root / 'n9' / 'run-control.py'))
assert control['BACKENDS'] == {
    'VSN9-PARENT': control['PARENT'],
    'VSN9-CANDIDATE': control['CANDIDATE'],
}
assert control['candidate_proof']() == proof
c, h = public['C'], public['C']['helpers']()
before = public['freeze'](h)  # Requires the fresh N9 pair and independent restoration.
c['save'](attempt / 'freeze-before.json', before)
for cell in public['CELLS']:
    argv = public['make_create'](cell, h)
    c['save'](attempt / (cell + '-create-argv.json'), argv)
    subprocess.run(['bash', '-n'], input=argv[-1], text=True, check=True)

# These suites exercise the inherited lifecycle/container/parser fixtures only;
# their audit hooks reject real process, network, and DRM access.
for path in (root / 'test-public-gate.py', root / 'n9' / 'test-control.py', root / 'test-first-frame.py'):
    sys.argv = [str(path)]
    try:
        runpy.run_path(str(path), run_name='__main__')
    except SystemExit as exc:
        assert exc.code in (None, 0), (path, exc.code)

after = public['freeze'](h)
assert before == after, 'code/reference/source/binary drift during CPU fixtures'
c['save'](attempt / 'freeze-after.json', after)
record = {
    'tests_passed': True,
    'scope': 'isolated public/N9 lifecycle and inherited/new native parser CPU fixtures',
    'native_contract': 'source and six-artifact build proof verified; no GPU execution',
    'runtime_not_exercised_by_this_suite': ['fresh paired N9', 'public qualification', 'trace capture'],
    'attempt': str(attempt),
    'freeze': after,
}
(attempt / 'isolated-tests.json').write_text(json.dumps(record, indent=2) + '\n')
c['save'](attempt / 'public-tested-code.json', record)
c['save'](root / 'preparation/public-tested-code.json', record)
print('CPU_PASSED_ISOLATED=' + str(attempt))
PY
status=$?
set -e
printf '%s\n' "$status" > "$attempt/exit-status.txt"
if (( status != 0 )); then
    tail -100 "$attempt/tests.log"
    printf 'CPU_FAILED_PRESERVED=%s\n' "$attempt"
    exit "$status"
fi
printf 'CPU_PASSED_ISOLATED=%s\n' "$attempt"
