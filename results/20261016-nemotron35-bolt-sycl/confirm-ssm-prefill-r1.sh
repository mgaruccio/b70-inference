#!/usr/bin/env bash
# Bounded fresh-launch baseline/candidate pair after the first successful candidate.
set -uo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
test "$(hostname)" = inference-host || exit 1
python3 -B - "$R" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
assert json.loads((root / 'ssm-prefill-r1-validated/summary.json').read_text())['status'] == 'passed'
PY
check=$?
test "$check" -eq 0 || exit "$check"
timeout --signal=TERM --kill-after=180s 3600s python3 -B -u "$R/run-bolt-cell-router-r1-v2.py" \
  --name prefill-router-r1-confirm-baseline \
  --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-router-r1 \
  --measure > "$R/prefill-router-r1-confirm-baseline.console.log" 2>&1
rc=$?
test "$rc" -eq 0 || exit "$rc"
timeout --signal=TERM --kill-after=180s 3600s python3 -B -u "$R/run-bolt-cell-ssm-prefill-r1.py" \
  --name ssm-prefill-r1-confirm-candidate \
  --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1 \
  --measure > "$R/ssm-prefill-r1-confirm-candidate.console.log" 2>&1
rc=$?
if test "$rc" -ne 0; then
  timeout --signal=TERM --kill-after=180s 1200s python3 -B -u "$R/run-bolt-cell-router-r1-v2.py" \
    --name ssm-prefill-r1-confirm-failure-recovery \
    --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-router-r1 \
    > "$R/ssm-prefill-r1-confirm-failure-recovery.console.log" 2>&1
  recovery=$?
  printf 'Candidate exit=%s; baseline recovery exit=%s\n' "$rc" "$recovery"
fi
exit "$rc"
