#!/usr/bin/env bash
# Fresh non-debug API gate before timing; never use the debug activation timings.
set -uo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
test "$(hostname)" = inference-host || exit 1
python3 -B - "$R" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
d = json.loads((root / 'ssm-prefill-r1-vmm-dispatch/summary.json').read_text())
assert d['status'] == 'passed' and d['debug'] and not d['measure']
assert d['prefill_activation']['usage']['prompt_tokens'] == 8270
PY
check=$?
test "$check" -eq 0 || exit "$check"
timeout --signal=TERM --kill-after=180s 3600s python3 -B -u "$R/run-bolt-cell-ssm-prefill-r1.py" \
  --name ssm-prefill-r1-validated \
  --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1 \
  --measure > "$R/ssm-prefill-r1-validated.console.log" 2>&1
rc=$?
if test "$rc" -ne 0; then
  timeout --signal=TERM --kill-after=180s 1200s python3 -B -u "$R/run-bolt-cell-router-r1-v2.py" \
    --name ssm-prefill-r1-quality-failure-recovery \
    --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-router-r1 \
    > "$R/ssm-prefill-r1-quality-failure-recovery.console.log" 2>&1
  recovery=$?
  printf 'Candidate exit=%s; baseline recovery exit=%s\n' "$rc" "$recovery"
fi
exit "$rc"
