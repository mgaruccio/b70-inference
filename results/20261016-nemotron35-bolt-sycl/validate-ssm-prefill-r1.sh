#!/usr/bin/env bash
# Native gate + debug activation only. Failure is retained; restore validated Bolt.
set -uo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
test "$(hostname)" = inference-host || exit 1
timeout --signal=TERM --kill-after=180s 1800s python3 -B -u "$R/run-bolt-cell-ssm-prefill-r1.py" \
  --name ssm-prefill-r1-dispatch \
  --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1 \
  --debug > "$R/ssm-prefill-r1-dispatch.console.log" 2>&1
rc=$?
if test "$rc" -ne 0; then
  timeout --signal=TERM --kill-after=180s 1200s python3 -B -u "$R/run-bolt-cell-router-r1-v2.py" \
    --name ssm-prefill-r1-native-failure-recovery \
    --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-router-r1 \
    > "$R/ssm-prefill-r1-native-failure-recovery.console.log" 2>&1
  recovery=$?
  printf 'Candidate exit=%s; baseline recovery exit=%s\n' "$rc" "$recovery"
fi
exit "$rc"
