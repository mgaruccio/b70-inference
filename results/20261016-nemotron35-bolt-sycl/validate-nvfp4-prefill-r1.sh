#!/usr/bin/env bash
set -euo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
BASE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1
CAND=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-prefill-pack-r1
test "$(hostname)" = inference-host
cd "$R"
recover() {
    status=$?
    if test "$status" -ne 0; then
        trap - EXIT
        echo "Candidate sequence failed ($status); preserving failure and restoring SSM winner."
        timeout --signal=TERM --kill-after=180s 1800s python3 -B -u run-bolt-cell-ssm-prefill-r1.py \
          --name nvfp4-prefill-r1-failure-recovery --source "$BASE" \
          > nvfp4-prefill-r1-failure-recovery.console.log 2>&1 || { echo 'BOLT RECOVERY FAILED'; exit 99; }
    fi
    exit "$status"
}
trap recover EXIT
python3 -B test-bolt-cell-nvfp4-prefill-r1.py > nvfp4-prefill-r1-controller-tests.log 2>&1
timeout --signal=TERM --kill-after=180s 2400s python3 -B -u run-bolt-cell-nvfp4-prefill-r1.py \
  --name nvfp4-prefill-r1-dispatch --source "$CAND" --debug \
  > nvfp4-prefill-r1-dispatch.console.log 2>&1
# Debug dispatch is mandatory before the fresh baseline/candidate timing pair.
timeout --signal=TERM --kill-after=180s 2400s python3 -B -u run-bolt-cell-nvfp4-prefill-r1.py \
  --name nvfp4-prefill-r1-pair-baseline --source "$BASE" --measure \
  > nvfp4-prefill-r1-pair-baseline.console.log 2>&1
timeout --signal=TERM --kill-after=180s 2400s python3 -B -u run-bolt-cell-nvfp4-prefill-r1.py \
  --name nvfp4-prefill-r1-pair-candidate --source "$CAND" --measure \
  > nvfp4-prefill-r1-pair-candidate.console.log 2>&1
printf 'NVFP4_PREFILL_SEQUENCE_PASSED\n'
