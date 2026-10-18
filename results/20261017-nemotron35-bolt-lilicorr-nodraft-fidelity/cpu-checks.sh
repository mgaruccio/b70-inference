#!/usr/bin/env bash
# Remote CPU-only preparation. Lead launches via bg_run, never a foreground suite.
set -euo pipefail
test "$(hostname)" = inference-host
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-nodraft-fidelity
attempt=${1:?unique preparation attempt, e.g. prepare-r1}
[[ $attempt =~ ^prepare-r[1-9][0-9]*$ ]]
S="$R/worker-scratch/$attempt"
test -d "$S"
log="$R/worker-scratch/$attempt.log"
(set -o noclobber; : > "$log")
exec > >(tee -a "$log") 2>&1
trap 'rc=$?; printf "CPU_SUITE_EXIT=%s\n" "$rc"' EXIT
set -x
cd "$S"
export PYTHONDONTWRITEBYTECODE=1
uname -sr
python3 --version
sha256sum run-a0.py analyze-a0.py test-a0.py cpu-checks.sh
available=$(df -Pk "$S" | awk 'NR==2 {print $4}')
test "$available" -gt 4194304
# No compilers, native tests, Docker, API, GPU, service, or hardware commands.
# Retain failure log and snapshot intact; next preparation uses a new attempt.
timeout --signal=TERM --kill-after=10s 240s python3 -B test-a0.py
