#!/usr/bin/env bash
# Remote CPU suite only. Invoke through bg_run; never a foreground timeout.
set -euo pipefail
test "$(hostname)" = inference-host
S=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-widths-single2/worker-scratch
attempt=${1:?preparation attempt, e.g. prepare-r1}
run=${2:?unique log suffix, e.g. r1}
[[ $attempt =~ ^prepare-r[1-9][0-9]*$ && $run =~ ^r[1-9][0-9]*$ ]]
log="$S/cpu-$run.log"
(set -o noclobber; : > "$log")
exec > >(tee -a "$log") 2>&1
trap 'rc=$?; printf "CPU_SUITE_EXIT=%s\n" "$rc"' EXIT
set -x
cd "$S/$attempt"
uname -sr
python3 --version
g++ --version
sha256sum router-widths-single2.patch run-trace.py analyze-trace.py test-trace.py test-widths.py
# CPU artifacts are temporary and confined to worker-scratch; leave 4GiB reserve.
available=$(df -Pk "$S" | awk 'NR==2 {print $4}')
test "$available" -gt 4194304
set +e
python3 -B test-trace.py
trace_rc=$?
python3 -B test-widths.py
widths_rc=$?
set -e
printf 'TRACE_EXIT=%s WIDTHS_EXIT=%s\n' "$trace_rc" "$widths_rc"
test "$trace_rc" -eq 0
test "$widths_rc" -eq 0
