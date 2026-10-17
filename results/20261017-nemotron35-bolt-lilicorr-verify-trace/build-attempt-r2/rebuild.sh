#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = inference-host
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-verify-trace
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-deterministic-r1
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-trace-r1
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
cd "$R"
exec > >(tee "$R/build-attempt-r2/build.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$R/build-attempt-r2/exit-code.txt"' EXIT
set -x
sha256sum -c "$R/accepted-before-build.sha256"
sha256sum -c "$R/parent-before-build.sha256"
python3 -B - <<'PY'
import runpy
m=runpy.run_path('run-trace.py')
assert m['binaries'](m['CANDIDATE'])==m['load'](m['ROOT']/'build-attempt-r1/candidate-binaries.json')
PY
(cd "$D" && git apply --reverse --check "$R/build-attempt-r1/verify-trace.patch" &&
 git apply --reverse "$R/build-attempt-r1/verify-trace.patch" &&
 git apply --check "$R/verify-trace.patch" && git apply "$R/verify-trace.patch")
# Reuse the exact checked build/dependency/hash tail, leaving the r1 attempt untouched.
export R PARENT D IMAGE
bash -euo pipefail <(sed -n '/^docker run --rm --name bolt-verify-trace-r1-build/,$p' "$R/build-candidate.sh")
