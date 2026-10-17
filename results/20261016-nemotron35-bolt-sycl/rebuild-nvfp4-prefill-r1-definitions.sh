#!/usr/bin/env bash
# Correct only native-test private definitions; preserve the initial failed build.
set -euo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-prefill-pack-r1
test "$(hostname)" = inference-host
(cd "$D" && git apply --check "$R/nvfp4-native-test-definitions-fix.patch" && git apply "$R/nvfp4-native-test-definitions-fix.patch" && git apply --reverse --check "$R/nvfp4-prefill-r1.patch")
# Reuse the frozen build/precheck commands verbatim, omitting archive creation/apply.
sed -n '/^docker run --rm --name bolt-nvfp4-prefill-r1-build/,$p' "$R/build-nvfp4-prefill-r1.sh" > "$R/nvfp4-prefill-r1-rebuild-body.sh"
export R D
export IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
bash -eu -o pipefail "$R/nvfp4-prefill-r1-rebuild-body.sh"
