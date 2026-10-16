#!/usr/bin/env bash
set -euo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-cache-r1
bash "$R/build-ssm-cache-r1.sh" --resume > "$R/ssm-cache-r1-debug-build.log" 2>&1
test -z "$(docker ps --format '{{.Names}}')"
docker run --rm --name bolt-ssm-native-diag --device /dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" -v "$D:/src:ro" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e GGML_SYCL_ENABLE_ESIMD=1 \
  -e GGML_SYCL_ENABLE_FUSION=1 -e GGML_SYCL_ENABLE_GRAPH=0 -e GGML_SYCL_DEBUG=1 \
  intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04 bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
exec /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o SSM_SCAN_CACHE_FUSION -p bolt_ssm_cache=
' > "$R/ssm-cache-r1-native-debug.log" 2>&1
