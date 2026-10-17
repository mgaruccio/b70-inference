#!/usr/bin/env bash
# Runs inside the isolated candidate container, after the inherited native385 gate.
set -euo pipefail
BIN=/src/build-sycl/bin/test-backend-ops
for fusion in 0 1; do
    echo "BOLT_OUTPROD_F32_FUSION${fusion}"
    GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_FUSION="$fusion" "$BIN" test -b SYCL0 -o '^OUT_PROD$' \
        -p '^type_a=f32,type_b=f32,' -j 1 2>&1 | tee "/tmp/outprod-f32-fusion${fusion}.log"
    grep -Fx '  80/80 tests passed' "/tmp/outprod-f32-fusion${fusion}.log"
    test "$(grep -Ec 'not supported.*\(expected\)' "/tmp/outprod-f32-fusion${fusion}.log")" -eq 3
    test "$(grep -c 'not supported' "/tmp/outprod-f32-fusion${fusion}.log")" -eq 3
    if grep -Ei 'skipping large tensors|FAILED|NaN at|ERR =|unexpected' "/tmp/outprod-f32-fusion${fusion}.log"; then exit 1; fi
    echo "BOLT_OUTPROD_Q1_FUSION${fusion}"
    GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_FUSION="$fusion" "$BIN" test -b SYCL0 -o '^OUT_PROD$' \
        -p '^type_a=q1_0,type_b=f32,.*nr=\[1,1\]' -j 1 2>&1 | tee "/tmp/outprod-q1-fusion${fusion}.log"
    grep -Fx '  18/18 tests passed' "/tmp/outprod-q1-fusion${fusion}.log"
    if grep -Ei 'not supported|skipping large tensors|FAILED|NaN at|ERR =' "/tmp/outprod-q1-fusion${fusion}.log"; then exit 1; fi
done
echo BOLT_OUTPROD_NEGATIVE_SUPPORT
"$BIN" support -b SYCL0 -o '^OUT_PROD$' -p '^type_a=f32,type_b=f32,.*layout=bad_' -j 1 \
    2>&1 | tee /tmp/outprod-support.log
test "$(grep -c '(expected)' /tmp/outprod-support.log)" -eq 3
if grep -F '(UNEXPECTED)' /tmp/outprod-support.log; then exit 1; fi
echo BOLT_LILICORR_OUTPROD_NATIVE_PASSED
