#!/usr/bin/env bash
# Run via PRETEST from verify-kernels after the serving run releases the GPU.
set -euo pipefail
OMP_NUM_THREADS=4 bash probe-python.sh attn -m pytest -q -s -p no:cacheprovider \
    attn/tests/test_attention.py::test_xpu_verify_attention_fp64 | tee "$PRETEST_DIR/attn-fp64.log"
test "$(grep -c '^verify_attention_fp64 ' "$PRETEST_DIR/attn-fp64.log")" -eq 3
