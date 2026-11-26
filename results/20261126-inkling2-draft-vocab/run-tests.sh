#!/usr/bin/env bash
# b70-dev only, with the GPU free: run the reduced-draft-vocab plugin tests on the B70.
set -euo pipefail
p=$HOME/inkling-vllm-xpu
runtime=$HOME/flint-vllm-spike
d=$p/vocab-probe
driver=$runtime/driver-26.40
cd "$d"
env -u VLLM_API_KEY \
    PYTHONPATH="$d/pytest-overlay:$d/tests:$d/plugin:$d/sparse:$runtime/serve" \
    LD_LIBRARY_PATH="$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib:$driver/usr/local/lib" \
    LIBRARY_PATH="$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib" CPATH="$driver/usr/include" \
    OCL_ICD_VENDORS="$runtime/opencl-vendors" TRITON_CACHE_DIR="$p/sparse-69444/triton" HF_HUB_OFFLINE=1 \
    "$p/sparse-69444/.venv/bin/python" -B -m pytest -q -rs -p no:cacheprovider tests/test_draft_vocab.py
