#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -eu
r=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-mtp-quant-aware-tiny
docker run --rm --network none --cpus 2 --memory 8g -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -v "$r:/experiment" -v /home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro --entrypoint bash vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f -lc 'set -eu; export B70_MTP_TEST_MODEL=/model; export B70_MTP_TEST_VLLM_ROOT=$(python3 -c "import importlib.util;print(next(iter(importlib.util.find_spec(\"vllm\").submodule_search_locations)))"); cd /experiment/source; python3 -m pytest -q -p no:cacheprovider tests/test_qwen38_mtp_training_runtime.py tests/test_qwen38_train_mtp.py; python3 qwen38_train_mtp.py --model /model --output /experiment/stock-mtp.safetensors --export-stock'
SH
