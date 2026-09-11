#!/usr/bin/env bash
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-reference-parity
H=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
IMAGE=vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4
cd "$R"
docker run --rm --pull=never --read-only --tmpfs /tmp \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$R:/parity:ro" -v "$R/official:/sources" \
  --entrypoint /opt/venv/bin/python "$IMAGE" /parity/replay-reference.py \
  --official-source /sources --fetch-source --check-imports --device cpu \
  > official-imports.json 2> official-imports.stderr
docker run --rm --pull=never --network=none --read-only --tmpfs /tmp \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$R:/parity:ro" \
  --entrypoint /opt/venv/bin/python "$IMAGE" /parity/test-reference-cpu.py \
  --official-source /parity/official > official-cpu-test.txt 2>&1
timeout --signal=INT --kill-after=90s 1500s python3 -u run-capture.py --approve-gpu-launch \
  --driver "$R/_dependencies/run-acceptance-diagnostics.py" --previous "$H" \
  --overlay "$R/_dependencies/patch-vllm-qwen38-dspark-bf16.py" \
  --out "$R/capture-01" --output-tokens 16 2>&1 | tee capture-driver.log
