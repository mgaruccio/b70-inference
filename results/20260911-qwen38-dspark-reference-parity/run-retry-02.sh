#!/usr/bin/env bash
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-reference-parity
H=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
cd "$R"
timeout --signal=INT --kill-after=90s 1500s python3 -u run-capture.py --approve-gpu-launch --driver "$R/_dependencies/run-acceptance-diagnostics.py" --previous "$H" --overlay "$R/_dependencies/patch-vllm-qwen38-dspark-bf16.py" --out "$R/capture-02" --output-tokens 16 2>&1 | tee capture-02-driver.log
bash run-reference-02-command.sh
