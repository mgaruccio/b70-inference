#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-grouped-quality
bash build-evaluator.sh
bash prepare-evaluator.sh
bash run-smoke.sh
