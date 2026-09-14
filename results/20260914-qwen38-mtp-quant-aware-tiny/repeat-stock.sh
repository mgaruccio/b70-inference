#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -eu
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-mtp-quant-aware-tiny
python3 -u source/qwen38_mtp_tune_probe.py --out stock-repeat --prompts corpus/heldout.jsonl --tokens 256
python3 -u source/qwen38_mtp_tune_probe.py --out stock-overlay-repeat --prompts corpus/heldout.jsonl --tokens 256 --runtime-patch source/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_mtp_training.py --weights stock-mtp.safetensors
SH
