#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -eu
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-mtp-quant-aware-tiny
patch=source/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_mtp_training.py
for cell in abba-stock-1 abba-tuned-1 abba-tuned-2 abba-stock-2; do
  case "$cell" in abba-stock-*) weights=stock-mtp.safetensors ;; *) weights=tuned-mtp.safetensors ;; esac
  python3 -u source/qwen38_mtp_tune_probe.py --out "$cell" --prompts corpus/heldout.jsonl --tokens 256 --generate --runtime-patch "$patch" --weights "$weights"
  python3 -u source/qwen38_mtp_tune_probe.py --check-outputs "$cell"
done
SH
