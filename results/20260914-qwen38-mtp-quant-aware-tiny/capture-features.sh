#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -eu
r=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-mtp-quant-aware-tiny
cd "$r"
patch=source/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_mtp_training.py
python3 -u source/qwen38_mtp_replay.py --source onpolicy-train onpolicy-heldout --out replay-off --runtime-patch "$patch"
python3 -u source/qwen38_mtp_replay.py --source onpolicy-train onpolicy-heldout --out replay-capture --runtime-patch "$patch" --capture
python3 source/qwen38_mtp_replay.py --compare replay-off replay-capture
docker run --rm --network none --entrypoint python3 -v "$r:/experiment" -w /experiment vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f source/qwen38_mtp_replay.py --finish-dataset replay-capture --out dataset
SH
