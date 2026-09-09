#!/usr/bin/env bash
set -euo pipefail
REPO=/home/mike/code/b70-inference
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard
ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host "mkdir -p '$R/code/scripts/experiments' '$R/code/betterbench'"
tar -C "$REPO" -cf - scripts/experiments/qwen38_standard_bench.py scripts/experiments/qwen38_dflash2_probe.py scripts/experiments/qwen38_lossy_probe.py scripts/patch-vllm-qwen38-dflash2-bf16.py scripts/patch-vllm-qwen38-xpu-prefill.py | ssh inference-host "tar -C '$R/code' -xf -"
tar -C /tmp/b70-betterbench-1de941d --exclude=.git --exclude=__pycache__ -cf - . | ssh inference-host "tar -C '$R/code/betterbench' -xf -"
ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'bash -s' <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard
P="$R/code/scripts/experiments/qwen38_standard_bench.py"
COMMON=(--betterbench "$R/code/betterbench" --patch "$R/code/scripts/patch-vllm-qwen38-dflash2-bf16.py" --prefill-patch "$R/code/scripts/patch-vllm-qwen38-xpu-prefill.py" --guard /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2/runner/patch_uniform_decode_prefill.py)
python3 -u "$P" "${COMMON[@]}" --draft-int4 /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-quant/rtn-int4-g128 --out "$R/int4-clients"
python3 -u "$P" "${COMMON[@]}" --out "$R/bf16-clients"
REMOTE
