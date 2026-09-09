#!/usr/bin/env bash
set -euo pipefail
REPO=/home/mike/code/b70-inference
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard
mkdir -p /tmp/qwen-standard-results
ssh -o BatchMode=yes inference-host "tar -C '$R' --exclude=__pycache__ -czf - int4-clients bf16-clients model-download-revisions.json pcie.json pcie-lspci.txt" | tar -C /tmp/qwen-standard-results -xzf -
ssh -o BatchMode=yes inference-host "mkdir -p '$R/code-v2'"
tar -C "$REPO" -cf - scripts/start-qwen38.sh scripts/experiments/qwen38_standard_bench.py scripts/experiments/qwen38_mtp_reference.py scripts/experiments/qwen38_long_context_bench.py scripts/experiments/qwen38_dflash2_probe.py scripts/experiments/qwen38_lossy_probe.py scripts/patch-vllm-qwen38-dflash2-bf16.py scripts/patch-vllm-qwen38-xpu-prefill.py tests/test_qwen38_standard_bench.py tests/test_qwen38_long_context_bench.py | ssh inference-host "tar -C '$R/code-v2' -xf -"
ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'bash -s' <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard
cd "$R/code-v2"
python3 -m unittest discover -s tests -p 'test_qwen38_*bench.py' -v > "$R/benchmark-client-tests.txt" 2>&1
cat "$R/benchmark-client-tests.txt"
P="$R/code-v2/scripts/experiments/qwen38_standard_bench.py"
COMMON=(--betterbench "$R/code/betterbench" --patch "$R/code-v2/scripts/patch-vllm-qwen38-dflash2-bf16.py" --prefill-patch "$R/code-v2/scripts/patch-vllm-qwen38-xpu-prefill.py" --guard /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2/runner/patch_uniform_decode_prefill.py)
python3 -u "$P" "${COMMON[@]}" --long-context-only --draft-int4 /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-quant/rtn-int4-g128 --out "$R/int4-long"
python3 -u "$P" "${COMMON[@]}" --long-context-only --out "$R/bf16-long"
python3 -u "$P" "${COMMON[@]}" --reference-mtp --out "$R/mtp4-clients"
python3 -u "$P" "${COMMON[@]}" --reference-mtp --long-context-only --out "$R/mtp4-long"
REMOTE
