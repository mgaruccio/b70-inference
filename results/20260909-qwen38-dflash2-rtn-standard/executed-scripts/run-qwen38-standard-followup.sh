#!/usr/bin/env bash
set -euo pipefail
REPO=/home/mike/code/b70-inference
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard
ssh -o BatchMode=yes inference-host "mkdir -p '$R/code-v3'"
tar -C "$REPO" -cf - scripts/start-qwen38.sh scripts/experiments/qwen38_standard_bench.py scripts/experiments/qwen38_mtp_reference.py scripts/experiments/qwen38_long_context_bench.py scripts/experiments/qwen38_dflash2_probe.py scripts/experiments/qwen38_lossy_probe.py scripts/patch-vllm-qwen38-dflash2-bf16.py scripts/patch-vllm-qwen38-xpu-prefill.py tests/test_qwen38_standard_bench.py tests/test_qwen38_long_context_bench.py | ssh inference-host "tar -C '$R/code-v3' -xf -"
ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'bash -s' <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard
cd "$R/code-v3"
python3 -m unittest discover -s tests -p 'test_qwen38_*bench.py' -v > "$R/benchmark-client-tests-v3.txt" 2>&1
cat "$R/benchmark-client-tests-v3.txt"
P="$R/code-v3/scripts/experiments/qwen38_standard_bench.py"
COMMON=(--betterbench "$R/code/betterbench" --patch "$R/code-v3/scripts/patch-vllm-qwen38-dflash2-bf16.py" --prefill-patch "$R/code-v3/scripts/patch-vllm-qwen38-xpu-prefill.py" --guard /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2/runner/patch_uniform_decode_prefill.py)
D=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-quant/rtn-int4-g128
FAILED=0
run_cell() {
  local name=$1; shift
  local rc=0
  python3 -u "$P" "${COMMON[@]}" "$@" --out "$R/$name" || rc=$?
  printf '%s\t%s\n' "$name" "$rc" | tee -a "$R/followup-cell-exits.tsv"
  if [ "$name" != bf16-long ] && [ "$rc" -ne 0 ]; then FAILED=1; fi
}
# The exact-boundary control is observational: retain either success or failure.
run_cell bf16-long --long-context-only
run_cell int4-long-32000 --long-context-only --near-limit 32000 --draft-int4 "$D"
run_cell bf16-long-32000 --long-context-only --near-limit 32000
run_cell mtp4-clients --reference-mtp
run_cell mtp4-long --reference-mtp --long-context-only
exit "$FAILED"
REMOTE
