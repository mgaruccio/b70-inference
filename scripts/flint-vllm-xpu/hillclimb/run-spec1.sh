#!/usr/bin/env bash
# spec1: install multi-row decode path (M <= 8), non-spec regression bench, then DFlash 2 benches.
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
export NO_RESTORE=1
cmp -s build-dec5/model.py $F/model.py || { echo venv model.py differs from build-dec5; exit 5; }
cp ktest3/xpu_decode.py $F/ops/xpu_decode.py
.venv/bin/python -I ktest3/apply_spec1_patch.py $F/model.py || exit 6
mkdir -p build-spec1
cp $F/model.py build-spec1/model.py; cp $F/ops/norm.py build-spec1/norm.py
cp $F/ops/xpu_router.py $F/ops/xpu_fused.py $F/ops/xpu_moe_decode.py $F/ops/xpu_decode.py build-spec1/
echo "== non-spec graph bench"
EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench
echo "NONSPEC_RUN=$(ls -td http-run.* | head -1)"
for k in ${KS:-3 7}; do
  echo "== dflash2 graph bench k=$k"
  NUM_SPEC=$k EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2
  echo "SPEC_RUN_k$k=$(ls -td http-run.* | head -1)"
done
