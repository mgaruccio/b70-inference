#!/usr/bin/env bash
# Gate + install + benchmark round-2 (shared expert folded into MoE decode kernels).
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
OFFLINE_SCRIPT=ktest3/decfinal.py OFFLINE_MAXSEC=900 bash maintenance.sh --offline
r=$(ls -td http-run.* | head -1); cat $r/decfinal.json || exit 3
grep -q '"PASS": true' $r/decfinal.json || { echo UNIT GATE FAILED; exit 4; }
cmp -s build-dec3/model.py $F/model.py || { echo venv model.py differs from build-dec3; exit 5; }
cp ktest3/xpu_decode.py $F/ops/xpu_decode.py
.venv/bin/python -I ktest3/apply_dec4_patch.py $F/model.py || exit 6
mkdir -p build-dec4
cp $F/model.py build-dec4/model.py; cp $F/ops/norm.py build-dec4/norm.py
cp $F/ops/xpu_router.py $F/ops/xpu_fused.py $F/ops/xpu_moe_decode.py $F/ops/xpu_decode.py build-dec4/
for i in 1 2; do
  echo "== graph bench $i"
  EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench
  echo "GRAPH_RUN_$i=$(ls -td http-run.* | head -1)"
done
echo "== eager bench"
READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench
echo "EAGER_RUN=$(ls -td http-run.* | head -1)"
