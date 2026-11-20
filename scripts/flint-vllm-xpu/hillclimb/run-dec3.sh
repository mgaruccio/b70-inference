#!/usr/bin/env bash
# Gate + install + benchmark round-3 decode kernels.
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
OFFLINE_SCRIPT=ktest3/decfinal.py OFFLINE_MAXSEC=900 bash maintenance.sh --offline
r=$(ls -td http-run.* | head -1); cat $r/decfinal.json || exit 3
grep -q '"PASS": true' $r/decfinal.json || { echo UNIT GATE FAILED; exit 4; }
cmp -s build-moe/model.py $F/model.py || { echo venv model.py differs from build-moe; exit 5; }
cp ktest3/xpu_decode.py $F/ops/xpu_decode.py
.venv/bin/python -I ktest3/apply_dec3_patch.py $F/model.py || exit 6
mkdir -p build-dec3
cp $F/model.py build-dec3/model.py; cp $F/ops/norm.py build-dec3/norm.py
cp $F/ops/xpu_router.py $F/ops/xpu_fused.py $F/ops/xpu_moe_decode.py $F/ops/xpu_decode.py build-dec3/
grep -q xpu_decode.py swap-build.sh || sed -i 's|^echo "installed $B"|[[ -f $B/xpu_decode.py ]] \&\& cp $B/xpu_decode.py $F/ops/xpu_decode.py\necho "installed $B"|' swap-build.sh
echo "== graph bench"
EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench
echo "GRAPH_RUN=$(ls -td http-run.* | head -1)"
echo "== eager bench"
READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench
echo "EAGER_RUN=$(ls -td http-run.* | head -1)"
