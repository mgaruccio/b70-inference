#!/usr/bin/env bash
# usage: swap-build.sh fused|prefused  (installs the saved Flint XPU files into the venv)
set -euo pipefail
cd "$HOME/flint-vllm-spike"
F=.venv/lib/python3.12/site-packages/vllm/models/flint
B=build-$1
cp $B/model.py $F/model.py
cp $B/norm.py $F/ops/norm.py
cp $B/xpu_router.py $F/ops/xpu_router.py
[[ -f $B/xpu_fused.py ]] && cp $B/xpu_fused.py $F/ops/xpu_fused.py
[[ -f $B/xpu_moe_decode.py ]] && cp $B/xpu_moe_decode.py $F/ops/xpu_moe_decode.py
[[ -f $B/xpu_decode.py ]] && cp $B/xpu_decode.py $F/ops/xpu_decode.py
echo "installed $B"
