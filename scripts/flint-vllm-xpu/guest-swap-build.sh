#!/usr/bin/env bash
# usage: swap-build.sh <name>  (installs the saved Flint XPU files from build-<name> into the venv)
set -euo pipefail
cd "$HOME/flint-vllm-spike"
F=.venv/lib/python3.12/site-packages/vllm/models/flint
Q=.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash2.py
B=build-$1
[[ -d $B ]]
cp $B/model.py $F/model.py
cp $B/norm.py $F/ops/norm.py
cp $B/xpu_router.py $F/ops/xpu_router.py
[[ -f $B/xpu_fused.py ]] && cp $B/xpu_fused.py $F/ops/xpu_fused.py
[[ -f $B/xpu_moe_decode.py ]] && cp $B/xpu_moe_decode.py $F/ops/xpu_moe_decode.py
[[ -f $B/xpu_decode.py ]] && cp $B/xpu_decode.py $F/ops/xpu_decode.py
[[ -f $B/xpu_draft_head.py ]] && cp $B/xpu_draft_head.py $F/ops/xpu_draft_head.py
# DFlash 2 drafter hook (spec3+); older builds get the stock file back.
if [[ -f $B/qwen3_dflash2.py ]]; then cp $B/qwen3_dflash2.py $Q; else cp qwen3_dflash2.py.orig $Q; fi
echo "installed $B"
