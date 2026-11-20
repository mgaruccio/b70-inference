#!/usr/bin/env bash
# spec3: int8 draft lm_head (drafter only) on build-spec2; std k3 parity + chat k3/4/5.
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
Q=.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash2.py
export NO_RESTORE=1
cmp -s build-spec2/model.py $F/model.py && cmp -s build-spec2/xpu_decode.py $F/ops/xpu_decode.py || { echo venv differs from build-spec2; exit 5; }
[ -f qwen3_dflash2.py.orig ] || cp $Q qwen3_dflash2.py.orig
cmp -s qwen3_dflash2.py.orig $Q || { echo qwen3_dflash2.py already modified; exit 6; }
cp ktest3/xpu_draft_head.py $F/ops/xpu_draft_head.py
.venv/bin/python -I ktest3/apply_drafthead_patch.py $Q || exit 7
mkdir -p build-spec3 && cp build-spec2/* build-spec3/ && cp $F/ops/xpu_draft_head.py $Q build-spec3/
env NUM_SPEC=3 EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2; echo "RUN_STD_K3=$(ls -td http-run.* | head -1)"
for k in 3 4 5; do
env PROBE_SCRIPT=ktest3/chat-bench.py NUM_SPEC=$k EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2; echo "RUN_CHAT_K$k=$(ls -td http-run.* | head -1)"
done
