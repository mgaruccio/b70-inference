#!/usr/bin/env bash
# spec4: + int8 drafter linears (M <= 64) on build-spec3; std k3 parity + chat k3/4.
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
Q=.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash2.py
export NO_RESTORE=1
cmp -s build-spec3/xpu_draft_head.py $F/ops/xpu_draft_head.py && cmp -s build-spec3/qwen3_dflash2.py $Q || { echo venv differs from build-spec3; exit 5; }
cp ktest3/xpu_draft_head.py $F/ops/xpu_draft_head.py
mkdir -p build-spec4 && cp build-spec3/* build-spec4/ && cp $F/ops/xpu_draft_head.py build-spec4/
env NUM_SPEC=3 EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2; echo "RUN_STD_K3=$(ls -td http-run.* | head -1)"
for k in 3 4; do
env PROBE_SCRIPT=ktest3/chat-bench.py NUM_SPEC=$k EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2; echo "RUN_CHAT_K$k=$(ls -td http-run.* | head -1)"
done
