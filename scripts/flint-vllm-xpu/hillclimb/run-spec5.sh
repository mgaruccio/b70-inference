#!/usr/bin/env bash
# spec5 (final): spec3 behaviour (int8 draft head; drafter linears opt-in, off); std k4 parity + chat k4.
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
export NO_RESTORE=1
cp ktest3/xpu_draft_head.py $F/ops/xpu_draft_head.py
mkdir -p build-spec5 && cp build-spec4/* build-spec5/ && cp $F/ops/xpu_draft_head.py build-spec5/
env NUM_SPEC=4 EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2; echo "RUN_STD_K4=$(ls -td http-run.* | head -1)"
env PROBE_SCRIPT=ktest3/chat-bench.py NUM_SPEC=4 EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2; echo "RUN_CHAT_K4=$(ls -td http-run.* | head -1)"
