#!/usr/bin/env bash
cd ~/flint-vllm-spike; export NO_RESTORE=1
for k in ${KS:-2 4 5 3}; do
env PROBE_SCRIPT=ktest3/chat-bench.py NUM_SPEC=$k EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2; echo "RUN_K$k=$(ls -td http-run.* | head -1)"
done
