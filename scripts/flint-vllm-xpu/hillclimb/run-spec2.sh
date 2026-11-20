#!/usr/bin/env bash
# spec2: grouped gate/up + row-shared qkv for M > 1; standard + chat benches.
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
export NO_RESTORE=1
cmp -s build-spec1/model.py $F/model.py || { echo venv model.py differs from build-spec1; exit 5; }
cp ktest3/xpu_decode.py $F/ops/xpu_decode.py
mkdir -p build-spec2 && cp build-spec1/* build-spec2/ && cp $F/ops/xpu_decode.py build-spec2/
G="EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800"
run() { echo "== $1"; shift; env "$@" bash maintenance.sh ${MODE}; echo "RUN=$(ls -td http-run.* | head -1)"; }
MODE=--bench run "nonspec std" EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096"
MODE=--bench-dflash2 run "spec k3 std" NUM_SPEC=3 EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none"
MODE=--bench run "nonspec chat" PROBE_SCRIPT=ktest3/chat-bench.py EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096"
for k in 3 7; do
MODE=--bench-dflash2 run "spec k$k chat" PROBE_SCRIPT=ktest3/chat-bench.py NUM_SPEC=$k EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none"
done
