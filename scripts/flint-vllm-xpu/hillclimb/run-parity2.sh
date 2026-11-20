#!/usr/bin/env bash
cd ~/flint-vllm-spike; export NO_RESTORE=1
env PROBE_SCRIPT=ktest3/parity2.py EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench; echo "RUN_NONSPEC=$(ls -td http-run.* | head -1)"
env PROBE_SCRIPT=ktest3/parity2.py NUM_SPEC=3 EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2; echo "RUN_K3=$(ls -td http-run.* | head -1)"
