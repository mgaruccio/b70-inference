#!/usr/bin/env bash
set -euo pipefail
for i in 1 2; do
    bash probe-python.sh router router/scripts/bench_router_topk.py --eight-row -o "$PRETEST_DIR/router-$i.json"
done
