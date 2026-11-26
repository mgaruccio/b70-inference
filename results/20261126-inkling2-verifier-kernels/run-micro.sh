#!/usr/bin/env bash
# Run as PRETEST inside hillclimb.sh. One fresh process per arm; don't mix DSOs.
set -euo pipefail
idx=0
for a in "$@"; do
    idx=$((idx+1))
    echo "run $idx $a"
    bash probe-python.sh "$a" bench_verify_nvfp4.py --repeats 7 --replays 32 -o "$PRETEST_DIR/micro-$idx-$a.json"
done
python3 compare-micro.py "$PRETEST_DIR"/micro-*.json
