#!/usr/bin/env bash
set -euo pipefail
bash probe-python.sh merge -m pytest -q -rs -p no:cacheprovider merge/tests/test_hybrid_prefix_sdpa.py -k 'xpu_hybrid_prefix_sdpa_page_boundary or xpu_fused_prefix_merge' | tee "$PRETEST_DIR/merge-tests.txt"
grep -Eq '^4 passed' "$PRETEST_DIR/merge-tests.txt"
bash probe-python.sh merge merge/scripts/bench_prefill_prefix_merge.py --nt 127 129 2048 --merges 28 --trace "$PRETEST_DIR/prefix-merge.trace.json" | tee "$PRETEST_DIR/prefix-merge.jsonl"
