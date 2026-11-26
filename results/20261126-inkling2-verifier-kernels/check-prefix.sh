#!/usr/bin/env bash
set -euo pipefail
bash probe-python.sh prefix -m pytest -q -rs -p no:cacheprovider prefix/tests/test_hybrid_prefix_sdpa.py -k 'xpu_fused_prefix_gather_fp8_bits or xpu_hybrid_prefix_sdpa_page_boundary' | tee "$PRETEST_DIR/prefix-tests.txt"
grep -Eq '^4 passed' "$PRETEST_DIR/prefix-tests.txt"
bash probe-python.sh prefix prefix/scripts/bench_prefix_kv_gather.py --trace "$PRETEST_DIR/gather.json" | tee "$PRETEST_DIR/gather-times.txt"
