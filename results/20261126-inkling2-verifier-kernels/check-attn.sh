#!/usr/bin/env bash
set -euo pipefail
bash probe-python.sh attn -m pytest -q -rs -p no:cacheprovider attn/tests/test_attention.py -k verify_bias_free_prefix | tee "$PRETEST_DIR/attn.pytest.txt"
grep -Eq '^3 passed' "$PRETEST_DIR/attn.pytest.txt"
bash probe-python.sh attn attn/scripts/bench_verify_attention.py | tee "$PRETEST_DIR/attn-times.txt"
