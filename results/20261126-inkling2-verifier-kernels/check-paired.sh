#!/usr/bin/env bash
# No serving arm runs unless every integrated XPU parity case executes and passes.
set -euo pipefail
bash probe-python.sh paired -m pytest -q -rs -p no:cacheprovider paired/tests/test_hybrid_prefix_sdpa.py -k xpu | tee "$PRETEST_DIR/paired-tests.txt"
grep -Eq '^7 passed' "$PRETEST_DIR/paired-tests.txt"
