#!/usr/bin/env bash
set -euo pipefail
bash probe-python.sh layout -m pytest -q -rs -p no:cacheprovider layout/tests/test_moe_xpu.py -k 'coalesced_scales or wrong_scale_layouts' | tee "$PRETEST_DIR/layout.pytest.txt"
grep -Eq '^2 passed' "$PRETEST_DIR/layout.pytest.txt"
bash probe-python.sh layout layout/scripts/bench-nvfp4-scale-layout.py --tokens 1 8 32 128 --iterations 64 | tee "$PRETEST_DIR/layout-times.txt"
