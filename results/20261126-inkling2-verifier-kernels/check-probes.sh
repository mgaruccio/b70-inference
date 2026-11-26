#!/usr/bin/env bash
# Must exercise the real FP64 comparison test (importorskip alone isn't a pass).
set -euo pipefail
for a in pf6 pf3 pf9; do
    bash probe-python.sh "$a" -m pytest -q -rs -p no:cacheprovider tests/test_moe_xpu.py -k 'test_xpu_cutlass_experts and nvfp4' | tee "$PRETEST_DIR/$a.pytest.txt"
    grep -Eq '^7 passed' "$PRETEST_DIR/$a.pytest.txt"
done
