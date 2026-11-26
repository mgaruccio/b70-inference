#!/usr/bin/env bash
# Same extension and runtime, fresh process per flag, no compiler environment.
set -euo pipefail
idx=0
for flag in 0 1 0; do
    idx=$((idx+1)); export INK2_NVFP4_FMA_SCALE_DECODE=$flag
    out=$PRETEST_DIR/dequant-$idx-$flag
    bash probe-python.sh dequant -m pytest -q -rs -p no:cacheprovider dequant/tests/test_moe_xpu.py -k 'nvfp4_cutlass_codec or (xpu_cutlass_experts and nvfp4)' | tee "$out.pytest.txt"
    grep -Eq '^8 passed' "$out.pytest.txt"
    bash probe-python.sh dequant dequant/scripts/bench_verify_nvfp4.py --repeats 7 --replays 32 -o "$out.json"
done
python3 compare-micro.py "$PRETEST_DIR"/dequant-*.json
