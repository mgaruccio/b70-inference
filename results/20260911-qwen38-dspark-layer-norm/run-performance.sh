#!/usr/bin/env bash
# Development run: only launch after inspecting the corrected numerical replay.
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-layer-norm
cd "$R"
# Same 36-request matrix/gates as the pre-fix campaign, then matched 64K input class.
timeout --signal=INT --kill-after=90s 2400s python3 -u run-native-sampling.py \
  --draft-sample-method greedy --cell dspark --out norm-eager 2>&1 | tee norm-eager-driver.log
timeout --signal=INT --kill-after=90s 2400s python3 -u run-native-sampling.py \
  --draft-sample-method greedy --graphs --cell dspark --out norm-graph 2>&1 | tee norm-graph-driver.log
timeout --signal=INT --kill-after=90s 2400s python3 -u run-native-sampling.py \
  --draft-sample-method greedy --graphs --long-context --cell dspark --out norm-graph-64k 2>&1 | tee norm-graph-64k-driver.log
