#!/usr/bin/env bash
# Development acceptance diagnostic; no timing/speedup claim for offloaded arms.
set -euo pipefail
set -o noclobber
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-quant-control
D=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-acceptance-diagnostics
H=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
N=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-layer-norm
CANONICAL_OVERLAY="$N/patch-dspark-native.py"
CANONICAL_SHA256=0640edc7a72c4b6650bb6846cdc988c87883dad7c0cb86d36684513a1c070643
printf '%s  %s\n' "$CANONICAL_SHA256" "$CANONICAL_OVERLAY" | sha256sum -c -
cd "$R"
# Current corrected eager cell is the same-target/no-offload diagnostic.
timeout --signal=INT --kill-after=90s 10800s python3 -B -u run-quant-control.py \
  --arm gptq --out "$R/gptq-offload8" \
  --driver "$D/run-acceptance-diagnostics.py" --previous-campaign "$H" --draft-dir "$H/draft" \
  --canonical-overlay "$CANONICAL_OVERLAY" --canonical-sha256 "$CANONICAL_SHA256" \
  --no-offload-reference "$N/norm-eager" \
  > "$R/gptq-offload8-driver.log" 2>&1
timeout --signal=INT --kill-after=90s 10800s python3 -B -u run-quant-control.py \
  --arm fp8 --out "$R/fp8-offload8" --target "$R/target-fp8" \
  --target-manifest "$R/download-result.json" --paired-with "$R/gptq-offload8" \
  --driver "$D/run-acceptance-diagnostics.py" --previous-campaign "$H" --draft-dir "$H/draft" \
  --canonical-overlay "$CANONICAL_OVERLAY" --canonical-sha256 "$CANONICAL_SHA256" \
  --no-offload-reference "$N/norm-eager" \
  > "$R/fp8-offload8-driver.log" 2>&1
