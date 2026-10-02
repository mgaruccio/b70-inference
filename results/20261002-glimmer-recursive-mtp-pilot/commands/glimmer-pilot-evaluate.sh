#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PATH="$HOME/mtp-env/bin:$PATH"
cd "$HOME/mtp-pilot-code"
RUN="$HOME/mtp-pilot-run"
set -x
timeout 5400 python scripts/experiments/glimmer_recursive_mtp.py evaluate --capture "$RUN/capture.pt" --heads "$RUN/heads/fixed-ce.pt" "$RUN/heads/shared-ce.pt" "$RUN/heads/shared-state.pt" --max-new-tokens 64 --repeats 1 --output "$RUN/evaluate.json" 2>&1 | tee "$RUN/evaluate.log"
