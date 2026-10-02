#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PATH="$HOME/mtp-env/bin:$PATH"
cd "$HOME/mtp-pilot-code"
RUN="$HOME/mtp-pilot-run/eager"
mkdir -p "$RUN"
P=scripts/experiments/glimmer_recursive_mtp.py
set -x
timeout 600 python "$P" capture --attention eager --output "$RUN/capture.pt" 2>&1 | tee "$RUN/capture.log"
timeout 1200 python "$P" train --attention eager --capture "$RUN/capture.pt" --output-dir "$RUN/heads" --rank 64 --updates 100 --batch-size 4 --seed 20261002 2>&1 | tee "$RUN/train.log"
timeout 4800 python "$P" evaluate --attention eager --capture "$RUN/capture.pt" --heads "$RUN/heads/fixed-ce.pt" "$RUN/heads/shared-ce.pt" "$RUN/heads/shared-state.pt" --max-new-tokens 64 --repeats 1 --output "$RUN/evaluate.json" 2>&1 | tee "$RUN/evaluate.log"
