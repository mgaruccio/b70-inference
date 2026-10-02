#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PATH="$HOME/mtp-env/bin:$PATH"
cd "$HOME/mtp-pilot-code"
RUN="$HOME/mtp-pilot-run"
set -x
timeout 1200 python scripts/experiments/glimmer_recursive_mtp.py train --capture "$RUN/capture.pt" --output-dir "$RUN/heads" --rank 64 --updates 100 --batch-size 4 --seed 20261002 2>&1 | tee "$RUN/train.log"
