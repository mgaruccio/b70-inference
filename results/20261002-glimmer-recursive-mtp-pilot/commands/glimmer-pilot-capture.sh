#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/mtp-pilot-code"
RUN="$HOME/mtp-pilot-run"
mkdir -p "$ROOT/scripts/experiments" "$ROOT/tests" "$RUN"
mv "$HOME/glimmer_recursive_mtp.py" "$HOME/glimmer_recursive_mtp.jsonl" "$HOME/glimmer_recursive_mtp.md" "$ROOT/scripts/experiments/"
mv "$HOME/test_glimmer_recursive_mtp.py" "$ROOT/tests/"
CACHE="$HOME/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
mkdir -p "$CACHE"
for f in "$HOME/glimmer-model/"*; do ln -s "$f" "$CACHE/$(basename "$f")"; done
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export PATH="$HOME/mtp-env/bin:$PATH"
cd "$ROOT"
printf '%s\n' b68a67ab > "$RUN/code-commit.txt"
python -m pip freeze > "$RUN/pip-freeze.txt"
python -m torch.utils.collect_env > "$RUN/collect-env.txt" 2>&1
nvidia-smi -q > "$RUN/nvidia-smi.txt"
set -x
python -m pytest -q -p no:cacheprovider tests/test_glimmer_recursive_mtp.py 2>&1 | tee "$RUN/pytest.log"
python -S scripts/experiments/glimmer_recursive_mtp.py validate 2>&1 | tee "$RUN/validate.log"
timeout 600 python scripts/experiments/glimmer_recursive_mtp.py capture --output "$RUN/capture.pt" 2>&1 | tee "$RUN/capture.log"
