#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PATH="$HOME/mtp-env/bin:$PATH"
cd "$HOME/mtp-pilot-code"
RUN="$HOME/mtp-pilot-run-r1"
mkdir -p "$RUN"
printf '%s\n' adbb09a264407d8986771e931a6a64f90fccbd8a > "$RUN/code-commit.txt"
python -m pip freeze > "$RUN/pip-freeze.txt"
python -m torch.utils.collect_env > "$RUN/collect-env.txt" 2>&1
nvidia-smi -q > "$RUN/nvidia-smi.txt"
P=scripts/experiments/glimmer_recursive_mtp.py
set -x
python -m pytest -q -p no:cacheprovider tests/test_glimmer_recursive_mtp.py 2>&1 | tee "$RUN/pytest.log"
python -S "$P" validate 2>&1 | tee "$RUN/validate.log"
timeout 1200 python "$P" train --capture "$RUN/capture.pt" --output-dir "$RUN/heads" --rank 64 --updates 100 --batch-size 4 --seed 20261002 2>&1 | tee "$RUN/train.log"
timeout 4500 python "$P" evaluate --record-divergence --capture "$RUN/capture.pt" --heads "$RUN/heads/fixed-ce.pt" "$RUN/heads/shared-ce.pt" "$RUN/heads/shared-state.pt" --max-new-tokens 64 --repeats 1 --output "$RUN/evaluate-diagnostic.json" 2>&1 | tee "$RUN/evaluate-diagnostic.log"
