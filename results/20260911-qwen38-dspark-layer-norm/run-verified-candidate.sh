#!/usr/bin/env bash
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-layer-norm
IMAGE=vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4
cd "$R"
timeout 180s docker run --rm --pull=never --network=none --read-only --tmpfs /tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e B70_DSPARK_SOURCE_ROOT=/opt/venv/lib/python3.12/site-packages/vllm \
  -v "$R:/input:ro" --entrypoint /bin/bash "$IMAGE" -lc \
  'mkdir -p /tmp/repo; cp -a /input/scripts /input/tests /tmp/repo/; cd /tmp/repo; /opt/venv/bin/python -B -m unittest discover -s tests -p test_qwen38_dspark_bf16.py -v' \
  2>&1 | tee pinned-image-tests.txt
if ! grep -q '^OK$' pinned-image-tests.txt || grep -q 'skipped=' pinned-image-tests.txt; then
  echo 'Full tensor suite must pass without skips' >&2
  exit 1
fi
bash run-candidate.sh
