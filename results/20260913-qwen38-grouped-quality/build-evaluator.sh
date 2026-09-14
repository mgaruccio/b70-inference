#!/usr/bin/env bash
set -uo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-grouped-quality
mkdir -p build-02
docker build --progress=plain --pull=false -t qwen38-quality:20260913 . 2>&1 | tee build-02/build.log
rc=${PIPESTATUS[0]}
echo "$rc" > build-02/exit-code.txt
if [ "$rc" -eq 0 ]; then
  docker image inspect qwen38-quality:20260913 > build-02/image.json
fi
exit "$rc"
