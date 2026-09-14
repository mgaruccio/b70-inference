#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-grouped-quality
R=$PWD
IMAGE=$(cat preparation-02/image-id.txt)
for arm in target native candidate; do
  python3 -u run-quality.py --arm "$arm" --data "$R/preparation-02/prepared/smoke.jsonl" --limit 4 --out "$R/smoke-$arm-01" --benchmark-timeout 7200 > "$R/smoke-$arm-01.console.log" 2>&1 || { rc=$?; echo "$rc" > "$R/smoke-$arm-01.exit-code.txt"; tail -60 "$R/smoke-$arm-01.console.log"; exit "$rc"; }
  echo 0 > "$R/smoke-$arm-01.exit-code.txt"
  mkdir -p "$R/smoke-$arm-01/scores"
  docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --cpus 4 --memory 8g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g --user "$(id -u):$(id -g)" -v "$R/preparation-02/prepared:/input/prepared:ro" -v "$R/smoke-$arm-01:/input/generation:ro" -v "$R/smoke-$arm-01/scores:/output:rw" "$IMAGE" score --task all --allow-code-execution --data /input/prepared --generation /input/generation/generation.jsonl --out /output/scores.json > "$R/smoke-$arm-01/score.log" 2>&1 || { rc=$?; echo "$rc" > "$R/smoke-$arm-01/score.exit-code.txt"; tail -60 "$R/smoke-$arm-01/score.log"; exit "$rc"; }
  echo 0 > "$R/smoke-$arm-01/score.exit-code.txt"
  echo "SMOKE $arm PASSED"
done
