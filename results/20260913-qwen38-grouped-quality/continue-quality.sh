#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-grouped-quality
R=$PWD
mkdir -p build-03
docker build --progress=plain --pull=false -t qwen38-quality:20260913 . > build-03/build.log 2>&1 || { rc=$?; echo "$rc" > build-03/exit-code.txt; tail -60 build-03/build.log; exit "$rc"; }
echo 0 > build-03/exit-code.txt
docker image inspect qwen38-quality:20260913 > build-03/image.json
bash score-cell.sh smoke-target-01
for arm in native candidate; do
  python3 -u run-quality.py --arm "$arm" --data "$R/preparation-02/prepared/smoke.jsonl" --limit 4 --out "$R/smoke-$arm-01" --benchmark-timeout 7200 > "$R/smoke-$arm-01.console.log" 2>&1 || { rc=$?; echo "$rc" > "$R/smoke-$arm-01.exit-code.txt"; tail -60 "$R/smoke-$arm-01.console.log"; exit "$rc"; }
  echo 0 > "$R/smoke-$arm-01.exit-code.txt"
  bash score-cell.sh "smoke-$arm-01"
  echo "SMOKE $arm PASSED"
done
# Only reach the full run after all three real generation/scoring smokes pass.
for arm in target native candidate; do
  python3 -u run-quality.py --arm "$arm" --data "$R/preparation-02/prepared" --out "$R/full-$arm-01" --benchmark-timeout 86400 > "$R/full-$arm-01.console.log" 2>&1 || { rc=$?; echo "$rc" > "$R/full-$arm-01.exit-code.txt"; tail -60 "$R/full-$arm-01.console.log"; exit "$rc"; }
  echo 0 > "$R/full-$arm-01.exit-code.txt"
  bash score-cell.sh "full-$arm-01"
  echo "FULL $arm PASSED"
done
python3 quality.py compare --target full-target-01/generation.jsonl --native full-native-01/generation.jsonl --candidate full-candidate-01/generation.jsonl --target-score full-target-01/scores/scores.json --native-score full-native-01/scores/scores.json --candidate-score full-candidate-01/scores/scores.json --out comparison.json
