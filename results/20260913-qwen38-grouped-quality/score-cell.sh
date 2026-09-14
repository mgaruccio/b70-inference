#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-grouped-quality
R=$PWD
CELL=${1:?cell directory required}
IMAGE=$(docker image inspect --format '{{.Id}}' qwen38-quality:20260913)
mkdir -p "$R/$CELL/scores"
printf '%s\n' "$IMAGE" > "$R/$CELL/scores/image-id.txt"
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --cpus 4 --memory 8g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g --user "$(id -u):$(id -g)" -v "$R/preparation-02/prepared:/input/prepared:ro" -v "$R/$CELL:/input/generation:ro" -v "$R/$CELL/scores:/output:rw" "$IMAGE" score --task all --allow-code-execution --data /input/prepared --generation /input/generation/generation.jsonl --out /output/scores.json > "$R/$CELL/score-02.log" 2>&1 || { rc=$?; echo "$rc" > "$R/$CELL/score-02.exit-code.txt"; tail -60 "$R/$CELL/score-02.log"; exit "$rc"; }
echo 0 > "$R/$CELL/score-02.exit-code.txt"
