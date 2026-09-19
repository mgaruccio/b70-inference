#!/usr/bin/env bash
# CPU-only sandbox. Samples are untrusted; no writable host mounts or network.
set -euo pipefail
ROOT=/home/mike/b70-evals/20260918-mtp-quality-longctx
IMAGE=$(cat "$ROOT/evalplus-fixed-image.txt")
SAMPLES=$(realpath "$1")
OUTPUT=$2
NAME="mtp-quality-eval-$$"
test ! -e "$OUTPUT"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
timeout --signal=TERM --kill-after=15s 1800s docker run --name "$NAME" \
 --network none --read-only --user 1000:1000 --cap-drop ALL \
 --security-opt no-new-privileges --pids-limit 256 --memory 8g --cpus 4 \
 --tmpfs /tmp:rw,nosuid,size=4g,mode=1777 \
 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
 -e HUMANEVAL_OVERRIDE_PATH=/data/HumanEvalPlus.jsonl -e EVALPLUS_ALLOW_CODE_EXECUTION=1 \
 -v "$ROOT/data/HumanEvalPlus.jsonl:/data/HumanEvalPlus.jsonl:ro" \
 -v "$SAMPLES:/samples.jsonl:ro" \
 -v "$ROOT/commands/evaluate.py:/evaluate.py:ro" \
 --entrypoint python "$IMAGE" /evaluate.py > "$OUTPUT" 2> "$OUTPUT.stderr"
python3 - "$OUTPUT" <<'PY'
import json,sys
result=json.load(open(sys.argv[1]))
assert result['returncode']==0 and result['results'] is not None
print('EVALUATION_COMPLETE',sys.argv[1])
PY
