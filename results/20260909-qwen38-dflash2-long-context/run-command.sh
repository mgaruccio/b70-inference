#!/usr/bin/env bash
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-long-context
scp -q /tmp/run-qwen38-long-context.sh "inference-host:$R/run-command.sh"
ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'bash -s' <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-long-context
CODE="$R/code"
DRAFT=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-quant/rtn-int4-g128
GUARD=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2/runner/patch_uniform_decode_prefill.py
BETTERBENCH=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard/code/betterbench
invariants() {
  [ -z "$(docker ps -q)" ]
  [ "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000 ]
  [ "$(docker inspect -f '{{.State.Running}}' glimmer-tb21-prefix-c8)" = false ]
  printf '63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4  /home/mike/inference/launchers/start-qwen38.sh\n' | sha256sum -c -
  printf 'POWER_MICROWATTS=275000000\nGLIMMER_RUNNING=false\nRUNNING_CONTAINERS=0\n'
}
invariants | tee "$R/preflight.txt"
cd "$CODE"
python3 scripts/experiments/qwen38_standard_bench.py --help > "$R/runner-help.txt"
failed=0
for CELL in int4-48k int4-64k bf16-48k bf16-64k; do
  case "$CELL" in *48k) LIMIT=49152 ;; *64k) LIMIT=65536 ;; esac
  EXTRA=()
  case "$CELL" in int4-*) EXTRA=(--draft-int4 "$DRAFT") ;; esac
  invariants > "$R/$CELL.preflight.txt"
  printf '\nBEGIN_CELL=%s CONTEXT=%s UTC=%s\n' "$CELL" "$LIMIT" "$(date -u +%FT%TZ)"
  COMMAND=(timeout --signal=TERM --kill-after=120s 60m python3 -u "$CODE/scripts/experiments/qwen38_standard_bench.py" --out "$R/$CELL" --betterbench "$BETTERBENCH" --guard "$GUARD" --patch "$CODE/scripts/patch-vllm-qwen38-dflash2-bf16.py" --prefill-patch "$CODE/scripts/patch-vllm-qwen38-xpu-prefill.py" --long-context-only --context "$LIMIT" "${EXTRA[@]}")
  printf '%q ' "${COMMAND[@]}" > "$R/$CELL.command.txt"; printf '\n' >> "$R/$CELL.command.txt"
  set +e
  "${COMMAND[@]}" 2>&1 | tee "$R/$CELL.console.txt"
  code=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\n' "$CELL" "$code" | tee -a "$R/cell-exits.tsv"
  invariants | tee "$R/$CELL.cleanup.txt"
  if [ "$code" -ne 0 ]; then failed=1; fi
  python3 - "$R/$CELL" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
summary = json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {}
print('CELL_OUTCOME=' + json.dumps({'cell':root.name, 'status':summary.get('status','no summary'), 'error':summary.get('error'), 'context':summary.get('context')}), flush=True)
path = root/'long-context/summary.json'
if path.exists():
    data = json.loads(path.read_text())
    for point in data['points']:
        print('POINT_OUTCOME=' + json.dumps({'cell':root.name, 'prompt':point['requested_length'], 'status':point['status'], 'valid_measured':sum(bool(row.get('valid')) for row in point['measurements']), 'summary':point.get('summary')}), flush=True)
PY
  printf 'END_CELL=%s EXIT=%s UTC=%s\n' "$CELL" "$code" "$(date -u +%FT%TZ)"
done
invariants | tee "$R/final-cleanup.txt"
printf 'CAMPAIGN_COMPLETE failed_cells_present=%s UTC=%s\n' "$failed" "$(date -u +%FT%TZ)"
exit "$failed"
REMOTE
