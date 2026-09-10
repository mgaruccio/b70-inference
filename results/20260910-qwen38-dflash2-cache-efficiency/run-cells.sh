#!/usr/bin/env bash
# Run on inference-host after copying the frozen code snapshot beside this file.
set -euo pipefail
R=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
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
[ "$#" -gt 0 ] || { echo 'Specify explicit cell names.' >&2; exit 2; }
cd "$CODE"
for CELL in "$@"; do
  EXTRA=()
  case "$CELL" in
    auto-8192-64k) BATCH=8192; LIMIT=65536 ;;
    auto-2048-64k) BATCH=2048; LIMIT=65536 ;;
    g8-8192-64k) BATCH=8192; LIMIT=65536; EXTRA=(--cache-group-size 8) ;;
    g8-2048-64k) BATCH=2048; LIMIT=65536; EXTRA=(--cache-group-size 8) ;;
    g8-4096-64k) BATCH=4096; LIMIT=65536; EXTRA=(--cache-group-size 8) ;;
    g8-2048-176k) BATCH=2048; LIMIT=180224; EXTRA=(--cache-group-size 8) ;;
    g8-2048-208k) BATCH=2048; LIMIT=212992; EXTRA=(--cache-group-size 8) ;;
    g8-4096-208k) BATCH=4096; LIMIT=212992; EXTRA=(--cache-group-size 8) ;;
    g8-2048-256k) BATCH=2048; LIMIT=262144; EXTRA=(--cache-group-size 8) ;;
    *) echo "Unknown cell: $CELL" >&2; exit 2 ;;
  esac
  [ ! -e "$R/$CELL" ] || { echo "Refusing to overwrite cell $CELL" >&2; exit 2; }
  if [ "$LIMIT" -eq 65536 ]; then
    EXTRA+=(--lengths 512 32768)
    DEADLINE=60m
  elif [ "$LIMIT" -eq 180224 ]; then
    EXTRA+=(--near-limit 180096)
    DEADLINE=180m
  elif [ "$LIMIT" -eq 212992 ]; then
    EXTRA+=(--lengths 512 8192 16384 32768 65536 120000 160000 190000 --near-limit 212864)
    DEADLINE=180m
  else
    EXTRA+=(--lengths 65536 190000 212864 --near-limit 262016)
    DEADLINE=180m
  fi
  invariants | tee "$R/$CELL.preflight.txt"
  printf '\nBEGIN_CELL=%s CONTEXT=%s PREFILL=%s UTC=%s\n' "$CELL" "$LIMIT" "$BATCH" "$(date -u +%FT%TZ)"
  COMMAND=(timeout --signal=TERM --kill-after=120s "$DEADLINE" python3 -u "$CODE/scripts/experiments/qwen38_standard_bench.py" --out "$R/$CELL" --betterbench "$BETTERBENCH" --guard "$GUARD" --patch "$CODE/scripts/patch-vllm-qwen38-dflash2-bf16.py" --prefill-patch "$CODE/scripts/patch-vllm-qwen38-xpu-prefill.py" --long-context-only --context "$LIMIT" --draft-int4 "$DRAFT" --max-num-batched-tokens "$BATCH" "${EXTRA[@]}")
  printf '%q ' "${COMMAND[@]}" > "$R/$CELL.command.txt"; printf '\n' >> "$R/$CELL.command.txt"
  set +e
  "${COMMAND[@]}" 2>&1 | tee "$R/$CELL.console.txt"
  code=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\n' "$CELL" "$code" | tee -a "$R/cell-exits.tsv"
  invariants | tee "$R/$CELL.cleanup.txt"
  python3 - "$R/$CELL" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
print('CELL_OUTCOME=' + json.dumps({'cell': root.name, 'status': summary.get('status'), 'error': summary.get('error'), 'context': summary.get('context')}), flush=True)
path = root / 'long-context/summary.json'
if path.exists():
    for point in json.loads(path.read_text())['points']:
        print('POINT_OUTCOME=' + json.dumps({'cell': root.name, 'prompt': point['requested_length'], 'status': point['status'], 'valid_measured': sum(bool(row.get('valid')) for row in point['measurements']), 'summary': point.get('summary')}), flush=True)
PY
  printf 'END_CELL=%s EXIT=%s UTC=%s\n' "$CELL" "$code" "$(date -u +%FT%TZ)"
  # Stop for diagnosis rather than hiding failures behind subsequent cells.
  [ "$code" -eq 0 ] || exit "$code"
done
invariants | tee "$R/final-cleanup.txt"
printf 'REQUESTED_CELLS_COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)"
