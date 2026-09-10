#!/usr/bin/env bash
# Execute on inference-host; each named cell is explicit and never overwritten.
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
[ "$#" -gt 0 ] || { echo 'Specify explicit cells.' >&2; exit 2; }
cd "$CODE"
for CELL in "$@"; do
  EXTRA=()
  case "$CELL" in
    baseline) ;;
    cap1) EXTRA=(--verification-cap 1) ;;
    cap3) EXTRA=(--verification-cap 3) ;;
    cap7) EXTRA=(--verification-cap 7) ;;
    adaptive) EXTRA=(--adaptive-verification) ;;
    *) echo "Unknown cell: $CELL" >&2; exit 2 ;;
  esac
  [ ! -e "$R/$CELL" ] || { echo "Refusing to overwrite $CELL" >&2; exit 2; }
  invariants | tee "$R/$CELL.preflight.txt"
  printf 'BEGIN_CELL=%s UTC=%s\n' "$CELL" "$(date -u +%FT%TZ)"
  COMMAND=(timeout --signal=TERM --kill-after=120s 120m python3 -u "$CODE/scripts/experiments/qwen38_standard_bench.py" --out "$R/$CELL" --betterbench "$BETTERBENCH" --guard "$GUARD" --patch "$CODE/scripts/patch-vllm-qwen38-dflash2-bf16.py" --prefill-patch "$CODE/scripts/patch-vllm-qwen38-xpu-prefill.py" --long-context-only --context 180224 --draft-int4 "$DRAFT" --max-num-batched-tokens 2048 --cache-group-size 8 --lengths 512 65536 160000 --near-limit 180096 "${EXTRA[@]}")
  printf '%q ' "${COMMAND[@]}" > "$R/$CELL.command.txt"; printf '\n' >> "$R/$CELL.command.txt"
  set +e
  "${COMMAND[@]}" 2>&1 | tee "$R/$CELL.console.txt"
  code=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\n' "$CELL" "$code" | tee -a "$R/cell-exits.tsv"
  invariants | tee "$R/$CELL.cleanup.txt"
  printf 'END_CELL=%s EXIT=%s UTC=%s\n' "$CELL" "$code" "$(date -u +%FT%TZ)"
  [ "$code" -eq 0 ] || exit "$code"
done
invariants | tee "$R/final-cleanup.txt"
printf 'REQUESTED_CELLS_COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)"
