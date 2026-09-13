#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
invariants() {
  date -Is
  [ -z "$(docker ps -q)" ]
  [ "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000 ]
  [ "$(docker inspect -f '{{.State.Running}}' glimmer-tb21-prefix-c8)" = false ]
  printf '63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4  /home/mike/inference/launchers/start-qwen38.sh\n' | sha256sum -c -
  for device in /dev/dri/renderD*; do
    if fuser -v "$device"; then echo "Render device occupied: $device" >&2; return 1; fi
  done
  echo 'IDLE_HOST_POWER275W_LAUNCHER_UNCHANGED'
}
for cell in mtp4 dspark dflash; do
  [ ! -e "$cell-01" ] || { echo "Refusing overwrite: $cell-01" >&2; exit 2; }
  invariants > "$cell.preflight.txt" 2>&1
  command=(timeout --signal=TERM --kill-after=120s 90m python3 -u run-comparison.py --cell "$cell" --out "$cell-01")
  printf '%q ' "${command[@]}" > "$cell.command.txt"; printf '\n' >> "$cell.command.txt"
  echo "BEGIN_CELL=$cell UTC=$(date -u +%FT%TZ)"
  set +e
  "${command[@]}" > "$cell.console.txt" 2>&1
  code=$?
  set -e
  printf '%s\t%s\n' "$cell" "$code" | tee -a cell-exits.tsv
  invariants > "$cell.cleanup.txt" 2>&1
  echo "END_CELL=$cell EXIT=$code UTC=$(date -u +%FT%TZ)"
  [ "$code" -eq 0 ] || exit "$code"
done
invariants > host-final.txt 2>&1
echo "ALL_THREE_COMPLETE UTC=$(date -u +%FT%TZ)"
