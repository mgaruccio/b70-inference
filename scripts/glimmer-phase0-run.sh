#!/usr/bin/env bash
# Phase 0 Glimmer instrumentation driver for inference-host.
# Public boundary: POST /v1/chat/completions stream=true on :18099.
set -euo pipefail

ROOT="${ROOT:-$HOME/inference}"
LAUNCHER="$ROOT/launchers/start-muse-glimmer.sh"
INSTR="$ROOT/launchers/glimmer-phase0-instrument.py"
LOG="$ROOT/logs/muse-glimmer.log"
STAMP="${STAMP:-$(date +%Y%m%dT%H%M%S)}"
OUT="${OUT:-$HOME/b70-evals/muse-glimmer/202608-program/$STAMP}"
PORT="${PORT:-18099}"
BASE="http://127.0.0.1:${PORT}/v1"

mkdir -p "$OUT"

wait_health() {
  local i
  for i in $(seq 1 90); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "health timeout" >&2
  return 1
}

record_launch() {
  local name="$1"
  {
    echo "cell=$name"
    echo "date=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "uname=$(uname -a)"
    echo "llama=$( "$ROOT/src/llama.cpp/build/bin/llama-server" --version 2>&1 | tr '\n' ' ' )"
    echo "git=$(git -C "$ROOT/src/llama.cpp" rev-parse HEAD)"
    echo "argv=$(tr '\0' ' ' < /proc/"$(pgrep -n -f '/llama-server -m')"/cmdline 2>/dev/null || true)"
    echo "dflash=${DFLASH:-}"
    echo "slots=${PARALLEL_SLOTS:-}"
    echo "slot_ctx=${SLOT_CTX:-}"
    echo "spec_n_max=${SPEC_N_MAX:-}"
    echo "weights=ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38"
    echo "draft=b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef"
  } > "$OUT/${name}.launch.txt"
}

start_cell() {
  local name="$1"
  echo "=== start $name DFLASH=${DFLASH:-0} SLOTS=${PARALLEL_SLOTS:-} CTX=${SLOT_CTX:-} NMAX=${SPEC_N_MAX:-} ==="
  tmux kill-session -t muse 2>/dev/null || true
  sleep 1
  : > "$LOG"
  tmux new -d -s muse "exec env DFLASH=${DFLASH:-0} PARALLEL_SLOTS=${PARALLEL_SLOTS:-8} SLOT_CTX=${SLOT_CTX:-32768} SPEC_N_MAX=${SPEC_N_MAX:-2} PORT=${PORT} $LAUNCHER > $LOG 2>&1"
  wait_health
  record_launch "$name"
}

run_wave() {
  local label="$1"
  shift
  python3 "$INSTR" --base "$BASE" --label "$label" --out "$OUT/${label}.json" --log "$LOG" "$@"
}

echo "OUT=$OUT"
curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null
record_launch "live-n2-np2"

# --- C1 on the live production cell (do not restart first) ---
run_wave c1-live-default --concurrency 1 --reps 3 --max-tokens 128 --warmup
run_wave c1-live-reason-low --concurrency 1 --reps 2 --max-tokens 128 --reasoning-strength low
run_wave c1-live-p2k --concurrency 1 --reps 1 --max-tokens 128 --prompt-tokens 2000
run_wave c1-live-p8k --concurrency 1 --reps 1 --max-tokens 128 --prompt-tokens 8000

# --- C4+ #27117 matrix on 8 slots / 32k, ub stays 512 ---
for nmax in 1 2 4; do
  DFLASH=1 PARALLEL_SLOTS=8 SLOT_CTX=32768 SPEC_N_MAX="$nmax" start_cell "c8-nmax${nmax}"
  run_wave "c4-n${nmax}-load4" --concurrency 4 --reps 1 --max-tokens 128 --warmup
  run_wave "c8-n${nmax}-load8" --concurrency 8 --reps 1 --max-tokens 128
done

# Healthy-control: 4 slots, n_max=2, 4 in flight
DFLASH=1 PARALLEL_SLOTS=4 SLOT_CTX=32768 SPEC_N_MAX=2 start_cell "c4-nmax2-np4"
run_wave "c4-n2-np4-load4" --concurrency 4 --reps 1 --max-tokens 128 --warmup

# Restore production 1-2 stream cell
DFLASH=1 PARALLEL_SLOTS=2 SLOT_CTX=131072 SPEC_N_MAX=2 start_cell "restore-n2-np2"
run_wave "c1-restore-smoke" --concurrency 1 --reps 1 --max-tokens 32

echo "PHASE0_DONE $OUT"
ls -la "$OUT"
