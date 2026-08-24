#!/usr/bin/env bash
# Name the Vulkan op inside the C4 n_max=2 30B verify.
# Runtime env only. Official DFlash. Do not rewrite inject.
# Public boundary: POST /v1/chat/completions stream=true on :18099.
set -euo pipefail

ROOT=/home/mike/inference
SERVER="$ROOT/src/llama.cpp-dflash2/build/bin/llama-server"
MODEL_DIR="$ROOT/models/Muse-Glimmer-30B-GGUF"
TARGET="$MODEL_DIR/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf"
DRAFT="$MODEL_DIR/dflash-Muse-Glimmer-30B-Q4_K_M.gguf"
INSTR="$ROOT/launchers/glimmer-phase0-instrument.py"
SUMMARIZE="$ROOT/launchers/b70-summarize-vkperf.py"
LOG="$ROOT/logs/muse-glimmer.log"
PORT=18099
STAMP=$(date +%Y%m%dT%H%M%S)
OUT="${OUT:-$HOME/b70-evals/muse-glimmer/${STAMP}-vkperf}"

mkdir -p "$OUT"
"$SERVER" --version > "$OUT/server-version.txt" 2>&1 || true
git -C "$ROOT/src/llama.cpp-dflash2" rev-parse HEAD > "$OUT/git.txt"
{
  echo "intel_gpu_top_xe=unsupported"
  echo "gputop=$(command -v gputop)"
  echo "GGML_VK_PERF_LOGGER=1"
  echo "GGML_VK_PERF_LOGGER_CONCURRENT=1"
  echo "GGML_VK_PERF_LOGGER_FREQUENCY=1"
} > "$OUT/tools.txt"

wait_health() {
  local i
  for i in $(seq 1 120); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "health timeout" >&2
  return 1
}

start_gputop() {
  local label="$1"
  stdbuf -oL -eL gputop -d 0.1 -n 400 > "$OUT/${label}.gputop.raw" 2>&1 &
  echo $! > "$OUT/${label}.gputop.pid"
}

stop_gputop() {
  local label="$1"
  local pidfile="$OUT/${label}.gputop.pid"
  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile")
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -f "$pidfile"
  fi
  python3 - "$OUT/${label}.gputop.raw" "$OUT/${label}.gputop.txt" <<'PY'
import re, sys
from pathlib import Path
raw, out = Path(sys.argv[1]), Path(sys.argv[2])
if raw.exists():
    text = raw.read_text(errors="replace")
    out.write_text(re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text))
PY
}

start_cell() {
  local name="$1"
  local slots="$2"
  local slot_ctx="$3"
  local nmax="$4"
  local vkperf="${5:-1}"
  local extra_flags="${6:---log-timestamps --perf}"
  local ctx=$((slots * slot_ctx))
  echo "=== start $name slots=$slots ctx=$ctx nmax=$nmax vkperf=$vkperf ==="
  tmux kill-session -t muse 2>/dev/null || true
  sleep 1
  : > "$LOG"
  local env_prefix=""
  if [ "$vkperf" = "1" ]; then
    env_prefix="GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_CONCURRENT=1 GGML_VK_PERF_LOGGER_FREQUENCY=1"
  fi
  tmux new -d -s muse "exec env $env_prefix $SERVER \
    -m $TARGET -a muse-glimmer-30b \
    -ngl 99 -c $ctx -np $slots --kv-unified -fa on \
    -ctk q8_0 -ctv q4_1 -b 512 -ub 512 --no-mmap \
    --jinja --temp 1.0 --top-p 0.95 --top-k 64 \
    --host 0.0.0.0 --port $PORT \
    -md $DRAFT --spec-type draft-dflash --spec-draft-n-max $nmax -ngld 99 \
    $extra_flags \
    > $LOG 2>&1"
  wait_health
  {
    echo "cell=$name"
    echo "date=$(date -Is)"
    echo "vkperf=$vkperf"
    echo "slots=$slots"
    echo "slot_ctx=$slot_ctx"
    echo "nmax=$nmax"
    echo "argv=$(tr '\0' ' ' < /proc/$(pgrep -n -f '/llama-server -m')/cmdline)"
    echo "environ_vk=$(tr '\0' '\n' < /proc/$(pgrep -n -f '/llama-server -m')/environ | grep '^GGML_VK' || true)"
    grep -E 'b70tick seq_rm|initializing,' "$LOG" | head -20
  } > "$OUT/${name}.launch.txt"
}

run_wave() {
  local label="$1"
  shift
  start_gputop "$label"
  python3 "$INSTR" --base "http://127.0.0.1:${PORT}/v1" \
    --label "$label" --out "$OUT/${label}.json" --log "$LOG" "$@"
  stop_gputop "$label"
  cp -a "$LOG" "$OUT/${label}.server.log"
  grep -E 'b70tick ' "$LOG" > "$OUT/${label}.b70tick.log" || true
  python3 "$SUMMARIZE" --log "$OUT/${label}.server.log" \
    --out "$OUT/${label}.vkperf.json" --label "$label" || true
}

echo "OUT=$OUT"

# C1 n2 busy baseline
start_cell c1-n2 2 131072 2 1
run_wave c1-n2 --concurrency 1 --reps 1 --max-tokens 64 --warmup

# C4 n1 busy concurrent
start_cell c8-n1 8 32768 1 1
run_wave c4-n1-load4 --concurrency 4 --reps 1 --max-tokens 64 --warmup

# C4 n2 idle concurrent — the bug
start_cell c8-n2 8 32768 2 1
run_wave c4-n2-load4 --concurrency 4 --reps 1 --max-tokens 64 --warmup

# Restore production 1-2 stream official n_max=2, no logger
start_cell restore-n2-np2 2 131072 2 0 ""
run_wave c1-restore-smoke --concurrency 1 --reps 1 --max-tokens 32

python3 "$SUMMARIZE" --compare \
  --c1 "$OUT/c1-n2.vkperf.json" \
  --c4n1 "$OUT/c4-n1-load4.vkperf.json" \
  --c4n2 "$OUT/c4-n2-load4.vkperf.json" \
  --out "$OUT/vkperf-compare.json" || true

echo "VKPERF_DONE $OUT"
ls -la "$OUT"
