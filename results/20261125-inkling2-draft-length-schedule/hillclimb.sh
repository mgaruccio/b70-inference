#!/usr/bin/env bash
# b70-dev only. Interleaved A/B of one launcher change against the live 409K/C32 config
# on an isolated loopback engine (:18080). Always restores production on exit.
# Usage: bash hillclimb.sh ctl cand ctl     (arms: ctl | cand | kN | sched with SCHEDULE=json; QUICK=1 for sweeps)
set -euo pipefail
umask 077
p=$HOME/inkling-vllm-xpu
runtime=$HOME/flint-vllm-spike
stage=$p/sparse-69444            # production source/plugin/venv (see ../20261125-inkling2-sparse-conv)
here=$(cd "$(dirname "$0")" && pwd)
weights=$p/weights-ed40e7da625f594fc9c4990f5f66be24873c22d7
live=$p/serve/serve-inkling.sh
svc=inkling-inference.service
out=$(mktemp -d "$here/run-$(date -u +%Y%m%d-%H%M%S).XXXXXX")
py=$stage/.venv/bin/python
candidate_pid=
restore=0
exec 9>"$stage/maintenance.lock"
flock -n 9
printf 'Artifacts: %s\n' "$out"
(($#)) || { echo "usage: $0 ARM..." >&2; exit 2; }

stop_candidate() {
    if [[ -n $candidate_pid ]]; then
        kill -TERM -- "-$candidate_pid" 2>/dev/null || true
        for ((i=0; i<30; i++)); do
            kill -0 -- "-$candidate_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$candidate_pid" 2>/dev/null || true
        wait "$candidate_pid" 2>/dev/null || true
        candidate_pid=
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT
    trap '' TERM INT HUP
    set +e
    stop_candidate
    if ((restore)); then
        if ! cmp -s "$out/saved-launcher.sh" "$live"; then
            echo 'Production launcher changed unexpectedly; not overwriting it' >&2
            rc=1
        fi
        if sudo -n systemctl start "$svc" && \
           timeout 300 bash "$HOME/cmp/live/live_check.sh" after-adaptive-verif >"$out/restored.log" 2>&1 && \
           systemctl is-active --quiet "$svc" && \
           [[ $(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models) == 401 ]]; then
            echo 'Production restored: authenticated smoke passed and unauthenticated model listing returned 401'
        else
            echo "ERROR: production restoration did not verify; see $out/restored.log" >&2
            rc=1
        fi
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

boot() {
    local arm=$1 dir=$2
    local base='"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"'
    local spec="{$base}"
    local -a extra=()
    case "$arm" in
        ctl) ;;
        cand) spec="{$base,\"enable_adaptive_verification\":true}" ;;
        k[1-6]) spec="{$base,\"num_speculative_tokens_per_batch_size\":[[1,32,${arm#k}]]}" ;;
        sched) [[ -n ${SCHEDULE:-} ]]; spec="{$base,\"num_speculative_tokens_per_batch_size\":$SCHEDULE}" ;;
        *) return 2 ;;
    esac
    mkdir -p "$dir"
    local driver=$runtime/driver-26.40
    # Identical to the live launcher except loopback host/port, no auth middleware,
    # --enable-prompt-tokens-details, and the arm's speculative config/env.
    local -a cmd=(env -u VLLM_API_KEY
        "PYTHONPATH=$stage/plugin:$stage/sparse:$runtime/serve" "TRITON_CACHE_DIR=$stage/triton"
        "LD_LIBRARY_PATH=$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib:$driver/usr/local/lib"
        "LIBRARY_PATH=$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib" "CPATH=$driver/usr/include"
        "OCL_ICD_VENDORS=$runtime/opencl-vendors" HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
        NEOReadDebugKeys=1 ForcePreemptionMode=3 VLLM_LOGGING_LEVEL=INFO
        INK2_CUTLASS_MOE=1 INK2_HYBRID_PREFIX_SDPA=1 INK2_SPARSE_CONV_HISTORY=1 "${extra[@]}"
        "$py" -B -m vllm.entrypoints.cli.main serve "$weights"
        --host 127.0.0.1 --port 18080 --served-model-name inkling-2-mini
        --language-model-only --dtype bfloat16 --max-model-len 409000
        --max-num-batched-tokens 2048 --gpu-memory-utilization 0.98 --max-num-seqs 32
        --block-size 64 --enable-prefix-caching --no-async-scheduling --kv-cache-dtype fp8
        --enable-prompt-tokens-details --generation-config auto --chat-template "$weights/chat_template.jinja"
        --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling
        -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY
        --speculative-config "$spec")
    printf '%q ' "${cmd[@]}" >"$dir/launch.txt"; printf '\n' >>"$dir/launch.txt"
    local t0=$SECONDS
    setsid "${cmd[@]}" >"$dir/server.log" 2>&1 9>&- &
    candidate_pid=$!
    for ((i=0; i<480; i++)); do
        kill -0 "$candidate_pid" 2>/dev/null || { tail -60 "$dir/server.log" >&2; return 1; }
        if curl -fsS --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
            echo "startup_s=$((SECONDS - t0))" >"$dir/startup.txt"
            curl -fsS --max-time 5 http://127.0.0.1:18080/v1/models >"$dir/models.json"
            grep -E 'Available KV cache memory|XPU KV cache size|KV cache size|adaptive|Adaptive' "$dir/server.log" | tee "$dir/capacity.txt" || true
            return 0
        fi
        sleep 1
    done
    tail -60 "$dir/server.log" >&2
    return 1
}

bench() {  # 3 paired trials per concurrency; seeds depend only on (n, trial)
    local dir=$1; shift
    PYTHONPATH="$stage/sparse:$stage" HF_HUB_OFFLINE=1 "$py" -B - "$dir" "$weights" "$@" <<'PY'
import sys
from pathlib import Path
from sweep import bench

root = Path(sys.argv[1])
for arg in sys.argv[3:]:
    for trial in (1, 2, 3):
        row = bench(root, sys.argv[2], int(arg), trial, timeout_s=300)
        if not row["passed"]:
            raise RuntimeError(f"benchmark check failed: {row}")
PY
}

[[ $(id -un) == mike && $(hostname -s) == b70-dev ]]
systemctl is-active --quiet "$svc"
sudo -n true
[[ -x $py && -s $stage/sweep.py && -s $here/checks.py ]]
if curl -fsS --max-time 3 http://127.0.0.1:18080/health >/dev/null 2>&1; then
    echo 'Loopback candidate port is already serving' >&2; exit 1
fi
cp -p "$live" "$out/saved-launcher.sh"
"$py" -I -c 'import importlib.metadata as m; print("\n".join(sorted(f"{d.name}=={d.version}" for d in m.distributions())))' >"$out/environment.txt"
{ uname -a; cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null || true; git -C "$stage/sparse" rev-parse HEAD 2>/dev/null || true; } >"$out/host.txt"
curl -fsS --max-time 5 http://127.0.0.1:8080/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ { seen++; if ($NF != 0) busy=1 } END { exit(seen < 2 || busy) }'
restore=1
sudo -n systemctl stop "$svc"
ref= arm_idx=0 full_needle_done=0
for arm in "$@"; do
    arm_idx=$((arm_idx + 1))
    dir=$out/$(printf '%02d' "$arm_idx")-$arm
    boot "$arm" "$dir"
    if [[ ${NEEDLE_ONLY:-0} == 1 ]]; then  # ~100% context gate only (375K requested ≈ 388K rendered)
        "$py" -B "$here/checks.py" needle "$dir" --tokens 375000 | tee "$dir/needle.log" || echo "NEEDLE FAILED 375000" | tee -a "$dir/needle.log"
        stop_candidate
        continue
    fi
    "$py" -B "$here/checks.py" parity "$dir" ${ref:+--reference "$ref"} | tee "$dir/parity.log"
    [[ -n $ref ]] || ref=$dir
    "$py" -B "$here/checks.py" chat "$dir" | tee "$dir/chat.log"
    if [[ ${QUICK:-0} == 1 ]]; then
        bench "$dir" 1 4 8 16 32 | tee "$dir/bench.log"
        for c in 8 32; do "$py" -B "$here/checks.py" chatc "$dir" --concurrency $c | tee -a "$dir/chat.log"; done
        stop_candidate
        continue
    fi
    "$py" -B "$here/checks.py" chat "$dir" --depth 30000 --cases 3 | tee -a "$dir/chat.log"
    bench "$dir" 1 4 8 16 32 | tee "$dir/bench.log"
    for c in 8 32; do "$py" -B "$here/checks.py" chatc "$dir" --concurrency $c | tee -a "$dir/chat.log"; done
    # Needle gate failures are recorded (needle-*.json "passed") but do not abort the A/B.
    "$py" -B "$here/checks.py" needle "$dir" --tokens 200000 | tee "$dir/needle.log" || echo "NEEDLE FAILED 200000" | tee -a "$dir/needle.log"
    if [[ $arm != ctl && $full_needle_done == 0 ]]; then
        "$py" -B "$here/checks.py" needle "$dir" --tokens 375000 | tee -a "$dir/needle.log" || echo "NEEDLE FAILED 375000" | tee -a "$dir/needle.log"
        full_needle_done=1
    fi
    stop_candidate
done
echo "B70 development A/B finished: $out"
