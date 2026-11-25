#!/usr/bin/env bash
# b70-dev only. Test isolated loopback engines; always restore production.
set -euo pipefail
umask 077
p=$HOME/inkling-vllm-xpu
runtime=$HOME/flint-vllm-spike
stage=$p/sparse-69444
weights=$p/weights-ed40e7da625f594fc9c4990f5f66be24873c22d7
live=$p/serve/serve-inkling.sh
svc=inkling-inference.service
out=$(mktemp -d "$stage/run-$(date -u +%Y%m%d-%H%M%S).XXXXXX")
py=$stage/.venv/bin/python
candidate_pid=
restore=0
exec 9>"$stage/maintenance.lock"
flock -n 9
printf 'Artifacts: %s\n' "$out"

stop_candidate() {
    if [[ -n $candidate_pid ]]; then
        kill -TERM -- "-$candidate_pid" 2>/dev/null || true
        for ((i=0; i<15; i++)); do
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
           timeout 240 bash "$HOME/cmp/live/live_check.sh" after-sparse-69444 >"$out/restored.log" 2>&1 && \
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
    local variant=$1 cap=$2 dir=$3 source plugin sparse=0
    mkdir -p "$dir"
    case "$variant" in
        old) source=$p/src/vllm-main-3146a9df41; plugin=$p/plugin-27474f21f0fba583d314deb400e5d052c0194567 ;;
        dense) source=$stage/dense; plugin=$stage/plugin ;;
        sparse) source=$stage/sparse; plugin=$stage/plugin; sparse=1 ;;
        *) return 2 ;;
    esac
    local driver=$runtime/driver-26.40
    local -a cmd=(env -u VLLM_API_KEY
        "PYTHONPATH=$plugin:$source:$runtime/serve" "TRITON_CACHE_DIR=$stage/triton"
        "LD_LIBRARY_PATH=$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib:$driver/usr/local/lib"
        "LIBRARY_PATH=$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib" "CPATH=$driver/usr/include"
        "OCL_ICD_VENDORS=$runtime/opencl-vendors" HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
        NEOReadDebugKeys=1 ForcePreemptionMode=3 VLLM_LOGGING_LEVEL=INFO
        INK2_CUTLASS_MOE=1 INK2_HYBRID_PREFIX_SDPA=1 "INK2_SPARSE_CONV_HISTORY=$sparse"
        "$py" -B -m vllm.entrypoints.cli.main serve "$weights"
        --host 127.0.0.1 --port 18080 --served-model-name inkling-2-mini
        --language-model-only --dtype bfloat16 --max-model-len 327680
        --max-num-batched-tokens 2048 --gpu-memory-utilization 0.98 --max-num-seqs "$cap"
        --block-size 64 --enable-prefix-caching --no-async-scheduling --kv-cache-dtype fp8
        --enable-prompt-tokens-details --generation-config auto --chat-template "$weights/chat_template.jinja"
        --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling
        -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY
        --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}')
    printf '%q ' "${cmd[@]}" >"$dir/launch.txt"; printf '\n' >>"$dir/launch.txt"
    setsid "${cmd[@]}" >"$dir/server.log" 2>&1 9>&- &
    candidate_pid=$!
    for ((i=0; i<240; i++)); do
        kill -0 "$candidate_pid" 2>/dev/null || { tail -60 "$dir/server.log" >&2; return 1; }
        if curl -fsS --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
            curl -fsS --max-time 5 http://127.0.0.1:18080/v1/models >"$dir/models.json"
            grep -E 'Available KV cache memory|XPU KV cache size' "$dir/server.log" | tee "$dir/capacity.txt"
            return 0
        fi
        sleep 1
    done
    tail -60 "$dir/server.log" >&2
    return 1
}

# Use the existing benchmark and near-limit workloads, without launching a full sweep.
bench() {
    local variant=$1 dir=$2; shift 2
    local source=$stage/$variant
    [[ $variant != old ]] || source=$p/src/vllm-main-3146a9df41
    PYTHONPATH="$source:$stage" HF_HUB_OFFLINE=1 "$py" -B - "$dir" "$weights" "$@" <<'PY'
import sys
from pathlib import Path
from sweep import bench, near_limit

root = Path(sys.argv[1])
for arg in sys.argv[3:]:
    if arg == "near":
        near_limit(root)
    else:
        row = bench(root, sys.argv[2], int(arg), 1, timeout_s=300 if arg == "64" else 100)
        print(row, flush=True)
        if not row["passed"]:
            raise RuntimeError(f"benchmark check failed: {row}")
PY
}

systemctl is-active --quiet "$svc"
[[ $(id -un) == mike && $(hostname -s) == b70-dev ]]
sudo -n true
[[ -x $py && -s $stage/parity.py && -s $stage/sweep.py ]]
compgen -G "$stage/plugin/vllm_inkling2_xpu/ops/_inkling_nvfp4_xe2*.so" >/dev/null
if [[ -n ${OLD_BASELINE:-} ]]; then
    [[ -s "$OLD_BASELINE/c32-r1.summary.json" && -s "$OLD_BASELINE/32771-extend.json" ]]
    cp -a "$OLD_BASELINE" "$out/old-c32"
    printf '%s\n' "$OLD_BASELINE" >"$out/old-baseline-source.txt"
fi
if [[ -n ${VERIFIED_C32_RUN:-} ]]; then
    [[ -z ${OLD_BASELINE:-} ]]
    for variant in old dense sparse; do
        src=$VERIFIED_C32_RUN/$variant-c32
        [[ -s $src/c1-r1.summary.json && -s $src/c32-r1.summary.json && -s $src/metrics.json ]]
        if [[ $variant != old ]]; then [[ -s $src/comparison.json ]]; fi
        cp -a "$src" "$out/$variant-c32"
    done
    printf '%s\n' "$VERIFIED_C32_RUN" >"$out/c32-baseline-source.txt"
fi
if curl -fsS --max-time 3 http://127.0.0.1:18080/health >/dev/null 2>&1; then
    echo 'Loopback candidate port is already serving' >&2; exit 1
fi
cp -p "$live" "$out/saved-launcher.sh"
"$py" -I -c 'import importlib.metadata as m; print("\n".join(sorted(f"{d.name}=={d.version}" for d in m.distributions())))' >"$out/environment.txt"
curl -fsS --max-time 5 http://127.0.0.1:8080/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ { seen++; if ($NF != 0) busy=1 } END { exit(seen < 2 || busy) }'
restore=1
sudo -n systemctl stop "$svc"
for variant in old dense sparse; do
    if [[ $variant == old && -n ${OLD_BASELINE:-} ]]; then continue; fi
    if [[ -n ${VERIFIED_C32_RUN:-} ]]; then continue; fi
    dir=$out/$variant-c32
    boot "$variant" 32 "$dir"
    args=()
    if [[ $variant == dense ]]; then args=(--reference "$out/old-c32"); fi
    if [[ $variant == sparse ]]; then args=(--reference "$out/dense-c32"); fi
    "$py" -B "$stage/parity.py" "$dir" "${args[@]}" | tee "$dir/parity.log"
    bench "$variant" "$dir" 1 32 | tee "$dir/bench-summary.log"
    stop_candidate
done
dir=$out/sparse-c64
boot sparse 64 "$dir"
bench sparse "$dir" 64 near | tee "$dir/bench-summary.log"
"$py" -B "$stage/parity.py" "$dir" --mixed
echo "B70 development validation finished: $out"
