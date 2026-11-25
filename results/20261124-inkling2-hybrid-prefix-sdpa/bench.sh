#!/usr/bin/env bash
# Run on the B70 guest with inkling-inference stopped. Loopback-only, no credentials.
# Usage: bash bench.sh /absolute/path/to/plugin-site /absolute/path/to/results
set -euo pipefail
plugin=$1
out=$2
p=$HOME/inkling-vllm-xpu
rt=$HOME/flint-vllm-spike
driver=$rt/driver-26.40
py=$rt/.venv/bin/python
weights=$p/weights-ed40e7da625f594fc9c4990f5f66be24873c22d7
[[ -d $plugin/vllm_inkling2_xpu && ! -e $out ]]
[[ $(systemctl is-active inkling-inference.service || :) == inactive ]]
[[ -z $(ss -Hltn 'sport = :18080') ]]
mkdir -p "$out"
export LD_LIBRARY_PATH=$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib:$driver/usr/local/lib
export LIBRARY_PATH=$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib CPATH=$driver/usr/include
export OCL_ICD_VENDORS=$rt/opencl-vendors PYTHONPATH=$plugin:$p/src/vllm-main-3146a9df41
export INK2_CUTLASS_MOE=1 TRITON_CACHE_DIR=$p/.triton HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export NEOReadDebugKeys=1 ForcePreemptionMode=3
unset VLLM_API_KEY
args=(serve "$weights" --host 127.0.0.1 --port 18080 --served-model-name inkling-2-mini
    --language-model-only --dtype bfloat16 --max-model-len 262144
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.98 --max-num-seqs 1
    --block-size 64 --no-enable-prefix-caching --no-async-scheduling --kv-cache-dtype fp8
    --generation-config auto --chat-template "$weights/chat_template.jinja"
    --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling
    -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}')
{
    date -uIs; uname -sr; lscpu | grep -E '^CPU\(s\):|^Model name:'; free -h
    grep -H . /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap || :
    "$py" -B -c 'import torch, triton, vllm, vllm_inkling2_xpu; print(torch.__version__, triton.__version__, vllm.__file__, vllm_inkling2_xpu.__file__, torch.xpu.get_device_name())'
    env | grep -E '^(INK2_CUTLASS_MOE|VLLM_NO_USAGE_STATS|NEOReadDebugKeys|ForcePreemptionMode|TRITON_CACHE_DIR|LD_LIBRARY_PATH|PYTHONPATH)=' | sort
} > "$out/environment.txt" 2>&1
pid=
stop_server() {
    if [[ -n $pid ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        pid=
    fi
}
trap stop_server EXIT
trap 'exit 143' TERM INT HUP
for flag in 0 1; do
    art=$out/flag-$flag
    mkdir "$art"
    export INK2_HYBRID_PREFIX_SDPA=$flag
    launch=$(printf '%q ' env INK2_CUTLASS_MOE=1 "INK2_HYBRID_PREFIX_SDPA=$flag" "$py" -B -m vllm.entrypoints.cli.main "${args[@]}")
    printf '%s\n' "${launch% }" > "$art/launch.txt"
    setsid "$py" -B -m vllm.entrypoints.cli.main "${args[@]}" > "$art/server.log" 2>&1 &
    pid=$!
    for ((i=0; i<150; i++)); do
        if curl -fsS --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then break; fi
        kill -0 "$pid" || { tail -40 "$art/server.log" >&2; exit 1; }
        sleep 2
    done
    curl -fsS --max-time 5 http://127.0.0.1:18080/v1/models | jq -e '[.data[]|select(.id=="inkling-2-mini" and .max_model_len==262144)]|length==1'
    "$py" -B "$(dirname "$0")/prefill.py" "$art" --trials "${TRIALS:-3}"
    stop_server
done
