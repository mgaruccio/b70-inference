#!/usr/bin/env bash
# Development-tier A/B/C comparison runner for Inkling 2 Mini on the B70 (guest b70-dev).
# Transient loopback server on :18080 only; the caller stops inkling-inference first.
# Usage: cmp.sh CONFIG SPEC LABEL MODE...
#   CONFIG: A (our hc-s1) | B (hc-ck1 + INK2_CUTLASS_MOE=1, SergiioB) | C (F3zz1k fork image, lossless env)
#           | Cbf16head (C with VLLM_INKLING2_HEAD=bf16, informational)
#   SPEC  : 0 | 1 (DSpark k=7)
#   MODE  : smoke | decode | depth6k | depth30k | prefill | parity | tf
# Env: TF_REF=<run dir with decode-0-*.json from config A, spec 0> (mode tf)
set -euo pipefail
umask 022
cfg=$1 spec=$2 label=$3; shift 3
p=$HOME/inkling-vllm-xpu
hc=$p/hc-scripts
cmp=$HOME/cmp
weights=$p/weights-ed40e7da625f594fc9c4990f5f66be24873c22d7
D=$HOME/flint-vllm-spike/driver-26.40
py=$HOME/flint-vllm-spike/.venv/bin/python
url=http://127.0.0.1:18080
MAXLEN=${MAXLEN:-32768} UTIL=0.92
art=$cmp/runs/$(date +%Y%m%d-%H%M%S)-$cfg-s$spec-$label
mkdir -p "$art"
[[ -z $(ss -Hltn 'sport = :18080') ]]
[[ -z $(systemctl is-active inkling-inference.service | grep -x active) ]]
specj='{"method":"dspark","num_speculative_tokens":7}'
common=(--served-model-name inkling-2-mini --host 127.0.0.1 --port 18080 --max-model-len $MAXLEN
        --gpu-memory-utilization $UTIL --max-num-seqs 1 --block-size 64 --no-enable-prefix-caching
        --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling)
pid= cname=
cleanup() {
    rc=$?
    trap - EXIT
    if [[ -n $cname ]]; then docker stop -t 20 "$cname" >/dev/null 2>&1 || true; fi
    if [[ -n $pid ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; sleep 3; kill -KILL -- "-$pid" 2>/dev/null || true; fi
    if ((rc != 0)); then tail -60 "$art/server.log" 2>/dev/null || true; fi
    printf 'ART %s rc=%s\n' "$art" "$rc"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM HUP INT
{ date -Is; uname -sr; cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null | sed 's/^/power1_cap_uW=/'; } > "$art/environment.txt"
case $cfg in
A|B|H1|H2)
    # H1: CUTLASS for >4-token batches, our decode GEMV (checkpoint NVFP4 layout) for <=4 tokens on
    #     NVFP4 layers; MXFP8 layers stay on CUTLASS. H2: as H1, MXFP8 layers fully on our kernels.
    tree=hc-s1; [[ $cfg == B ]] && tree=hc-ck1; [[ $cfg == H* ]] && tree=hc-hy1
    export LD_LIBRARY_PATH=$D/usr/lib/x86_64-linux-gnu:$D/usr/lib:$D/usr/local/lib LIBRARY_PATH=$D/usr/lib/x86_64-linux-gnu:$D/usr/lib CPATH=$D/usr/include OCL_ICD_VENDORS=$HOME/flint-vllm-spike/opencl-vendors
    export PYTHONPATH=$p/src/$tree:$p/src/transformers-mini/src HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TRITON_CACHE_DIR=$p/.triton
    export NEOReadDebugKeys=1 ForcePreemptionMode=3
    [[ $cfg != A ]] && export INK2_CUTLASS_MOE=1
    [[ $cfg == H* ]] && export INK2_HYBRID_DEC_MAX=4
    [[ $cfg == H2 ]] && export INK2_CUTLASS_MXFP8=0
    args=(serve "$weights" "${common[@]}" --language-model-only --dtype bfloat16 --max-num-batched-tokens "${CHUNK:-2048}"
          -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY)
    [[ $spec == 1 ]] && args+=(--speculative-config "$specj")
    { "$py" -c 'import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__, torch.xpu.get_device_name())'; echo "tree=$tree"; env | grep -E '^(INK2|INKLING|VLLM|TRITON|SYCL|IGC|NEO|Force)' | sort; } >> "$art/environment.txt" 2>&1
    printf '%q ' "$py" -B -m vllm.entrypoints.cli.main "${args[@]}" > "$art/launch.txt"; echo >> "$art/launch.txt"
    setsid "$py" -B -m vllm.entrypoints.cli.main "${args[@]}" > "$art/server.log" 2>&1 &
    pid=$! ;;
C|C0|Cbf16head)
    # C: lossless env + upstream vLLM PR #69186 (XPU fp32-output lm_head mm) bind-mounted, as config A has it.
    # C0: lossless env, unpatched image (fp32 head = per-step weight cast). Cbf16head: bf16 head (informational).
    head=fp32; [[ $cfg == Cbf16head ]] && head=bf16
    cname=f3-cmp
    docker rm -f "$cname" >/dev/null 2>&1 || true
    pr=()
    [[ $cfg == C ]] && pr=(-v "$cmp/f3-pr69186/logits_processor.py:/workspace/vllm/vllm/model_executor/layers/logits_processor.py:ro")
    dargs=(run --rm --name "$cname" --network host --ipc=host --privileged --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path "${pr[@]}"
           --shm-size 32g -v "$weights":/models/inkling-2-mini:ro
           -e LD_PRELOAD=/opt/b70/libl0_owner_residency.so -e ZE_ENABLE_TRACING_LAYER=1
           -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 -e VLLM_USE_V2_MODEL_RUNNER=1
           -e NEOReadDebugKeys=1 -e ClDeviceGlobalMemSizeAvailablePercent=100
           -e EnableDeferBacking=1 -e DeferBackingMemoryPressurePercent=0
           -e CCL_SYCL_ALLREDUCE_TMP_BUF=1 -e CCL_SYCL_ALLGATHERV_TMP_BUF=1
           -e CCL_SYCL_REDUCE_SCATTER_TMP_BUF=1 -e CCL_SYCL_BROADCAST_TMP_BUF=1
           -e ExperimentalCopyThroughLock=0 -e ExperimentalH2DCpuCopyThreshold=0 -e ExperimentalD2HCpuCopyThreshold=0
           -e VLLM_INKLING2_EXPERTS=nvfp4 -e VLLM_INKLING2_MXFP8=bf16 -e VLLM_INKLING2_DENSE=bf16
           -e VLLM_INKLING2_HEAD=$head -e VLLM_INKLING2_PREFILL_EXPERTS=nvfp4 -e VLLM_NO_USAGE_STATS=1
           -e ZE_AFFINITY_MASK=0 --entrypoint vllm vllm-xpu:b70-inkling2
           serve /models/inkling-2-mini "${common[@]}" --trust-remote-code --tensor-parallel-size 1
           --max-num-batched-tokens "${CHUNK:-4096}" --limit-mm-per-prompt '{"image":0}')
    [[ $spec == 1 ]] && dargs+=(--speculative-config "$specj")
    { docker image inspect vllm-xpu:b70-inkling2 --format '{{.Id}} {{.Created}}'; echo "f3_commit=edd722c73517b43680483187afca93fb6ab41958"; } >> "$art/environment.txt"
    printf '%q ' docker "${dargs[@]}" > "$art/launch.txt"; echo >> "$art/launch.txt"
    setsid docker "${dargs[@]}" > "$art/server.log" 2>&1 &
    pid=$! ;;
*) echo "bad config $cfg"; exit 2 ;;
esac
t0=$(date +%s)
for _ in $(seq 1 900); do
    if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then break; fi
    kill -0 "$pid" 2>/dev/null || { echo 'server died'; exit 1; }
    sleep 2
done
curl -fsS --max-time 3 "$url/health" >/dev/null
echo "startup_s $(( $(date +%s) - t0 ))" | tee "$art/startup.txt"
for m in "$@"; do
    echo "== $m $(date -Is)" | tee -a "$art/modes.txt"
    case $m in
    smoke)
        curl -sS --fail-with-body --max-time 300 -H 'Content-Type: application/json' -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 37 plus 58? Reply with only the number."}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"reasoning_effort":"low"}}' "$url/v1/chat/completions" > "$art/chat.json"
        jq -e '.choices[0] | .finish_reason == "stop" and (.message.content | test("^\\s*95\\s*$"))' "$art/chat.json" ;;
    decode) "$py" -B "$hc/decode_bench.py" "$art" --tokens 256 ;;
    depth6k) "$py" -B "$hc/decode_bench.py" "$art" --tokens 256 --depth 6000 --cases 3 ;;
    depth30k) "$py" -B "$hc/decode_bench.py" "$art" --tokens 256 --depth 30000 --cases 3 ;;
    prefill) "$py" -B "$hc/xpu_prefill_bench.py" "$art" "$weights/tokenizer.json" ;;
    parity) "$py" -B "$cmp/collect.py" "$art/parity" && "$py" -B "$cmp/summarize.py" "$art/parity" "$cmp/ref" ;;
    tf) "$py" -B "$hc/tf_bench.py" "${TF_REF:?}" "$art" ;;
    *) echo "unknown mode $m"; exit 2 ;;
    esac
done
grep -E 'Model loading took|KV cache size|Available KV|GPU KV cache size|Maximum concurrency' "$art/server.log" > "$art/memory.txt" || true
