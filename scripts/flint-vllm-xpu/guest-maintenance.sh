#!/usr/bin/env bash
# One bounded local experiment. Run on the B70 guest, never as root.
set -euo pipefail
umask 077
cd "$HOME/flint-vllm-spike"
exec 9>maintenance.lock
flock -n 9
live=flint-inference.service
candidate_pattern='flint-vllm-smoke-*.service'
live_bin="$HOME/.local/share/b70-smoke/llama-b12499/llama-server"
cred_source=/run/b70-flint/api-key
live_args=("$live_bin" --model "$HOME/models/flint/Flint-Q4_K_M.gguf" --alias flint
           --device Vulkan0 --gpu-layers all --fit off --ctx-size 16384 --parallel 1
           --port 8080 --jinja --temp 1 --top-p 0.95 --top-k 20 --host 0.0.0.0
           --api-key-file /run/credentials/flint-inference.service/api-key)

sudo -n test -s "$cred_source"
# NO_RESTORE=1: llama.cpp is not restored on exit and may already be stopped
# (vLLM is the intended full-time server).
no_restore=${NO_RESTORE:-0}
live_up=1
systemctl is-active --quiet "$live" || live_up=0
(( live_up || no_restore ))
active_candidates=$(systemctl list-units --no-legend --plain --state=active,activating,deactivating 'flint-vllm-smoke.service' "$candidate_pattern")
if [[ -n $active_candidates ]]; then
    echo 'A candidate service is already running' >&2
    exit 1
fi
listeners=$(ss -H -ltn)
if grep -Eq '(^|[[:space:]])[^[:space:]]*:18080[[:space:]]' <<< "$listeners"; then
    echo 'Candidate port already in use' >&2
    exit 1
fi
if (( live_up )); then
pid=$(systemctl show "$live" -p MainPID --value)
mapfile -d '' -t running_argv < "/proc/$pid/cmdline"
[[ $(declare -p running_argv | cut -d= -f2-) == "$(declare -p live_args | cut -d= -f2-)" ]]
[[ $(systemctl show "$live" -p Environment --value) == "LD_LIBRARY_PATH=${live_bin%/*}" ]]
[[ $(systemctl show "$live" -p User --value) == "$USER" ]]
[[ $(systemctl show "$live" -p FragmentPath --value) == /run/systemd/transient/flint-inference.service ]]
fi
if [[ ${1:-} == --check-rollback ]]; then
    printf 'Live transient unit, credential source, launch arguments and environment match rollback.\n'
    exit 0
fi
mode=${1:-}
case "$mode" in
    --run|--bench|--profile|--probe|--offline) ;;
    --dflash2|--bench-dflash2|--profile-dflash2|--probe-dflash2) [[ -s dflash2/model.safetensors && -s dflash2/config.json ]] ;;
    *) exit 2 ;;
esac
[[ -s preflight.ok && -s gptq/verified.ok ]]
read -r driver_path < preflight.ok
[[ $driver_path == "$PWD/driver-26.40/usr/lib/x86_64-linux-gnu:$PWD/driver-26.40/usr/lib" ]]
run=$(mktemp -d "$PWD/http-run.XXXXXX")
candidate="flint-vllm-smoke-${run##*.}.service"
printf 'Test artifacts: %s\n' "$run"

(( live_up )) && python3 -I - "$pid" <<'PY'
import http.client
import json
import pathlib
import sys

args = pathlib.Path(f'/proc/{sys.argv[1]}/cmdline').read_bytes().decode().split('\0')
with pathlib.Path(args[args.index('--api-key-file') + 1]).open() as f:
    key = f.readline(4096).strip()
assert key
conn = http.client.HTTPConnection('127.0.0.1', 8080, timeout=3)
conn.request('GET', '/slots', headers={'Authorization': 'Bearer ' + key})
resp = conn.getresponse()
assert resp.status == 200
slots = json.load(resp)
assert slots and all(s.get('is_processing') is False for s in slots), 'live slot busy; aborting maintenance'
conn.close()
PY

restore() {
    local rc=$?
    trap - EXIT HUP INT TERM
    set +e
    sudo -n timeout 28s systemctl stop "$candidate" >/dev/null 2>&1
    if (( no_restore )); then
        echo 'NO_RESTORE=1: llama.cpp left stopped.'
        exit "$rc"
    fi
    if ! systemctl is-active --quiet "$live"; then
        sudo -n systemd-run --quiet --collect --unit="$live" --uid="$USER" \
            --property=Type=exec --property="LoadCredential=api-key:$cred_source" \
            --setenv="LD_LIBRARY_PATH=${live_bin%/*}" -- "${live_args[@]}"
        if [[ $? -ne 0 ]]; then
            printf 'ROLLBACK FAILED to start %s\n' "$live" >&2
            exit 91
        fi
    fi
    for (( n=0; n<50; n++ )); do
        if curl -fsS --max-time 1 http://127.0.0.1:8080/health > "$run/restored-health.json" 2>/dev/null; then
            systemctl show "$live" -p MainPID -p InvocationID -p ActiveState
            echo 'Existing llama.cpp serving again on :8080.'
            exit "$rc"
        fi
        sleep 1
    done
    printf 'ROLLBACK FAILED: %s not ready\n' "$live" >&2
    exit 92
}
trap restore EXIT
trap 'exit 130' HUP INT TERM
(( live_up )) && sudo -n timeout 100s systemctl stop "$live"
# No speculative arguments. No remote code. Candidate is not reachable off VM.
cmd=("$PWD/.venv/bin/vllm" serve "$PWD/gptq"
     --host 127.0.0.1 --port 18080 --served-model-name flint-xpu-test
     --dtype bfloat16 --kv-cache-dtype auto --attention-backend TRITON_ATTN
     --gpu-memory-utilization 0.90 --max-model-len 4096
     --max-num-seqs 1 --max-num-batched-tokens 512 --disable-prefix-caching
     --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47)
[[ -n ${EAGER:-1} && ${EAGER:-1} != 0 ]] && cmd+=(--enforce-eager)
# shellcheck disable=SC2206
[[ -n ${EXTRA_ARGS:-} ]] && cmd+=($EXTRA_ARGS)
if [[ $mode == *dflash2 ]]; then
    cmd+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$PWD/dflash2\",\"num_speculative_tokens\":${NUM_SPEC:-7}}")
fi
if [[ $mode == --profile* ]]; then
    mkdir -p "$run/trace"
    cmd+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$run/trace\",\"torch_profiler_activities\":[\"CPU\",\"XPU\"],\"torch_profiler_with_stack\":false,\"torch_profiler_dump_cuda_time_total\":false}")
fi
if [[ $mode == --offline ]]; then
    cmd=("$PWD/.venv/bin/python" -I "$PWD/${OFFLINE_SCRIPT:-offline.py}" "$run" ${OFFLINE_ARGS:-})
fi
printf '%q ' "${cmd[@]}" > "$run/argv.txt"
printf '\n' >> "$run/argv.txt"
if [[ $mode == --offline ]]; then
    sudo -n systemd-run --quiet --collect --unit="$candidate" --uid="$USER" --working-directory="$PWD" \
        --property=Type=exec --property=RuntimeMaxSec=${OFFLINE_MAXSEC:-1200} --property=TimeoutStopSec=20 \
        --setenv="LD_LIBRARY_PATH=$driver_path:$PWD/driver-26.40/usr/local/lib" \
        --setenv="LIBRARY_PATH=$driver_path" \
        --setenv="CPATH=$PWD/driver-26.40/usr/include" \
        --setenv="OCL_ICD_VENDORS=$PWD/opencl-vendors" \
        --setenv=NEOReadDebugKeys=1 --setenv=ForcePreemptionMode=3 \
        --setenv=VLLM_XPU_INC_WNA16_BACKEND=w4a16 \
        --setenv=HF_HUB_OFFLINE=1 --setenv=VLLM_NO_USAGE_STATS=1 \
        --setenv=VLLM_LOGGING_LEVEL=INFO ${OFFLINE_ENV:+$(printf -- "--setenv=%s " $OFFLINE_ENV)} -- "${cmd[@]}"
    for (( n=0; n<610; n++ )); do
        systemctl is-active --quiet "$candidate" || break
        sleep 2
    done
    journalctl -u "$candidate" --since "$(stat -c %y "$run/argv.txt")" --no-pager -o cat > "$run/server.log"
    [[ -s $run/offline.done ]] && rc=0 || rc=34
    printf 'Offline exit=%s; log: %s/server.log\n' "$rc" "$run"
    exit "$rc"
fi
sudo -n systemd-run --quiet --collect --unit="$candidate" --uid="$USER" --working-directory="$PWD" \
    --property=Type=exec --property=RuntimeMaxSec=${SERVE_MAXSEC:-500} --property=TimeoutStopSec=20 \
    --setenv="LD_LIBRARY_PATH=$driver_path:$PWD/driver-26.40/usr/local/lib" \
    --setenv="LIBRARY_PATH=$driver_path" \
    --setenv="CPATH=$PWD/driver-26.40/usr/include" \
    --setenv="OCL_ICD_VENDORS=$PWD/opencl-vendors" \
    --setenv=NEOReadDebugKeys=1 --setenv=ForcePreemptionMode=3 \
    --setenv=VLLM_XPU_INC_WNA16_BACKEND=w4a16 \
    --setenv=HF_HUB_OFFLINE=1 --setenv=VLLM_NO_USAGE_STATS=1 \
    --setenv=VLLM_LOGGING_LEVEL=INFO ${SERVE_ENV:+$(printf -- "--setenv=%s " $SERVE_ENV)} -- "${cmd[@]}"
ready=0
for (( n=0; n<${READY_WAIT:-120}; n++ )); do
    if curl -fsS --max-time 1 http://127.0.0.1:18080/health >/dev/null 2>&1; then
        ready=1
        break
    fi
    systemctl is-active --quiet "$candidate" || break
    sleep 1
 done
rc=33
if (( ready )); then
    rc=0
    timeout 120s python3 -I http-smoke.py > "$run/http.jsonl" 2> "$run/http.stderr" || rc=$?
    if (( rc == 0 )); then
        if [[ -n ${PROBE_SCRIPT:-} ]]; then
            timeout 400s python3 -I "$PROBE_SCRIPT" "$run" > "$run/probe.stdout" 2> "$run/probe.stderr" || rc=$?
        elif [[ $mode == --probe* ]]; then
            timeout 400s python3 -I probe.py "$run" > "$run/probe.stdout" 2> "$run/probe.stderr" || rc=$?
            timeout 400s python3 -I parity.py "$run" > "$run/parity.stdout" 2> "$run/parity.stderr" || rc=$?
        elif [[ $mode == --profile* ]]; then
            timeout 240s python3 -I flint_profile.py "$run" > "$run/profile.stdout" 2> "$run/profile.stderr" || rc=$?
        elif [[ $mode == --bench* ]]; then
            timeout 240s python3 -I dev-bench.py "$run" "$mode" > "$run/bench.jsonl" 2> "$run/bench.stderr" || rc=$?
            timeout 400s python3 -I parity.py "$run" > "$run/parity.stdout" 2> "$run/parity.stderr" || rc=$?
        elif [[ $mode != --run ]]; then
            timeout 240s python3 -I dev-bench.py "$run" "$mode" > "$run/bench.jsonl" 2> "$run/bench.stderr" || rc=$?
        fi
    fi
fi
journalctl -u "$candidate" --since "$(stat -c %y "$run/argv.txt")" --no-pager -o cat > "$run/server.log"
printf 'Ready=%s HTTP smoke exit=%s; log: %s/server.log\n' "$ready" "$rc" "$run"
exit "$rc"
