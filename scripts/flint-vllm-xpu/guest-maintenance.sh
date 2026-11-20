#!/usr/bin/env bash
# One bounded local experiment. Run on the B70 guest, never as root.
set -euo pipefail
umask 077
cd "$HOME/flint-vllm-spike"
exec 9>maintenance.lock
flock -n 9
candidate_pattern='flint-vllm-smoke-*.service'
# Live Flint server: flint-inference.service (vLLM) or its rollback flint-llamacpp.service.
# Whichever is active is stopped for the experiment and restarted on exit.
# NO_RESTORE=1: allow none to be active and leave it stopped on exit.
no_restore=${NO_RESTORE:-0}
live=
for u in flint-inference.service flint-llamacpp.service; do
    systemctl is-active --quiet "$u" && live=$u
done
live_up=0
[[ -n $live ]] && live_up=1
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
if [[ ${1:-} == --check-rollback ]]; then
    systemctl show flint-inference.service flint-llamacpp.service -p Id -p ActiveState -p UnitFileState
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

if (( live_up )); then
sudo -n cat /etc/b70-flint/api-key | python3 -I -c '
import http.client, json, re, sys
key = sys.stdin.readline().strip()
def get(path, auth):
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=3)
    c.request("GET", path, headers={"Authorization": "Bearer " + key} if auth else {})
    r = c.getresponse(); body = r.read(); c.close()
    return r.status, body
st, body = get("/metrics", False)
if st == 200 and b"vllm:num_requests_running" in body:
    busy = sum(float(v) for v in re.findall(rb"^vllm:num_requests_(?:running|waiting)\{[^}]*\} ([0-9.e+]+)$", body, re.M))
    assert busy == 0, "live vLLM busy; aborting maintenance"
else:
    st, body = get("/slots", True)
    assert st == 200
    slots = json.loads(body)
    assert slots and all(s.get("is_processing") is False for s in slots), "live slot busy; aborting maintenance"
'
fi

restore() {
    local rc=$?
    trap - EXIT HUP INT TERM
    set +e
    sudo -n timeout 28s systemctl stop "$candidate" >/dev/null 2>&1
    if (( no_restore )) || [[ -z $live ]]; then
        echo 'NO_RESTORE=1: live Flint server left stopped.'
        exit "$rc"
    fi
    if ! sudo -n systemctl start "$live"; then
        printf 'RESTORE FAILED to start %s\n' "$live" >&2
        exit 91
    fi
    for (( n=0; n<900; n++ )); do
        if curl -fsS --max-time 1 http://127.0.0.1:8080/health > "$run/restored-health.json" 2>/dev/null; then
            systemctl show "$live" -p MainPID -p InvocationID -p ActiveState
            echo "Live $live serving again on :8080."
            exit "$rc"
        fi
        sleep 1
    done
    printf 'RESTORE FAILED: %s not ready\n' "$live" >&2
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
