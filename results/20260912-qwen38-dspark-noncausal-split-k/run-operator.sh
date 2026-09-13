#!/usr/bin/env bash
# Run from the staged remote campaign; no package installs or persistent changes.
set -euo pipefail
out=${1:?usage: bash run-operator.sh NEW_OUTPUT_DIRECTORY}
test ! -e "$out"
mkdir "$out"
root=$(pwd)
image=vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4
launcher=/home/mike/inference/launchers/start-qwen38.sh
power=/sys/class/drm/card0/device/hwmon/hwmon2/power1_cap
name=b70-dspark-noncausal-operator
snapshot() {
    sha256sum "$launcher"
    cat "$power"
    docker ps --format '{{.Names}}'
    docker inspect -f '{{.State.Running}}' glimmer-tb21-prefix-c8
}
snapshot > "$out/host-before.txt"
test "$(sha256sum "$launcher" | cut -d' ' -f1)" = 63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4
test "$(cat "$power")" = 275000000
test -z "$(docker ps -q)"
test "$(docker inspect -f '{{.State.Running}}' glimmer-tb21-prefix-c8)" = false
if fuser /dev/dri/renderD128 > "$out/device-before.txt" 2>&1; then
    echo 'Render device is occupied' >&2
    exit 1
fi
if docker inspect "$name" >/dev/null 2>&1; then
    echo 'Reserved container name already exists' >&2
    exit 1
fi
sha256sum check-noncausal.py qwen38_noncausal_split_k.py run-operator.sh > "$out/source-sha256.txt"
cleanup() {
    rc=$?
    trap - EXIT
    docker rm -f "$name" > "$out/cleanup.txt" 2>&1 || true
    snapshot > "$out/host-after.txt"
    if ! cmp -s "$out/host-before.txt" "$out/host-after.txt"; then rc=1; fi
    printf '%s\n' "$rc" > "$out/exit-code.txt"
    exit "$rc"
}
trap cleanup EXIT
command=(docker run --rm --name "$name" --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)"
    --ipc=host -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0
    -e PYTHONPATH=/experiment -v "$root:/experiment:ro" -v "$root/$out:/output"
    --entrypoint /opt/venv/bin/python "$image" -P /experiment/check-noncausal.py --out /output)
printf '%q ' "${command[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
"${command[@]}" > "$out/console.log" 2>&1
