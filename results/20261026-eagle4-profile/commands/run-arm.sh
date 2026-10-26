#!/usr/bin/env bash
# Serial public-boundary validation and unprofiled measurement for one immutable arm.
set -Eeuo pipefail
ROOT=/home/mike/b70-evals/20261026-eagle4-profile
ARM=${1:?unique arm name}
KIND=${2:?eagle or mtp}
REPEATS=${3:-6}
DO_CLI=${4:-0}
OUT="$ROOT/$ARM"
LABEL="profile-$ARM"
HEALTH_PID=
case "$KIND" in
 eagle) NAME="intel15-e4-$LABEL" ;;
 mtp) NAME="intel15-mtp-$LABEL" ;;
 *) exit 2 ;;
esac
mkdir "$OUT/measure"
cleanup() {
 if [[ -n "$HEALTH_PID" ]]; then kill "$HEALTH_PID" 2>/dev/null || true; wait "$HEALTH_PID" 2>/dev/null || true; fi
 if docker container inspect "$NAME" >/dev/null 2>&1; then docker stop -t 30 "$NAME" >/dev/null || true; fi
}
trap cleanup EXIT
if [[ "$KIND" == eagle ]]; then
 bash "$ROOT/launch-$ARM.sh" /home/mike/inference/models/intel15-Qwen3.8-27B \
  /home/mike/inference/models/EAGLE4-Qwen3.8-27B "$OUT" "$LABEL" 0.9 > "$OUT/measure/server.log" 2>&1 &
else
 bash "$ROOT/launch-native-mtp.sh" /home/mike/inference/models/intel15-Qwen3.8-27B \
  "$LABEL" mtp 0.9 > "$OUT/measure/server.log" 2>&1 &
fi
SERVER_PID=$!
curl --silent --show-error --fail --retry 180 --retry-all-errors --retry-delay 5 \
 --retry-max-time 1800 --connect-timeout 3 --max-time 10 http://127.0.0.1:18110/health -o /dev/null &
HEALTH_PID=$!
wait -n -p FINISHED "$SERVER_PID" "$HEALTH_PID"
if [[ "$FINISHED" == "$SERVER_PID" ]]; then exit 1; fi
HEALTH_PID=
python3 "$ROOT/control/eagle-source/tests/intel15_http_probe.py" run \
 --baseurl http://127.0.0.1:18110 --model Qwen3.8-27B --fixture "$ROOT/fixture.json" \
 --out "$OUT/measure/http" --repeats "$REPEATS" --timeout 600 | tee "$OUT/measure/http.log"
python3 /home/mike/b70-evals/20261025-intel15-eagle4/chat-canaries.py "$OUT/measure/semantic" | tee "$OUT/measure/semantic.log"
if [[ "$DO_CLI" == 1 ]]; then
 curl --silent --show-error --fail http://127.0.0.1:18110/metrics > "$OUT/measure/metrics-before.txt"
 docker exec "$NAME" /bin/bash -lc '
set -Eeo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force >/dev/null
mkdir /tmp/profile-bench
vllm bench serve --model Qwen3.8-27B --tokenizer /llm/models/Qwen3.8-27B \
 --served-model-name Qwen3.8-27B --host 127.0.0.1 --port 8000 --backend vllm \
 --request-rate inf --max-concurrency 1 --ignore-eos --trust-remote-code \
 --dataset-name random --num-prompts 10 --num-warmups 1 \
 --random-input-len 32768 --random-output-len 512 --random-range-ratio 0 \
 --temperature 0 --seed 0 --save-result --save-detailed \
 --result-dir /tmp/profile-bench --result-filename result.json --metric-percentiles 50,90,99
' | tee "$OUT/measure/benchmark.log"
 docker cp "$NAME:/tmp/profile-bench/result.json" "$OUT/measure/benchmark.json"
 curl --silent --show-error --fail http://127.0.0.1:18110/metrics > "$OUT/measure/metrics-after.txt"
fi
cleanup
wait "$SERVER_PID"
trap - EXIT
docker ps --format '{{.Names}}'
ss -ltn 'sport = :18110'
df -B1 /home/mike
