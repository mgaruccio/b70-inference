#!/bin/bash
set -euo pipefail
R=/home/mike/b70-evals/muse-glimmer/concurrency-sweep-20260906T002815Z
C=${1:?concurrency required}
[[ "$C" =~ ^[0-9]+$ ]] || exit 2
D="$R/c$C";NAME="glimmer-csweep-20260906-c$C"
if docker container inspect "$NAME" >/dev/null 2>&1; then echo name-already-exists; exit 1; fi
cleanup() { docker logs "$NAME" > "$D/server.log" 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
NAME="$NAME" bash "$D/start.sh"
ready=0
for ((attempt=0; attempt<180; attempt++)); do
  if curl -sf -m 3 http://127.0.0.1:18080/v1/models >/dev/null; then ready=1; break; fi
  if [ "$(docker inspect --format '{{.State.Running}}' "$NAME")" != true ]; then
    echo "SERVER_EXITED_BEFORE_READY $NAME"; docker logs "$NAME" 2>&1 | tail -60; exit 1
  fi
  sleep 5
done
[ "$ready" -eq 1 ] || { echo HEALTH_TIMEOUT; exit 1; }
python "$R/measure.py" "$C" "$D"
quality_rc=0
python /home/mike/b70-evals/muse-glimmer/20260905-concurrency/quality.py "$C" "$D/quality.json" > "$D/quality.log" 2>&1 || quality_rc=$?
printf 'C%s QUALITY_EXIT=%s\n' "$C" "$quality_rc"
python - "$D/quality.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]));r=j['rows']
print('QUALITY_SUMMARY',json.dumps({'concurrency':j['concurrency'],'passed':sum(x['score']['quoteable'] for x in r),'total':len(r),'completed':sum(x['score']['finish_ok'] and x['score']['content_ok'] for x in r)}),flush=True)
assert all(x['score']['finish_ok'] and x['score']['content_ok'] for x in r), 'Incomplete/empty completed-answer request'
PY
