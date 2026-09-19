#!/usr/bin/env bash
# Run on inference-host only, after the official evaluator canary passes.
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20260918-mtp-quality-longctx
OLD=/home/mike/b70-evals/20260915-mtp-scale-attempt
IMAGE=vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
EVALIMAGE=$(cat "$ROOT/evalplus-fixed-image.txt")
EXPECTED=63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4
WEIGHTS="$OLD/training-lr1e6_decay/tuned-mtp.safetensors"
SERVER_PID=
CELL=
cleanup() {
 if test -n "$SERVER_PID"; then
  OWNER=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/profile"}}{{.Source}}{{end}}{{end}}' qwen38 2>/dev/null || true)
  if test "$OWNER" = "$CELL"; then docker stop -t 20 qwen38 >/dev/null; fi
  wait "$SERVER_PID" || true
  SERVER_PID=
 fi
 test "$(sha256sum /home/mike/inference/launchers/start-qwen38.sh | cut -d' ' -f1)" = "$EXPECTED"
}
trap cleanup EXIT
MODE=${1:?quality or long}
case "$MODE" in quality) LABELS='A1 B1 B2 A2';; long) LABELS='A B';; long-candidate) MODE=long; LABELS='B';; *) exit 2;; esac
python3 - "$ROOT/eval-canary-fixed.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['returncode']==0
tasks=r['results']['eval'];assert len(tasks)==164
for key,rows in tasks.items():
 assert len(rows)==1 and rows[0]['base_status']==rows[0]['plus_status']=='pass',(key,rows)
PY
for LABEL in $LABELS; do
 test -z "$(docker ps -q)"
 test "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000
 test "$(sha256sum /home/mike/inference/launchers/start-qwen38.sh | cut -d' ' -f1)" = "$EXPECTED"
 test "$(sha256sum "$WEIGHTS" | cut -d' ' -f1)" = 1af9142095c1d387847c83f27bdc330df8d34f81d0d73683c593ef5768babde5
 CELL="$ROOT/$MODE-$LABEL"
 if test "$MODE" = long; then CELL="$ROOT/long-verified-$LABEL"; fi
 EXTRA=()
 case "$LABEL" in B*) EXTRA=(--weights "$WEIGHTS");; esac
 docker run --rm --network none --user 1000:1000 -v "$ROOT:$ROOT" -v "$OLD:$OLD:ro" \
  -v /home/mike/inference/launchers:/home/mike/inference/launchers:ro \
  -v /home/mike/inference/src/intel-arc-pro-b70-inference-cookbook/patches:/cookbook:ro \
  -v "$ROOT/commands/prefill-guard.py:/guard.py:ro" --entrypoint python "$IMAGE" \
  "$ROOT/source/scripts/experiments/qwen38_mtp_native_corpus.py" prepare \
  --launcher /home/mike/inference/launchers/start-qwen38.sh --guard /guard.py --reference-patches /cookbook \
  --no-capture --max-total-tokens 2097152 --max-requests 1024 --output "$CELL" "${EXTRA[@]}"
 if test "$MODE" = long; then
  python3 - "$CELL" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);launcher=p/'launcher.sh';text=launcher.read_text()
assert text.count('--enable-prefix-caching')==1
text=text.replace('--enable-prefix-caching','--no-enable-prefix-caching')
launcher.write_text(text)
c=json.loads((p/'launch-config.json').read_text());c['prefix_caching']=False
c['temporary_launcher_sha256']=hashlib.sha256(text.encode()).hexdigest()
c['intentional_differences'].append('disable prefix cache for cold long-context sweep; renderer disables thinking')
(p/'launch-config.json').write_text(json.dumps(c,indent=2)+'\n')
PY
 fi
 bash -n "$CELL/launcher.sh"
 bash "$CELL/launcher.sh" > "$CELL/server.log" 2>&1 &
 SERVER_PID=$!
 python3 - "$SERVER_PID" "$MODE-$LABEL" <<'PY'
import json,os,sys,time,urllib.request
pid=int(sys.argv[1]);deadline=time.monotonic()+900
while time.monotonic()<deadline:
 try:os.kill(pid,0)
 except ProcessLookupError:raise SystemExit('Server exited; see server.log')
 try:
  with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2) as r:
   if r.status==200:break
 except OSError:pass
 time.sleep(3)
else:raise SystemExit('Readiness exceeded 900 seconds')
with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=10) as r:
 assert any(x['id']=='qwen38' and x['max_model_len']==212992 for x in json.load(r)['data'])
print('SERVER_READY '+sys.argv[2],flush=True)
PY
 if test "$MODE" = quality; then
  docker run --rm --network host --user 1000:1000 -v "$ROOT:$ROOT" --entrypoint python "$IMAGE" \
   "$ROOT/source/scripts/experiments/qwen38_mtp_native_corpus.py" generate \
   --server-config "$CELL/launch-config.json" --capture-off --synthetic --max-tokens 128 --measure \
   --max-total-tokens 2097152 --output "$ROOT/quality-$LABEL-warmup" > "$ROOT/quality-$LABEL-warmup.log" 2>&1
  docker run --rm --network host --user 1000:1000 -v "$ROOT:$ROOT" --entrypoint python "$IMAGE" \
   "$ROOT/source/scripts/experiments/qwen38_mtp_native_corpus.py" generate \
   --server-config "$CELL/launch-config.json" --capture-off --measure \
   --records "$ROOT/data/public-tasks.jsonl" --sequence-limit 8192 --min-response-tokens 2048 \
   --max-tokens 2048 --max-total-tokens 2097152 --max-requests 1024 \
   --output "$ROOT/quality-$LABEL-output" > "$ROOT/quality-$LABEL.log" 2>&1
  cat "$ROOT/quality-$LABEL-output/summary.json"
  cleanup
  docker run --rm --network none --user 1000:1000 -e HOME=/tmp \
   -e HUMANEVAL_OVERRIDE_PATH="$ROOT/data/HumanEvalPlus.jsonl" \
   -v "$ROOT:$ROOT" --entrypoint python "$EVALIMAGE" "$ROOT/commands/sanitize.py" "$ROOT" "$LABEL"
  bash "$ROOT/commands/evaluate.sh" "$ROOT/quality-$LABEL-samples.jsonl" "$ROOT/quality-$LABEL-evaluation.json"
 else
  python3 - "$CELL" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);c=json.loads((p/'launch-config.json').read_text());t=(p/'launcher.sh').read_text()
assert c['prefix_caching'] is False
assert hashlib.sha256(t.encode()).hexdigest()==c['temporary_launcher_sha256']
assert '--no-enable-prefix-caching' in t and '--enable-prefix-caching' not in t
assert 'enable_prefix_caching=False' in (p/'server.log').read_text()
print('CACHE_DISABLED_VERIFIED launcher hash and live runtime initialization log',flush=True)
PY
  python3 "$ROOT/source/scripts/experiments/qwen38_long_context_bench.py" \
   --out "$ROOT/long-verified-$LABEL-output" --lengths 512 8192 16384 32768 65536 120000 160000 \
   --near-limit 212000 --confirm-prefix-cache-disabled > "$ROOT/long-verified-$LABEL.log" 2>&1
  cleanup
 fi
 echo "CELL_COMPLETE=$MODE-$LABEL"
done
sha256sum /home/mike/inference/launchers/start-qwen38.sh "$WEIGHTS"
docker ps --format '{{.Names}}'
df -h /home/mike
echo "ROUND_COMPLETE=$MODE"
