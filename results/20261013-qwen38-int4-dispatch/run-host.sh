#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REMOTE=/home/mike/b70-evals/20261013-qwen38-int4-dispatch
SSH=(-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no)
timeout -k 2s 40s scp "${SSH[@]}" "$HERE/run-diagnostic.py" "$HERE/analyze-shapes.py" "mike@inference-host:$REMOTE/"
# nohup protects host-side restoration if an intermittent SSH transport drops.
timeout -k 10s 3000s ssh -T "${SSH[@]}" mike@inference-host 'bash -s' <<'HOST'
set -uo pipefail
OUT=/home/mike/b70-evals/20261013-qwen38-int4-dispatch
nohup python3 -Bu "$OUT/run-diagnostic.py" --output "$OUT/native-01" > "$OUT/controller.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/controller.pid"
wait "$PID"
RESULT=$?
cat "$OUT/controller.log"
if test "$RESULT" = 0; then
  python3 -B "$OUT/analyze-shapes.py" "$OUT/native-01" > "$OUT/shapes.json"
  RESULT=$?
  if test "$RESULT" = 0; then
    python3 - "$OUT/shapes.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(json.dumps({k:r[k] for k in ('chunks','native_operator_calls','native_ops_missing_shapes','total_device_kernel_count','unmapped_generic_gemm_kernels','native_kernels_missing_shapes','onednn_verbose_lines','onednn_dispatch_lines')}))
print('Top exact shapes:',[(s['M'],s['K'],s['N'],s['device_ms']) for s in r['shapes'][:8]])
PY
  fi
fi
exit "$RESULT"
HOST
