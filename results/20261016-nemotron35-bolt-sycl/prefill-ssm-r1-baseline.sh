#!/usr/bin/env bash
# Run on inference-host only; unchanged resident staged-SSM winner.
set -euo pipefail
ROOT=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
OUT="$ROOT/prefill-ssm-r1-baseline"
test "$(hostname)" = inference-host
test "$(docker inspect -f '{{index .Config.Labels "b70.experiment"}}' nemotron-bolt-probe)" = bolt-fusion
test "$(docker inspect -f '{{.State.Running}}' nemotron-bolt-probe)" = true
test ! -e "$OUT"
mkdir "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
trap 'docker logs nemotron-bolt-probe > "$OUT/server.log" 2>&1' EXIT
docker inspect nemotron-bolt-probe > "$OUT/container-inspect.json"
python3 -B - "$ROOT" "$OUT" <<'PY'
import json, pathlib, runpy, sys
root, out = map(pathlib.Path, sys.argv[1:])
runner = runpy.run_path(str(root / 'run-bolt-cell-ssm-prefill-r1.py'))
state = runner['hardware']()
(out / 'hardware-before.json').write_text(json.dumps(state, indent=2) + '\n')
inspect = json.loads((out / 'container-inspect.json').read_text())[0]
assert next(m['Source'] for m in inspect['Mounts'] if m['Destination'] == '/src') == str(runner['PREFILL_SSM'])
assert 'GGML_SYCL_DEBUG=1' not in inspect['Config']['Env']
(out / 'probe-api.py').write_bytes((root / 'probe-api.py').read_bytes())
assert "('json-lookup'," in (out / 'probe-api.py').read_text()
PY
for mode in quality bench long; do
  timeout --signal=TERM --kill-after=30s 2400s python3 -B "$OUT/probe-api.py" --mode "$mode" --out "$OUT/$mode"
done
python3 -B - "$ROOT" "$OUT" <<'PY'
import json, pathlib, runpy, sys
root, out = map(pathlib.Path, sys.argv[1:])
state = runpy.run_path(str(root / 'run-bolt-cell-ssm-prefill-r1.py'))['hardware']()
(out / 'hardware-after.json').write_text(json.dumps(state, indent=2) + '\n')
assert state == json.loads((out / 'hardware-before.json').read_text())
print('PREFILL_BASELINE_PASSED')
PY
