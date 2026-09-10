#!/usr/bin/env bash
# Explicit continuation after the baseline/prefill-only cells; no silent retries.
set -euo pipefail
R=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$R/run-cells.sh" g8-8192-64k g8-2048-64k g8-4096-64k
capacity_attempt() {
  local cell=$1 code
  set +e
  bash "$R/run-cells.sh" "$cell"
  code=$?
  set -e
  if [ "$code" -eq 0 ]; then return 0; fi
  # A cleanly rejected startup is retained, not a successful capacity result.
  # Any runtime/functional/cleanup failure stops the campaign for diagnosis.
  python3 - "$R/$cell" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
s = json.loads((root / 'summary.json').read_text())
log = (root / 'server.log').read_text()
assert s['status'] == 'failed' and 'during startup' in s.get('error', '')
assert 'To serve at least one request' in log and 'larger than the available KV cache' in log
assert s['launcher_unchanged'] and s['power_unchanged'] and s['glimmer_running_after'] == 'false'
assert (root.parent / (root.name + '.cleanup.txt')).is_file()
print('RETAINED_STARTUP_CAPACITY_FAILURE=' + root.name, flush=True)
PY
}
capacity_attempt g8-2048-208k
capacity_attempt g8-2048-256k
# The measured prefill-only gain was 0.54 GiB, less than the initial projection.
# If 208K could not start, explicitly measure a separate 180224-token limit;
# never label that result as a successful 212992/262144-token request.
if python3 - "$R/g8-2048-208k/summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['status'] == 'failed' else 1)
PY
then
  bash "$R/run-cells.sh" g8-2048-176k
fi
printf 'CANDIDATE_CAMPAIGN_COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)"
