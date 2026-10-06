#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic
python3 - "$ROOT" <<'PY'
import json,sys,os
from pathlib import Path
root=Path(sys.argv[1]); p=json.loads((root/'pilot-protocol.json').read_text())
assert not (root/'provider-instance.json').exists() and not (root/'provider-create-request.json').exists(), 'One create only; inspect existing state'
r=json.loads((root/'preflight-ready.json').read_text()); assert r['ready'] and r['optimizer_updates']==0
assert r['approved_hours']==2 and r['approved_max_cost_usd']==6.6
assert (root/'archived-teacher-inputs.json').is_file()
assert (root/'backup-private.json').stat().st_mode & 0o777 == 0o600
print('ONE_H100_DIAGNOSTIC_AUTHORIZED_2H_6.60_ZERO_UPDATES',flush=True)
PY
python3 "$ROOT/provider.py"
bash "$ROOT/launch-bootstrap.sh"
