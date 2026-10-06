#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261006-qwen38-serving-recurrence-validation
python3 - "$ROOT" <<'PY'
import json,time,sys
from pathlib import Path
root=Path(sys.argv[1])
assert not (root/'provider-instance.json').exists(),'Creation already attempted; inspect/reuse exact lease, never duplicate'
p=json.loads((root/'pilot-protocol.json').read_text())
assert p['authorized'] and p['mode']=='correctness_and_profiling_only'
assert p['optimizer_updates']==0 and not p['full_training_authorized'] and not p['test_data_deployed']
assert p['lease_cap_hours']==4 and p['lease_cap_usd']==13.2 and p['hourly_price_cents']==330
assert p['restored_sequences']=={'train':374,'dev':64}
ready=json.loads((root/'preflight-ready.json').read_text())
assert ready['cpu_checks_passed'] and ready['workflow_checks_passed'] and ready['implementation_commit']
assert ready['teacher_parity_workflow_complete'] and ready['optimizer_updates_authorized']==0
assert time.time()-ready['checked_unix']<3600,'Refresh preflight after source changes or stale validation'
for name in ('backup-private.json','restore-private.json'):
 path=root/name
 assert path.stat().st_mode & 0o077==0
 assert time.time()-path.stat().st_mtime<3600,'Signed URL manifest is stale; refresh privately'
assert 'preparation incomplete' not in (root/'validate-recurrence.py').read_text().lower()
print('AUTHORIZED_ONE_H100_MAX4H_OR_USD13_20_CORRECTNESS_ONLY_ZERO_UPDATES',flush=True)
PY
python3 "$ROOT/provider.py" | tee "$ROOT/provision.log"
bash "$ROOT/launch-bootstrap.sh"
