#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261006-qwen38-kl-dev-checkpoint-sweep
# A new, named lease only. provider.py refuses duplicate creation and carries
# provider auto-delete at four hours or $13.20; never rerun this create wrapper.
python3 - "$ROOT" <<'PY'
import json,time,sys
from pathlib import Path
root=Path(sys.argv[1])
assert not (root/'provider-instance.json').exists(),'Creation was already attempted; inspect/reuse exact lease, never duplicate'
for name in ['backup-private.json','restore-private.json']:
 path=root/name
 assert path.stat().st_mode & 0o077 == 0,'Private URL file permissions unsafe'
 assert time.time()-path.stat().st_mtime<3600,'Signed URLs are not fresh; refresh before lease, never print them'
p=json.loads((root/'pilot-protocol.json').read_text())
assert p['optimizer_updates']==0 and p['split']=='dev' and p['requests_per_cell']==64
assert sorted(p['checkpoint_order'])==list(range(374,3741,374))
assert p['lease_cap_hours']==4 and p['lease_cap_usd']==13.2
print('AUTHORIZED_ONE_H100_CAP4H_USD13_20_DEV64_D8_TEN_SAVED_EPOCHS_ZERO_TRAINING',flush=True)
PY
python3 "$ROOT/provider.py" | tee "$ROOT/provision.log"
bash "$ROOT/launch-bootstrap.sh"
bash "$ROOT/launch-ablation.sh"
