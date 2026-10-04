#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/mtp-env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd "$HOME/mtp-training-code"
RUN="$HOME/mtp-training-run"
P=scripts/experiments/glimmer_recursive_mtp.py
INIT="$RUN/state-supervision/shared-state-best.pt"
OUT="$RUN/amplitude-pilot"
test -f "$INIT"
test -f "$RUN/state-supervision/train-manifest.json"
test ! -e "$OUT"
python - "$RUN/state-preflight.json" <<'PY'
import json,sys
with open(sys.argv[1]) as f: assert json.load(f)['passed']
PY
COMMON=(--capture "$RUN/capture/index.json" --init-head "$INIT" --variants shared-state shared-state-norm --rank 128 --train-depth 1 --schedule-updates 2000 --warmup-updates 100 --batch-size 64 --lr 3e-4 --ce-weight 0.25 --kl-weight 1.0 --state-weight 0.2 --seed 20261002)
# Actual update/save/reload before the pilot. Neither this nor the pilot trains recursive depth.
timeout 600 python "$P" train "${COMMON[@]}" --output-dir "$RUN/amplitude-smoke" --updates 1 --checkpoint-every 1 > "$RUN/state-smoke.log" 2>&1
python - "$RUN" "$P" <<'PY'
import importlib.util,json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('mtp',sys.argv[2]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
initial=m.load_head_checkpoint(root/'state-supervision/shared-state-best.pt')
a=m.load_head_checkpoint(root/'amplitude-smoke/shared-state-last.pt');b=m.load_head_checkpoint(root/'amplitude-smoke/shared-state-norm-last.pt')
for c,norm in [(a,0.),(b,.2)]:
 assert c['update']==1 and c['root_exposures']==64 and c['state_weight']==.2 and c['state_norm_weight']==norm
 assert all(float(s['step'])==1 for s in c['optimizer']['state'].values())
 assert any(not torch.equal(c['head'][k],initial['head'][k]) for k in initial['head'])
assert a['sampler'].keys()==b['sampler'].keys() and a['init_head_identity']==b['init_head_identity']
for k,v in a['sampler'].items():
 assert (torch.equal(v,b['sampler'][k]) if torch.is_tensor(v) else v==b['sampler'][k]),k
report={'passed':True,'updates_per_arm':1,'state_weights':[.2,.2],'state_norm_weights':[0.,.2],'same_sampler_state':True,'checkpoints_reloaded':True}
(root/'state-smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
PY
# Before spending on more updates, probe the existing directional head through depth 4.
mkdir -p "$RUN/prior-recursion"
timeout 900 python "$P" validate-head --capture "$RUN/capture/index.json" --head "$INIT" --probe-depth 4 --validation-roots 1024 --batch-size 8 --max-new-tokens 64 --record-divergence --output "$RUN/prior-recursion/validation.json" > "$RUN/prior-recursion.log" 2>&1
# Short matched pilot; both fresh optimizers/schedules and identical batches.
timeout 1500 python "$P" train "${COMMON[@]}" --output-dir "$OUT" --updates 2000 --checkpoint-every 500 --validation-every 500 --validation-roots 1024 --validation-batch-size 8 --validation-seed 314159 --validation-prompts "$HOME/mtp-training-code/scripts/experiments/glimmer_mtp_validation.jsonl" --probe-every 1000 --probe-new-tokens 64 --record-divergence > "$RUN/state-training.log" 2>&1
for arm in shared-state shared-state-norm; do
 for depth in 1 2 4; do
  mkdir -p "$OUT/$arm/depth-$depth"
  timeout 900 python "$P" validate-head --capture "$RUN/capture/index.json" --head "$OUT/$arm-best.pt" --probe-depth "$depth" --validation-roots 1024 --batch-size 8 --max-new-tokens 64 --record-divergence --output "$OUT/$arm/depth-$depth/validation.json" > "$RUN/$arm-depth-$depth.log" 2>&1
 done
done
python - "$OUT" <<'PY'
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);arms={}
for arm in ('shared-state','shared-state-norm'):
 s=json.loads((root/(arm+'.json')).read_text());assert s['updates_completed']==2000 and s['root_exposures']==128000
 rows=[]
 for depth in (1,2,4):
  v=json.loads((root/arm/f'depth-{depth}/validation.json').read_text());p=v['probe'];exact=[r for r in p['pairs'] if r['exact_token_identity']]
  assert v['trained_depth']==1 and p['depth']==depth
  rows.append({'depth':depth,'offline':v['offline']['per_depth'],'accepted_drafts':p['accepted_drafts'],'proposed_drafts':p['proposed_drafts'],'verification_passes':p['verification_passes'],'mean_accepted_drafts_per_pass':p['mean_accepted_drafts_per_pass'],'conditional_acceptance':p['conditional_acceptance'],'acceptance':p['draft_acceptance_rate'],'fidelity':p['fidelity'],'median_decode_speedup_exact_pairs_only':statistics.median(r['candidate']['decode_tokens_per_s']/r['baseline']['decode_tokens_per_s'] for r in exact) if exact else None})
 arms[arm]={'updates':s['updates_completed'],'best_update':s['best_update'],'state_weight':s['state_weight'],'state_norm_weight':s['state_norm_weight'],'depths':rows}
report={'tier':'development','scope':'2k-update amplitude-loss ablation with recursive deployment probes; all heads TRAINED ONLY AT DEPTH 1','initialization':'same prior shared-state 10k best checkpoint; optimizers reset','arms':arms}
(root/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,allow_nan=False),flush=True)
PY
