#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/mtp-env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd "$HOME/mtp-training-code"
RUN="$HOME/mtp-training-run"
P=scripts/experiments/glimmer_recursive_mtp.py
INIT="$RUN/recursive-depth2-continue20k/checkpoint-last.pt"
OUT="$RUN/recursive-depth4-20k"
test -f "$INIT"; test -f "$RUN/recursive-depth2-continue20k/train-manifest.json"; test ! -e "$OUT"
COMMON=(--capture "$RUN/capture/index.json" --init-head "$INIT" --variants shared-state-norm --rank 128 --train-depth 4 --schedule-updates 20000 --warmup-updates 200 --batch-size 64 --lr 3e-4 --ce-weight 0.25 --kl-weight 1.0 --state-weight 0.2 --seed 20261002)
timeout 600 python "$P" train "${COMMON[@]}" --output-dir "$RUN/continuation-smoke" --updates 1 --checkpoint-every 1 > "$RUN/continuation-smoke.log" 2>&1
python - "$RUN" "$P" <<'PY'
import importlib.util,json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('mtp',sys.argv[2]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.load_head_checkpoint(root/'recursive-depth2-continue20k/checkpoint-last.pt')
b=m.load_head_checkpoint(root/'continuation-smoke/checkpoint-last.pt')
assert a['update']==20000 and a['max_depth']==2
assert b['update']==1 and b['max_depth']==4 and b['root_exposures']==64 and b['loss_position_exposures']==256
assert b['state_weight']==b['state_norm_weight']==.2
assert all(float(v['step'])==1 for v in b['optimizer']['state'].values())
assert a['head'].keys()==b['head'].keys() and any(not torch.equal(a['head'][k],b['head'][k]) for k in a['head'])
assert sum(v.numel() for v in a['head'].values())==sum(v.numel() for v in b['head'].values())
report={'passed':True,'source_trained_depth':2,'source_update':20000,'new_stage_actual_updates':1,'trained_depth':4,'checkpoint_reloaded':True,'fresh_optimizer_step':1,'weights_changed':True,'same_parameter_count':True}
(root/'continuation-smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
PY
echo 'Starting 20000 depth4-unrolled updates from the completed depth2 head; fresh optimizer/schedule, no approval pause.'
python "$P" train "${COMMON[@]}" --output-dir "$OUT" --updates 20000 --checkpoint-every 1000 --validation-every 2000 --validation-roots 1024 --validation-batch-size 8 --validation-seed 314159 --validation-prompts "$HOME/mtp-training-code/scripts/experiments/glimmer_mtp_validation.jsonl" --probe-every 0 --record-divergence > "$RUN/continuation-training.log" 2>&1
for phase in initial final; do
 head="$INIT"; if [[ $phase == final ]]; then head="$OUT/checkpoint-last.pt"; fi
 for depth in 1 2 4; do
  mkdir -p "$RUN/eval-$phase/depth-$depth"
  timeout 600 python "$P" validate-head --capture "$RUN/capture/index.json" --head "$head" --probe-depth "$depth" --validation-roots 1024 --batch-size 8 --max-new-tokens 64 --record-divergence --output "$RUN/eval-$phase/depth-$depth/validation.json" > "$RUN/$phase-depth-$depth.log" 2>&1
 done
done
python - "$RUN" <<'PY'
import json,statistics,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);out=root/'recursive-depth4-20k';s=json.loads((out/'shared-state-norm.json').read_text())
c=torch.load(out/'checkpoint-last.pt',map_location='cpu',weights_only=True)
assert s['updates_completed']==c['update']==20000 and s['max_depth']==c['max_depth']==4
assert c['root_exposures']==1280000 and c['loss_position_exposures']==5120000
assert all(float(v['step'])==20000 for v in c['optimizer']['state'].values())
rows=[]
for phase in ('initial','final'):
 for depth in (1,2,4):
  v=json.loads((root/f'eval-{phase}'/f'depth-{depth}'/'validation.json').read_text());p=v['probe'];exact=[r for r in p['pairs'] if r['exact_token_identity']]
  assert v['trained_depth']==(2 if phase=='initial' else 4) and p['depth']==depth and len(p['pairs'])==12
  rows.append({'phase':phase,'depth':depth,'offline':v['offline']['per_depth'],'acceptance':p['draft_acceptance_rate'],'mean_accepted_drafts_per_pass':p['mean_accepted_drafts_per_pass'],'accepted_drafts':p['accepted_drafts'],'proposed_drafts':p['proposed_drafts'],'verification_passes':p['verification_passes'],'fidelity':p['fidelity'],'diagnostic_median_decode_ratio_exact_pairs_only':statistics.median(r['candidate']['decode_tokens_per_s']/r['baseline']['decode_tokens_per_s'] for r in exact) if exact else None})
report={'tier':'development','source_stage_update':20000,'prior_depth2_updates_in_lineage':22000,'additional_actual_depth4_updates':20000,'trained_depth':4,'root_exposures_this_stage':1280000,'loss_position_exposures_this_stage':5120000,'initialization':'completed depth2 continuation checkpoint; fresh optimizer/schedule and restarted deterministic sampler, not exact optimizer resume','evaluation':'initial depth2 and final depth4 heads on same physical GPU; fixed12 validation prompts,64 new tokens; final update used; no production promotion','evaluations':rows}
(out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,allow_nan=False),flush=True)
PY
