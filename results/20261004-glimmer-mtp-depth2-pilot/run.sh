#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/mtp-env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd "$HOME/mtp-training-code"
RUN="$HOME/mtp-training-run"
P=scripts/experiments/glimmer_recursive_mtp.py
INIT="$RUN/amplitude-pilot/shared-state-norm-best.pt"
OUT="$RUN/recursive-depth2"
test -f "$INIT"; test -f "$RUN/amplitude-pilot/train-manifest.json"; test ! -e "$OUT"
COMMON=(--capture "$RUN/capture/index.json" --init-head "$INIT" --variants shared-state-norm --rank 128 --schedule-updates 2000 --warmup-updates 100 --batch-size 64 --lr 3e-4 --ce-weight 0.25 --kl-weight 1.0 --state-weight 0.2 --seed 20261002)
# Public-CLI actual optimizer updates, CUDA targets frozen, save/reload before full run.
for spec in control-one-step:1 recurrent-two-step:2; do
 arm=${spec%:*}; depth=${spec#*:}
 timeout 600 python "$P" train "${COMMON[@]}" --train-depth "$depth" --output-dir "$RUN/depth2-smoke/$arm" --updates 1 --checkpoint-every 1 > "$RUN/$arm-smoke.log" 2>&1
done
python - "$RUN" "$P" <<'PY'
import importlib.util,json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('mtp',sys.argv[2]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
initial=m.load_head_checkpoint(root/'amplitude-pilot/shared-state-norm-best.pt')
a=m.load_head_checkpoint(root/'depth2-smoke/control-one-step/checkpoint-last.pt')
b=m.load_head_checkpoint(root/'depth2-smoke/recurrent-two-step/checkpoint-last.pt')
for c,d in [(a,1),(b,2)]:
 assert c['update']==1 and c['root_exposures']==64 and c['max_depth']==d
 assert c['state_weight']==.2 and c['state_norm_weight']==.2
 assert all(float(v['step'])==1 for v in c['optimizer']['state'].values())
 assert any(not torch.equal(c['head'][k],initial['head'][k]) for k in initial['head'])
assert a['init_head_identity']==b['init_head_identity'] and a['sampler'].keys()==b['sampler'].keys()
for k,v in a['sampler'].items():assert (torch.equal(v,b['sampler'][k]) if torch.is_tensor(v) else v==b['sampler'][k]),k
assert sum(x.numel() for x in a['head'].values())==sum(x.numel() for x in b['head'].values())
report={'passed':True,'actual_updates_per_arm':1,'trained_depths':[1,2],'same_initialization_and_sampler':True,'same_parameter_count':True,'checkpoints_reloaded':True}
(root/'depth2-smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
PY
# No pauses or approvals between smoke gate and the requested real training.
for spec in control-one-step:1 recurrent-two-step:2; do
 arm=${spec%:*}; depth=${spec#*:}
 echo "Starting actual training arm=$arm recursive_loss_depth=$depth updates=2000"
 timeout 1200 python "$P" train "${COMMON[@]}" --train-depth "$depth" --output-dir "$OUT/$arm" --updates 2000 --checkpoint-every 500 --validation-every 500 --validation-roots 1024 --validation-batch-size 8 --validation-seed 314159 --validation-prompts "$HOME/mtp-training-code/scripts/experiments/glimmer_mtp_validation.jsonl" --probe-every 0 --record-divergence > "$RUN/$arm-training.log" 2>&1
done
# Compare fixed final-update checkpoints, not arm-specific best-selection objectives.
for arm in control-one-step recurrent-two-step; do
 for depth in 1 2 4; do
  mkdir -p "$OUT/$arm/depth-$depth"
  timeout 600 python "$P" validate-head --capture "$RUN/capture/index.json" --head "$OUT/$arm/checkpoint-last.pt" --probe-depth "$depth" --validation-roots 1024 --batch-size 8 --max-new-tokens 64 --record-divergence --output "$OUT/$arm/depth-$depth/validation.json" > "$RUN/$arm-depth-$depth.log" 2>&1
 done
done
python - "$OUT" <<'PY'
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);arms={}
for arm,trained in [('control-one-step',1),('recurrent-two-step',2)]:
 s=json.loads((root/arm/'shared-state-norm.json').read_text());assert s['updates_completed']==2000 and s['root_exposures']==128000 and s['max_depth']==trained
 rows=[]
 for depth in (1,2,4):
  v=json.loads((root/arm/f'depth-{depth}/validation.json').read_text());p=v['probe'];exact=[r for r in p['pairs'] if r['exact_token_identity']]
  assert v['trained_depth']==trained and p['depth']==depth
  rows.append({'depth':depth,'offline':v['offline']['per_depth'],'accepted_drafts':p['accepted_drafts'],'proposed_drafts':p['proposed_drafts'],'verification_passes':p['verification_passes'],'mean_accepted_drafts_per_pass':p['mean_accepted_drafts_per_pass'],'conditional_acceptance':p['conditional_acceptance'],'acceptance':p['draft_acceptance_rate'],'fidelity':p['fidelity'],'diagnostic_median_decode_ratio_exact_pairs_only':statistics.median(r['candidate']['decode_tokens_per_s']/r['baseline']['decode_tokens_per_s'] for r in exact) if exact else None})
 arms[arm]={'updates':s['updates_completed'],'root_exposures':s['root_exposures'],'loss_position_exposures':s['loss_position_exposures'],'trained_depth':trained,'evaluated_update':2000,'state_weight':s['state_weight'],'state_norm_weight':s['state_norm_weight'],'depths':rows}
report={'tier':'development','scope':'Matched 2k-update control versus genuinely depth2-unrolled shared-block training; no production promotion','initialization':'same prior shared-state-norm 2k best checkpoint; fresh optimizers/schedules','comparison':'same updates and roots, intentionally twice the supervised loss positions at depth2; evaluate final update2000 for both','arms':arms}
(root/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,allow_nan=False),flush=True)
PY
