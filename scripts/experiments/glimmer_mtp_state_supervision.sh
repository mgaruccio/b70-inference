#!/usr/bin/env bash
# Development ablation: matched one-step control/state-loss arms, not recursive success.
set -euo pipefail
export PATH="$HOME/mtp-env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd "$HOME/mtp-training-code"
RUN="$HOME/mtp-training-run"
P=scripts/experiments/glimmer_recursive_mtp.py
INIT="$RUN/stage1/checkpoint-best.pt"
OUT="$RUN/state-supervision"
test -f "$INIT"
test -f "$RUN/stage1/train-manifest.json"
test ! -e "$OUT"
python - "$RUN/state-preflight.json" <<'PY'
import json, sys
with open(sys.argv[1]) as preflight:
    assert json.load(preflight)['passed'], 'Actual-batch preflight did not pass'
PY
# Exercise actual CLI warm-start, optimizer update, and checkpoint reload for both arms first.
timeout 600 python "$P" train \
  --capture "$RUN/capture/index.json" --output-dir "$RUN/state-smoke" \
  --init-head "$INIT" --variants shared-ce shared-state --rank 128 \
  --train-depth 1 --updates 1 --schedule-updates 10000 --warmup-updates 500 \
  --batch-size 64 --lr 3e-4 --ce-weight 0.25 --kl-weight 1.0 --state-weight 0.2 \
  --seed 20261002 --checkpoint-every 1 \
  > "$RUN/state-smoke.log" 2>&1
python - "$RUN" "$P" <<'PY'
import importlib.util, json, sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('mtp', sys.argv[2]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
initial = m.load_head_checkpoint(root / 'stage1/checkpoint-best.pt')
control = m.load_head_checkpoint(root / 'state-smoke/shared-ce-last.pt')
state = m.load_head_checkpoint(root / 'state-smoke/shared-state-last.pt')
for checkpoint, weight in ((control, 0.), (state, .2)):
    assert checkpoint['update'] == 1 and checkpoint['root_exposures'] == 64
    assert checkpoint['state_weight'] == weight
    assert all(float(s['step']) == 1 for s in checkpoint['optimizer']['state'].values())
    assert any(not torch.equal(checkpoint['head'][k], initial['head'][k]) for k in initial['head'])
assert control['sampler'].keys() == state['sampler'].keys()
for key, value in control['sampler'].items():
    other = state['sampler'][key]
    assert (torch.equal(value, other) if torch.is_tensor(value) else value == other), \
        'Matched sampler state differs: ' + key
assert control['init_head_identity'] == state['init_head_identity']
assert any(not torch.equal(control['head'][k], state['head'][k]) for k in control['head'])
report = {'passed': True, 'updates_per_arm': 1, 'same_sampler_state': True,
          'weights_changed': True, 'checkpoints_reloaded': True, 'state_weights': [0., .2]}
(root / 'state-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
PY
# Both arms reset optimizer/schedule and see identical seeded batches.
# shared-ce ignores state_weight; shared-state uses 0.2*(normalized MSE + cosine).
timeout 2400 python "$P" train \
  --capture "$RUN/capture/index.json" --output-dir "$OUT" \
  --init-head "$INIT" --variants shared-ce shared-state --rank 128 \
  --train-depth 1 --updates 10000 --schedule-updates 10000 --warmup-updates 500 \
  --batch-size 64 --lr 3e-4 --ce-weight 0.25 --kl-weight 1.0 --state-weight 0.2 \
  --seed 20261002 --checkpoint-every 1000 --validation-every 1000 \
  --validation-roots 1024 --validation-batch-size 8 --validation-seed 314159 \
  --validation-prompts "$HOME/mtp-training-code/scripts/experiments/glimmer_mtp_validation.jsonl" \
  --probe-every 5000 --probe-new-tokens 64 --record-divergence \
  > "$RUN/state-training.log" 2>&1
for arm in shared-ce shared-state; do
  mkdir -p "$OUT/$arm"
  timeout 600 python "$P" validate-head \
    --capture "$RUN/capture/index.json" --head "$OUT/$arm-best.pt" \
    --split validation --validation-roots 1024 --batch-size 8 --seed 314159 \
    --max-new-tokens 64 --record-divergence --output "$OUT/$arm/validation.json" \
    > "$RUN/$arm-validation.log" 2>&1
done
python - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1]); results = {}
for arm in ('shared-ce', 'shared-state'):
    summary = json.loads((root / (arm + '.json')).read_text())
    validation = json.loads((root / arm / 'validation.json').read_text())
    probe = validation['probe']
    exact = [p for p in probe['pairs'] if p['exact_token_identity']]
    assert summary['updates_completed'] == 10000 and summary['root_exposures'] == 640000
    results[arm] = {
        'updates': summary['updates_completed'], 'best_update': summary['best_update'],
        'state_weight': summary['state_weight'], 'offline': validation['offline'],
        'acceptance': probe['draft_acceptance_rate'], 'accepted_drafts': probe['accepted_drafts'],
        'proposed_drafts': probe['proposed_drafts'], 'categories': probe['categories'],
        'fidelity': probe['fidelity'], 'exact_pairs': len(exact),
        'median_decode_speedup_exact_pairs_only': statistics.median(
            p['candidate']['decode_tokens_per_s'] / p['baseline']['decode_tokens_per_s']
            for p in exact) if exact else None,
    }
report = {'tier': 'development', 'scope': 'Matched one-step state-supervision ablation; not recursive success or a lossless serving claim',
          'initialization': 'same previously selected 48k checkpoint; both optimizers reset',
          'selection': 'minimum fixed-validation teacher KL; held-out test remains sealed',
          'arms': results,
          'acceptance_delta': results['shared-state']['acceptance'] - results['shared-ce']['acceptance']}
(root / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, allow_nan=False), flush=True)
PY
