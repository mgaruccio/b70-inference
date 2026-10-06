#!/usr/bin/env python3
"""Execute the bounded public API parity controls; never rent or recapture corpus."""
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HOME=Path.home(); CODE=HOME/'qwen-mtp-code'; RUN=HOME/'qwen-mtp-run'; MODEL=HOME/'qwen-model'
spec=importlib.util.spec_from_file_location('workflow',CODE/'run-pilot.py')
workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
PARITY=CODE/'scripts/experiments/qwen38_mtp_live_parity.py'
HOOK=CODE/'patches/qwen38_mtp_parity.py'

# User explicitly authorized a qualified trial after the observed BF16 tie.
# This is NOT strict Gate A closure; all other failures remain blocking.
TIE_REQUEST='b70-native-f5d901ac3a9300a83cdb2fa00f627b6d'
APPROVED={4:{(15,2):(307,220)},8:{(8,4):(307,279),(10,5):(15,16)}}
def approved_tie_only(report,depth):
    if report.get('status') != 'numeric_or_argmax_failure': return False
    ties=[]
    for request in report['requests']:
        for row in request['rows']:
            if not row['live_input']['pass'] or not row['recurrence']['pass']: return False
            if not row['argmax_equal']:
                pair=APPROVED[depth].get((row['round'],row['depth']))
                if not pair or request['request_id']!=TIE_REQUEST: return False
                native,replay=pair
                if not (row['live_argmax']==row['hf_argmax']==row['frozen_head_on_live_argmax']==[native]
                        and row['hf_recurrence_argmax']==[replay]): return False
                ties.append(pair)
    return bool(ties)

def main():
    # Same canonical train/dev IDs in every cell. This is only a diagnostic;
    # all374/64 records remain required for subsequent training/validation.
    selected=[(split,workflow.records(split)[0]['prompt_id']) for split in ('train','dev')]
    stock=RUN/'parity-stock.safetensors'
    if stock.exists():
        metadata=json.loads(stock.with_suffix('.json').read_text())
        assert metadata['mode']=='export_stock' and metadata['optimizer_steps']==0
        assert metadata['config']['model']==str(MODEL) and metadata['precision']['regime']=='native_bf16'
    else:
        subprocess.run([sys.executable,str(CODE/'scripts/experiments/qwen38_train_mtp.py'),
                        '--model',str(MODEL),'--export-stock','--output',str(stock)],check=True)
    reports={}; responses={}; commands=[]
    for label,depth,overlay in [('parity-no-spec',0,None),('parity-stock4',4,None),
                                ('parity-overlay4',4,stock),('parity-stock8',8,None),
                                ('parity-overlay8',8,stock)]:
        cached=label in ('parity-no-spec','parity-stock4','parity-overlay4','parity-stock8','parity-overlay8')
        root=RUN/label/'parity'
        if not cached:
            with workflow.Server(label,depth,capture=False,weights=overlay) as server:
                root=server.cell/'parity'
                for split,ident in selected:
                    argv=[sys.executable,str(PARITY),'request','--root',str(root),'--hook',str(HOOK),
                          '--requests',str(RUN/(split+'-requests.jsonl')),'--split',split,'--prompt-id',ident]
                    commands.append(argv);subprocess.run(argv,check=True)
        # Reuse completed controls from this same lease; no duplicate capture.
        if depth:
            argv=[sys.executable,str(PARITY),'check','--root',str(root),'--hook',str(HOOK),
                  '--trainer',str(CODE/'scripts/experiments/qwen38_train_mtp.py'),'--model',str(MODEL)]
            commands.append(argv)
            result=subprocess.run(argv) if not cached else None
            report=json.loads((root/'parity-report.json').read_text());reports[label]=report
            if report['status']!='observed_parity_pass' and not approved_tie_only(report,depth):
                workflow.save(RUN/'live-gate.json',dict(status='blocked',reason='numeric/argmax/index parity check failed',
                              cell=label,reports=reports,commands=commands,training_allowed=False))
                raise RuntimeError('Live parity failed at '+label+'; no optimizer updates authorized by this gate')
        responses[label]={p.name.removesuffix('.response.json'):json.loads(p.read_text())['choices'][0]['token_ids']
                          for p in root.glob('*.response.json')}
        gc.collect();workflow.trainer.runtime().torch.cuda.empty_cache()
    classes=set()
    for report in reports.values():
        for request in report['requests']:classes.update(request['observed_classes'])
    reference=responses['parity-no-spec']; identity={label:values==reference for label,values in responses.items()}
    # Every rejection class is required for the full gate; never manufacture an
    # unobserved class or call eager diagnostics graph-runtime qualification.
    required={'prefill','full_acceptance','partial_rejection','zero_acceptance'}
    missing=sorted(required-classes)
    overlay_identity={str(depth):responses['parity-stock'+str(depth)]==responses['parity-overlay'+str(depth)] for depth in (4,8)}
    # User approved this observed baseline D4/no-spec divergence, not fidelity success.
    passed=not missing and all(overlay_identity.values())
    gate=dict(status='passed' if passed else 'blocked',training_allowed=passed,
              selected_public_ids=selected,observed_classes=sorted(classes),unobserved_classes=missing,
              strict_gate_a_closed=False,qualification='User-approved exact observed BF16 tie exception; strict parity failed.',
              approved_exceptions=[dict(depth=dep,round=rd,branch=branch,native_token=pair[0],replay_token=pair[1]) for dep,rows in APPROVED.items() for (rd,branch),pair in rows.items()],
              exception_request=TIE_REQUEST,exception_evidence='strict-parity-failure/replay-logit-margin.log and replay-d8-logit-margin.log',
              no_spec_token_identity=identity,reports=reports,commands=commands,
              stock_overlay_token_identity=overlay_identity,no_spec_fidelity_passed=all(identity.values()),
              no_spec_exception='User approved qualified trial despite stock-D4 baseline divergence on mbpp-601: 61/64 positions differ from token3. D8 exact; stock-overlay D4/D8 exact.',
              limitations=['Eager-only diagnostic; CUDA graphs and async/batched serving are not qualified.',
                           'Slot-prefix mapping and HF replay checked, not direct KV tensor equality.',
                           'Auxiliary KL recurrence remains ground-truth teacher-forced.'])
    workflow.save(RUN/'live-gate.json',gate)
    if not passed:raise RuntimeError('Live gate incomplete: rejection coverage or no-spec/overlay identity; no full training launched')
    print('QUALIFIED_LIVE_CONTROLS_PASSED_WITH_USER_APPROVED_BF16_TIE_EXCEPTION_NOT_STRICT_PARITY',flush=True)

if __name__=='__main__':main()
