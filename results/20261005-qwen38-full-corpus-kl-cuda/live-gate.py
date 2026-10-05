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

def main():
    # Same canonical train/dev IDs in every cell. This is only a diagnostic;
    # all374/64 records remain required for subsequent training/validation.
    selected=[(split,workflow.records(split)[0]['prompt_id']) for split in ('train','dev')]
    stock=RUN/'parity-stock.safetensors'
    subprocess.run([sys.executable,str(CODE/'scripts/experiments/qwen38_train_mtp.py'),
                    '--model',str(MODEL),'--export-stock','--output',str(stock)],check=True)
    reports={}; responses={}; commands=[]
    for label,depth,overlay in [('parity-no-spec',0,None),('parity-stock4',4,None),
                                ('parity-overlay4',4,stock),('parity-stock8',8,None),
                                ('parity-overlay8',8,stock)]:
        with workflow.Server(label,depth,capture=False,weights=overlay) as server:
            root=server.cell/'parity'
            for split,ident in selected:
                argv=[sys.executable,str(PARITY),'request','--root',str(root),'--hook',str(HOOK),
                      '--requests',str(RUN/(split+'-requests.jsonl')),'--split',split,'--prompt-id',ident]
                commands.append(argv);subprocess.run(argv,check=True)
        # Native serving is stopped before HF replay loads the MTP model/head.
        if depth:
            argv=[sys.executable,str(PARITY),'check','--root',str(root),'--hook',str(HOOK),
                  '--trainer',str(CODE/'scripts/experiments/qwen38_train_mtp.py'),'--model',str(MODEL)]
            commands.append(argv)
            result=subprocess.run(argv)
            report=json.loads((root/'parity-report.json').read_text());reports[label]=report
            if result.returncode:
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
    passed=not missing and all(identity.values())
    gate=dict(status='passed' if passed else 'blocked',training_allowed=passed,
              selected_public_ids=selected,observed_classes=sorted(classes),unobserved_classes=missing,
              no_spec_token_identity=identity,reports=reports,commands=commands,
              limitations=['Eager-only diagnostic; CUDA graphs and async/batched serving are not qualified.',
                           'Slot-prefix mapping and HF replay checked, not direct KV tensor equality.',
                           'Auxiliary KL recurrence remains ground-truth teacher-forced.'])
    workflow.save(RUN/'live-gate.json',gate)
    if not passed:raise RuntimeError('Live gate incomplete: rejection coverage or no-spec/overlay identity; no full training launched')
    print('LIVE_GATE_FOUR_PARITY_CONTROLS_TWO_PUBLIC_REQUESTS_ALL_CLASSES_AND_NO_SPEC_IDENTITY_PASSED',flush=True)

if __name__=='__main__':main()
