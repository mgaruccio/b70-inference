#!/usr/bin/env python3
"""Select bounded actual verifier histories, including corrected-cache continuations."""
import json
import os
from pathlib import Path
import sys

os.umask(0o077)
HOME=Path.home(); CODE=HOME/'qwen-mtp-code'; RUN=HOME/'qwen-mtp-run'
sys.path.insert(0,str(CODE/'scripts/experiments'))
import qwen38_mtp_live_parity as parity
import qwen38_train_mtp as trainer

def main():
    torch=trainer.runtime().torch
    hook=parity.module_at(CODE/'patches/qwen38_mtp_parity.py','diagnostic_hook')
    destination=RUN/'verifier-histories'; destination.mkdir(mode=0o700,exist_ok=False)
    manifest=[]; seen=set(); classes=set()
    for label in ('parity-stock4','parity-stock8'):
        root=RUN/label/'parity'
        for control_path in sorted(root.glob('*.control.json')):
            key=control_path.name.removesuffix('.control.json')
            control=hook.validate_control(json.loads(control_path.read_text()),key)
            paths=sorted(root.glob(key+'.*.pt'))
            assert 0<len(paths)<=hook.MAX_ROUNDS
            cursor=0; stream=[]; previous=[]; previous_kind=None
            for path in paths:
                trace=torch.load(path,weights_only=True,map_location='cpu')
                keep,kind=parity.round_layout(trace,control['prompt_token_ids'],cursor,stream,previous)
                classes.add(kind)
                positions=parity.positions(trace['target_positions'])
                ids=trace['target_ids'].tolist(); start=positions[0]
                assert positions==list(range(start,start+len(ids)))
                prefix=stream[:start]+ids
                assert len(prefix)==start+len(ids)
                categories=[kind]
                if previous_kind in ('zero_acceptance','partial_rejection'):
                    categories.append('after-'+previous_kind)
                for category in categories:
                    signature=(label,category)
                    if signature in seen or len(ids)<4: continue
                    seen.add(signature)
                    name='verifier-'+str(len(manifest)).zfill(3)
                    reference=trace['target_hidden'][-4:].clone()
                    assert reference.shape==(4,5120) and reference.dtype==torch.bfloat16
                    row={'name':name,'source_id':control['prompt_id'],'split':control['split'],'source_label':label,'request_id':key,'round':trace['step'],'acceptance_kind':kind,'category':category,'input_ids':prefix,'loss_mask':[False]*(len(prefix)-3)+[True]*3,'root':len(prefix)-5,'scored_teacher_positions':list(range(len(prefix)-4,len(prefix))),'native_verifier_rows':str(Path('verifier-histories')/(name+'.pt'))}
                    assert row['root']>=0
                    torch.save({'teacher_rows':reference,'positions':torch.tensor(row['scored_teacher_positions']),'prefix':torch.tensor(prefix)},destination/(name+'.pt'))
                    manifest.append(row)
                previous=[sample['ids'].item() for sample in trace['samples']]
                cursor+=keep; previous_kind=kind
    assert 0<len(manifest)<=12
    (RUN/'verifier-inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (RUN/'verifier-history-selection.json').write_text(json.dumps({'status':'completed','histories':len(manifest),'observed_classes':sorted(classes),'categories':sorted({r['category'] for r in manifest}),'optimizer_updates':0,'source':'existing live native verifier traces; no manufactured acceptance classes'},indent=2)+'\n')
    print('ACTUAL_VERIFIER_HISTORY_SELECTION='+str(len(manifest)),flush=True)

if __name__=='__main__': main()
