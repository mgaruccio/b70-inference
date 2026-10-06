#!/usr/bin/env python3
"""Compare actual implementation teacher rows with native exact-token prefill captures."""
import json
import os
from pathlib import Path
import sys

os.umask(0o077)
HOME=Path.home(); CODE=HOME/'qwen-mtp-code'; RUN=HOME/'qwen-mtp-run'; MODEL=HOME/'qwen-model'
sys.path.insert(0,str(CODE/'scripts/experiments'))
import qwen38_train_mtp as trainer
import qwen38_mtp_live_parity as parity

def main():
    torch=trainer.runtime().torch; device=torch.device('cuda')
    checkpoint=trainer.Checkpoint(MODEL)
    embedding,head=trainer.frozen_heads(checkpoint,device)
    manifest=json.loads((RUN/'teacher-inputs.json').read_text())
    native_outputs=json.loads((RUN/'prefill-replay-on/replay-outputs.json').read_text())
    rows=[]
    for record in manifest:
        name=record['name']
        native=torch.load(RUN/'prefill-replay-on/features'/(name+'.pt'),weights_only=True,map_location='cpu')
        reference=torch.load(RUN/'teacher-histories'/(name+'.pt'),weights_only=True,map_location='cpu')
        ids=torch.tensor(record['input_ids'],dtype=torch.long)
        assert torch.equal(native['input_ids'],ids) and torch.equal(reference['prefix'],ids)
        assert torch.equal(native['positions'],torch.arange(ids.numel()))
        assert record['scored_teacher_positions']==list(range(ids.numel()-4,ids.numel()))
        native_rows=native['target_last_hidden_states'][-4:]
        hf_rows=reference['teacher_rows']
        assert native_rows.dtype==hf_rows.dtype==torch.bfloat16
        error=parity.errors(hf_rows,native_rows)
        with torch.no_grad(),trainer.autocast(device,dtype=torch.bfloat16):
            native_logits=head(native_rows.to(device)).float()
            hf_logits=head(hf_rows.to(device)).float()
        native_ids=native_logits.argmax(-1).tolist(); hf_ids=hf_logits.argmax(-1).tolist()
        margins={}
        for label,logits in [('native',native_logits),('hf',hf_logits)]:
            values,indices=logits.topk(2,dim=-1)
            margins[label]={'top2_ids':indices.tolist(),'top2_logits':values.tolist(),'margins':(values[:,0]-values[:,1]).tolist()}
        kl=trainer.kl_divergence(hf_logits,native_logits,1.0).detach().cpu().tolist()
        emitted=native_outputs[name]['token_ids']
        assert len(emitted)==1
        rows.append({'name':name,'source_id':record['source_id'],'split':record['split'],'root':record['root'],'prefix_tokens':len(record['input_ids']),'positions':record['scored_teacher_positions'],'errors':error,'hf_argmax':hf_ids,'native_frozen_head_argmax':native_ids,'argmax_equal':hf_ids==native_ids,'native_sampling_last_argmax_equal':emitted==native_ids[-1:],'native_next_token_ids':emitted,'top2':margins,'teacher_native_to_hf_kl':kl})
        if record.get('native_verifier_rows'):
            verifier=torch.load(RUN/record['native_verifier_rows'],weights_only=True,map_location='cpu')
            assert torch.equal(verifier['prefix'],ids)
            assert verifier['positions'].tolist()==record['scored_teacher_positions']
            verifier_rows=verifier['teacher_rows']
            assert verifier_rows.shape==native_rows.shape and verifier_rows.dtype==torch.bfloat16
            with torch.no_grad(),trainer.autocast(device,dtype=torch.bfloat16):
                verifier_logits=head(verifier_rows.to(device)).float()
            verifier_ids=verifier_logits.argmax(-1).tolist()
            rows[-1]['verifier']={'native_prefill_errors':parity.errors(native_rows,verifier_rows),'hf_errors':parity.errors(hf_rows,verifier_rows),'argmax':verifier_ids,'native_prefill_argmax_equal':verifier_ids==native_ids,'hf_argmax_equal':verifier_ids==hf_ids,'category':record['category'],'source_label':record['source_label'],'round':record['round']}
    verifier_checks=[r['verifier'] for r in rows if 'verifier' in r]
    assert verifier_checks, 'Actual native verifier histories required'
    numeric=all(r['errors']['pass'] for r in rows) and all(v['native_prefill_errors']['pass'] and v['hf_errors']['pass'] for v in verifier_checks)
    argmax=all(r['argmax_equal'] and r['native_sampling_last_argmax_equal'] for r in rows) and all(v['native_prefill_argmax_equal'] and v['hf_argmax_equal'] for v in verifier_checks)
    report={'status':'numeric_and_argmax_pass' if numeric and argmax else 'teacher_parity_gate_open','optimizer_updates':0,'training_allowed':False,'teacher_rows':len(rows)*4,'branches':len(rows),'native_verifier_histories':len(verifier_checks),'numeric_passed':numeric,'argmax_passed':argmax,'fresh_near_tie_waivers':False,'tolerance':parity.TOLERANCE,'rows':rows,'scope':'implementation FrozenTarget.replay versus native fresh target prefill and actual incremental verifier rows, including corrected-cache continuations'}
    (RUN/'teacher-parity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('TEACHER_PARITY_STATUS='+report['status']+' ZERO_UPDATES',flush=True)
    if not numeric: raise RuntimeError('Teacher numeric parity failed; no full training allowed')

if __name__=='__main__': main()
