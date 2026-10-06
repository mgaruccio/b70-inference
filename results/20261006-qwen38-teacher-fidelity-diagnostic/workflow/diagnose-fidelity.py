#!/usr/bin/env python3
"""Same-lease teacher execution-path diagnostics; no optimizer or training entrypoint."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.umask(0o077)
HOME=Path.home(); CODE=HOME/'qwen-mtp-code'; RUN=HOME/'qwen-mtp-run'; MODEL=HOME/'qwen-model'
spec=importlib.util.spec_from_file_location('serving',CODE/'run-pilot.py')
serving=importlib.util.module_from_spec(spec);spec.loader.exec_module(serving)
trainer=serving.trainer
sys.path.insert(0,str(CODE/'scripts/experiments'))
import qwen38_mtp_live_parity as parity
PARITY=CODE/'scripts/experiments/qwen38_mtp_live_parity.py'
DIAG=CODE/'scripts/experiments/qwen38_teacher_fidelity.py'
HOOK=CODE/'patches/qwen38_mtp_parity.py'

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def invoke(label,argv,required=True):
    save(RUN/(label+'-command.json'),argv)
    started=time.monotonic()
    with (RUN/(label+'.log')).open('w') as log:
        result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    report={'command':argv,'returncode':result.returncode,'elapsed_s':time.monotonic()-started}
    save(RUN/(label+'-execution.json'),report)
    print('STEP='+label+' RETURN='+str(result.returncode),flush=True)
    if required and result.returncode:raise RuntimeError('Diagnostic step failed: '+label)
    return report

def native_histories():
    selected=[(split,serving.records(split)[0]['prompt_id']) for split in ('train','dev')]
    for label,depth in [('parity-stock4',4),('parity-stock8',8)]:
        with serving.Server(label,depth,capture=False,eager=True) as server:
            root=server.cell/'parity'
            for split,ident in selected:
                invoke(label+'-'+split,[sys.executable,str(PARITY),'request','--root',str(root),'--hook',str(HOOK),'--requests',str(RUN/(split+'-requests.jsonl')),'--split',split,'--prompt-id',ident])
        invoke(label+'-check',[sys.executable,str(PARITY),'check','--root',str(root),'--hook',str(HOOK),'--trainer',str(CODE/'scripts/experiments/qwen38_train_mtp.py'),'--model',str(MODEL),'--device','cuda'],required=False)
        assert (root/'parity-report.json').is_file(),'Checker must retain an actual report'
    invoke('verifier-histories',[sys.executable,str(CODE/'verifier-histories.py')])
    current=json.loads((RUN/'verifier-inputs.json').read_text())
    fixed=json.loads((RUN/'archived-teacher-inputs.json').read_text())
    fixed=[r for r in fixed if not r.get('native_verifier_rows')]
    assert len(fixed)==48 and 0<len(current)<=12
    manifest=fixed+current
    assert len({r['name'] for r in manifest})==len(manifest)
    assert sum(len(r['input_ids']) for r in manifest)<=1048576
    save(RUN/'teacher-inputs.json',manifest)
    return manifest

def native_prefill(manifest):
    partial=next((r for r in manifest if r.get('category')=='partial_rejection'),None)
    assert partial is not None,'Need actual partial-rejection capture control'
    controls=[manifest[0],partial]
    outputs={}
    for eager in (False,True):
        mode='eager' if eager else 'compiled'
        for capture,records in [(False,controls),(True,manifest)]:
            label='prefill-'+mode+('-on' if capture else '-off');rows={}
            with serving.Server(label,0,replay=capture,eager=eager) as server:
                control=server.cell/'features/capture-request.json'
                for record in records:
                    if capture:save(control,{k:record[k] for k in ('name','input_ids','loss_mask')})
                    body={'model':'qwen38','prompt':record['input_ids'],'max_tokens':1,'temperature':0,'seed':42,'return_token_ids':True}
                    save(server.cell/(record['name']+'-request.json'),body)
                    response=serving.probe.post('/v1/completions',body)
                    save(server.cell/(record['name']+'-response.json'),response)
                    assert response['usage']['completion_tokens']==1
                    ids=response['choices'][0].get('token_ids',[]);assert len(ids)==1
                    rows[record['name']]=ids
                    if capture:
                        assert (server.cell/'features'/(record['name']+'.pt')).is_file()
                        control.unlink()
                save(server.cell/'replay-outputs.json',rows)
            outputs[label]=rows
        assert all(outputs['prefill-'+mode+'-on'][r['name']]==outputs['prefill-'+mode+'-off'][r['name']] for r in controls),'Capture changed output'
    save(RUN/'target-capture-identity.json',{'status':'passed','controls_per_mode':2,'requests_per_capture_cell':len(manifest),'exact_output_identity':True,'optimizer_updates':0})

def compare(manifest):
    torch=trainer.runtime().torch;device=torch.device('cuda')
    checkpoint=trainer.Checkpoint(MODEL);embedding,head=trainer.frozen_heads(checkpoint,device)
    groups={k:[] for k in ('hf_full_cached','hf_full_compiled','hf_full_eager','hf_cached_compiled','hf_cached_eager','native_compiled_eager','native_compiled_verifier','native_eager_verifier','hf_full_verifier','hf_cached_verifier')}
    native_sampling=[]
    native_outputs={mode:json.loads((RUN/('prefill-'+mode+'-on/replay-outputs.json')).read_text()) for mode in ('compiled','eager')}
    def pair(group,name,actual,reference):
        assert actual.shape==reference.shape==(4,checkpoint.config.hidden_size)
        assert actual.dtype==reference.dtype==torch.bfloat16
        error=parity.errors(actual,reference)
        with torch.no_grad(),trainer.autocast(device,dtype=torch.bfloat16):
            a=head(actual.to(device)).float();b=head(reference.to(device)).float()
        aid=a.argmax(-1).tolist();bid=b.argmax(-1).tolist()
        margins={}
        for label,logits in [('actual',a),('reference',b)]:
            values,ids=logits.topk(2,dim=-1)
            margins[label]={'top2_ids':ids.tolist(),'top2_logits':values.tolist(),'margins':(values[:,0]-values[:,1]).tolist()}
        groups[group].append({'name':name,'errors':error,'actual_argmax':aid,'reference_argmax':bid,'argmax_equal':aid==bid,'top2':margins})
    for record in manifest:
        name=record['name'];ids=torch.tensor(record['input_ids'],dtype=torch.long)
        hf=torch.load(RUN/'hf-replay'/(name+'.pt'),weights_only=True,map_location='cpu')
        assert torch.equal(hf['prefix'],ids)
        native={}
        for mode in ('compiled','eager'):
            row=torch.load(RUN/('prefill-'+mode+'-on/features')/(name+'.pt'),weights_only=True,map_location='cpu')
            assert torch.equal(row['input_ids'],ids) and torch.equal(row['positions'],torch.arange(ids.numel()))
            assert record['scored_teacher_positions']==list(range(ids.numel()-4,ids.numel()))
            native[mode]=row['target_last_hidden_states'][-4:]
            with torch.no_grad(),trainer.autocast(device,dtype=torch.bfloat16):
                projected=head(native[mode].to(device)).float().argmax(-1).tolist()
            emitted=native_outputs[mode][name]
            assert len(emitted)==1
            native_sampling.append({'name':name,'mode':mode,'projected_last_argmax':projected[-1:],'API_token_ids':emitted,'equal':projected[-1:]==emitted})
        pair('hf_full_cached',name,hf['full_rows'],hf['cached_rows'])
        for mode in ('compiled','eager'):
            pair('hf_full_'+mode,name,hf['full_rows'],native[mode])
            pair('hf_cached_'+mode,name,hf['cached_rows'],native[mode])
        pair('native_compiled_eager',name,native['compiled'],native['eager'])
        if record.get('native_verifier_rows'):
            verifier=torch.load(RUN/record['native_verifier_rows'],weights_only=True,map_location='cpu')
            assert torch.equal(verifier['prefix'],ids) and verifier['positions'].tolist()==record['scored_teacher_positions']
            for mode in ('compiled','eager'):pair('native_'+mode+'_verifier',name,native[mode],verifier['teacher_rows'])
            pair('hf_full_verifier',name,hf['full_rows'],verifier['teacher_rows'])
            pair('hf_cached_verifier',name,hf['cached_rows'],verifier['teacher_rows'])
    summary={k:{'histories':len(v),'rows':4*len(v),'numeric_fail_histories':sum(not x['errors']['pass'] for x in v),'argmax_disagree_rows':sum(a!=b for x in v for a,b in zip(x['actual_argmax'],x['reference_argmax'])),'max_relative_l2':max(n for x in v for n in x['errors']['relative_l2']),'numeric_passed':all(x['errors']['pass'] for x in v),'argmax_passed':all(x['argmax_equal'] for x in v)} for k,v in groups.items()}
    save(RUN/'native-head-sampling.json',{'optimizer_updates':0,'training_allowed':False,'all_equal':all(r['equal'] for r in native_sampling),'rows':native_sampling})
    save(RUN/'paired-comparisons.json',{'status':'completed_diagnostic_comparisons','optimizer_updates':0,'training_allowed':False,'training_ready':False,'fresh_waivers':False,'tolerance':parity.TOLERANCE,'summary':summary,'groups':groups,'scope':'Same lease, same prefixes/weights; native compile/eager bundle, HF full/cached path. No SSM precision or GDN backend changes, no throughput claim.'})
    print('PAIRED_FIDELITY_SUMMARY='+json.dumps(summary),flush=True)
    del embedding,head;gc.collect();torch.cuda.empty_cache()
    return summary

def main():
    p=json.loads((RUN/'pilot-protocol.json').read_text())
    assert p['authorized'] and p['lease_cap_hours']==2 and p['lease_cap_usd']==6.6
    assert p['optimizer_updates']==0 and not p['full_training_authorized'] and not p['test_data_deployed']
    started=time.monotonic();manifest=native_histories()
    native_prefill(manifest)
    execution=invoke('hf-full-cached',[sys.executable,str(DIAG),'hf','--model',str(MODEL),'--manifest',str(RUN/'teacher-inputs.json'),'--output-dir',str(RUN/'hf-replay'),'--device','cuda'],required=False)
    hf=json.loads((RUN/'hf-replay/fidelity.json').read_text())
    assert hf['optimizer_updates']==0 and not hf['training_allowed'] and hf['entries']==len(manifest)
    assert hf['mode']=='hf_cached' and hf['status'] in ('hf_full_cached_internal_pass','hf_full_cached_internal_gate_open')
    internal_pass=hf['numeric_passed'] and hf['argmax_passed']
    assert execution['returncode']==(0 if internal_pass else 1), 'Only complete diagnostic gate failure may continue; structural errors stop'
    assert all((RUN/'hf-replay'/(r['name']+'.pt')).is_file() for r in manifest)
    result=compare(manifest)
    save(RUN/'diagnostic.json',{'status':'completed_execution_path_diagnosis','optimizer_updates':0,'training_ready':False,'full_training_authorized':False,'test_data_used':False,'histories':len(manifest),'current_verifier_histories':sum(bool(r.get('native_verifier_rows')) for r in manifest),'native_cells_completed':6,'performance_improvement_claimed':False,'elapsed_s':time.monotonic()-started,'comparisons':result})
    serving.archive('final')

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(RUN/'failure.json',{'type':type(error).__name__,'reason':str(error),'optimizer_updates':0,'training_allowed':False})
        print('DIAGNOSIS_FAILED='+type(error).__name__,flush=True)
        try:serving.archive('final')
        except Exception as backup_error:print('FAILURE_ARCHIVE_FAILED='+type(backup_error).__name__,flush=True)
        raise
