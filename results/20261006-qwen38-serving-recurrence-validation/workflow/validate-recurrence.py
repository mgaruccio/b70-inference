#!/usr/bin/env python3
"""Owned-lease correctness/profiling only; zero optimizer updates, no test data."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.umask(0o077)
HOME=Path.home(); CODE=HOME/'qwen-mtp-code'; RUN=HOME/'qwen-mtp-run'; MODEL=HOME/'qwen-model'
spec=importlib.util.spec_from_file_location('workflow',CODE/'run-pilot.py')
workflow=importlib.util.module_from_spec(spec); spec.loader.exec_module(workflow)
trainer=workflow.trainer
PARITY=CODE/'scripts/experiments/qwen38_mtp_live_parity.py'; HOOK=CODE/'patches/qwen38_mtp_parity.py'

def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def invoke(label,argv,required=True):
    started=time.monotonic(); log=RUN/(label+'.log')
    save(RUN/(label+'-command.json'),argv)
    with log.open('w') as handle:
        result=subprocess.run(argv,stdout=handle,stderr=subprocess.STDOUT)
    report={'command':argv,'returncode':result.returncode,'elapsed_s':time.monotonic()-started,'log':log.name}
    save(RUN/(label+'-execution.json'),report)
    print('VALIDATION_STEP='+label+' RETURN='+str(result.returncode),flush=True)
    if required and result.returncode:
        raise RuntimeError('Validation step failed; retained log: '+label)
    return report

def diagnostic_records():
    restore=json.loads((RUN/'cache-restore.json').read_text())
    assert restore['restored_sequences']=={'train':374,'dev':64}
    config=trainer.native_config(json.loads((MODEL/'config.json').read_text()))
    chosen={}; inventories={}
    for split,folder in [('train','capture-train/train'),('dev','capture-dev/heldout')]:
        eligible=[]
        for row in restore['lengths'][split]:
            record=trainer.load_record(RUN/folder/row['file'],config,2048)
            roots=trainer.sample_roots(record,4,8,random.Random(42))
            if roots: eligible.append((row['tokens'],row['id'],row['file'],roots))
        eligible.sort(); assert len(eligible)>=3
        indices=[0,len(eligible)//2,len(eligible)-1]
        values=[eligible[i] for i in indices]
        assert len({v[1] for v in values})==3
        target=RUN/(split+'-diagnostics'); target.mkdir(mode=0o700,exist_ok=False)
        for length,ident,name,roots in values:
            os.link(RUN/folder/name,target/name)
        inventories[split]=[{'id':ident,'file':name,'tokens':length,'eligible_seed42_roots':roots} for length,ident,name,roots in values]
        chosen[split]=target
    save(RUN/'diagnostic-inputs.json',{'selection':'shortest/median/longest with recursive roots','source_full_counts':restore['restored_sequences'],'diagnostic_only':True,'optimizer_updates':0,'records':inventories})
    return chosen

def cli_validation(chosen):
    reports={}
    for label,greedy in [('off',False),('on',True)]:
        root=RUN/label; root.mkdir(mode=0o700,exist_ok=False); output=root/'tuned-mtp.safetensors'
        argv=[sys.executable,str(CODE/'scripts/experiments/qwen38_train_mtp.py'),'--model',str(MODEL),'--train-dir',str(chosen['train']),'--eval-dir',str(chosen['dev']),'--output',str(output),'--validate-only','--recursive-depth','4','--roots','8','--depth-weights','1','1','.8','.8','--device','cuda','--seed','42','--lr','1e-6','--max-length','2048','--grad-accum','1','--logits-chunk','64','--kl-weight','1','--kl-temperature','1']
        if greedy: argv+=['--greedy-kl']
        execution=invoke('cli-'+label,argv)
        metadata=json.loads(output.with_suffix('.json').read_text())
        assert metadata['optimizer_steps']==0,'No optimizer updates authorized'
        reports[label]={'execution':execution,'metadata':metadata}
    torch=trainer.runtime().torch; safe=trainer.runtime().safe_open
    stock=RUN/'reference-heads/tuned-mtp.step0000.safetensors'
    with safe(str(stock),framework='pt',device='cpu') as source:
        for label in ('off','on'):
            with safe(str(RUN/label/'tuned-mtp.safetensors'),framework='pt',device='cpu') as target:
                assert set(source.keys())==set(target.keys())
                assert all(torch.equal(source.get_tensor(k),target.get_tensor(k)) for k in source.keys()),'Zero-update export changed stock weights'
    save(RUN/'cli-validation.json',{'status':'completed','optimizer_updates':0,'stock_tensor_identity':True,'reports':reports})
    return reports

def native_trace_validation():
    selected=[(split,workflow.records(split)[0]['prompt_id']) for split in ('train','dev')]
    stock=RUN/'on/tuned-mtp.safetensors'; reports={}; responses={}
    for label,depth,overlay in [('parity-no-spec',0,None),('parity-stock4',4,None),('parity-overlay4',4,stock),('parity-stock8',8,None),('parity-overlay8',8,stock)]:
        with workflow.Server(label,depth,capture=False,weights=overlay) as server:
            root=server.cell/'parity'
            for split,ident in selected:
                argv=[sys.executable,str(PARITY),'request','--root',str(root),'--hook',str(HOOK),'--requests',str(RUN/(split+'-requests.jsonl')),'--split',split,'--prompt-id',ident]
                invoke(label+'-'+split,argv)
        if depth:
            argv=[sys.executable,str(PARITY),'check','--root',str(root),'--hook',str(HOOK),'--trainer',str(CODE/'scripts/experiments/qwen38_train_mtp.py'),'--model',str(MODEL),'--device','cuda']
            execution=invoke(label+'-check',argv,required=False)
            report_path=root/'parity-report.json'
            if not report_path.exists(): raise RuntimeError('Native parity checker produced no report')
            reports[label]={'execution':execution,'report':json.loads(report_path.read_text())}
        responses[label]={p.name.removesuffix('.response.json'):json.loads(p.read_text())['choices'][0]['token_ids'] for p in root.glob('*.response.json')}
        gc.collect(); trainer.runtime().torch.cuda.empty_cache()
    identity={label:values==responses['parity-no-spec'] for label,values in responses.items()}
    overlay={str(depth):responses['parity-stock'+str(depth)]==responses['parity-overlay'+str(depth)] for depth in (4,8)}
    assert all(overlay.values()),'Stock-overlay output identity failed'
    classes=set()
    for item in reports.values():
        for result in item['report'].get('requests',[]): classes.update(result.get('observed_classes',[]))
    required={'prefill','full_acceptance','partial_rejection','zero_acceptance'}
    strict=all(item['report']['status']=='observed_parity_pass' for item in reports.values()) and required<=classes
    save(RUN/'live-gate.json',{'status':'strict_pass' if strict else 'strict_gate_open','strict_gate_a_closed':strict,'training_allowed':False,'fresh_parity_waivers':False,'observed_classes':sorted(classes),'missing_classes':sorted(required-classes),'no_spec_token_identity':identity,'stock_overlay_identity':overlay,'selected':selected,'reports':reports})
    return reports

def api_controls():
    stock=RUN/'on/tuned-mtp.safetensors'; reports={}
    for label,depth,overlay in [('dev-no-spec',0,None),('dev-stock4',4,None),('dev-stock-overlay4',4,stock),('dev-stock8',8,None),('dev-stock-overlay8',8,stock)]:
        assert len(workflow.records('dev'))==64
        reports[label]=workflow.evaluate(label,depth,'dev',weights=overlay)
        assert reports[label]['requests']==64
    exact={}
    for depth in (4,8):
        left=RUN/('dev-stock'+str(depth)); right=RUN/('dev-stock-overlay'+str(depth))
        differences=[]
        for record in workflow.records('dev'):
            name=record['prompt_id']+'-result.json'
            a=json.loads((left/name).read_text()); b=json.loads((right/name).read_text())
            if a['token_ids']!=b['token_ids']: differences.append(record['prompt_id'])
        exact[str(depth)]={'matches':64-len(differences),'requests':64,'different_ids':differences}
    no_spec={}
    for label in ('dev-stock4','dev-stock-overlay4','dev-stock8','dev-stock-overlay8'):
        different=[]
        for record in workflow.records('dev'):
            name=record['prompt_id']+'-result.json'
            a=json.loads((RUN/'dev-no-spec'/name).read_text()); b=json.loads((RUN/label/name).read_text())
            if a['token_ids']!=b['token_ids']: different.append(record['prompt_id'])
        no_spec[label]={'matches':64-len(different),'requests':64,'different_ids':different}
    save(RUN/'api-controls.json',{'status':'completed','requests':320,'optimizer_updates':0,'cells':reports,'stock_overlay_output_identity':exact,'no_spec_output_identity':no_spec,'performance_improvement_claimed':False})
    assert all(row['matches']==64 for row in exact.values()), 'Stock-overlay full-dev identity failed'
    return no_spec

def teacher_prefill_validation():
    invoke('verifier-histories',[sys.executable,str(CODE/'verifier-histories.py')])
    invoke('teacher-histories',[sys.executable,str(CODE/'teacher-histories.py')])
    manifest=json.loads((RUN/'teacher-inputs.json').read_text())
    assert 0<len(manifest)<=128
    outputs={}
    for label,capture in [('prefill-replay-off',False),('prefill-replay-on',True)]:
        rows={}
        with workflow.Server(label,0,capture=False,replay=capture) as server:
            control=server.cell/'features/capture-request.json'
            for record in manifest:
                if capture: save(control,{k:record[k] for k in ('name','input_ids','loss_mask')})
                body={'model':'qwen38','prompt':record['input_ids'],'max_tokens':1,'temperature':0,'seed':42,'return_token_ids':True}
                save(server.cell/(record['name']+'-request.json'),body)
                response=workflow.probe.post('/v1/completions',body)
                save(server.cell/(record['name']+'-response.json'),response)
                choice=response['choices'][0]
                assert response['usage']['completion_tokens']==1 and len(choice.get('token_ids',[]))==1
                rows[record['name']]={'token_ids':choice['token_ids'],'text':choice['text']}
                if capture:
                    assert (server.cell/'features'/(record['name']+'.pt')).is_file()
                    control.unlink()
                print('EXACT_PREFIX_REPLAY='+label+' '+record['name'],flush=True)
            save(server.cell/'replay-outputs.json',rows)
        outputs[label]=rows
    assert outputs['prefill-replay-off']==outputs['prefill-replay-on'],'Target replay capture changed outputs'
    save(RUN/'target-capture-identity.json',{'status':'passed','requests_per_cell':len(manifest),'exact_output_identity':True,'optimizer_updates':0})
    invoke('teacher-compare',[sys.executable,str(CODE/'teacher-compare.py')])
    return json.loads((RUN/'teacher-parity.json').read_text())

def main():
    protocol=json.loads((RUN/'pilot-protocol.json').read_text())
    assert protocol['optimizer_updates']==0 and not protocol['full_training_authorized']
    assert protocol['mode']=='correctness_and_profiling_only' and not protocol['test_data_deployed']
    started=time.monotonic()
    chosen=diagnostic_records()
    cli_validation(chosen)
    native_trace_validation()
    teacher=teacher_prefill_validation()
    no_spec=api_controls()
    live=json.loads((RUN/'live-gate.json').read_text())
    fidelity=all(no_spec_row['matches']==64 for no_spec_row in no_spec.values()) and all(live['no_spec_token_identity'].values())
    save(RUN/'validation.json',{'status':'completed_development_validation','optimizer_updates':0,'full_training_authorized':False,'test_data_used':False,'cli_diagnostic_records':{'train':3,'dev':3},'restored_full_corpus':{'train':374,'dev':64},'teacher_branches':teacher['branches'],'teacher_numeric_passed':teacher['numeric_passed'],'teacher_argmax_passed':teacher['argmax_passed'],'native_strict_gate_closed':live['strict_gate_a_closed'],'no_spec_token_identity':live['no_spec_token_identity'],'no_spec_fidelity_passed':fidelity,'serving_cells_completed':5,'measured_dev_requests':320,'elapsed_s':time.monotonic()-started,'performance_improvement_claimed':False,'training_ready':teacher['numeric_passed'] and teacher['argmax_passed'] and live['strict_gate_a_closed'] and fidelity})
    workflow.archive('final')
    print('CORRECTNESS_PROFILING_COMPLETE_ZERO_UPDATES_FIVE_DEV64_CELLS',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        save(RUN/'failure.json',{'type':type(error).__name__,'reason':str(error),'optimizer_updates':0,'full_training_authorized':False})
        print('VALIDATION_FAILED='+type(error).__name__,flush=True)
        try:
            workflow.archive('final')
        except Exception as backup_error:
            print('FAILURE_ARCHIVE_FAILED='+type(backup_error).__name__,flush=True)
        raise
