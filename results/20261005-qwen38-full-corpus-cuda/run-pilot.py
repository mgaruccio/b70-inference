#!/usr/bin/env python3
"""Bounded public-only native CUDA API baseline, recursive-head pilot and ABBA."""
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

HOME=Path.home(); RUN=HOME/'qwen-mtp-run'; CODE=HOME/'qwen-mtp-code'; MODEL=HOME/'qwen-model'; UPLOAD=HOME/'qwen-mtp-upload'
sys.path.insert(0,str(CODE/'scripts/experiments'))
import qwen38_mtp_native_corpus as native
import qwen38_mtp_tune_probe as tune
import qwen38_lossy_probe as probe
import qwen38_train_mtp as trainer
spec=importlib.util.spec_from_file_location('baseline_api',CODE/'baseline-api.py'); api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
probe.BASE='http://127.0.0.1:8000'
native.PATCHES=CODE/'patches'
IMAGE='vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967'
probe.IMAGE=IMAGE  # Existing sandbox uses the already-pulled CUDA image, without GPU access.
original_command=probe.command
def command(*args,**kwargs):
    if args and args[0]=='docker':
        args=tuple('python3' if item=='python' else item for item in args)
        return original_command('sudo','-n',*args,**kwargs)
    return original_command(*args,**kwargs)
probe.command=command

def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False))

def docker(*args,**kwargs):
    return command('docker',*args,**kwargs)

def archive(phase):
    if (HOME/'qwen-pilot.log').exists():shutil.copyfile(HOME/'qwen-pilot.log',RUN/'pilot.log')
    path=UPLOAD/(phase+'.tgz')
    excludes=['--exclude=capture-train','--exclude=capture-dev','--exclude=teacher-capture','--exclude=*.safetensors'] if phase=='evaluation' else []
    subprocess.run(['tar','--exclude=*private*.json','--exclude=__pycache__',*excludes,'-cf',str(path),'-C',str(RUN),'.'],check=True)
    subprocess.run(['python3',str(CODE/'upload-archive.py'),str(UPLOAD),phase],check=True)
    print('ARCHIVED_PHASE='+phase,flush=True)

def records(split):
    return [json.loads(line) for line in (RUN/(split+'-requests.jsonl')).read_text().splitlines() if line.strip()]

def stop_owned(name,cell,proc):
    inspected=subprocess.run(['sudo','-n','docker','inspect','-f','{{range .Mounts}}{{if eq .Destination "/profile"}}{{.Source}}{{end}}{{end}}',name],capture_output=True,text=True)
    if inspected.returncode==0:
        if inspected.stdout.strip()!=str(cell):raise RuntimeError('Container ownership mismatch; refuse cleanup')
        docker('stop','--time','30',name,timeout=60)
    if proc is not None:proc.wait(timeout=90)

class Server:
    def __init__(self,label,depth,capture=False,weights=None):
        self.label=label;self.depth=depth;self.capture=capture;self.weights=weights;self.cell=RUN/label;self.cell.mkdir(mode=0o700)
        self.name='qwen-cuda-mtp-'+label.lower();self.proc=None;self.log=None
    def __enter__(self):
        if docker('ps','-q').strip():raise RuntimeError('GPU host must be idle; do not stop unrelated workloads')
        argv=['sudo','-n','docker','run','--rm','--name',self.name,'--gpus','all','--ipc','host','-p','127.0.0.1:8000:8000','-v',str(MODEL)+':/model:ro','-v',str(CODE/'patches')+':/patches:ro','-v',str(self.cell)+':/profile','-e','VLLM_USE_V2_MODEL_RUNNER=0','-e','VLLM_ENABLE_CUDA_COMPATIBILITY=1']
        if self.weights:argv+=['-v',str(self.weights)+':/mtp.safetensors:ro','-e','B70_MTP_WEIGHTS=/mtp.safetensors']
        if self.capture:
            (self.cell/'features').mkdir(mode=0o700)
            argv+=['-e','B70_MTP_NATIVE_CAPTURE_DIR=/profile/features','-e','B70_MTP_NATIVE_MAX_TOKENS=2097152','-e','B70_MTP_NATIVE_MAX_REQUESTS=1024']
        serve=['vllm','serve','/model','--dtype','bfloat16','--max-model-len','8192','--gpu-memory-utilization','0.92','--kv-cache-dtype','auto','--max-num-seqs','1','--max-num-batched-tokens','2048','--mamba-cache-mode','align','--performance-mode','balanced','--compilation-config','{"cudagraph_capture_sizes":[1,2,4,8,16]}','--no-async-scheduling','--enable-prefix-caching' if self.capture else '--no-enable-prefix-caching','--chat-template-content-format','openai','--default-chat-template-kwargs','{"enable_thinking":false}','--reasoning-parser','qwen3','--served-model-name','qwen38','--language-model-only','--port','8000']
        if self.depth:serve+=['--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':self.depth})]
        import shlex
        setup='set -e; python3 /patches/patch_mtp_training.py; python3 /patches/patch_mtp_native_capture.py; exec '+shlex.join(serve)
        argv+=['--entrypoint','bash',IMAGE,'-c',setup]
        save(self.cell/'command.json',argv)
        save(self.cell/'config.json',dict(tier='development',image=IMAGE,depth=self.depth,weights=str(self.weights) if self.weights else 'stock',dtype='bfloat16',kv_dtype='auto',quantization=None,context=8192,concurrency=1,capture=self.capture,prefix_caching=self.capture,thinking=False))
        print('SERVER_START='+self.label,flush=True)
        self.log=(self.cell/'server.log').open('w');self.proc=subprocess.Popen(argv,stdout=self.log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+900
            while True:
                if self.proc.poll() is not None:raise RuntimeError('Native CUDA server exited; inspect '+str(self.cell/'server.log'))
                try:probe.get('/health');break
                except OSError:pass
                if time.monotonic()>deadline:raise TimeoutError('Server readiness exceeded 900 seconds')
                time.sleep(3)
            models=json.loads(probe.get('/v1/models'));save(self.cell/'models.json',models)
            if not any(m['id']=='qwen38' and m['max_model_len']==8192 for m in models['data']):raise RuntimeError('Wrong public model/context')
            save(self.cell/'runtime.json',dict(container=docker('inspect',self.name),versions=docker('exec',self.name,'python','-c','import vllm,torch; print(torch.__version__); print(vllm.__version__); print(torch.cuda.get_device_properties(0))',timeout=60),gpu=command('nvidia-smi','--query-gpu=name,memory.used,power.draw,temperature.gpu','--format=csv')))
            print('SERVER_READY='+self.label,flush=True)
            return self
        except BaseException:
            stop_owned(self.name,self.cell,self.proc);self.log.close();raise
    def __exit__(self,*args):
        stop_owned(self.name,self.cell,self.proc)
        if self.log:self.log.close()

def evaluate(label,depth,split,weights=None):
    with Server(label,depth,weights=weights) as server:
        client=SimpleNamespace(out=server.cell,rows=[])
        canary=api.request(client,'canary','What is 19 + 23? Reply with only the integer.',32)
        if canary['content'].strip()!='42' or canary['finish_reason']!='stop':raise RuntimeError('Functional canary failed')
        api.request(client,'warmup',records(split)[0]['messages'][0]['content'],128)
        measured=[]; checks={r['id']:r['checks'] for r in [json.loads(line) for line in (RUN/(split+'-checks.jsonl')).read_text().splitlines()]}
        for record in records(split):
            ident=record['prompt_id'];row=api.request(client,ident,record['messages'][0]['content'],512)
            source=row['content'].strip();fenced=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',source,flags=re.DOTALL)
            if fenced:source=fenced.group(1)
            row['functional']=probe.sandbox(source,checks[ident]) if row['finish_reason']=='stop' else {'pass':False,'reason':'output_budget_exhausted'}
            save(server.cell/(ident+'-result.json'),row);measured.append(row)
            print('TEST_REQUEST_DONE='+json.dumps(dict(cell=label,id=ident,completed=len(measured),total=len(records(split)),decode_tps=row['decode_tps'],functional_pass=row['functional']['pass'])),flush=True)
        accepted=tune.aggregate(measured) if depth else None
        if not depth and any(native.metric_total(r['metric_deltas'],'spec_decode_num_drafts_total') for r in measured):raise RuntimeError('No-spec control drafted tokens')
        report=dict(status='completed',requests=len(measured),accepted=accepted,median_decode_tps=statistics.median(r['decode_tps'] for r in measured),median_e2e_tps=statistics.median(r['usage']['completion_tokens']/r['elapsed_s'] for r in measured),functional_pass=sum(r['functional']['pass'] for r in measured),functional_total=len(measured),finish_reasons=[r['finish_reason'] for r in measured])
        save(server.cell/'summary.json',report);print('EVAL_COMPLETED='+label+' '+json.dumps(report),flush=True)
        return report

def capture(splits=('train','dev'),label='teacher-capture'):
    if (RUN/label).exists():raise RuntimeError('Capture cell already exists; use a fresh owned cell before loading the teacher')
    for split in splits:
        if (RUN/('capture-'+split)).exists():raise RuntimeError('Capture output exists; preserve incomplete attempt before restarting')
        for record in records(split):
            native.validate_record(dict(record,split='train' if split=='train' else 'heldout'),512)
    with Server(label,4,capture=True) as server:
        config=server.cell/'native-launch-config.json'
        save(config,dict(capture=True,speculative_tokens=4,prefix_caching=True,thinking=False,max_requests=1024,max_total_tokens=2097152,teacher_repository='Qwen/Qwen3.8-27B',teacher_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',teacher_dtype='bfloat16',teacher_quantization=None))
        rt=native.runtime();finalize=rt.finalize_request
        def finalize_bf16(*args):
            payload=finalize(*args);payload['metadata'].update(source='native-bf16-verifier',teacher_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',teacher_quantization=None);return payload
        rt.finalize_request=finalize_bf16
        for split in splits:
            capture_records=RUN/(split+'-capture-requests.jsonl')
            normalized=[dict(record,split='train' if split=='train' else 'heldout') for record in records(split)]
            capture_records.write_text(''.join(json.dumps(record)+'\n' for record in normalized))
            # Only the native capture split label changes; canonical IDs/messages stay immutable.
            for record in normalized:native.validate_record(record,512)
            args=SimpleNamespace(server_config=config,capture_dir=server.cell/'features',output=RUN/('capture-'+split),base_url=probe.BASE,model='qwen38',synthetic=False,records=capture_records,trace_split=None,max_tokens=512,sequence_limit=2048,min_response_tokens=1,max_total_tokens=2097152,max_requests=1024,timeout=300,measure=False)
            counts=native.generate(args,rt)
            if counts['requests']!=len(records(split)) or counts['skipped_long']:raise RuntimeError('Public capture incomplete; do not silently train a smaller corpus')
            print('CAPTURE_COMPLETED='+split+' '+json.dumps(counts),flush=True)
            folder=RUN/('capture-'+split)/('train' if split=='train' else 'heldout')
            if len(list(folder.glob('*.pt')))!=len(records(split)):
                raise RuntimeError('Physical capture count differs from full corpus; refuse reduced-data training')

def training():
    common=['--model',str(MODEL),'--train-dir',str(RUN/'capture-train/train'),'--eval-dir',str(RUN/'capture-dev/heldout'),'--recursive-depth','4','--roots','8','--depth-weights','1','1','0.8','0.8','--max-length','2048','--grad-accum','1','--logits-chunk','64','--device','cuda','--seed','42','--lr','0.000001']
    for name,steps,interval in [('cuda-smoke',1,1),('training',3740,374)]:
        dest=RUN/name;dest.mkdir(mode=0o700)
        budget=['--epochs','10'] if name=='training' else ['--steps',str(steps)]
        argv=[sys.executable,str(CODE/'scripts/experiments/qwen38_train_mtp.py'),*common,*budget,'--checkpoint-every',str(interval),'--output',str(dest/'tuned-mtp.safetensors')]
        save(dest/'command.json',argv);print('TRAINING_START='+name+' UPDATES='+str(steps),flush=True)
        with (dest/'train.log').open('w') as log:
            with subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                for line in process.stdout:
                    log.write(line); log.flush(); print(line,end='',flush=True)
                if process.wait()!=0: raise RuntimeError('Head training failed; retained train.log')
        result=json.loads((dest/'tuned-mtp.json').read_text())
        if result['optimizer_steps']!=steps or len(result['train_steps'])!=steps:raise RuntimeError('Optimizer update accounting failed')
        if result['precision']['regime']!='native_bf16' or any(c['metrics']['RTN_effective_dense'] is not None for c in result['checkpoints']):raise RuntimeError('Incorrect precision regime')
        from safetensors.torch import load_file
        stock=load_file(str(dest/'tuned-mtp.step0000.safetensors'));trained=load_file(str(dest/'tuned-mtp.safetensors'))
        changed=[key for key in stock if not trainer.runtime().torch.equal(stock[key],trained[key])]
        if not changed:raise RuntimeError('No BF16 exported parameter changed after optimizer updates')
        save(dest/'update-verification.json',dict(optimizer_steps=steps,changed_mtp_keys=changed,trainable_parameters=sum(v.numel() for v in trained.values()),no_target_model_constructed=True))
        print('TRAINING_COMPLETED='+name+' UPDATES='+str(steps),flush=True)
    selected=result['dev_selection']
    shutil.copyfile(selected['path'],RUN/'training/selected-mtp.safetensors')
    save(RUN/'training/selected-checkpoint.json',selected)
    print('DEV_SELECTED_CHECKPOINT='+json.dumps(selected),flush=True)
    archive('training')
    items=json.loads((UPLOAD/'backup-private.json').read_text())
    for name in ['tuned-mtp.safetensors','selected-mtp.safetensors']:
        shutil.copyfile(RUN/'training'/name,UPLOAD/name)
        save(UPLOAD/'phase-private.json',[next(x for x in items if x['path']==name)])
        subprocess.run(['python3',str(CODE/'r2-transfer.py'),'--run-dir',str(UPLOAD),'--manifest',str(UPLOAD/'phase-private.json'),'--upload'],check=True)

def drift():
    torch=trainer.runtime().torch;checkpoint=trainer.Checkpoint(MODEL)
    embedding,head=trainer.frozen_heads(checkpoint,'cuda');files=sorted((RUN/'capture-dev/heldout').glob('*.pt'))[:4];reports={}
    for label,weights in [('stock',RUN/'training/tuned-mtp.step0000.safetensors'),('candidate',RUN/'training/selected-mtp.safetensors')]:
        from safetensors.torch import load_file
        model=trainer.build_native_mtp(checkpoint.config,load_file(str(weights)),'cuda');model.eval();values={str(i):[] for i in range(1,5)}
        with torch.no_grad():
            for file in files:
                record=trainer.load_record(file,checkpoint.config,2048)
                roots=[r for r in trainer.sample_roots(record,4,8,random.Random(42)) if r+4<record['target_last_hidden_states'].shape[0]]
                if not roots:raise RuntimeError('No observed teacher-state roots for drift diagnosis')
                outputs=trainer.sequence_depths(model,embedding,record,'cuda',4,roots)
                for d,(hidden,_,_) in enumerate(outputs,1):
                    predicted=hidden[roots] if d==1 else hidden
                    actual=record['target_last_hidden_states'][[r+d for r in roots]].to('cuda')
                    p=predicted.float();a=actual.float()
                    cosine=torch.nn.functional.cosine_similarity(p,a,dim=-1);l2=(p-a).norm(dim=-1);relative=l2/a.norm(dim=-1).clamp_min(1e-12)
                    with trainer.autocast('cuda'):
                        pl=head(predicted);al=head(actual)
                    target=torch.softmax(al.float(),dim=-1);kl=(target*(torch.log_softmax(al.float(),dim=-1)-torch.log_softmax(pl.float(),dim=-1))).sum(dim=-1)
                    for j in range(len(roots)):values[str(d)].append({'cosine':cosine[j].item(),'l2':l2[j].item(),'relative_l2':relative[j].item(),'predicted_norm':p[j].norm().item(),'teacher_norm':a[j].norm().item(),'next_token_kl_teacher_to_head':kl[j].item()})
        reports[label]={key:{'roots':len(rows),**{metric:statistics.mean(r[metric] for r in rows) for metric in rows[0]}} for key,rows in values.items()}
        del model
    save(RUN/'state-drift.json',dict(mode='teacher-forced native shared-head recursion on four validation captures; not free-running drift or serving acceptance',depths=[1,2,3,4],data_split='dev',results=reports))

def main():
    os.umask(0o077);signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt('terminated')))
    protocol=json.loads((RUN/'pilot-protocol.json').read_text()); assert protocol['training']['optimizer_updates']==3740 and protocol['training']['epochs']==10 and len(records('train'))==374 and len(records('dev'))==len(records('test'))==64
    try:
        baseline={str(depth):evaluate('stock-depth'+str(depth),depth,'dev') for depth in (0,4,8)}
        save(RUN/'stock-baseline.json',baseline);archive('baseline')
        capture();archive('capture')
        training();drift()
        import gc
        gc.collect();trainer.runtime().torch.cuda.empty_cache()  # Release diagnostic allocations before native serving.
        candidate=RUN/'training/selected-mtp.safetensors'; cells={}
        cells['no-spec']=evaluate('no-spec',0,'test')
        pairs=[]
        for depth in (4,8):
            prefix='D'+str(depth)+'-'
            for label,weights in [('A1-stock',None),('B1-candidate',candidate),('B2-candidate',candidate),('A2-stock',None)]:
                cells[prefix+label]=evaluate(prefix+label,depth,'test',weights)
                archive('evaluation')  # Small per-cell raw results; model/capture archives are already safe.
            pairs.extend([(prefix+'A1-stock',prefix+'A2-stock'),(prefix+'B1-candidate',prefix+'B2-candidate'),(prefix+'A1-stock',prefix+'B1-candidate'),(prefix+'A2-stock',prefix+'B2-candidate'),('no-spec',prefix+'A1-stock'),('no-spec',prefix+'B1-candidate')])
        pairs.extend([('D4-A1-stock','D8-A1-stock'),('D4-B1-candidate','D8-B1-candidate')])
        matches={}
        for a,b in pairs:
            rows=[]
            for record in records('test'):
                key=record['prompt_id']; first=json.loads((RUN/a/(key+'-result.json')).read_text()); second=json.loads((RUN/b/(key+'-result.json')).read_text())
                if first['prompt_token_ids']!=second['prompt_token_ids']:raise RuntimeError('Cross-cell rendered prompt mismatch')
                rows.append(dict(id=key,exact=first['token_ids']==second['token_ids'],first_pass=first['functional']['pass'],second_pass=second['functional']['pass']))
            matches[a+':'+b]=rows
        selected=json.loads((RUN/'training/selected-checkpoint.json').read_text())
        save(RUN/'comparison.json',dict(status='completed',tier='development',stock_baseline=baseline,test_cells=cells,token_matches=matches,dev_selection=selected,promotion=False))
        print('FULL_CORPUS_CUDA_TRAINING_AND_TESTS_COMPLETED_NO_PROMOTION',flush=True)
    except BaseException as error:
        import traceback
        save(RUN/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        archive('final')

if __name__=='__main__':main()
