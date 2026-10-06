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
    if (HOME/'qwen-fidelity.log').exists():shutil.copyfile(HOME/'qwen-fidelity.log',RUN/'fidelity-driver.log')
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
    def __init__(self,label,depth,capture=False,weights=None,replay=False,eager=None):
        assert not (replay and (capture or depth)), 'Target-only replay cannot speculate/native-capture'
        self.replay=replay; self.eager=label.startswith('parity-') if eager is None else eager
        self.label=label;self.depth=depth;self.capture=capture;self.weights=weights;self.cell=RUN/label;self.cell.mkdir(mode=0o700)
        self.name='qwen-cuda-mtp-'+label.lower();self.proc=None;self.log=None
    def __enter__(self):
        if docker('ps','-q').strip():raise RuntimeError('GPU host must be idle; do not stop unrelated workloads')
        argv=['sudo','-n','docker','run','--rm','--name',self.name,'--gpus','all','--ipc','host','-p','127.0.0.1:8000:8000','-v',str(MODEL)+':/model:ro','-v',str(CODE/'patches')+':/patches:ro','-v',str(self.cell)+':/profile','-e','VLLM_USE_V2_MODEL_RUNNER=0','-e','VLLM_ENABLE_CUDA_COMPATIBILITY=1']
        if self.weights:argv+=['-v',str(self.weights)+':/mtp.safetensors:ro','-e','B70_MTP_WEIGHTS=/mtp.safetensors']
        if self.capture:
            (self.cell/'features').mkdir(mode=0o700)
            argv+=['-e','B70_MTP_NATIVE_CAPTURE_DIR=/profile/features','-e','B70_MTP_NATIVE_MAX_TOKENS=2097152','-e','B70_MTP_NATIVE_MAX_REQUESTS=1024']
        if self.replay:
            (self.cell/'features').mkdir(mode=0o700)
            argv+=['-e','B70_MTP_CAPTURE_DIR=/profile/features','-e','B70_MTP_CAPTURE_MAX_TOKENS=1048576']
        serve=['vllm','serve','/model','--dtype','bfloat16','--max-model-len','8192','--gpu-memory-utilization','0.92','--kv-cache-dtype','auto','--max-num-seqs','1','--max-num-batched-tokens','2048','--mamba-cache-mode','align','--performance-mode','balanced','--compilation-config','{"cudagraph_capture_sizes":[1,2,4,8,16]}','--no-async-scheduling','--enable-prefix-caching' if self.capture else '--no-enable-prefix-caching','--chat-template-content-format','openai','--default-chat-template-kwargs','{"enable_thinking":false}','--reasoning-parser','qwen3','--served-model-name','qwen38','--language-model-only','--port','8000']
        if self.depth:serve+=['--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':self.depth})]
        serve+=['--gdn-prefill-backend','flashinfer','--mamba-cache-dtype','auto','--mamba-ssm-cache-dtype','auto']
        if self.eager:
            offset=serve.index('--compilation-config'); del serve[offset:offset+2]; serve+=['--enforce-eager']
        if self.label.startswith('parity-'):
            (self.cell/'parity').mkdir(mode=0o700)
            if self.depth:
                argv+=['-e','QWEN38_MTP_PARITY_DIR=/profile/parity','-e','QWEN38_MTP_PARITY_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','-e','QWEN38_MTP_PARITY_IMAGE='+IMAGE]
                if self.depth==4:
                    (self.cell/'parity/native').mkdir(mode=0o700)
                    argv+=['-e','B70_MTP_NATIVE_CAPTURE_DIR=/profile/parity/native','-e','B70_MTP_NATIVE_MAX_TOKENS=2200','-e','B70_MTP_NATIVE_MAX_REQUESTS=2']
        import shlex
        setup='set -e; python3 /patches/patch_mtp_training.py; python3 /patches/patch_mtp_native_capture.py; exec '+shlex.join(serve)
        if self.label.startswith('parity-') and self.depth:setup=setup.replace('exec vllm','python3 /patches/qwen38_mtp_parity.py; exec vllm')
        argv+=['--entrypoint','bash',IMAGE,'-c',setup]
        save(self.cell/'command.json',argv)
        save(self.cell/'config.json',dict(tier='development',image=IMAGE,depth=self.depth,weights=str(self.weights) if self.weights else 'stock',dtype='bfloat16',kv_dtype='auto',quantization=None,context=8192,concurrency=1,capture=self.capture,replay_capture=self.replay,prefix_caching=self.capture,thinking=False,enforce_eager=self.eager,gdn_prefill_backend='flashinfer',mamba_cache_dtype='auto',mamba_ssm_cache_dtype='auto',expected_ssm_state_dtype='bfloat16'))
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

if __name__ == '__main__':
    raise SystemExit('Diagnostic serving library only; no training entrypoint')
