import importlib.util,json
from pathlib import Path
import torch
C=Path.home()/'qwen-mtp-code'; R=Path.home()/'qwen-mtp-run/parity-stock8/parity'
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
p=load(C/'scripts/experiments/qwen38_mtp_live_parity.py','parity_diag'); t=p.module_at(C/'scripts/experiments/qwen38_train_mtp.py','trainer_diag')
c=t.Checkpoint(Path.home()/'qwen-model'); m=t.build_native_mtp(c.config,c.mtp_state(),'cuda').to(torch.bfloat16).eval(); m.requires_grad_(False); e,h=t.frozen_heads(c,'cuda')
key='b70-native-f5d901ac3a9300a83cdb2fa00f627b6d'; control=json.loads((R/(key+'.control.json')).read_text()); response=json.loads((R/(key+'.response.json')).read_text()); traces=[torch.load(x,map_location='cpu',weights_only=True) for x in sorted(R.glob(key+'.*.pt'))]
class Head:
 def __init__(self):self.calls=0
 def __call__(self,x):
  y=h(x); self.calls+=1
  # Three calls per depth: live-input replay, recursive replay, frozen head on native.
  if self.calls in (202,203,204,253,254,255):
   v,i=y.float().topk(5,dim=-1);print(json.dumps({'head_call':self.calls,'top_ids':i.tolist(),'top_logits':v.tolist()}),flush=True)
  return y
with torch.inference_mode():
 result=p.replay(traces,control,response['choices'][0]['token_ids'],t,m,e,Head(),'cuda')
print(json.dumps({'observed_numeric_pass':result['observed_numeric_pass'],'failures':[r for r in result['rows'] if not r['argmax_equal']]}),flush=True)
