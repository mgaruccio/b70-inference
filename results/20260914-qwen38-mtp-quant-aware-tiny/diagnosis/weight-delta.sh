#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -eu
r=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-mtp-quant-aware-tiny
docker run --rm -i --network none --cpus 2 --memory 8g -e OMP_NUM_THREADS=2 -v "$r:/experiment:ro" --entrypoint python3 vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f - <<'PY'
import ast,json,pathlib,torch
from safetensors import safe_open
p=pathlib.Path('/experiment')
source=p/'source/results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches/patch_draft_mtp_int4.py'
tree=ast.parse(source.read_text()); helper=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HELPER_SOURCE' for t in n.targets)); fn=next(n for n in ast.parse(helper).body if isinstance(n,ast.FunctionDef) and n.name=='quantize_to_int4')
ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
quant=ns['quantize_to_int4']; totals={'parameters':0,'bf16_changed':0,'int4_elements':0,'int4_codes_changed':0,'scales':0,'scales_changed':0}
with safe_open(p/'stock-mtp.safetensors',framework='pt',device='cpu') as a, safe_open(p/'tuned-mtp.safetensors',framework='pt',device='cpu') as b:
 for name in a.keys():
  x=a.get_tensor(name);y=b.get_tensor(name);changed=int((x!=y).sum());n=x.numel()
  row={'name':name,'parameters':n,'bf16_changed':changed}
  totals['parameters']+=n;totals['bf16_changed']+=changed
  if x.ndim==2:
   qx,sx,_,_=quant(x.half());qy,sy,_,_=quant(y.half())
   codes=sum(int((((qx>>shift)&15)!=((qy>>shift)&15)).sum()) for shift in range(0,32,4))
   sc=int((sx!=sy).sum());row.update(int4_codes_changed=codes,int4_codes_changed_pct=100*codes/n,scales_changed=sc,scales=sx.numel())
   totals['int4_elements']+=n;totals['int4_codes_changed']+=codes;totals['scales']+=sx.numel();totals['scales_changed']+=sc
   del qx,qy,sx,sy
  print(json.dumps(row),flush=True)
  del x,y
print('TOTAL='+json.dumps(totals),flush=True)
PY
SH
