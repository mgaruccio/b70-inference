#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -eu
r=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-mtp-quant-aware-tiny
test -z "$(docker ps -q)"
gid=$(stat -c '%g' /dev/dri/renderD128)
docker run --rm -i --name mtp-frozen-diagnosis --network none --ipc=host --device /dev/dri --group-add "$gid" -v /dev/dri:/dev/dri:ro -v "$r:/experiment:ro" -v /home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 -e OMP_NUM_THREADS=4 --entrypoint python3 vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f - <<'PY'
import sys,json,pathlib,torch,gc
sys.path.insert(0,'/experiment/source')
import qwen38_train_mtp as t
from safetensors.torch import load_file
p=pathlib.Path('/experiment'); cp=t.Checkpoint('/model');device='xpu'
embedding,head=t.frozen_heads(cp,device)
files={split:sorted((p/'dataset'/split).glob('*.pt')) for split in ('train','heldout')}
for variant in ('stock','tuned'):
 for packed in (False,True):
  state=load_file(str(p/(variant+'-mtp.safetensors')))
  if packed:
   state={k:(t.rtn_effective_lm_head(v,device='cpu',row_chunk=512).float() if v.ndim==2 else v.float()) for k,v in state.items()}
  model=t.build_native_mtp(cp.config,state,device);model.eval();del state
  for split,paths in files.items():
   loss=0.;count=0;correct=0
   with torch.no_grad():
    for path in paths:
     record=t.load_record(path,cp.config,1024)
     h,labels,mask=t.sequence_hidden(model,embedding,record,device)
     idx=mask.nonzero(as_tuple=True)[0]
     for ids in idx.split(32):
      with t.autocast(device):logits=head(h.index_select(0,ids))
      loss+=torch.nn.functional.cross_entropy(logits.float(),labels[ids],reduction='sum').item()
      correct+=(logits.argmax(-1)==labels[ids]).sum().item();count+=ids.numel()
      del logits
     del record,h,labels,mask,idx
   print(json.dumps({'variant':variant,'core':'RTN_effective_dense' if packed else 'BF16_export_FP16_compute','split':split,'sequences':len(paths),'tokens':count,'ce':loss/count,'argmax_agreement':correct/count,'correct':correct}),flush=True)
  del model;gc.collect();torch.xpu.empty_cache()
print('DONE: frozen offline diagnostic; dense dequant GEMM is not the actual serving INT4 kernel.',flush=True)
PY
SH
