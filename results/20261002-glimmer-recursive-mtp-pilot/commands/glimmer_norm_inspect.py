import json,struct,statistics
from pathlib import Path
root=Path('/home/mike/inference/models/Muse-Glimmer-30B-GPTQ-Int4-sym-G128')
key='model.language_model.norm.weight'
idx=json.loads((root/'model.safetensors.index.json').read_text())
file=root/idx['weight_map'][key]
with file.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    assert n<16*1024*1024
    h=json.loads(f.read(n))[key]
    assert h['dtype']=='BF16' and h['shape']==[6656]
    start,end=h['data_offsets']; f.seek(8+n+start); data=f.read(end-start)
values=[struct.unpack('<f',struct.pack('<I',x[0]<<16))[0] for x in struct.iter_unpack('<H',data)]
print(json.dumps({'source':str(root),'tensor':key,'dtype':h['dtype'],'shape':h['shape'],'negative_channels':sum(x<0 for x in values),'zero_channels':sum(x==0 for x in values),'min':min(values),'max':max(values),'mean':statistics.mean(values),'values':values}))
