#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -euo pipefail
[ -z "$(docker ps -q)" ]
[ "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000 ]
docker run --pull=never --rm -i --name b70-profile-support --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/renderD128)" -v /tmp/b70-step-profile-support-02:/output -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 --entrypoint /opt/venv/bin/python vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4 -P - <<'PY'
import torch,json,pathlib
out=pathlib.Path('/output'); out.mkdir(exist_ok=True)
a=torch.randn(256,256,device='xpu'); b=torch.randn_like(a)
for _ in range(3): c=a@b
torch.xpu.synchronize()
g=torch.xpu.XPUGraph()
with torch.xpu.graph(g): c=a@b
for _ in range(3): g.replay()
torch.xpu.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.XPU],with_stack=False) as p:
 for _ in range(3):
  with torch.profiler.record_function('b70_graph_replay_support'):
   g.replay()
 torch.xpu.synchronize()
p.export_chrome_trace(str(out/'support.json'))
d=json.loads((out/'support.json').read_text()); events=d['traceEvents']; cats={}
for e in events:
 k=e.get('cat','<none>'); cats[k]=cats.get(k,0)+1
print(json.dumps({'torch':torch.__version__,'device':torch.xpu.get_device_name(),'categories':cats,'kernels':[{'name':e['name'],'dur':e.get('dur'),'args':e.get('args')} for e in events if e.get('cat')=='kernel']},indent=2))
starts=[torch.xpu.Event(enable_timing=True) for _ in range(5)]
ends=[torch.xpu.Event(enable_timing=True) for _ in range(5)]
for start,end in zip(starts,ends):
 start.record(); g.replay(); end.record()
torch.xpu.synchronize()
print('GRAPH_EVENT_MS', [start.elapsed_time(end) for start,end in zip(starts,ends)])
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.XPU],with_stack=False) as ep:
 for _ in range(3):
  with torch.profiler.record_function('b70_eager_support'): c=a@b
 torch.xpu.synchronize()
ep.export_chrome_trace(str(out/'support-eager.json'))
es=json.loads((out/'support-eager.json').read_text())['traceEvents']
print('EAGER_CATEGORIES', {cat:sum(e.get('cat','<none>')==cat for e in es) for cat in sorted({e.get('cat','<none>') for e in es})})
PY
SH
