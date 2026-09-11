#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -euo pipefail
[ -z "$(docker ps -q)" ]
docker run --pull=never --rm -i --name b70-profile-support --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/renderD128)" -v /tmp/b70-step-profile-overlay-support:/output -e PYTHONPATH=/output -e B70_STEP_TIMING=1 -e B70_STEP_TIMING_OUT=/output/timing.json -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 --entrypoint /opt/venv/bin/python vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4 -P - <<'PY'
import torch,json,pathlib
import qwen38_step_timing_overlay as mod
overlay=mod.install()
a=torch.randn(256,256,device='xpu'); b=torch.randn_like(a)
for _ in range(3): c=a@b
torch.xpu.synchronize()
g=torch.xpu.XPUGraph()
with torch.xpu.graph(g): c=a@b
for _ in range(3): g.replay()
torch.xpu.synchronize()
overlay.start(kind='support')
for _ in range(5): g.replay()
overlay.stop(reason='support_complete')
d=json.loads(pathlib.Path('/output/timing.json').read_text());print(json.dumps(d,indent=2))
PY
SH
