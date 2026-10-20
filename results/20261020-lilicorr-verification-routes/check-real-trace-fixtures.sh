#!/usr/bin/env bash
set -euo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no inference-host 'bash -s' <<'BASH'
set -euo pipefail
python3 -B - <<'PY'
import runpy
from pathlib import Path
root=Path('/home/mike/b70-evals/20261019-lilicorr-q8-drafter')
m=runpy.run_path(str(root/'run-performance.py'))
runtime,analysis,c=m['bind']()
p,k,h=runtime['load_dependencies']()
t=runpy.run_path('/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons/trace-probe.py')
checks=[(root/'verification-routes-20261020-r1/VSPF-A1','VSTB-A')]
checks += [(root/'timing-20261020-baseline-r1'/cell,'VSTB-W3' if 'W3' in cell else 'VSTB-A') for cell in c['CELLS']]
for directory,cell in checks:
    raw=b''.join((directory/('native-and-startup.raw.'+stream+'.log')).read_bytes() for stream in ('stdout','stderr'))
    result=t['check_trace_startup'](raw.decode('utf-8'),cell,h)
    assert result['evidence']=='unaltered raw startup and actual trace PID1'
    print('REAL_FROZEN_NATIVE_CALLBACK_STARTUP_ACCEPTED',str(directory),len(raw),'bytes',flush=True)
print('ALL_FIVE_REAL_STARTUP_FIXTURES_ACCEPTED_NO_STUBS_NO_GPU_WORK_NO_SERVICE_CHANGES',flush=True)
PY
BASH
