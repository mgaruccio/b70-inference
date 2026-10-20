#!/usr/bin/env bash
set -euo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no inference-host 'bash -s' <<'BASH'
set -euo pipefail
python3 -B - <<'PY'
import json,runpy
from pathlib import Path
package=Path('/home/mike/b70-evals/20261020-lilicorr-verification-routes')
a=runpy.run_path(str(package/'run-performance.py'))
runtime,analysis,c=a['bind']();p,k,h=runtime['load_dependencies']()
instance=runtime['Runtime'](h,a['OUT'])
assert instance._trace_p is c['_public']() is p
assert p['CANDIDATE']==a['q8_qualified_source']()
clean=Path('/home/mike/b70-evals/20261019-lilicorr-q8-drafter/timing-20261020-baseline-r1')
oldtrace=Path('/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons/trace-short55-r1')
for cell in a['CELLS']:
    base=a['base_create_argv'](cell,h);derived=c['make_create'](cell,h)
    assert derived[-4]==base[-4] and derived[-3:-1]==['bash','-lc']
    assert derived.index(str(a['capture_path'](cell))+':/capture')<len(derived)-4
    assert c['server_argv'](base)==c['server_argv'](derived)
    baseline=json.loads((clean/cell/'pid-before.json').read_text())
    p['C']['check_pid'](baseline,a['role_for'](cell),h)
    tracecell='VSTB-W3' if a['role_for'](cell)=='W3' else 'VSTB-A'
    traced=json.loads((oldtrace/tracecell/'pid-before.json').read_text())
    a['check_trace_pid'](traced,baseline)
    raw=b''.join((clean/cell/('native-and-startup.raw.'+s+'.log')).read_bytes() for s in ('stdout','stderr'))
    a['check_trace_startup'](raw.decode(),cell,h)
    server=b''.join((oldtrace/tracecell/('server.'+s+'.log')).read_bytes() for s in ('stdout','stderr'))
    a['check_trace_server_log'](server.decode())
    print('REAL_ADAPTER_NATIVE_STARTUP_PID_SERVER_ARGV_ACCEPTED',cell,flush=True)
r1=clean.parent/'verification-routes-20261020-r1/VSPF-A1'
raw=b''.join((r1/('native-and-startup.raw.'+s+'.log')).read_bytes() for s in ('stdout','stderr'))
a['check_trace_startup'](raw.decode(),'VSPF-A1',h)
print('REAL_ADAPTER_VALIDATION_PASSED_NO_STUBS_NO_GPU_WORK_NO_CONTAINER_CREATION_OR_SERVICE_CHANGES',flush=True)
PY
BASH
