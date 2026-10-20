#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-validate}
case "$MODE" in validate|capture|after) ;; *) exit 2;; esac
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no inference-host "bash -s -- $MODE" <<'BASH'
set -euo pipefail
python3 -B - "$1" <<'PY'
import json, runpy, socket, sys
from pathlib import Path
mode = sys.argv[1]
path = Path('/home/mike/b70-evals/20261019-lilicorr-q8-drafter/run-performance.py')
m = runpy.run_path(str(path)); g = m['bind'].__globals__
out = path.parent / 'verification-routes-20261020-r1'
prefix = 'bolt-lilicorr-verification-routes-20261020-r1-'
assert __debug__ and socket.gethostname() == 'inference-host'
assert g['HERE'] == g['ROOT'] == path.parent
g.update(OUT=out, PREFIX=prefix)
runtime, analysis, c = m['bind']()
base_make = c['make_create']
old = 'unset BOLT_VERIFY_CAPTURE_DIR LLAMA_TRACE'
new = 'unset LLAMA_TRACE; export BOLT_VERIFY_CAPTURE_DIR=/tmp/bolt-lilicorr-verification-routes'

def diagnostic_make(cell, h):
    baseline = list(base_make(cell, h))
    derived = list(baseline)
    assert baseline[-1].count(old) == 1
    derived[-1] = baseline[-1].replace(old, new)
    assert derived[:-1] == baseline[:-1]
    assert c['server_argv'](baseline) == c['server_argv'](derived)
    assert c['argv_diff'](baseline, derived) == [dict(index=len(derived)-1, base=baseline[-1], derived=derived[-1])]
    return derived

c['make_create'] = diagnostic_make
p, _k, h = runtime['load_dependencies']()
assert c['make_create'] is diagnostic_make
assert c['OUT'] == out and c['PREFIX'] == prefix and c['ROOT'] == path.parent
assert p['CANDIDATE'] == g['QUALIFIED_SOURCE'], 'qualified ancestry must remain original'
plans = {cell: dict(baseline=base_make(cell,h), diagnostic=diagnostic_make(cell,h), server_argv=c['server_argv'](diagnostic_make(cell,h))) for cell in c['CELLS']}
assert list(plans) == ['VSPF-A1','VSPF-W3a','VSPF-W3b','VSPF-A2']
if mode == 'validate':
    print('CONFIG_VALIDATED_ONLY_NO_GPU_WORK_OR_SERVICE_CHANGE', flush=True)
    print(json.dumps(dict(out=str(out), prefix=prefix, source=str(p['CANDIDATE']), cells=list(plans), only_delta=dict(before=old,after=new)), indent=2), flush=True)
elif mode == 'capture':
    out.mkdir(exist_ok=False)
    c['save'](out/'diagnostic-launch-inputs.json', dict(tier='development diagnostic, no clean speed claim', approved='Run route capture', source=str(p['CANDIDATE']), old_prelude=old, new_prelude=new, plans=plans))
    (out/'declaration.md').write_text('Existing-controller route diagnosis on original Q8. Private OUT/PREFIX plus host-only capture logging export only; source/binaries/models/server argv/native/public/fidelity/restoration gates unchanged. No builds, installations, math changes, W7 or persistent launcher changes. Full A1/W3a/W3b/A2, 25 public jobs/cell, short55/long8270, warmup plus three measured, temp0 seed42 cache_prompt=false max128 ignoreEOS. Timings diagnostic only. Separate-process restoration required before analysis.\n\nPrimary research: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html — host submission is asynchronous; route/host-wall records are not GPU event times.\n')
    result = runtime['capture'](runtime['Runtime'](h,out))
    assert result['status'] == 'captured' and result['restored'], result
    print('VERIFICATION_ROUTE_CAPTURED_AUTOMATICALLY_RESTORED_INDEPENDENT_CHECK_PENDING', flush=True)
else:
    assert mode == 'after'
    restored = runtime['Runtime'](h,out).verify_restored(out/'analysis-live-restoration.json')
    assert restored['passed'], restored
    (out/'analysis-r1').mkdir(exist_ok=False)
    result = analysis['analyze'](out,p,h)
    assert result['status'] == 'analyzed' and result['independently_restored'] and not result['errors'], result.get('errors')
    assert len(result['rows']) == 24
    for row in result['rows']:
        if row['cell'].startswith('VSPF-W3'):
            assert tuple(row['timings'][k] for k in ('draft_n','draft_n_accepted')) == {'short':(129,84),'long':(133,82)}[row['point']]
    print('VERIFICATION_ROUTE_CAPTURE_ANALYZED_NATIVE_100PUBLIC_24MEASURED_EXACT_COUNTS_INDEPENDENT_RESTORATION_PASSED', flush=True)
PY
BASH
