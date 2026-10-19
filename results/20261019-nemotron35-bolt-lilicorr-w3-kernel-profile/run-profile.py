#!/usr/bin/env python3
"""Two instrumented requests using the existing native/lifecycle/public boundary."""
import argparse
import copy
import hashlib
import json
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-w3-kernel-profile')
Q8 = HERE.parent / '20261019-nemotron35-bolt-lilicorr-w3-batched-q8'
OUT = ROOT / 'capture-r1'
TOOLS = ROOT / 'tools'
CELLS = ('VSPF-A1', 'VSPF-W3a')
PREFIX = 'bolt-lilicorr-kernel-profile-r1-'
TOOL_HASHES = {
    'unitrace': '99ffdad3f3ae86aeee8e52f4b687a9d41e5e67491993952de37c112544a0ff30',
    'libunitrace_tool.so': '4bb8d93a64d9091867babd6e14d3b8ed7647b1f4c649802cc1e8570ad0aa2d04',
}
OWN_FILES = ('README.md', 'build-profiler.sh', 'run-profile.py', 'test-profile.py')
SERVER = 'exec /src/build-sycl/bin/llama-server '
WRAPPED = ('export ZE_ENABLE_TRACING_LAYER=1; '
           'unset UNITRACE_KernelMetrics UNITRACE_ChromeKmdLogging; '
           'exec /tools/unitrace -d -h -s -v --chrome-kernel-logging '
           '--session {session} --start-paused --result-dir /profile/results '
           '/src/build-sycl/bin/llama-server ')


def tool_proof():
    assert (HERE / 'tool-build-exit-code.txt').read_text().strip() == '0'
    actual = {name: hashlib.sha256((TOOLS / name).read_bytes()).hexdigest()
              for name in TOOL_HASHES}
    assert actual == TOOL_HASHES, 'profiler binary drift'
    return actual


def session_name(cell):
    assert cell in CELLS
    return 'boltq8profile1' + cell.replace('-', '')


def collect_request(runtime, c, cell):
    name, session = c['name'](cell), session_name(cell)
    runtime.command('docker', 'exec', name, '/tools/unitrace', '--resume', session, timeout=30)
    try:
        return runtime.timed_request(cell, 'short', 'profile', 0)
    finally:
        runtime.command('docker', 'exec', name, '/tools/unitrace', '--pause', session, timeout=30)


def extra_mounts(cell):
    return ((TOOLS, '/tools', 'ro'), (OUT / 'profiler' / cell, '/profile', 'rw'))


def wrap_argv(argv, cell):
    result = list(argv)
    assert result[-3:-1] == ['bash', '-lc']
    assert result[-1].count(SERVER) == 1
    assert '/tools/unitrace' not in result[-1]
    bindings = [item for source, destination, mode in extra_mounts(cell)
                for item in ('-v', f'{source}:{destination}:{mode}')]
    # The pinned image precedes the shell. Add only these declared binds.
    result[-4:-4] = bindings
    result[-1] = result[-1].replace(SERVER, WRAPPED.format(session=session_name(cell)))
    return result


def expected_reference(reference, cell):
    """Derive expected extra mounts, never change an observed record."""
    result = copy.deepcopy(reference)
    for source, destination, mode in extra_mounts(cell):
        assert not any(m['Destination'] == destination for m in result['Mounts'])
        result['HostConfig']['Binds'].append(f'{source}:{destination}:{mode}')
        result['Mounts'].append(dict(Type='bind', Source=str(source), Destination=destination,
                                     Mode=mode, RW=mode == 'rw', Propagation='rprivate'))
    return result


def bind():
    prior = runpy.run_path(str(Q8 / 'run-performance.py'))
    runtime, _, c = prior['bind']()
    old_make, old_codes, old_hashes = c['make_create'], c['adapter_code_hashes'], c['source_binary_hashes']
    original_load = c['load']
    c.update(ROOT=ROOT, OUT=OUT, PREFIX=PREFIX, CELLS=CELLS)
    c['planned_names'] = lambda: [c['name'](cell) for cell in CELLS]
    c['make_create'] = lambda cell, h: wrap_argv(old_make(cell, h), cell)

    def check(record, cell, h):
        context = dict(c)
        reference_path = c['qualified_dir'](c['role_for'](cell)) / 'container-before.json'
        def load_expected(path):
            assert Path(path) == reference_path
            return expected_reference(original_load(path), cell)
        context['load'] = load_expected
        return prior['check_container'](context, record, cell, h)

    def codes():
        result = old_codes()
        assert result == original_load(Q8 / 'timing-r1/adapter-code-hashes-before.json')
        return dict(result, **{str(HERE / name): c['sha'](HERE / name) for name in OWN_FILES})

    c.update(check_container=check, adapter_code_hashes=codes,
             source_binary_hashes=lambda h: dict(validated_backend=old_hashes(h), profiler=tool_proof()))
    return prior, runtime, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-capture', action='store_true', required=True)
    parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == ROOT
    prior, r, c = bind()
    p, k, h = r['load_dependencies']()
    assert p['CANDIDATE'] == prior['PARENT']
    tool_proof()
    c['adapter_code_hashes']()
    OUT.mkdir(exist_ok=False)
    for cell in CELLS:
        (OUT / 'profiler' / cell).mkdir(parents=True, exist_ok=False)
    rt = r['Runtime'](h, OUT)
    rt.mode = 'profiling'  # Existing lifecycle summary must not permit timing claims.
    paired = {}

    def request(runtime, cell):
        runtime.reference_dir = c['qualified_dir']('A')
        runtime.reference_label = 'frozen qualified VSB-A; backend already qualified'
        result = collect_request(runtime, c, cell)
        view = c['result_view'](result)
        if paired:
            assert view == paired['VSPF-A1'], 'instrumented A/W3 output mismatch'
        paired[cell] = view
        c['save'](OUT / cell / 'profile-check.json', dict(
            passed=True, actual=view, matched_frozen_a=True,
            matched_fresh_a=cell != 'VSPF-A1', timing_permitted=False,
            full_qualification_rerun=False, native_commands_instrumented=False))
        return []

    result = k['capture'](rt, CELLS, request)
    assert result['status'] == 'captured' and result['restored'] and not result['timing_permitted']
    rt.verify_restored(OUT / 'analysis-live-restoration.json')
    files = {}
    for cell in CELLS:
        folder = OUT / 'profiler' / cell
        device = list(folder.rglob('device_timing.txt'))
        chrome = list(folder.rglob('chrome_trace.json'))
        assert device and chrome, ('missing profiler outputs', cell)
        assert all(path.stat().st_size > 0 for path in device + chrome)
        files[cell] = [str(path.relative_to(OUT)) for path in device + chrome]
    c['save'](OUT / 'profiler-files.json', dict(files=files, analysis_pending=True,
                                               timing_permitted=False))
    print('KERNEL_PROFILE_CAPTURED_RESTORED_ATTRIBUTION_ANALYSIS_PENDING', flush=True)


if __name__ == '__main__':
    main()
