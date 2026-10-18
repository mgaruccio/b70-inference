#!/usr/bin/env python3
"""ONE true no-draft capture, lead-only. No build; restoration precedes analysis."""
import hashlib
import json
import re
import runpy
import shlex
import shutil
import signal
import socket
import subprocess
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-nodraft-fidelity')
REFERENCE = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-router-widths-single2'
P = runpy.run_path(str(REFERENCE / 'run-trace.py'))
OLD = runpy.run_path(str(P['WIDTH1'] / 'run-controls.py'))
CANDIDATE = P['CANDIDATE']
ACCEPTED = P['ACCEPTED']
NAME = 'bolt-lilicorr-nodraft-fidelity-r1-a0'
EXPECTED = {
    'llama-server': '22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d',
    'libllama-server-impl.so': '43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede',
    'libggml-sycl.so.0': '64d6a58f17cb92ea9fe6354d9c553f74724565a58fc018d2f4bbd013d090a72e',
    'test-backend-ops': 'bdd86349fac5ab43fa32d465aad27cec7f31dcad0651f7611403d8a783f10c7d',
    'libllama-common.so': '934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812',
}
SOURCE_EXPECTED = {
    'ggml/src/ggml-sycl/ggml-sycl.cpp': '22ca5fdf535a86ecff94dac10e5b56a683f92f6629610d0ff86b76234b62dd89',
    'tests/test-backend-ops.cpp': 'd5aa69f915f03268846b2cba3f6a83981def184be7441b10caab047a5f4c736a',
}
load, save = P['load'], P['save']


def make_create(s0, capture):
    """Pure argv transform; no rebinding of imported function globals."""
    assert capture == ROOT / 'run-r1/A0/capture'
    argv = list(s0)
    assert argv[argv.index('--name') + 1] == P['PREFIX'] + 's0'
    argv[argv.index('--name') + 1] = NAME
    old_capture = str(REFERENCE / 'run-r1/S0/capture') + ':/capture'
    assert argv.count(old_capture) == 1
    argv[argv.index(old_capture)] = str(capture) + ':/capture'
    draft_mount = str(OLD['DRAFT'] / 'draft') + ':/draft:ro'
    assert argv.count(draft_mount) == 1
    i = argv.index(draft_mount)
    assert argv[i - 1] == '-v'
    del argv[i - 1:i + 1]
    suffix = ' ' + shlex.join(P['spec_for_arm']('S0'))
    assert argv[-1].endswith(suffix)
    argv[-1] = argv[-1][:-len(suffix)]
    assert_no_draft_argv(argv)
    return argv


def assert_no_draft_argv(argv):
    prefix, server = argv[-1].rsplit('exec ', 1)
    words = shlex.split(server)
    assert not any('draft' in word.lower() or 'spec-' in word.lower() for word in words)
    assert all(':/draft' not in word for word in argv[:-1])
    assert argv.count(str(CANDIDATE) + ':/src:ro') == 1
    assert argv[argv.index('--name') + 1] == NAME
    assert argv.count(str(ROOT / 'run-r1/A0/capture') + ':/capture') == 1
    for script in ('run-trace.py', 'analyze-trace.py'):
        assert argv.count(str(REFERENCE / script) + ':/iq2-single2-gate/' + script + ':ro') == 1
    assert argv.count(str(OLD['PRIOR'] / 'native-outprod.sh') + ':/native-outprod.sh:ro') == 1
    for key, value in P['ENV']:
        assert argv.count(key + '=' + value) == 1
    for key in ('GGML_SYCL_ENABLE_ESIMD=1', 'GGML_SYCL_ENABLE_FUSION=1', 'GGML_SYCL_ENABLE_GRAPH=0'):
        assert argv.count(key) == 1
    for flag, value in (('--ctx-size', '65536'), ('--parallel', '1'), ('--batch-size', '1024'),
                        ('--ubatch-size', '1024'), ('--cache-type-k', 'q8_0'), ('--cache-type-v', 'q8_0')):
        assert words.count(flag) == 1 and words[words.index(flag) + 1] == value
    assert '--cache-type-r' not in words  # pinned default F32, verified at startup
    assert P['ROUTER_NATIVE_COMMAND'] in prefix and 'bash /native-outprod.sh;' in prefix
    assert P['MARKER'] in prefix


def check_startup(log):
    assert log.count(P['MARKER']) == 1
    native = runpy.run_path(str(P['WIDTH1'] / 'analyze-controls.py'))
    native['parse_native_gate'](log, 'C0')  # native385 + OUTPROD190/support9, NOT an arm label
    for name in ('iq2', 'q8', 'attention', 'attention_view', 'router'):
        P['check_' + name + '_native_gate'](log)
    assert 'call ggml_sycl_ssm_scan_prefill:' not in log.split(P['MARKER'], 1)[0]
    return check_model_startup(log.split(P['MARKER'], 1)[1])


def check_model_startup(model):
    OLD['check_model_log'](model, 'A')
    assert [int(m.group(1)) for m in OLD['N_RS_SEQ_RE'].finditer(model)] == [0]
    for key in ('n_outputs_max', 'n_outputs_max_per_seq'):
        assert re.findall(r'llama_context: ' + key + r'\s*=\s*(\d+)', model) == ['1']
    assert 'no implementations specified for speculative decoding' in model
    # Paths contain "lilicorr" legitimately. Reject activation/load markers, not path names.
    for pattern in (r'adding speculative implementation', r'common_speculative_impl_draft',
                    r'/draft/', r'loading draft', r'load.*draft model', r'draft.*context.*initial'):
        assert not re.search(pattern, model, re.I), pattern
    assert 'R (f32)' in model and 'S (f32)' in model
    for key, value in (('DNN', 1), ('FUSION', 1), ('ESIMD', 1), ('OPT', 1), ('VMM', 1), ('GRAPH', 0)):
        assert f'GGML_SYCL_ENABLE_{key}: {value}' in model
    assert 'BOLT_VERIFY error' not in model
    return dict(label='A0', target_n_rs_seq=0, n_outputs_max=1, n_outputs_max_per_seq=1,
                speculative_implementation=False, draft_loaded=False)


def check_sources_and_binaries():
    protected = load(REFERENCE / 'protected-source-hashes.json')
    proof = load(REFERENCE / 'source-proof.json')
    assert proof['candidate_source'] == SOURCE_EXPECTED
    expected_source = dict(protected['parent'], **SOURCE_EXPECTED)
    assert P['source_hashes'](CANDIDATE) == expected_source, 'candidate source changed'
    assert P['source_hashes'](ACCEPTED) == protected['accepted'], 'accepted source changed'
    assert P['binaries'](CANDIDATE) == EXPECTED == load(REFERENCE / 'candidate-binaries.json')
    accepted = load(REFERENCE / 'accepted-binaries.json')
    assert P['binaries'](ACCEPTED) == accepted
    assert 'BOLT_VERIFY_BUILD_PASSED' in (REFERENCE / 'post-build-verification.log').read_text()
    assert 'BOLT_VERIFY_DEPENDENCIES_PASSED' in (REFERENCE / 'runtime-precheck.log').read_text()
    return dict(candidate_binaries=EXPECTED, accepted_binaries=accepted,
                candidate_source_files=len(expected_source), accepted_source_files=len(protected['accepted']),
                candidate_changed_source=SOURCE_EXPECTED, all_source_hashes_match=True)


def frozen_requests():
    payload = load(REFERENCE / 'run-r1/frozen-payload.json')
    diagnostic = OLD['diagnostic_payload'](payload)
    for label in ('S0', 'C0'):
        cell = REFERENCE / 'run-r1' / label
        assert payload == load(cell / 'ordinary.request.json') == load(cell / 'warmup.request.json')
        assert diagnostic == load(cell / 'diagnostic.request.json')
    assert payload['max_tokens'] == 128 and payload['temperature'] == 0 and payload['seed'] == 42
    assert payload['cache_prompt'] is False and payload['ignore_eos'] is True
    assert diagnostic['backend_sampling'] is False
    return payload, diagnostic


def main():
    assert socket.gethostname() == 'inference-host'
    helper = runpy.run_path(str(OLD['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(OLD['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD['OLD'] / 'probe-api.py'))
    command, ready = helper['command'], helper['ready']
    hashes = check_sources_and_binaries()
    payload, diagnostic = frozen_requests()
    assert shutil.disk_usage(ROOT).free >= P['CAPTURE_RESERVE_BYTES']
    before_hw = runner['hardware']()
    assert before_hw == load(REFERENCE / 'run-r1/hardware-before.json')
    original = json.loads(command('docker', 'inspect', OLD['ORIGINAL']))[0]
    assert original['Id'] == P['ORIGINAL_ID'] and original['State']['Running']
    saved_original = load(REFERENCE / 'run-r1/original-container.json')
    P['check_restored'](saved_original, original)
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {OLD['ORIGINAL']}
    assert NAME not in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    s0 = load(REFERENCE / 'run-r1/S0/create-argv.json')
    # Reconstruct the reference without executing it; detect unexpected command drift.
    base = load(OLD['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json')
    assert base[-1] == original['Config']['Cmd'][-1]
    assert s0 == P['make_create'](base, 'S0', runner['IMAGE'], REFERENCE / 'run-r1/S0/capture', OLD)
    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)  # exactly one fresh launch, never retry/overwrite this run
    cell = out / 'A0'
    cell.mkdir()
    capture = cell / 'capture'
    capture.mkdir(mode=0o755)
    argv = make_create(s0, capture)
    save(cell / 'create-argv.json', argv)
    save(out / 'original-container.json', original)
    save(out / 'hardware-before.json', before_hw)
    save(out / 'hash-checks-before.json', hashes)
    save(out / 'frozen-payload.json', payload)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    (out / 'declaration.md').write_bytes((ROOT / 'README.md').read_bytes())
    summary = dict(status='prepared', order=['A0'], tier='development-fidelity', restored=False,
                   timing_permitted=False, promotion_permitted=False, analysis_pending=True,
                   parent_invariance_required=False)
    stopped = created = False

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped = True
        command('docker', 'stop', '-t', '90', OLD['ORIGINAL'])
        created = True
        command(*argv, timeout=60)
        command('docker', 'start', NAME)
        summary['models'] = ready(api, NAME)
        startup = command('docker', 'logs', NAME)
        (cell / 'native-and-startup.log').write_text(startup)
        summary['startup'] = check_startup(startup)
        (cell / 'quality.log').write_text(command('python3', '-B', str(OLD['OLD'] / 'probe-api.py'),
                                                '--out', str(cell / 'quality'), '--mode', 'quality', timeout=1200))
        for stem in ('warmup', 'ordinary'):
            api['chat'](OLD['URL'], payload['messages'][0]['content'], cell / stem,
                        count=128, stream=True, ignore_eos=True)
        save(cell / 'diagnostic.request.json', diagnostic)
        assert shutil.disk_usage(capture).free >= P['CAPTURE_RESERVE_BYTES']
        pre = command('docker', 'logs', '--timestamps', NAME)
        (cell / 'pre-diagnostic.log').write_text(pre)
        save(cell / 'diagnostic.response.json', api['request'](OLD['URL'], '/v1/chat/completions', diagnostic))
        command('docker', 'stop', '-t', '90', NAME)
        post = command('docker', 'logs', '--timestamps', NAME)
        (cell / 'through-diagnostic.log').write_text(post)
        # Delimit only. All independent parsing/consistency/route/quality work is post-restore.
        assert post.startswith(pre), 'Docker diagnostic log prefix changed'
        (cell / 'diagnostic-window.log').write_text(post[len(pre):])
        summary['status'] = 'captured'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        if hasattr(exc, 'read'):
            (out / 'http-error.txt').write_bytes(exc.read())
        raise
    finally:
        errors = []
        try:
            if created:
                for action in ('stop', 'logs', 'inspect'):
                    try:
                        if action == 'stop':
                            command('docker', 'stop', '-t', '90', NAME)
                        elif action == 'logs':
                            (cell / 'server.log').write_text(command('docker', 'logs', '--timestamps', NAME))
                        else:
                            save(cell / 'container.json', json.loads(command('docker', 'inspect', NAME))[0])
                    except BaseException as exc:
                        errors.append(dict(action=action, error=repr(exc)))
        finally:
            try:
                if stopped:
                    assert NAME not in command('docker', 'ps', '--format', '{{.Names}}').splitlines()
                    command('docker', 'start', OLD['ORIGINAL'])
                    summary['restored_models'] = ready(api, OLD['ORIGINAL'])
                    smoke = api['chat'](OLD['URL'], 'Reply with exactly this word and nothing else: READY',
                                        out / 'restored-smoke', count=32)
                    save(out / 'restored-smoke.result.json', smoke)
                    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
                    live = json.loads(command('docker', 'inspect', OLD['ORIGINAL']))[0]
                    save(out / 'restored-container.json', live)
                    P['check_restored'](original, live)
                    assert api['request'](OLD['URL'], '/health')['status'] == 'ok'
                    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {OLD['ORIGINAL']}
                save(out / 'hash-checks-after.json', check_sources_and_binaries())
                after_hw = runner['hardware']()
                save(out / 'hardware-after.json', after_hw)
                assert after_hw == before_hw
                summary['restored'] = True
                if errors:
                    summary['cleanup_errors'] = errors
                    raise RuntimeError('cleanup evidence incomplete')
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                save(out / 'summary.json', summary)
    print('BOLT_A0_CAPTURED_AND_RESTORED_ANALYSIS_PENDING_NO_TIMING', flush=True)


if __name__ == '__main__':
    main()
