#!/usr/bin/env python3
"""LEAD ONLY: one diagnostics-off A0/C0 pair, then restore; no fidelity analysis."""
import argparse
import json
import runpy
import signal
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
ROOT, OLD, P = C['ROOT'], C['OLD'], C['P']
load, save = C['load'], C['save']
HTTP_LIMIT = 4 * 1024**2


def artifact(stem, suffix):
    return Path(str(stem) + suffix)


def exchange(endpoint, payload, stem, opener=urllib.request.urlopen):
    """Exact wire bytes, status, headers, partial failures; no inference retries."""
    wire = json.dumps(payload).encode()
    save(artifact(stem, '.request.json'), payload)
    artifact(stem, '.request-body').write_bytes(wire)
    request = urllib.request.Request(OLD['URL'] + endpoint, data=wire,
                                    headers={'Content-Type': 'application/json'})
    save(artifact(stem, '.http-request.json'), dict(url=request.full_url, method=request.get_method(),
                                                  headers=dict(request.header_items())))
    record = dict(status=None, headers=[], complete=False)
    with artifact(stem, '.http-body').open('xb') as output:
        try:
            try:
                response = opener(request, timeout=1200)
            except urllib.error.HTTPError as exc:
                response = exc  # Keep error headers/body before failing closed.
            with response:
                record.update(status=response.status, headers=list(response.headers.items()))
                total = 0
                while True:
                    block = response.read(min(65536, HTTP_LIMIT + 1 - total))
                    if not block:
                        break
                    output.write(block)
                    output.flush()
                    total += len(block)
                    assert total <= HTTP_LIMIT, 'HTTP evidence safety cap exceeded'
                record.update(complete=True, bytes=total)
                assert response.status == 200, ('HTTP failure', response.status)
        except BaseException as exc:
            record['error'] = repr(exc)
            raise
        finally:
            save(artifact(stem, '.http.json'), record)
    return artifact(stem, '.http-body')


def public_tokenize(payload, stem, send=exchange):
    template = C['L']['apply_request'](payload)
    rendered = json.loads(send('/apply-template', template, artifact(stem, '.template')).read_bytes())
    save(artifact(stem, '.template.response.json'), rendered)
    tokenize = C['L']['tokenize_request'](rendered)
    response = json.loads(send('/tokenize', tokenize, artifact(stem, '.tokenize')).read_bytes())
    save(artifact(stem, '.tokenize.response.json'), response)
    return C['L']['token_ids'](response)


def check_capacity(out, label, index, jobs):
    record = C['capacity'](ROOT, jobs)
    save(out / f'capacity-{label}-{index:02}.json', record)
    assert record['fits'], ('insufficient disk; no cleanup/reduced reserve', record)
    return record


def serving_pid(cell, label, phase, command):
    record = json.loads(command('docker', 'exec', C['PREFIX'] + label.lower(),
                                'python3', '-B', '-c', C['PID_PROBE']))
    save(cell / ('serving-pid-' + phase + '.json'), record)
    C['check_pid'](record, label)


def require(condition):
    assert condition
    return True


def restore(out, original, before_hw, created, stopped, command, ready, api, hardware):
    """Attempt every restoration check even if one fails; stopped containers stay."""
    errors, records = [], {}

    def attempt(name, fn):
        try:
            value = fn()
            records[name] = True
            return value
        except BaseException as exc:
            errors.append(dict(action=name, error=repr(exc)))

    for label in created:
        name, cell = C['PREFIX'] + label.lower(), out / label
        attempt('stop-' + label, lambda name=name: command('docker', 'stop', '-t', '90', name))
        log = attempt('logs-' + label, lambda name=name: command('docker', 'logs', '--timestamps', name))
        if log is not None:
            attempt('save-log-' + label, lambda cell=cell, log=log: (cell / 'server.log').write_text(log))
        container = attempt('inspect-' + label, lambda name=name: json.loads(command('docker', 'inspect', name))[0])
        if container is not None:
            attempt('save-inspect-' + label, lambda cell=cell, container=container: save(cell / 'container.json', container))
    if stopped:
        attempt('start-original', lambda: command('docker', 'start', OLD['ORIGINAL']))
        attempt('ready', lambda: ready(api, OLD['ORIGINAL']))
        smoke = attempt('smoke', lambda: api['chat'](OLD['URL'], 'Reply with exactly this word and nothing else: READY',
                                                    out / 'restored-smoke', count=32))
        if smoke is not None:
            attempt('save-smoke', lambda: save(out / 'restored-smoke.result.json', smoke))
            attempt('smoke-check', lambda: require(smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'))
    live = attempt('inspect-original', lambda: json.loads(command('docker', 'inspect', OLD['ORIGINAL']))[0])
    if live is not None:
        attempt('save-original', lambda: save(out / 'restored-container.json', live))
        attempt('exact-restore', lambda: P['check_restored'](original, live))
    attempt('health', lambda: require(api['request'](OLD['URL'], '/health')['status'] == 'ok'))
    attempt('sole-live', lambda: require(set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {OLD['ORIGINAL']}))
    hashes = attempt('source-binaries', C['N']['check_sources_and_binaries'])
    if hashes is not None:
        attempt('save-hashes', lambda: save(out / 'hash-checks-after.json', hashes))
    attempt('endpoint-source', C['source_contracts'])
    after_hw = attempt('hardware', hardware)
    if after_hw is not None:
        attempt('save-hardware', lambda: save(out / 'hardware-after.json', after_hw))
        attempt('hardware-check', lambda: require(after_hw == before_hw))
    attempt('capacity-after', lambda: check_capacity(out, 'restored', 0, []))
    save(out / 'restoration-checks.json', dict(passed=not errors, errors=errors, checks=list(records)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-run-one-pair', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host' and Path(__file__).resolve().parent == ROOT
    tested = C['check_tested_code']()
    source = C['source_contracts']()
    hashes = C['N']['check_sources_and_binaries']()
    jobs = C['frozen_jobs']()
    initial = C['capacity'](ROOT, jobs)
    assert initial['fits'], initial
    # Exclusivity before ANY live action, including restoration smoke.
    assert not (ROOT / 'run-r1').exists()
    helper = runpy.run_path(str(OLD['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(OLD['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD['OLD'] / 'probe-api.py'))
    command, ready = helper['command'], helper['ready']
    before_hw = runner['hardware']()
    assert before_hw == load(C['REFERENCE'] / 'run-r1/hardware-before.json')
    original = json.loads(command('docker', 'inspect', OLD['ORIGINAL']))[0]
    assert original['Id'] == P['ORIGINAL_ID']
    P['check_restored'](load(C['REFERENCE'] / 'run-r1/original-container.json'), original)
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {OLD['ORIGINAL']}
    assert api['request'](OLD['URL'], '/health')['status'] == 'ok'
    existing = command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    assert not any(C['PREFIX'] + label.lower() in existing for label in C['ARMS'])
    base = load(OLD['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json')
    assert base[-1] == original['Config']['Cmd'][-1]
    for label in ('S0', 'C0'):
        capture = C['REFERENCE'] / 'run-r1' / label / 'capture'
        assert load(capture.parent / 'create-argv.json') == P['make_create'](base, label, runner['IMAGE'], capture, OLD)
    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)
    for name, value in (('tested-code', tested), ('original-container', original), ('hardware-before', before_hw),
                        ('hash-checks-before', hashes), ('source-contracts', source), ('frozen-requests', jobs),
                        ('initial-capacity', initial)):
        save(out / (name + '.json'), value)
    (out / 'declaration.md').write_bytes((ROOT / 'README.md').read_bytes())
    summary = dict(status='prepared', restored=False, order=list(C['ARMS']), completed=[],
                   tier='development-fidelity', analysis_pending=True, timing_permitted=False,
                   promotion_permitted=False, raw_vector_claimed=False, fresh_route_equivalence_claimed=False)
    created, stopped = [], False

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped = True
        command('docker', 'stop', '-t', '90', OLD['ORIGINAL'])
        for label in C['ARMS']:
            check_capacity(out, label, 0, jobs)
            assert C['N']['check_sources_and_binaries']() == hashes
            cell = out / label
            cell.mkdir()
            for directory in ('quality', 'public', 'short', 'long'):
                (cell / directory).mkdir()
            argv = C['make_create'](label)
            save(cell / 'create-argv.json', argv)
            name = C['PREFIX'] + label.lower()
            created.append(label)
            command(*argv, timeout=60)
            command('docker', 'start', name)
            save(cell / 'models.json', ready(api, name))
            startup = command('docker', 'logs', '--timestamps', name)
            (cell / 'native-and-startup.log').write_text(startup)
            save(cell / 'startup.json', C['check_startup'](startup, label))
            serving_pid(cell, label, 'before', command)
            public = []
            for i, job in enumerate(jobs):
                check_capacity(out, label + '-public', i, jobs)
                ids = public_tokenize(job['payload'], cell / 'public' / f'{i:02}')
                public.append(dict(ordinal=i, stem=job['stem'], prompt_ids=ids, prompt_tokens=len(ids)))
            save(cell / 'public/summary.json', public)
            # No A0 response normalization/comparison can block C0. Only native,
            # environment, HTTP/public-contract, source, and capacity failures can.
            for i, job in enumerate(jobs):
                check_capacity(out, label, i + 1, jobs[i:])
                exchange('/v1/chat/completions', job['payload'], cell / job['stem'])
            serving_pid(cell, label, 'after', command)
            command('docker', 'stop', '-t', '90', name)
            log = command('docker', 'logs', '--timestamps', name)
            (cell / 'server.log').write_text(log)
            C['check_off_log'](log)
            summary['completed'].append(label)
        summary['status'] = 'captured'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        raise
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        errors = restore(out, original, before_hw, created, stopped, command, ready, api, runner['hardware'])
        summary['restored'] = not errors
        summary['restoration_errors'] = errors
        if errors:
            summary['status'] = 'failed'
        save(out / 'summary.json', summary)
        assert not errors, errors
    print('BOLT_OFF_PAIR_CAPTURED_RESTORED_ANALYSIS_PENDING_NO_TIMING', flush=True)


if __name__ == '__main__':
    main()
