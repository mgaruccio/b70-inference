#!/usr/bin/env python3
"""LEAD ONLY: one fresh A0/C0 pair. Never build, remove containers, or analyze fidelity."""
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


def send(payload, stem):
    """Capture the exact public HTTP response; interpretation is post-restoration."""
    save(stem.with_suffix('.request.json'), payload)
    request = urllib.request.Request(OLD['URL'] + '/v1/chat/completions',
                                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    # Keep partial streams, including failed attempts. Do not regenerate payloads.
    with stem.with_suffix('.http-body').open('xb') as output:
        try:
            with urllib.request.urlopen(request, timeout=1200) as response:
                save(stem.with_suffix('.http.json'), dict(status=response.status, headers=dict(response.headers)))
                while True:
                    block = response.read(65536)
                    if not block:
                        break
                    output.write(block)
                    output.flush()
        except urllib.error.HTTPError as exc:
            output.write(exc.read())
            save(stem.with_suffix('.http.json'), dict(status=exc.code, error=str(exc)))
            raise


def check_capacity(out, label, index, jobs, capture=None):
    record = C['capacity'](ROOT, jobs)
    if capture is not None:
        record['capture'] = C['capture_inventory'](capture)
    save(out / f'capacity-{label}-{index:02}.json', record)
    assert record['fits'], ('insufficient disk; no deletion or reduced reserve permitted', record)
    return record


def restore(out, original, before_hw, created, stopped, command, ready, api, hardware):
    """Best-effort ALL restoration/evidence checks even if an earlier check fails."""
    errors, records = [], {}

    def attempt(name, fn):
        try:
            value = fn()
            records[name] = value
            return value
        except BaseException as exc:
            errors.append(dict(action=name, error=repr(exc)))
            return None

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
            save(out / 'restored-smoke.result.json', smoke)
            attempt('smoke-check', lambda: require(smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'))
    live = attempt('inspect-original', lambda: json.loads(command('docker', 'inspect', OLD['ORIGINAL']))[0])
    if live is not None:
        save(out / 'restored-container.json', live)
        attempt('exact-restore', lambda: P['check_restored'](original, live))
    attempt('health', lambda: require(api['request'](OLD['URL'], '/health')['status'] == 'ok'))
    attempt('sole-live', lambda: require(set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {OLD['ORIGINAL']}))
    hashes = attempt('source-binaries', C['N']['check_sources_and_binaries'])
    if hashes is not None:
        save(out / 'hash-checks-after.json', hashes)
    attempt('endpoint-source', C['source_contracts'])
    after_hw = attempt('hardware', hardware)
    if after_hw is not None:
        save(out / 'hardware-after.json', after_hw)
        attempt('hardware-check', lambda: require(after_hw == before_hw))
    save(out / 'restoration-checks.json', dict(passed=not errors, errors=errors,
                                             checks=list(records), stored_evidence_not_live_health=True))
    return errors


def require(condition):
    assert condition
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-run-one-pair', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host'
    assert Path(__file__).resolve().parent == ROOT
    tested = C['check_tested_code']()
    C['source_contracts']()
    hashes = C['N']['check_sources_and_binaries']()
    jobs = C['frozen_jobs']()
    # Refuse before touching the live server; recheck A0, C0 and every request too.
    initial_capacity = C['capacity'](ROOT, jobs)
    assert initial_capacity['fits'], initial_capacity
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
        old_capture = C['REFERENCE'] / 'run-r1' / label / 'capture'
        assert load(old_capture.parent / 'create-argv.json') == P['make_create'](base, label, runner['IMAGE'], old_capture, OLD)
    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)  # Never overwrite/retry a partial pair.
    save(out / 'tested-code.json', tested)
    save(out / 'original-container.json', original)
    save(out / 'hardware-before.json', before_hw)
    save(out / 'hash-checks-before.json', hashes)
    save(out / 'frozen-requests.json', jobs)
    save(out / 'initial-capacity.json', initial_capacity)
    (out / 'declaration.md').write_bytes((ROOT / 'README.md').read_bytes())
    summary = dict(status='prepared', restored=False, order=list(C['ARMS']), completed=[],
                   tier='development-fidelity', analysis_pending=True, timing_permitted=False,
                   promotion_permitted=False, parent_invariance_required=False)
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
            # No analysis of A0 is required to enter C0. Actual A0 bytes may refuse C0.
            check_capacity(out, label, 0, jobs)
            cell = out / label
            cell.mkdir()
            (cell / 'quality').mkdir()
            (cell / 'public').mkdir()
            capture = cell / 'capture'
            capture.mkdir(mode=0o755)
            argv = C['make_create'](label, capture)
            save(cell / 'create-argv.json', argv)
            name = C['PREFIX'] + label.lower()
            created.append(label)  # include partial docker-create failures in finally
            command(*argv, timeout=60)
            command('docker', 'start', name)
            save(cell / 'models.json', ready(api, name))
            startup = command('docker', 'logs', '--timestamps', name)
            (cell / 'native-and-startup.log').write_text(startup)
            save(cell / 'startup.json', C['check_startup'](startup, label))
            # Render/tokenize every exact request BEFORE this arm's first inference.
            public = []
            for i, job in enumerate(jobs):
                ids = C['public_tokenize'](job['payload'], cell / 'public' / f'{i:02}', api['request'])
                public.append(dict(ordinal=i, stem=job['stem'], prompt_tokens=len(ids),
                                   historical_expected=job.get('historical_prompt_tokens')))
            save(cell / 'public/summary.json', public)
            pre = None
            for i, job in enumerate(jobs):
                check_capacity(out, label, i + 1, jobs[i:], capture)
                if job['stem'] == 'diagnostic':
                    pre = command('docker', 'logs', '--timestamps', name)
                send(job['payload'], cell / job['stem'])
            command('docker', 'stop', '-t', '90', name)
            post = command('docker', 'logs', '--timestamps', name)
            (cell / 'server.log').write_text(post)
            # Even a bad boundary is deferred to analysis; never discard a full pair.
            boundary = dict(prefix_ok=pre is not None and post.startswith(pre),
                            before_bytes=len(pre.encode()) if pre is not None else None,
                            before_sha256=C['hashlib'].sha256(pre.encode()).hexdigest() if pre is not None else None,
                            through_bytes=len(post.encode()))
            save(cell / 'diagnostic-boundary.json', boundary)
            if boundary['prefix_ok']:
                (cell / 'diagnostic-window.log').write_text(post[len(pre):])
            save(cell / 'capture-inventory.json', C['capture_inventory'](capture))
            summary['completed'].append(label)
        summary['status'] = 'captured'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        if hasattr(exc, 'read'):
            (out / 'http-error.txt').write_bytes(exc.read())
        raise
    finally:
        # Repeated signals must not interrupt restoration itself.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        errors = restore(out, original, before_hw, created, stopped, command, ready, api, runner['hardware'])
        summary['restored'] = not errors
        summary['restoration_errors'] = errors
        if errors:
            summary['status'] = 'failed'
        save(out / 'summary.json', summary)
        assert not errors, errors
    print('BOLT_LONG_PAIR_CAPTURED_RESTORED_ANALYSIS_PENDING_NO_TIMING', flush=True)


if __name__ == '__main__':
    main()
