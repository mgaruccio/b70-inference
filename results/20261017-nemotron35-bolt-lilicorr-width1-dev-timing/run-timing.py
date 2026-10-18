#!/usr/bin/env python3
"""LEAD ONLY: O1,A1,C1,C2,A2,O2; restore before any statistical analysis."""
import argparse
import hashlib
import json
import re
import runpy
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
ROOT, ORDER, POINTS = C['ROOT'], C['ORDER'], C['POINTS']
save, load = C['save'], C['load']


def artifact(stem, suffix):
    return Path(str(stem) + suffix)


def stream_chat(url, payload, stem, normalize, opener=urllib.request.urlopen, clock=time.monotonic):
    """probe-api.py::chat timing semantics, exact frozen payload + partial evidence.

    Not a direct chat() call: that helper rebuilds payloads and loses partial SSE
    on exceptions. Disk persistence is outside the client timing interval here.
    """
    assert payload['stream'] is True and payload['stream_options'] == {'include_usage': True}
    wire = json.dumps(payload).encode()
    save(artifact(stem, '.request.json'), payload)
    artifact(stem, '.request-body').write_bytes(wire)
    request = urllib.request.Request(url + '/v1/chat/completions', data=wire,
                                    headers={'Content-Type': 'application/json'})
    save(artifact(stem, '.http-request.json'), dict(url=request.full_url, method=request.get_method(),
                                                  headers=dict(request.header_items())))
    raw, events, total, first = [], [], 0, None
    record = dict(status=None, headers=[], complete=False)
    result = None
    started = clock()
    try:
        try:
            response = opener(request, timeout=1200)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            record.update(status=response.status, headers=list(response.headers.items()))
            for line in response:
                raw.append(line)
                total += len(line)
                assert total <= 4 * 1024**2, 'HTTP evidence cap exceeded'
                if response.status != 200 or not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    continue  # Include complete stream consumption; reject trailing data in normalize.
                row = json.loads(data)
                at = clock() - started
                events.append(dict(elapsed_s=at, body=row))
                if first is None and any(c.get('delta', {}).get('content') for c in row.get('choices', [])):
                    first = at
        elapsed = clock() - started
        record.update(complete=True, bytes=total)
        assert response.status == 200, ('HTTP failure', response.status)
        result = normalize(b''.join(raw), payload)  # Requires DONE, finish and final usage.
        result.update(elapsed_s=elapsed, ttfc_s=first)
        C['timing_metrics'](result)
        return result
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        # Even interrupted or malformed streams are retained, never replaced.
        record.update(bytes=total, elapsed_s=locals().get('elapsed', clock() - started), ttfc_s=first)
        artifact(stem, '.http-body').write_bytes(b''.join(raw))
        save(artifact(stem, '.http.json'), record)
        save(artifact(stem, '.events.json'), events)
        if result is not None:
            save(artifact(stem, '.result.json'), result)


def sampled_request(sampler, fn, stem, clock=time.monotonic):
    started = clock()
    try:
        sampler.start()
        return fn()
    finally:
        # Includes persistence overhead, deliberately NOT the client timing window.
        try:
            sampler.stop()
        finally:
            save(artifact(stem, '.telemetry.json'), C['telemetry'](sampler.samples, clock() - started))


class Runtime:
    def __init__(self, h, out):
        self.h, self.out, self.off = h, out, h['off']
        self.command = h['grouped']['command']
        self.original_name = self.off['OLD']['ORIGINAL']
        self.original = load(C['OFF'] / 'run-r1/original-container.json')
        self.contract = C['original_contract'](self.original, h)
        self.created, self.prefixes = [], {}
        self.capacity_index = 0

    def inspect(self, name):
        return json.loads(self.command('docker', 'inspect', name))[0]

    def names(self):
        return set(self.command('docker', 'ps', '--format', '{{.Names}}').splitlines())

    def capacity(self, tag):
        record = C['capacity']()
        save(self.out / 'capacity' / f'{self.capacity_index:03}-{tag}.json', record)
        self.capacity_index += 1
        assert record['fits'], ('insufficient free space; no cleanup/retry', record)

    def logs(self, name):
        # Preserve separate stream ordering for exact prefix subtraction.
        proc = subprocess.run(['docker', 'logs', '--timestamps', name], capture_output=True, timeout=90, check=True)
        return dict(stdout=proc.stdout, stderr=proc.stderr)

    def log_window(self, cell):
        name = self.original_name if cell.startswith('O') else C['PREFIX'] + cell.lower()
        return C['fresh_logs'](self.prefixes.get(cell, dict(stdout=b'', stderr=b'')), self.logs(name))

    def save_logs(self, cell, phase):
        streams = self.log_window(cell)
        for key, data in streams.items():
            (self.out / cell / (phase + '.' + key + '.log')).write_bytes(data)
        return (streams['stdout'] + streams['stderr']).decode('utf-8', errors='replace')

    def ownership(self, expected):
        # Existing fuser mechanism, root visibility; Docker alone is not isolation.
        assert self.command('sudo', '-n', 'id', '-u').strip() == '0', 'cannot inspect all host DRM clients'
        devices = sorted(str(p) for p in Path('/dev/dri').glob('*') if p.name.startswith(('card', 'renderD')))
        assert devices
        proc = subprocess.run(['sudo', '-n', 'fuser', '-v', *devices], capture_output=True, text=True, timeout=15)
        assert proc.returncode in (0, 1), proc.stderr
        assert not re.search(r'(?i)denied|cannot|not found|failed|error', proc.stderr), proc.stderr
        pids = {int(s) for s in proc.stdout.split()}
        want = set() if expected is None else {self.inspect(expected)['State']['Pid']}
        record = dict(devices=devices, pids=sorted(pids), allowed_pids=sorted(want), stderr=proc.stderr,
                      returncode=proc.returncode, observation='root fuser snapshot; not continuous ownership proof')
        assert pids <= want, ('competing host GPU client', record)
        if expected is not None:
            assert pids == want, ('expected GPU owner not visible', record)
        return record

    def telemetry_mapping(self):
        card = Path('/sys/class/drm/card0/device')
        hwmon = card / 'hwmon/hwmon2'
        pci = card.resolve().name
        identity = self.command('lspci', '-Dnn', '-s', pci)
        assert 'B70' in identity and 'Intel' in identity, ('card0 not verified B70', identity)
        assert (card / 'vendor').read_text().strip() == '0x8086'
        assert (card / 'driver').resolve().name == 'xe'
        assert hwmon.resolve().parent.parent == card.resolve()
        assert (hwmon / 'name').read_text().strip() == 'xe'
        sampler = self.h['sampler']['GpuSampler'](interval_s=0.2)
        paths = {k: getattr(sampler, k) for k in ('energy1_path', 'energy2_path', 'freq0_path', 'freq1_path', 'temp_path')}
        assert paths['energy1_path'] == str(hwmon / 'energy1_input')
        assert paths['energy2_path'] == str(hwmon / 'energy2_input')
        assert paths['temp_path'] == str(hwmon / 'temp2_input')
        assert paths['freq0_path'] == str(card / 'tile0/gt0/freq0/cur_freq')
        assert paths['freq1_path'] == str(card / 'tile0/gt1/freq0/cur_freq')
        values = {k: self.h['sampler']['read_text'](v) for k, v in paths.items()}
        assert values['temp_path'], 'thermal observation unavailable; do not invent stability'
        return dict(pci_identity=identity, pci=pci, hwmon_name='xe', paths=paths, values=values,
                    missing=[k for k, v in values.items() if not v], interval_s=0.2,
                    thermal_assumption='no enforced equilibrium; record drift, balanced fresh loads, no waiting loop')

    def observe(self, tag, expected):
        record = dict(hardware=self.h['runner']['hardware'](), telemetry=self.telemetry_mapping(),
                      ownership=self.ownership(expected), live=sorted(self.names()))
        save(self.out / (tag + '.json'), record)
        assert record['hardware'] == self.before_hw
        assert record['live'] == ([] if expected is None else [expected])
        return record

    def preflight(self):
        self.capacity('initial')
        save(self.out / 'tested-code.json', C['check_frozen'](self.h))
        self.hashes = self.off['N']['check_sources_and_binaries']()
        self.before_hw = self.h['runner']['hardware']()
        assert self.before_hw == load(self.off['REFERENCE'] / 'run-r1/hardware-before.json')
        live = self.inspect(self.original_name)
        self.off['P']['check_restored'](self.original, live)
        self.original = live
        assert self.h['api']['request'](self.off['OLD']['URL'], '/health')['status'] == 'ok'
        existing = self.command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
        assert not any(C['PREFIX'] + cell.lower() in existing for cell in ORDER if C['label'](cell))
        save(self.out / 'original-container.json', self.original)
        save(self.out / 'hash-checks-before.json', self.hashes)
        save(self.out / 'hardware-before.json', self.before_hw)
        save(self.out / 'source-contracts.json', C['source_evidence'](self.h))
        for name, path, passed in (
            ('model', self.off['OLD']['OLD'] / 'model-verified.json', 'pass_'),
            ('draft', self.off['P']['WIDTH1'] / 'run-r1/draft-verified.json', 'pass')):
            record = load(path)
            assert record[passed] and Path(record['path']).stat().st_size == record['bytes']
            save(self.out / (name + '-provenance.json'), dict(record, verification='prior full hash; current size, no weight rewrite'))
        env = dict(kernel=self.command('uname', '-a'), os_release=Path('/etc/os-release').read_text(),
                   cpu=self.command('lscpu'), memory=Path('/proc/meminfo').read_text(),
                   driver=self.command('modinfo', 'xe'), source_ids={str(s): (s / 'UPSTREAM_COMMIT').read_text()
                       for s in (self.off['N']['ACCEPTED'], self.off['CANDIDATE'])})
        save(self.out / 'environment.json', env)
        self.observe('run-before', self.original_name)

    def begin(self, cell):
        self.capacity(cell)
        self.observe(cell + '-before', self.original_name if cell == 'O1' else None)
        assert self.off['N']['check_sources_and_binaries']() == self.hashes
        directory = self.out / cell
        directory.mkdir()
        if cell.startswith('O'):
            if cell == 'O1':
                self.command('docker', 'stop', '-t', '90', self.original_name)
            assert not self.inspect(self.original_name)['State']['Running']
            self.ownership(None)
            before = self.logs(self.original_name)
            assert self.logs(self.original_name) == before, 'stopped original log prefix not drained'
            self.prefixes[cell] = before
            save(directory / 'stopped-log-prefix.json', {k: dict(bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for k, v in before.items()})
            name = self.original_name
        else:
            name = C['PREFIX'] + cell.lower()
            argv = C['make_create'](cell, self.off)
            save(directory / 'create-argv.json', argv)
            self.created.append(cell)  # Account for a create that succeeds before transport failure.
            self.command(*argv, timeout=60)
        self.command('docker', 'start', name)
        save(directory / 'models.json', self.h['grouped']['ready'](self.h['api'], name))
        startup = self.save_logs(cell, 'native-and-startup')
        if cell.startswith('O'):
            checked = C['check_original_log'](startup, self.contract)
            self.off['P']['check_restored'](self.original, self.inspect(name))
        else:
            checked = self.off['check_startup'](startup, C['label'](cell))
        save(directory / 'startup.json', checked)
        self.pid(cell, 'before')
        self.observe(cell + '-ready', name)
        save(directory / 'startup-buffers.json', [line for line in startup.splitlines()
              if any(s in line.lower() for s in ('buffer size', 'kv cache size', 'recurrent'))])
        save(directory / 'container-before.json', self.inspect(name))

    def pid(self, cell, phase):
        name = self.original_name if cell.startswith('O') else C['PREFIX'] + cell.lower()
        record = json.loads(self.command('docker', 'exec', name, 'python3', '-B', '-c', self.off['PID_PROBE']))
        save(self.out / cell / ('pid-' + phase + '.json'), record)
        if cell.startswith('O'):
            assert record['argv'] == self.contract['server_argv']
            assert record['exe'] == '/src/build-sycl/bin/llama-server'
            env = dict(record['diagnostics'])
            assert 'BOLT_VERIFY_CAPTURE_DIR' not in env and 'LLAMA_TRACE' not in env
            assert env.get('GGML_SYCL_DEBUG', '0') == env.get('BOLT_TRACE_TENSORS', '0') == '0'
            assert record['scoped_tmp_rows'] == []
        else:
            self.off['check_pid'](record, C['label'](cell))
        maps = self.command('docker', 'exec', name, 'cat', '/proc/1/maps')
        (self.out / cell / ('libraries-' + phase + '.txt')).write_text('\n'.join(sorted({s.split()[-1] for s in maps.splitlines() if '.so' in s})) + '\n')

    def public(self, cell, job):
        self.capacity(cell + '-public')
        stem = self.out / cell / 'public' / job['stem']
        stem.parent.mkdir(parents=True, exist_ok=True)
        ids = self.h['http']['public_tokenize'](job['payload'], stem)
        assert ids == self.off['reference_prompt'](C['label'](cell) or 'A0', job)
        return ids

    def prevalidate(self, cell):
        pre, points = C['jobs'](self.off)
        checks = []
        for job in pre:
            ids = self.public(cell, job)
            self.capacity(cell + '-quality')
            stem = self.out / cell / job['stem']
            stem.parent.mkdir(parents=True, exist_ok=True)
            raw = self.h['http']['exchange']('/v1/chat/completions', job['payload'], stem).read_bytes()
            result = self.h['analysis']['normalize'](raw, job['payload'])
            save(artifact(stem, '.result.json'), result)
            assert result['usage']['prompt_tokens'] == len(ids) and result['timings']['cache_n'] == 0
            functional = (self.off['L']['functional_anchor'](result) if job['stem'] == 'anchor' else
                          self.off['L']['functional_quality'](job['stem'].split('/')[1], result))
            ref = C['reference'](self.h, C['label'](cell) or 'A0', job)[1]
            checks.append(dict(stem=job['stem'], functional=functional,
                               equals_candidate_reference=C['view'](result) == C['view'](ref), actual=C['view'](result)))
            save(self.out / cell / 'quality-checks.json', checks)
            assert functional, ('quality/anchor failure', cell, job['stem'])
        for point in POINTS:
            ordinary, diagnostic = (points[point][k] for k in ('ordinary', 'diagnostic'))
            self.public(cell, ordinary)
            # O identity/repeatability is not an assumption; no O diagnostic requirement.
            if C['label'](cell):
                ids = self.public(cell, diagnostic)
                self.capacity(cell + '-diagnostic')
                stem = self.out / cell / diagnostic['stem']
                stem.parent.mkdir(parents=True, exist_ok=True)
                raw = self.h['http']['exchange']('/v1/chat/completions', diagnostic['payload'], stem).read_bytes()
                actual = self.off['diagnostic_view'](json.loads(raw), ids)
                ref_raw, ref = C['reference'](self.h, C['label'](cell), diagnostic)
                expected = self.off['diagnostic_view'](json.loads(ref_raw), ids)
                checked = dict(actual=actual, expected=expected, passed=actual == expected)
                # Counts can vary: fidelity is IDs/pieces/text/finish/usage, not drafting efficiency.
                checked['passed'] = all(actual[k] == expected[k] for k in ('ids', 'pieces', 'text', 'finish_reason', 'usage'))
                save(artifact(stem, '.check.json'), checked)
                assert checked['passed'], 'candidate diagnostic IDs differ from diagnostics-off reference'
                assert C['view'](ref) == C['view'](C['reference'](self.h, C['label'](cell), ordinary)[1])
        save(self.out / cell / 'prevalidation.json', dict(passed=True, quality_count=20, anchor_tokens=8238,
                                                       actual_completion_ids_only_for_candidate_diagnostics=True))

    def request(self, cell, point, phase, ordinal):
        self.capacity(cell + '-' + phase)
        job = C['jobs'](self.off)[1][point]['ordinary']
        stem = self.out / cell / point / f'{phase}-{ordinal:02}'
        stem.parent.mkdir(parents=True, exist_ok=True)
        fn = lambda: stream_chat(self.off['OLD']['URL'], job['payload'], stem, self.h['analysis']['normalize'])
        result = (sampled_request(self.h['sampler']['GpuSampler'](interval_s=0.2), fn, stem)
                  if phase == 'measured' else fn())
        ref = C['reference'](self.h, C['label'](cell) or 'A0', job)[1]
        C['check_result'](result, point, ref if C['label'](cell) else None)
        save(artifact(stem, '.check.json'), dict(passed=True, equals_candidate_reference=C['view'](result) == C['view'](ref)))

    def end(self, cell):
        name = self.original_name if cell.startswith('O') else C['PREFIX'] + cell.lower()
        self.pid(cell, 'after')
        self.observe(cell + '-after', name)
        assert self.off['N']['check_sources_and_binaries']() == self.hashes
        if cell != 'O2':
            self.command('docker', 'stop', '-t', '90', name)
        log = self.save_logs(cell, 'server')
        if C['label'](cell):
            self.off['check_off_log'](log)
        else:
            C['check_original_log'](log, self.contract)
        save(self.out / cell / 'container-after.json', self.inspect(name))

    def restore(self):
        errors, checks = [], []
        def attempt(name, fn):
            try:
                value = fn()
                checks.append(name)
                return value
            except BaseException as exc:
                errors.append(dict(action=name, error=repr(exc)))
        for cell in self.created:
            name = C['PREFIX'] + cell.lower()
            attempt('stop-' + cell, lambda name=name: self.command('docker', 'stop', '-t', '90', name))
            attempt('logs-' + cell, lambda cell=cell: self.save_logs(cell, 'restored-final'))
            attempt('inspect-' + cell, lambda cell=cell, name=name: save(self.out / cell / 'preserved-container.json', self.inspect(name)))
        for cell in self.prefixes:
            attempt('original-log-' + cell, lambda cell=cell: self.save_logs(cell, 'restoration-boundary'))
        original = attempt('inspect-original', lambda: self.inspect(self.original_name))
        if original is None or not original['State']['Running']:
            attempt('start-original', lambda: self.command('docker', 'start', self.original_name))
        attempt('ready', lambda: self.h['grouped']['ready'](self.h['api'], self.original_name))
        def smoke():
            result = self.h['api']['chat'](self.off['OLD']['URL'], 'Reply with exactly this word and nothing else: READY', self.out / 'restored-smoke', count=32)
            save(self.out / 'restored-smoke.result.json', result)
            assert result['text'].strip() == 'READY' and result['finish_reason'] == 'stop'
        attempt('smoke', smoke)
        attempt('independent-live-restoration', lambda: self.verify_restored(self.out / 'restoration-live.json'))
        attempt('capacity-after', lambda: self.capacity('restored'))
        save(self.out / 'restoration.json', dict(passed=not errors, checks=checks, errors=errors))
        return errors

    def verify_restored(self, destination):
        record, errors = dict(independently_live=True), []
        def attempt(key, fn):
            try:
                record[key] = fn()
            except BaseException as exc:
                errors.append(dict(check=key, error=repr(exc)))
        def original():
            live = self.inspect(self.original_name)
            self.off['P']['check_restored'](load(self.out / 'original-container.json'), live)
            return live
        def health():
            result = self.h['api']['request'](self.off['OLD']['URL'], '/health')
            assert result['status'] == 'ok'
            return result
        def sole():
            result = self.names()
            assert result == {self.original_name}
            return sorted(result)
        def hardware():
            result = self.h['runner']['hardware']()
            assert result == load(self.out / 'hardware-before.json')
            return result
        def hashes():
            result = self.off['N']['check_sources_and_binaries']()
            assert result == load(self.out / 'hash-checks-before.json')
            return result
        for key, fn in (('container', original), ('health', health), ('sole_live', sole),
                        ('hardware', hardware), ('hashes', hashes),
                        ('ownership', lambda: self.ownership(self.original_name)),
                        ('telemetry', self.telemetry_mapping)):
            attempt(key, fn)
        record.update(passed=not errors, errors=errors)
        save(destination, record)
        assert not errors, errors
        return record


def capture(runtime, out):
    summary = dict(status='prepared', order=list(ORDER), completed=[], measured_attempts=[],
                   restored=False, tier='development', analysis_pending=True, timing_permitted=True,
                   promotion_permitted=False, raw_vector_equality_claimed=False)
    try:
        runtime.preflight()
        for cell in ORDER:
            runtime.begin(cell)
            runtime.prevalidate(cell)
            for job in (j for j in C['schedule']() if j['cell'] == cell):
                if job['phase'] == 'measured':
                    summary['measured_attempts'].append(job)
                    save(out / 'summary.json', summary)
                runtime.request(**job)
            runtime.end(cell)
            summary['completed'].append(cell)
        summary['status'] = 'captured'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            save(out / 'failed-command.json', dict(argv=exc.cmd, returncode=exc.returncode,
                 stdout=str(exc.stdout), stderr=str(exc.stderr)))
    finally:
        # Only real runner installs handlers; CPU fixtures never change signals.
        if isinstance(runtime, Runtime):
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
        try:
            errors = runtime.restore()
        except BaseException as exc:
            errors = [dict(action='restore-unhandled', error=repr(exc))]
        summary.update(restored=not errors, restoration_errors=errors)
        if errors:
            summary['status'] = 'failed'
        save(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-run-six-cells', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host' and Path(__file__).resolve().parent == ROOT
    h = C['helpers']()
    C['check_frozen'](h)
    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)  # No resume, retries or replacement run hidden here.
    (out / 'declaration.md').write_bytes((ROOT / 'README.md').read_bytes())
    def interrupted(signum, frame):
        raise InterruptedError(f'interrupted by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    summary = capture(Runtime(h, out), out)
    assert summary['status'] == 'captured' and summary['restored'], summary
    print('SIX_CELLS_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)


if __name__ == '__main__':
    main()
