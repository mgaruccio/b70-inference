#!/usr/bin/env python3
"""Lead-only lifecycle, public-boundary, stream, and cleanup machinery."""
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


def stream_chat(url, payload, stem, normalize, opener=urllib.request.urlopen,
                clock=time.monotonic):
    """Stream the exact frozen request and retain raw bytes/events on every exit."""
    assert payload['stream'] is True
    assert payload['stream_options'] == {'include_usage': True}
    wire = json.dumps(payload).encode()
    C['save'](C['artifact'](stem, '.request.json'), payload)
    C['artifact'](stem, '.request-body').write_bytes(wire)
    request = urllib.request.Request(url + '/v1/chat/completions', data=wire,
                                     headers={'Content-Type': 'application/json'})
    C['save'](C['artifact'](stem, '.http-request.json'),
              dict(url=request.full_url, method=request.get_method(),
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
                assert total <= C['HTTP_LIMIT'], 'HTTP evidence cap exceeded'
                if response.status != 200 or not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    continue
                row = json.loads(data)
                at = clock() - started
                events.append(dict(elapsed_s=at, body=row))
                if first is None and any(choice.get('delta', {}).get('content')
                                         for choice in row.get('choices', [])):
                    first = at
        elapsed = clock() - started
        record.update(complete=True, bytes=total)
        assert response.status == 200, ('HTTP failure', response.status)
        result = normalize(b''.join(raw), payload)
        result.update(elapsed_s=elapsed, ttfc_s=first)
        C['timing_metrics'](result)
        return result
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record.update(bytes=total, elapsed_s=locals().get('elapsed', clock() - started),
                      ttfc_s=first)
        C['artifact'](stem, '.http-body').write_bytes(b''.join(raw))
        C['save'](C['artifact'](stem, '.http.json'), record)
        C['save'](C['artifact'](stem, '.events.json'), events)
        if result is not None:
            C['save'](C['artifact'](stem, '.result.json'), result)


def sampled_request(sampler, fn, stem, clock=time.monotonic):
    started = clock()
    try:
        sampler.start()
        return fn()
    finally:
        try:
            sampler.stop()
        finally:
            C['save'](C['artifact'](stem, '.telemetry.json'),
                      C['telemetry'](sampler.samples, clock() - started))


class Runtime:
    """A single exclusive case lifecycle; never creates a second service."""
    def __init__(self, h, out, mode, qualified_widths=()):
        self.h = h
        self.out = Path(out)
        self.mode = mode
        self.qualified_widths = tuple(sorted(int(x) for x in qualified_widths))
        self.off = h['off']
        self.command = h['grouped']['command']
        self.ready = h['grouped']['ready']
        self.original_name = self.off['OLD']['ORIGINAL']
        self.created = []
        self.prefixes = {}
        self.capacity_index = 0
        self.active = None
        self.original = None
        self.before_hw = None
        self.hashes = None
        self.original_was_running = False

    def inspect(self, name):
        return json.loads(self.command('docker', 'inspect', name))[0]

    def names(self):
        return set(self.command('docker', 'ps', '--format', '{{.Names}}').splitlines())

    def logs(self, name):
        proc = subprocess.run(['docker', 'logs', '--timestamps', name],
                              capture_output=True, timeout=90, check=True)
        return dict(stdout=proc.stdout, stderr=proc.stderr)

    def save_logs(self, cell, phase, name=None):
        name = name or (self.original_name if cell == 'ORIGINAL' else C['PREFIX'] + cell.lower())
        streams = self.logs(name)
        if cell in self.prefixes:
            streams = C['fresh_logs'](self.prefixes[cell], streams)
        for key, data in streams.items():
            (self.out / cell / (phase + '.' + key + '.log')).write_bytes(data)
        return (streams['stdout'] + streams['stderr']).decode('utf-8', errors='replace')

    def capacity(self, tag):
        record = C['capacity'](self.out)
        C['save'](self.out / 'capacity' / f'{self.capacity_index:03}-{tag}.json', record)
        self.capacity_index += 1
        assert record['fits'], ('insufficient free space; no cleanup/retry', record)
        return record

    def ownership(self, expected):
        assert self.command('sudo', '-n', 'id', '-u').strip() == '0'
        devices = sorted(str(path) for path in Path('/dev/dri').glob('*')
                         if path.name.startswith(('card', 'renderD')))
        assert devices
        proc = subprocess.run(['sudo', '-n', 'fuser', '-v', *devices],
                              capture_output=True, text=True, timeout=15)
        assert proc.returncode in (0, 1), proc.stderr
        assert not re.search(r'(?i)denied|cannot|not found|failed|error', proc.stderr), proc.stderr
        pids = {int(value) for value in proc.stdout.split()}
        wanted = set() if expected is None else {self.inspect(expected)['State']['Pid']}
        record = dict(devices=devices, pids=sorted(pids), allowed_pids=sorted(wanted),
                      stderr=proc.stderr, returncode=proc.returncode,
                      observation='root fuser snapshot; not continuous ownership proof')
        C['save'](self.out / 'ownership' / f'{len(list((self.out / "ownership").glob("*.json"))):03}.json', record)
        assert pids <= wanted, ('competing host GPU client', record)
        if expected is not None:
            assert pids == wanted, ('expected GPU owner not visible', record)
        return record

    def telemetry_mapping(self):
        card = Path('/sys/class/drm/card0/device')
        hwmon = card / 'hwmon/hwmon2'
        pci = card.resolve().name
        identity = self.command('lspci', '-Dnn', '-s', pci)
        assert 'B70' in identity and 'Intel' in identity, identity
        assert (card / 'vendor').read_text().strip() == '0x8086'
        assert (card / 'driver').resolve().name == 'xe'
        assert hwmon.resolve().parent.parent == card.resolve()
        assert (hwmon / 'name').read_text().strip() == 'xe'
        sampler = self.h['sampler']['GpuSampler'](interval_s=0.2)
        paths = {key: getattr(sampler, key) for key in
                 ('energy1_path', 'energy2_path', 'freq0_path', 'freq1_path', 'temp_path')}
        assert paths['energy1_path'] == str(hwmon / 'energy1_input')
        assert paths['energy2_path'] == str(hwmon / 'energy2_input')
        assert paths['temp_path'] == str(hwmon / 'temp2_input')
        assert paths['freq0_path'] == str(card / 'tile0/gt0/freq0/cur_freq')
        assert paths['freq1_path'] == str(card / 'tile0/gt1/freq0/cur_freq')
        values = {key: self.h['sampler']['read_text'](path) for key, path in paths.items()}
        assert all(values.values()), 'thermal/frequency/energy mapping unavailable'
        return dict(pci_identity=identity, pci=pci, hwmon_name='xe', paths=paths,
                    values=values, missing=[key for key, value in values.items() if not value],
                    interval_s=0.2,
                    thermal_assumption='no enforced equilibrium; record drift only')

    def observe(self, tag, expected):
        record = dict(hardware=self.h['runner']['hardware'](),
                      telemetry=self.telemetry_mapping(),
                      ownership=self.ownership(expected), live=sorted(self.names()))
        C['save'](self.out / (tag + '.json'), record)
        assert record['hardware'] == self.before_hw
        expected_live = [] if expected is None else [expected]
        assert record['live'] == expected_live, (tag, record['live'], expected_live)
        return record

    def preflight(self):
        self.capacity('initial')
        C['check_tested'](self.h)
        self.hashes = C['source_binary_hashes'](self.h)
        self.before_hw = self.h['runner']['hardware']()
        frozen_hw = C['load'](self.off['REFERENCE'] / 'run-r1/hardware-before.json')
        assert self.before_hw == frozen_hw, 'hardware/capability snapshot differs from accepted prerequisite'
        frozen_original = C['load'](self.off['ROOT'] / 'run-r1/original-container.json')
        live = self.inspect(self.original_name)
        assert C['same_container'](frozen_original, live)
        assert live['State']['Running']
        self.original = live
        self.original_was_running = True
        assert self.h['api']['request'](self.off['OLD']['URL'], '/health')['status'] == 'ok'
        existing = self.command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
        assert not any(name.startswith(C['PREFIX']) for name in existing)
        C['save'](self.out / 'original-container.json', self.original)
        C['save'](self.out / 'hash-checks-before.json', self.hashes)
        C['save'](self.out / 'hardware-before.json', self.before_hw)
        C['save'](self.out / 'source-contracts.json', C['source_evidence'](self.h))
        env = dict(kernel=self.command('uname', '-a'),
                   os_release=Path('/etc/os-release').read_text(),
                   cpu=self.command('lscpu'), memory=Path('/proc/meminfo').read_text(),
                   driver=self.command('modinfo', 'xe'),
                   source_ids={str(source): (source / 'UPSTREAM_COMMIT').read_text()
                               for source in (self.off['N']['ACCEPTED'], self.off['N']['CANDIDATE'])})
        C['save'](self.out / 'environment.json', env)
        C['save'](self.out / 'original-live.json', live)
        self.observe('run-before', self.original_name)

    def _stop_original_for_case(self):
        live = self.inspect(self.original_name)
        assert C['same_container'](self.original, live)
        if live['State']['Running']:
            self.command('docker', 'stop', '-t', '90', self.original_name)
        assert not self.inspect(self.original_name)['State']['Running']
        self.ownership(None)
        prefix = self.logs(self.original_name)
        assert self.logs(self.original_name) == prefix, 'stopped original log prefix was drained'
        self.prefixes['ORIGINAL'] = prefix
        self.out.joinpath('ORIGINAL').mkdir(exist_ok=True)
        for key, value in prefix.items():
            (self.out / 'ORIGINAL' / ('stopped-prefix.' + key + '.log')).write_bytes(value)
        C['save'](self.out / 'ORIGINAL' / 'stopped-log-prefix.json',
                  {key: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                   for key, value in prefix.items()})

    def pid(self, cell, phase):
        name = C['PREFIX'] + cell.lower()
        record = json.loads(self.command('docker', 'exec', name, 'python3', '-B', '-c', C['PID_PROBE']))
        C['save'](self.out / cell / ('pid-' + phase + '.json'), record)
        C['check_pid'](record, cell, self.h)
        maps = self.command('docker', 'exec', name, 'cat', '/proc/1/maps')
        (self.out / cell / ('libraries-' + phase + '.txt')).write_text(
            '\n'.join(sorted({line.split()[-1] for line in maps.splitlines() if '.so' in line})) + '\n')
        return record

    def begin(self, cell):
        assert self.active is None
        self.capacity(cell)
        if not self.prefixes:
            self._stop_original_for_case()
        else:
            assert not self.inspect(self.original_name)['State']['Running']
            self.ownership(None)
        name = C['PREFIX'] + cell.lower()
        argv = C['make_create'](cell, self.h)
        cell_dir = self.out / cell
        cell_dir.mkdir()
        C['save'](cell_dir / 'create-argv.json', argv)
        C['save'](cell_dir / 'argv-diff.json', C['argv_diff'](
            self.off['make_create'](C['cell_label'](cell)), argv))
        self.created.append(cell)
        self.command(*argv, timeout=60)
        self.command('docker', 'start', name)
        C['save'](cell_dir / 'models.json', self.ready(self.h['api'], name))
        startup = self.save_logs(cell, 'native-and-startup', name)
        label = C['cell_label'](cell)
        self.off['check_startup'](startup, label)
        assert 'BOLT_VERIFY_SERVER_START' not in startup
        startup_contract = dict(label=label, width=C['width_for'](cell),
                                evidence='PID1 argv below; startup text is retained, not forged')
        pid_record = self.pid(cell, 'before')
        startup_contract['pid_argv'] = pid_record['argv']
        startup_contract['pid_width_evidence'] = 'actual /proc/1/cmdline'
        C['save'](cell_dir / 'startup.json', startup_contract)
        self.observe(cell + '-ready', name)
        C['save'](cell_dir / 'startup-buffers.json',
                  [line for line in startup.splitlines()
                   if any(word in line.lower() for word in ('buffer size', 'kv cache size', 'recurrent'))])
        C['save'](cell_dir / 'container-before.json', self.inspect(name))
        self.active = cell

    def _reference(self, job):
        stem = self.off['ROOT'] / 'run-r1/A0' / job['stem']
        raw = self.h['analysis']['read_http'](stem, job['payload'], '/v1/chat/completions')
        if C['qualification_kinds'](job) == 'diagnostic':
            return raw, self.off['diagnostic_view'](json.loads(raw),
                                                       self.off['reference_prompt']('A0', job))
        return raw, self.h['analysis']['normalize'](raw, job['payload'])

    def public(self, cell, job):
        stem = self.out / cell / 'public' / job['stem']
        stem.parent.mkdir(parents=True, exist_ok=True)
        try:
            ids = self.h['http']['public_tokenize'](job['payload'], stem)
        except BaseException:
            raise
        expected = self.off['reference_prompt']('A0', job)
        comparison = C['first_difference'](ids, expected)
        C['save'](C['artifact'](stem, '.prompt-comparison.json'),
                  dict(actual_ids=ids, expected_ids=expected, comparison=comparison,
                       retokenized_output_as_ids_claimed=False))
        if ids != expected:
            raise C['FidelityFailure']('apply-template/tokenize prompt IDs differ from frozen A')
        return ids

    def _exchange(self, payload, stem):
        return self.h['http']['exchange']('/v1/chat/completions', payload, stem).read_bytes()

    def run_job(self, cell, job, expected_for_compare=True):
        ids = self.public(cell, job)
        stem = self.out / cell / job['stem']
        stem.parent.mkdir(parents=True, exist_ok=True)
        raw = self._exchange(job['payload'], stem)
        kind = C['qualification_kinds'](job)
        ref_raw, reference = self._reference(job)
        if kind == 'diagnostic':
            actual = self.off['diagnostic_view'](json.loads(raw), ids)
            expected = reference
            comparison = C['diagnostic_compare'](actual, expected)
            C['save'](C['artifact'](stem, '.diagnostic-view.json'), actual)
            C['save'](C['artifact'](stem, '.diagnostic-comparison.json'),
                      dict(actual=actual, expected=expected, comparison=comparison,
                           raw_f32_equality_claimed=False,
                           probabilities_compared=False,
                           actual_ids_source='choices[0].logprobs.content[*].id'))
            if expected_for_compare and not comparison['passed']:
                raise C['FidelityFailure']('diagnostic actual IDs/pieces/text/finish/usage differ')
            return dict(kind=kind, actual=actual, expected=expected,
                        comparison=comparison,
                        speculation=C['check_speculation_counts'](actual))
        result = self.h['analysis']['normalize'](raw, job['payload'])
        C['save'](C['artifact'](stem, '.result.json'), result)
        comparison = dict(equal=C['result_view'](result) == C['result_view'](reference),
                          actual=C['result_view'](result), expected=C['result_view'](reference),
                          raw_f32_equality_claimed=False)
        C['save'](C['artifact'](stem, '.comparison.json'), comparison)
        C['check_output'](result, job, reference if expected_for_compare else None, ids)
        functional = C['check_functional'](self.h, job, result)
        if not functional:
            raise C['FidelityFailure']('functional quality/stop/length gate failed')
        return dict(kind=kind, actual=C['result_view'](result), expected=C['result_view'](reference),
                    functional=functional, comparison=comparison, result=result)

    def run_qualification_cell(self, cell):
        jobs = C['qualification_jobs'](self.h)
        report = dict(cell=cell, width=C['width_for'](cell), passed=False,
                      baseline=(cell == 'A'), jobs=[], errors=[], fidelity_failures=[],
                      behavior=dict(natural_stop_cases=0, fixed_length_cases=0,
                                    diagnostics_exercised=0, rejected_draft_observed=0,
                                    not_exercised=[]),
                      no_synthetic_acceptance=True,
                      raw_f32_equality_claimed=False)
        for job in jobs:
            row = dict(stem=job['stem'], kind=C['qualification_kinds'](job), passed=False)
            try:
                result = self.run_job(cell, job)
                row.update(result, passed=True)
                if row['kind'] in ('quality', 'anchor'):
                    report['behavior']['natural_stop_cases'] += 1
                elif row['kind'] == 'ordinary':
                    report['behavior']['fixed_length_cases'] += 1
                elif row['kind'] == 'diagnostic':
                    report['behavior']['diagnostics_exercised'] += 1
                    if result['speculation']['rejected_draft_observed']:
                        report['behavior']['rejected_draft_observed'] += 1
                report['jobs'].append(row)
            except C['FidelityFailure'] as exc:
                row.update(passed=False, failure=repr(exc))
                report['jobs'].append(row)
                report['fidelity_failures'].append(dict(stem=job['stem'], error=repr(exc)))
            except BaseException as exc:
                row.update(passed=False, error=repr(exc))
                report['jobs'].append(row)
                report['errors'].append(dict(stem=job['stem'], error=repr(exc)))
                raise C['InfraFailure'](f'API/infrastructure failure at {job["stem"]}: {exc}') from exc
        if report['behavior']['diagnostics_exercised'] == 0:
            report['behavior']['not_exercised'].append('diagnostic actual IDs')
        if report['width'] is not None and report['behavior']['rejected_draft_observed'] == 0:
            report['behavior']['not_exercised'].append('real rejected draft count')
            report['fidelity_failures'].append(dict(scope='behavior',
                                                    error='no real rejected draft count observed'))
        report['passed'] = (not report['errors'] and not report['fidelity_failures'] and
                            report['behavior']['natural_stop_cases'] == 21 and
                            report['behavior']['fixed_length_cases'] == 2 and
                            (report['width'] is None or
                             report['behavior']['rejected_draft_observed'] > 0))
        C['save'](self.out / cell / 'qualification.json', report)
        if cell == 'A' and not report['passed']:
            raise C['InfraFailure']('fresh no-draft A failed frozen output/functional gate')
        return report

    def prevalidate_timing_cell(self, cell):
        report = self.run_qualification_cell(cell)
        assert report['passed'], ('timed cell quality/diagnostic gate failed', cell, report)
        C['save'](self.out / cell / 'timing-gate.json',
                  dict(passed=True, timing_not_started=True, qualification=report))

    def timed_request(self, cell, point, phase, ordinal):
        job = C['timing_jobs'](self.h)[point]
        stem = self.out / cell / point / f'{phase}-{ordinal:02}'
        stem.parent.mkdir(parents=True, exist_ok=True)
        fn = lambda: stream_chat(self.off['OLD']['URL'], job['payload'], stem,
                                 self.h['analysis']['normalize'])
        if phase == 'measured':
            sampler = self.h['sampler']['GpuSampler'](interval_s=0.2)
            result = sampled_request(sampler, fn, stem)
            telemetry = C['load'](C['artifact'](stem, '.telemetry.json'))
            assert telemetry == C['telemetry'](sampler.samples, telemetry['window_duration_s'])
        else:
            result = fn()
        try:
            C['check_output'](result, job, self._reference(job)[1],
                              self.off['reference_prompt']('A0', job))
        except (AssertionError, C['FidelityFailure']) as exc:
            raise C['InfraFailure'](f'timed response fidelity failure at {cell}/{point}/{phase}-{ordinal}: {exc}') from exc
        C['save'](C['artifact'](stem, '.check.json'),
                  dict(passed=True, warmup_excluded=(phase == 'warmup'),
                       metrics=C['timing_metrics'](result),
                       raw_f32_equality_claimed=False))
        return result

    def run_timing_cell(self, cell):
        self.prevalidate_timing_cell(cell)
        rows = []
        for point in C['POINTS']:
            self.timed_request(cell, point, 'warmup', 0)
            for ordinal in range(3):
                result = self.timed_request(cell, point, 'measured', ordinal)
                stem = self.out / cell / point / f'measured-{ordinal:02}'
                rows.append(dict(cell=cell, point=point, phase='measured', ordinal=ordinal,
                                 artifact=str(stem), passed=True,
                                 actual=C['result_view'](result), timings=result['timings'],
                                 metrics=C['timing_metrics'](result),
                                 equals_no_draft=True))
        C['save'](self.out / cell / 'timing-rows.json', rows)
        return rows

    def end(self, cell):
        name = C['PREFIX'] + cell.lower()
        self.pid(cell, 'after')
        self.observe(cell + '-after', name)
        assert C['source_binary_hashes'](self.h) == self.hashes
        startup = (self.out / cell / 'native-and-startup.stdout.log').read_text(errors='replace') + \
                  (self.out / cell / 'native-and-startup.stderr.log').read_text(errors='replace')
        C['save'](self.out / cell / 'server-duration-stats.json', C['log_duration_stats'](startup))
        C['save'](self.out / cell / 'container-after.json', self.inspect(name))
        self.command('docker', 'stop', '-t', '90', name)
        server = self.save_logs(cell, 'server', name)
        self.off['check_off_log'](server)
        C['save'](self.out / cell / 'server-duration-stats.json', C['log_duration_stats'](server))
        assert not self.inspect(name)['State']['Running']
        self.ownership(None)
        self.active = None

    def verify_restored(self, destination):
        record, errors = dict(independently_live=True), []
        def attempt(key, fn):
            try:
                record[key] = fn()
            except BaseException as exc:
                errors.append(dict(check=key, error=repr(exc)))
        def original():
            live = self.inspect(self.original_name)
            assert C['same_container'](C['load'](self.out / 'original-container.json'), live)
            assert live['State']['Running']
            return live
        def health():
            response = self.h['api']['request'](self.off['OLD']['URL'], '/health')
            assert response['status'] == 'ok'
            return response
        def sole():
            result = self.names()
            assert result == {self.original_name}
            return sorted(result)
        def hardware():
            result = self.h['runner']['hardware']()
            assert result == C['load'](self.out / 'hardware-before.json')
            return result
        def hashes():
            result = C['source_binary_hashes'](self.h)
            assert result == C['load'](self.out / 'hash-checks-before.json')
            return result
        for key, fn in (
            ('container', original), ('health', health), ('sole_live', sole),
            ('hardware', hardware), ('source_binaries', hashes),
            ('ownership', lambda: self.ownership(self.original_name)),
            ('telemetry', self.telemetry_mapping),
        ):
            attempt(key, fn)
        record.update(passed=not errors, errors=errors)
        C['save'](destination, record)
        if errors:
            raise C['InfraFailure'](f'independent restoration check failed: {errors}')
        return record

    def _stop_if_running(self, name):
        live = self.inspect(name)
        if live.get('State', {}).get('Running', False):
            self.command('docker', 'stop', '-t', '90', name)

    def restore(self):
        errors, checks = [], []
        def attempt(name, fn):
            try:
                fn()
                checks.append(name)
            except BaseException as exc:
                record = dict(action=name, error=repr(exc))
                if isinstance(exc, subprocess.CalledProcessError):
                    record.update(argv=exc.cmd, returncode=exc.returncode,
                                  stdout=(exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else exc.stdout),
                                  stderr=(exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else exc.stderr),
                                  output=(exc.output.decode(errors='replace') if isinstance(getattr(exc, 'output', None), bytes) else getattr(exc, 'output', None)))
                errors.append(record)
        for cell in self.created:
            name = C['PREFIX'] + cell.lower()
            attempt('stop-' + cell, lambda name=name: self._stop_if_running(name))
            attempt('logs-' + cell, lambda cell=cell, name=name: self._preserve_restore_log(cell, name))
            attempt('inspect-' + cell, lambda cell=cell, name=name:
                    C['save'](self.out / cell / 'preserved-container.json', self.inspect(name)))
        if self.original_was_running:
            live = None
            try:
                live = self.inspect(self.original_name)
            except BaseException as exc:
                errors.append(dict(action='inspect-original-before-start', error=repr(exc)))
            if live is None or not live.get('State', {}).get('Running', False):
                attempt('start-original', lambda: self.command('docker', 'start', self.original_name))
            attempt('ready-original', lambda: self.ready(self.h['api'], self.original_name))
            def smoke():
                result = self.h['api']['chat'](
                    self.off['OLD']['URL'],
                    'Reply with exactly this word and nothing else: READY',
                    self.out / 'restored-smoke', count=32)
                C['save'](self.out / 'restored-smoke.result.json', result)
                assert result['text'].strip() == 'READY' and result['finish_reason'] == 'stop'
            attempt('smoke-original', smoke)
        attempt('independent-live-restoration',
                lambda: self.verify_restored(self.out / 'restoration-live.json'))
        attempt('capacity-restored', lambda: self.capacity('restored'))
        C['save'](self.out / 'restoration.json', dict(passed=not errors,
                                                       checks=checks, errors=errors))
        return errors

    def _preserve_restore_log(self, cell, name):
        stream = self.logs(name)
        for key, data in stream.items():
            (self.out / cell / ('restored-final.' + key + '.log')).write_bytes(data)


def _save_exception(out, exc):
    if isinstance(exc, subprocess.CalledProcessError):
        def text(value):
            if value is None:
                return None
            return value.decode(errors='replace') if isinstance(value, bytes) else str(value)
        C['save'](Path(out) / 'failed-command.json',
                  dict(argv=exc.cmd, returncode=exc.returncode,
                       stdout=text(exc.stdout), stderr=text(exc.stderr),
                       output=text(getattr(exc, 'output', None))))
    else:
        C['save'](Path(out) / 'failed-exception.json', dict(type=type(exc).__name__, error=repr(exc)))


def capture(runtime, cells, callback):
    out = runtime.out
    summary = dict(status='prepared', mode=runtime.mode, order=list(cells), completed=[],
                   measured_attempts=[], restored=False, analysis_pending=True,
                   timing_permitted=runtime.mode == 'timing', promotion_permitted=False,
                   raw_f32_equality_claimed=False)
    def interrupted(signum, frame):
        raise InterruptedError(f'interrupted by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        runtime.preflight()
        for cell in cells:
            runtime.begin(cell)
            try:
                rows = callback(runtime, cell)
                if runtime.mode == 'timing':
                    summary['measured_attempts'].extend(rows)
            finally:
                runtime.end(cell)
            summary['completed'].append(cell)
            C['save'](out / 'summary.json', summary)
        summary['status'] = 'captured'
    except BaseException as exc:
        failure = exc
        summary.update(status='failed', error=repr(exc))
        _save_exception(out, exc)
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            errors = runtime.restore()
        except BaseException as exc:
            errors = [dict(action='restore-unhandled', error=repr(exc))]
        summary.update(restored=not errors, restoration_errors=errors)
        if errors:
            summary['status'] = 'failed'
        C['save'](out / 'summary.json', summary)
    if failure is not None:
        raise failure
    assert summary['status'] == 'captured' and summary['restored'], summary
    return summary
EOF