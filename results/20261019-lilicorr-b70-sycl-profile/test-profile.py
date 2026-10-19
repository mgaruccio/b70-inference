#!/usr/bin/env python3
"""CPU-only tests for the second diagnostic; no Docker, GPU, network, or remote mutation."""
import copy
import gzip
import hashlib
import json
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / '20261019-lilicorr-b70-profile'
D = runpy.run_path(str(HERE / 'run-profile.py'))
S = D['S']
A = runpy.run_path(str(HERE / 'analyze-profile.py'))
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pins():
    return S['load'](HERE / 'source-hashes.json')


def fixture_bytes(name):
    return gzip.decompress((HERE / 'fixtures' / (name + '.gz')).read_bytes())


def saved_stderr(directory):
    directory = Path(directory)
    raw = directory / 'server.stderr.log'
    packed = directory / 'server.stderr.log.gz'
    if not raw.is_file() and not packed.is_file():
        raise AssertionError('missing retained stderr')
    raw_data = raw.read_bytes() if raw.is_file() else None
    packed_data = gzip.decompress(packed.read_bytes()) if packed.is_file() else None
    if raw_data is not None and packed_data is not None:
        assert raw_data == packed_data, 'retained log representations differ'
    return (raw_data if raw_data is not None else packed_data).decode()


def combined_fixture():
    data = gzip.decompress((HERE / 'fixtures/combined-startup-measured.json.gz').read_bytes())
    return json.loads(data)


def strip_blocks(text, marker):
    return re.sub(r'^[ \t]*// ' + re.escape(marker) + r'_BEGIN\n.*?^[ \t]*// ' +
                  re.escape(marker) + r'_END\n', '', text, flags=re.M | re.S)


class PatchTests(unittest.TestCase):
    def test_exact_fixtures_and_zero_fuzz_forward_reverse(self):
        p = pins()
        self.assertEqual(sorted(S['SOURCE_SCOPE']), sorted(p['parent']))
        S['check_fixtures']()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = {}
            for name in S['SOURCE_SCOPE']:
                data = fixture_bytes(name)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                originals[name] = data
            for flags in (['--dry-run', '--forward'], ['--forward']):
                result = subprocess.run(
                    ['patch', '--fuzz=0', '--batch', '-p1', *flags,
                     '-i', str(HERE / 'lilicorr-profile.patch')],
                    cwd=root, check=True, capture_output=True, text=True)
                self.assertNotRegex(result.stdout.lower(), r'fuzz|offset|failed')
            for name, original in originals.items():
                data = (root / name).read_bytes()
                self.assertEqual(p['candidate'][name], hashlib.sha256(data).hexdigest())
                text = data.decode()
                marker = ('LILICORR_SYCL_PROFILE_ONLY' if name.startswith('ggml/')
                          else 'LILICORR_PROFILE_ONLY')
                self.assertEqual(original, strip_blocks(text, marker).encode())
                if name.startswith('ggml/'):
                    self.assertNotIn('std::chrono', text)
                    self.assertNotIn('steady_clock', text)
                    self.assertIn('return ggml_time_us();', text)
            result = subprocess.run(
                ['patch', '--fuzz=0', '--batch', '--reverse', '-p1',
                 '-i', str(HERE / 'lilicorr-profile.patch')],
                cwd=root, check=True, capture_output=True, text=True)
            self.assertNotRegex(result.stdout.lower(), r'fuzz|offset|failed')
            for name, original in originals.items():
                self.assertEqual(original, (root / name).read_bytes())

    def test_source_scope_build_script_and_expected_binary_delta(self):
        script = (HERE / 'build-candidate.sh').read_text()
        self.assertNotRegex(script, r'--device|/dev/dri|--privileged|docker cp|ssh ')
        self.assertIn('CMAKE_HOME_DIRECTORY:INTERNAL=/src', script)
        self.assertIn('--fuzz=0', script)
        self.assertIn('lilicorr-profile.patch', script)
        subprocess.run(['bash', '-n', str(HERE / 'build-candidate.sh')], check=True)
        self.assertEqual(sorted(S['SOURCE_SCOPE']), [
            'common/speculative.cpp', 'ggml/src/ggml-sycl/ggml-sycl.cpp',
            'src/llama-context.cpp'])
        self.assertEqual(S['CHANGED_BINARIES'], [
            'libggml-sycl.so.0', 'libllama-common.so', 'libllama.so.0.7.0'])

    def test_retained_log_accepts_raw_or_gzip_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'profile log\n'
            (root / 'server.stderr.log').write_bytes(payload)
            self.assertEqual(saved_stderr(root), payload.decode())
            (root / 'server.stderr.log.gz').write_bytes(gzip.compress(payload))
            self.assertEqual(saved_stderr(root), payload.decode())
            (root / 'server.stderr.log').write_bytes(b'different\n')
            with self.assertRaises(AssertionError):
                saved_stderr(root)

    def test_backend_gate_scope_order_and_no_synchronous_logging(self):
        original = fixture_bytes('ggml/src/ggml-sycl/ggml-sycl.cpp').decode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in S['SOURCE_SCOPE']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(fixture_bytes(name))
            path = root / 'ggml/src/ggml-sycl/ggml-sycl.cpp'
            subprocess.run(['patch', '--fuzz=0', '--batch', '--forward', '-p1',
                             '-i', str(HERE / 'lilicorr-profile.patch')],
                            cwd=root, check=True, capture_output=True)
            patched = path.read_text()
        self.assertIn('std::getenv("LLAMA_LILICORR_PROFILE")', patched)
        self.assertIn('lilicorr_sycl_profile_graph_marker', patched)
        self.assertIn('lilicorr_sycl_profile_scope', patched)
        self.assertIn('profile_records.reserve', patched)
        self.assertIn('LLAMA_LILICORR_SYCL_PROFILE ', patched)
        self.assertNotIn('std::chrono', patched)
        self.assertNotIn('steady_clock', patched)
        self.assertIn('return ggml_time_us();', patched)
        self.assertNotRegex(patched, r'LLAMA_LILICORR_SYCL_PROFILE\n')
        added = '\n'.join(re.findall(
            r'^[ \t]*// LILICORR_SYCL_PROFILE_ONLY_BEGIN\n(.*?)^[ \t]*// LILICORR_SYCL_PROFILE_ONLY_END\n',
            patched, flags=re.M | re.S))
        self.assertNotRegex(added, r'\.wait\s*\(|queue\.wait|sycl::event|memcpy\s*\(')
        self.assertLess(patched.index('bool ok = ggml_sycl_compute_forward'),
                        patched.index('stage=op'))
        self.assertEqual(original, strip_blocks(patched, 'LILICORR_SYCL_PROFILE_ONLY'))

    def test_shared_clock_helper_stub_compiles_and_runs(self):
        source = r'''
#include <cstdint>
static int64_t ggml_time_us() { return 1234; }
static int64_t lilicorr_sycl_profile_now_us() { return ggml_time_us(); }
int main() { return lilicorr_sycl_profile_now_us() == 1234 ? 0 : 1; }
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpp = root / 'clock-helper.cpp'
            exe = root / 'clock-helper'
            cpp.write_text(source)
            subprocess.run(['c++', '-std=c++17', '-Wall', str(cpp), '-o', str(exe)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(exe)], check=True)


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.prefix = '2026-10-19T00:00:00Z '

    def lines(self, bad=None):
        rows = [
            'LLAMA_LILICORR_SYCL_PROFILE v=1 stage=gate graph=7 requested=1 enabled=1 marker=1 graph_nodes=5 marker_name=lilicorr.out_head',
            'LLAMA_LILICORR_SYCL_PROFILE v=1 stage=op graph=7 first_index=0 last_index=1 count=2 begin_us=110 end_us=120 elapsed_host_us=10 group=compute_forward first_op=MUL_MAT last_op=MUL_MAT first_name=a last_name=b src0_name=lilicorr.out_head src0_type=Q8_0 src1_name=x src1_type=F32 dst_type=F32 ne0=4 ne1=2 ne2=1 ne3=1 src0_ne0=8 src0_ne1=4 src0_ne2=1 src0_ne3=1 dst_ne0=4 dst_ne1=2 dst_ne2=1 dst_ne3=1',
            'LLAMA_LILICORR_SYCL_PROFILE v=1 stage=op graph=7 first_index=3 last_index=3 count=1 begin_us=125 end_us=135 elapsed_host_us=10 group=topk first_op=TOP_K last_op=TOP_K first_name=c last_name=c src0_name=- src0_type=F32 src1_name=- src1_type=- dst_type=F32 ne0=4 ne1=1 ne2=1 ne3=1 src0_ne0=4 src0_ne1=1 src0_ne2=1 src0_ne3=1 dst_ne0=4 dst_ne1=1 dst_ne2=1 dst_ne3=1',
            'LLAMA_LILICORR_SYCL_PROFILE v=1 stage=graph graph=7 requested=1 enabled=1 marker=1 graph_nodes=5 covered_nodes=3 uncovered_nodes=2 skipped_nodes=2 spans=2 first_index=0 last_index=3 begin_us=105 end_us=145 core_wall_us=40 event_wall_us=20 uncounted_wall_us=20 log_wall_us=4 covered_ranges=0-1,3 uncovered_ranges=2,4 marker_name=lilicorr.out_head',
        ]
        if bad is not None:
            rows[bad[0]] = bad[1]
        return [self.prefix + row for row in rows]

    def phase_events(self):
        return [
            dict(stage='process_chunk', line=3, begin_us=95, end_us=160, rows=5),
            dict(stage='ubatch', line=4, begin_us=100, end_us=150, build_alloc_us=2,
                 inputs_us=3, submit_us=48, rows=5, gtype=1, nodes=5),
        ]

    def test_event_parser_and_phase_correlation(self):
        rows = A['parse_sycl'](self.lines(), first_line=20)
        self.assertEqual([r['stage'] for r in rows], ['gate', 'op', 'op', 'graph'])
        request = {'lines': self.lines(), 'first_line': 20}
        phase = {'events': self.phase_events()}
        report = A['request_sycl'](request, phase)
        self.assertEqual(report['graph_count'], 1)
        self.assertEqual(report['graph_core_wall_us'], 40)
        self.assertEqual(report['operation_wall_us'], 20)
        self.assertEqual(report['graphs'][0]['phase']['stage'], 'process')
        self.assertEqual(report['scheduler_coverage']['expected_ubatches'], 1)
        self.assertEqual(report['scheduler_coverage']['covered_ubatches'], 1)
        self.assertEqual(len(report['operator_totals']), 2)
        self.assertEqual(len(report['operator_totals'][0]['raw_refs']), 1)

    def test_ggml_view_names_with_spaces_remain_lossless(self):
        line = self.lines()[1].replace('first_name=a ', 'first_name=Kcur (view) ')
        line = line.replace('src1_name=x ', 'src1_name=inp_g_embeddings (reshaped) ')
        row = A['event'](line)
        self.assertEqual('Kcur (view)', row['first_name'])
        self.assertEqual('inp_g_embeddings (reshaped)', row['src1_name'])
        with self.assertRaises(AssertionError):
            A['event'](line + ' ne0=5')
        with self.assertRaises(AssertionError):
            A['event'](line + ' unknown=5')

    def test_parser_rejects_overlap_missing_coverage_negative_duration_and_marker_drift(self):
        rows = self.lines()
        overlap = rows[2].replace('begin_us=125 end_us=135', 'begin_us=115 end_us=135 elapsed_host_us=20')
        with self.assertRaises(AssertionError):
            A['request_sycl']({'lines': [rows[0], rows[1], overlap, rows[3]], 'first_line': 1},
                              {'events': self.phase_events()})
        missing = rows[3].replace('covered_nodes=3 uncovered_nodes=2', 'covered_nodes=2 uncovered_nodes=3')
        with self.assertRaises(AssertionError):
            A['request_sycl']({'lines': [rows[0], rows[1], rows[2], missing], 'first_line': 1},
                              {'events': self.phase_events()})
        negative = rows[1].replace('end_us=120 elapsed_host_us=10', 'end_us=109 elapsed_host_us=-1')
        with self.assertRaises(AssertionError):
            A['event'](negative)
        bad_marker = rows[0].replace('marker_name=lilicorr.out_head', 'marker_name=not_lilicorr')
        with self.assertRaises(AssertionError):
            A['event'](bad_marker)

    def test_gate_disabled_is_not_claimed_as_coverage(self):
        rows = self.lines()
        disabled_gate = rows[0].replace('enabled=1 marker=1', 'enabled=0 marker=0').replace(
            'marker_name=lilicorr.out_head', 'marker_name=-')
        disabled_graph = (rows[3].replace('enabled=1 marker=1', 'enabled=0 marker=0')
                           .replace('covered_nodes=3 uncovered_nodes=2', 'covered_nodes=0 uncovered_nodes=5')
                           .replace('skipped_nodes=2 spans=2', 'skipped_nodes=0 spans=0')
                           .replace('first_index=0 last_index=3', 'first_index=-1 last_index=-1')
                           .replace('begin_us=105 end_us=145 core_wall_us=40 event_wall_us=20 uncounted_wall_us=20',
                                    'begin_us=0 end_us=0 core_wall_us=0 event_wall_us=0 uncounted_wall_us=0')
                           .replace('log_wall_us=4 covered_ranges=0-1,3 uncovered_ranges=2,4 marker_name=lilicorr.out_head',
                                    'log_wall_us=0 covered_ranges=- uncovered_ranges=all marker_name=-'))
        report = A['_unattributed_sycl'](A['parse_sycl']([disabled_gate, disabled_graph], 1))
        self.assertEqual(0, report['enabled_graph_count'])
        self.assertEqual(1, report['disabled_graph_count'])
        with self.assertRaises(AssertionError):
            A['request_sycl']({'lines': [disabled_gate, disabled_graph], 'first_line': 1},
                              {'events': self.phase_events()})

    def test_combined_startup_warmup_fixture_covers_all_real_rounds(self):
        fixture = combined_fixture()
        self.assertTrue(fixture['source']['stderr_raw_equals_gzip'])
        source = fixture['source']
        source_profile = FIRST / 'profile-r1/analysis-r1/profile.json'
        source_directory = FIRST / 'profile-r1/VSPF-W3a'
        source_gzip = source_directory / 'server.stderr.log.gz'
        source_bytes = saved_stderr(source_directory).encode()
        self.assertEqual(source['profile_sha256'], hashlib.sha256(source_profile.read_bytes()).hexdigest())
        self.assertEqual(source['stderr_sha256'], hashlib.sha256(source_bytes).hexdigest())
        if source_gzip.is_file():
            self.assertEqual(source['stderr_gzip_sha256'], hashlib.sha256(source_gzip.read_bytes()).hexdigest())
        startup = A['_startup_sycl'](fixture['startup_lines'])
        self.assertEqual(1, startup['graph_count'])
        self.assertEqual(0, startup['enabled_graph_count'])
        self.assertEqual(1, startup['disabled_graph_count'])
        self.assertEqual(8, len(fixture['requests']))
        for request in fixture['requests']:
            parsed = A['request_sycl'](
                {'lines': request['lines'], 'first_line': request['first_line']},
                request['phase_request'])
            draft_rounds = 43 if request['point'] == 'short' else 45
            scheduler = parsed['scheduler_coverage']
            self.assertEqual(89 if request['point'] == 'short' else 101,
                             scheduler['expected_ubatches'])
            self.assertEqual(draft_rounds, scheduler['expected_draft_ubatches'])
            self.assertEqual(draft_rounds, scheduler['covered_draft_ubatches'])
            self.assertEqual(0, scheduler['uncovered_draft_ubatches'])
            self.assertEqual(0, scheduler['uncovered_ubatches'])
            self.assertGreater(scheduler['uncovered_nodes'], 0)
            self.assertEqual(scheduler['expected_nodes'],
                             scheduler['covered_nodes'] + scheduler['uncovered_nodes'])

    def test_combined_fixture_missing_whole_graph_and_duplicate_owner_fail_closed(self):
        fixture = combined_fixture()
        request = fixture['requests'][0]
        marker = A['MARKER']
        first_graph = [line for line in request['lines'] if marker in line and 'graph=1001 ' in line]
        self.assertEqual(4, len(first_graph))
        dropped = [line for line in request['lines'] if line != first_graph[-1]]
        with self.assertRaises(AssertionError):
            A['request_sycl']({'lines': dropped, 'first_line': request['first_line']},
                              request['phase_request'])
        duplicate = [line.replace('graph=1001 ', 'graph=999999 ') for line in first_graph]
        with self.assertRaises(AssertionError):
            A['request_sycl']({'lines': request['lines'] + duplicate,
                               'first_line': request['first_line']}, request['phase_request'])
        ambiguous = copy.deepcopy(request['phase_request'])
        ambiguous['events'].append(copy.deepcopy(next(
            event for event in ambiguous['events'] if event['stage'] == 'process_chunk')))
        with self.assertRaises(AssertionError):
            A['request_sycl']({'lines': request['lines'], 'first_line': request['first_line']},
                              ambiguous)

    def test_uninstrumented_injection_and_partial_scheduler_are_explicit(self):
        request = combined_fixture()['requests'][1]
        bundles = A['_graph_bundles'](A['parse_sycl'](request['lines'], request['first_line']))
        draft_ids = {b['graph'] for b in bundles
                     if A['_phase_parent'](b, request['phase_request']['events'])['stage'] == 'draft'}
        lines = [line for line in request['lines'] if A['MARKER'] not in line
                 or A['event'](line)['graph'] in draft_ids]
        parsed = A['request_sycl']({'lines': lines, 'first_line': request['first_line']},
                                   request['phase_request'])
        coverage = parsed['scheduler_coverage']
        self.assertEqual(43, coverage['covered_draft_ubatches'])
        self.assertGreater(coverage['uncovered_ubatches'], 0)
        self.assertGreater(coverage['uninstrumented_scheduler_nodes'], 0)
        missing_id = next(iter(draft_ids))
        missing = [line for line in lines if A['MARKER'] not in line
                   or A['event'](line)['graph'] != missing_id]
        with self.assertRaisesRegex(AssertionError, 'missing measured draft'):
            A['request_sycl']({'lines': missing, 'first_line': request['first_line']},
                              request['phase_request'])
        phase = self.phase_events()
        phase[1]['nodes'] = 7
        partial = A['request_sycl']({'lines': self.lines(), 'first_line': 1}, {'events': phase})
        self.assertEqual(2, partial['scheduler_coverage']['uninstrumented_scheduler_nodes'])

    def test_full_pipeline_retains_public_spans_and_rejects_absent_measured_records(self):
        fixture = combined_fixture()
        text = saved_stderr(FIRST / 'profile-r1/VSPF-W3a')
        requests, _ = A['BASE']['requests'](text)
        blocks = []
        for index, item in enumerate(fixture['requests'], 25):
            old = '\n'.join(requests[index]['lines'])
            self.assertEqual(1, text.count(old))
            by_begin = {}
            events = A['parse_sycl'](item['lines'], item['first_line'])
            for bundle in A['_graph_bundles'](events):
                owner = A['_phase_parent'](bundle, item['phase_request']['events'])
                ubatch = next(row for row in item['phase_request']['events']
                              if row['stage'] == 'ubatch' and row['line'] == owner['ubatch_line'])
                records = [line for line in item['lines'] if A['MARKER'] in line
                           and A['event'](line)['graph'] == bundle['graph']]
                by_begin.setdefault(ubatch['begin_us'], []).extend(records)
            lines = []
            for line in item['lines']:
                if A['MARKER'] in line:
                    continue
                if A['BASE']['MARKER'] in line:
                    event = A['BASE']['event'](line)
                    if event['stage'] == 'ubatch':
                        lines.extend(by_begin.pop(event['begin_us'], []))
                lines.append(line)
            self.assertFalse(by_begin)
            blocks.append('\n'.join(lines))
            text = text.replace(old, blocks[-1])
        public_records = [line.replace('graph=1001 ', 'graph=777777 ')
                          for line in fixture['requests'][0]['lines']
                          if A['MARKER'] in line and 'graph=1001 ' in line]
        self.assertEqual(4, len(public_records))
        text = text.replace(requests[0]['lines'][-1],
                            '\n'.join(public_records) + '\n' + requests[0]['lines'][-1], 1)
        # Synthetic backend records over real request/phase boundaries are parser
        # tests only. They do not establish live producer behavior or performance.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            c = D['bind']()[2]
            phase_report = dict(status='analyzed', errors=[], requests=[], startup_events={})
            for cell in c['CELLS']:
                (out / cell).mkdir()
                stderr = text if 'W3' in cell else saved_stderr(FIRST / 'profile-r1' / cell)
                (out / cell / 'server.stderr.log').write_text(stderr)
                (out / cell / 'server.stdout.log').write_text('')
                if 'W3' in cell:
                    for item in fixture['requests']:
                        phase_report['requests'].append(dict(cell=cell, phase=item['phase'],
                            point=item['point'], ordinal=item['ordinal'], **copy.deepcopy(item['phase_request'])))
            with mock.patch.dict(A['BASE'], {'analyze': lambda *_args: copy.deepcopy(phase_report)}):
                result = A['analyze'](out, c)
                self.assertEqual('analyzed', result['status'], result['errors'])
                public = next(r for r in result['sycl_diagnostic']['public'] if 'W3' in r['cell'])
                self.assertEqual(1, public['requests'][0]['enabled_graph_count'])
                self.assertEqual(12, len(result['sycl_diagnostic']['graphs']))
                old = blocks[1]
                clean = '\n'.join(line for line in old.splitlines() if A['MARKER'] not in line)
                path = out / 'VSPF-W3a/server.stderr.log'
                self.assertEqual(1, path.read_text().count(old))
                path.write_text(path.read_text().replace(old, clean, 1))
                failed = A['analyze'](out, c)
                self.assertEqual('failed', failed['status'])
                self.assertIn('missing SYCL diagnostic events', str(failed['errors']))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.analysis, self.c = D['bind']()
        self.public = dict(OUT=QUALIFIED, C={}, CANDIDATE=S['PARENT'],
                      make_create=lambda cell, _h: S['load'](QUALIFIED / cell / 'create-argv.json'))
        self.c['configure'](self.public)
        self.source_bind = D['_FIRST_RUN']['source_bind']

    def _container_fixture(self, cell):
        record = S['load'](self.c['qualified_dir'](self.c['role_for'](cell)) / 'container-before.json')
        record.update(Name='/' + self.c['name'](cell), Id='b' * 64)
        record['Config']['Hostname'] = 'b' * 12
        record['Config']['Cmd'] = self.c['make_create'](cell, None)[-3:]
        if self.c['role_for'](cell) == 'W3':
            record['Config']['Env'].append(D['PROFILE_ENV'] + '=1')
            substitutions = {
                D['Q']['old_draft_bind'](): D['Q']['candidate_draft_bind'](),
                self.source_bind(S['PARENT']): self.source_bind(S['CANDIDATE']),
            }
            record['HostConfig']['Binds'] = [substitutions.get(x, x)
                                              for x in record['HostConfig']['Binds']]
            record['Mounts'] = D['Q']['_expected_mounts'](record, cell)
            next(m for m in record['Mounts'] if m['Destination'] == '/src')['Source'] = str(S['CANDIDATE'])
        return record

    def test_actual_root_out_prefix_source_bind_and_container_identity(self):
        runtime, analysis, contract = D['bind']()
        self.assertIsNotNone(runtime)
        self.assertIsNotNone(analysis)
        self.assertEqual(contract['ROOT'], D['ROOT'])
        self.assertEqual(contract['OUT'], D['OUT'])
        self.assertEqual(contract['PREFIX'], D['PREFIX'])
        self.assertEqual(D['ROOT'], Path('/home/mike/b70-evals/20261019-lilicorr-b70-sycl-profile'))
        self.assertEqual(D['OUT'], D['ROOT'] / 'profile-r1')
        self.assertEqual(D['PREFIX'], 'bolt-lilicorr-b70-sycl-profile-r1-')
        self.assertEqual(self.source_bind(S['PARENT']), str(S['PARENT']) + ':/src:ro')
        self.assertEqual(self.source_bind(S['CANDIDATE']), str(S['CANDIDATE']) + ':/src:ro')
        for cell in self.c['CELLS']:
            record = self._container_fixture(cell)
            before = copy.deepcopy(record)
            self.assertTrue(self.c['check_container'](record, cell, None))
            self.assertEqual(before, record)
        pristine = self._container_fixture('VSPF-W3a')
        bad = copy.deepcopy(pristine)
        bad['Config']['Env'].append(D['PROFILE_ENV'] + '=0')
        with self.assertRaises(AssertionError):
            self.c['check_container'](bad, 'VSPF-W3a', None)

    def test_actual_make_create_wrapper_has_exact_w3_delta_and_no_qualified_rebind(self):
        for cell in self.c['CELLS']:
            base_cell = self.c['base_cell'](cell)
            base = self.public['make_create'](base_cell, None)
            argv = self.c['make_create'](cell, None)
            self.assertEqual(base[-3:], argv[-3:])
            self.assertEqual(self.c['server_argv'](base), self.c['server_argv'](argv))
            if self.c['role_for'](cell) == 'W3':
                self.assertIn(self.source_bind(S['CANDIDATE']), argv)
                self.assertNotIn(self.source_bind(S['PARENT']), argv)
                self.assertEqual(1, sum(D['PROFILE_ENV'] in item for item in argv))
            else:
                self.assertNotIn(self.source_bind(S['CANDIDATE']), argv)
                self.assertFalse(any(D['PROFILE_ENV'] in item for item in argv))
        self.assertEqual(S['PARENT'], D['S']['PARENT'])

    def test_actual_source_binary_model_wrappers_reject_same_size_mutations(self):
        runtime, analysis, c = D['bind']()
        with tempfile.TemporaryDirectory(dir=HERE / '.scratch') as directory:
            root = Path(directory)
            parent, candidate = root / 'parent', root / 'candidate'
            for tree in (parent, candidate):
                tree.mkdir()
                (tree / 'UPSTREAM_COMMIT').write_text(pins()['upstream_commit'] + '\n')
                for name in S['SOURCE_SCOPE']:
                    path = tree / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(fixture_bytes(name))
                for name in S['ARTIFACTS']:
                    path = tree / 'build-sycl/bin' / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((name + ('profile' if tree == candidate and name in S['CHANGED_BINARIES'] else '')).encode())
            subprocess.run(['patch', '--fuzz=0', '--batch', '-p1', '-i', str(HERE / 'lilicorr-profile.patch')],
                           cwd=candidate, check=True, capture_output=True)
            for name in ('source-hashes.json', 'lilicorr-profile.patch'):
                (root / name).write_bytes((HERE / name).read_bytes())
            for name in S['SOURCE_SCOPE']:
                archive = root / 'fixtures' / (name + '.gz')
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes((HERE / 'fixtures' / (name + '.gz')).read_bytes())
            save(root / 'parent-binaries.json', S['binaries'](parent))
            save(root / 'candidate-binaries.json', S['binaries'](candidate))
            save(root / 'source-diff.json', dict(changed_relative_to_parent=S['SOURCE_SCOPE']))
            (root / 'build-attempt-r1').mkdir()
            (root / 'build-attempt-r1/exit-code.txt').write_text('0\n')
            (root / 'runtime-precheck.log').write_text('LILICORR_PROFILE_DEPENDENCIES_PASSED')
            (root / 'post-build-verification.log').write_text('LILICORR_PROFILE_BUILD_PASSED')
            save(root / 'build-attempt-r1/image.json', {'fixture': True})
            qualified = dict(source=S['source_map'](parent), binaries=S['binaries'](parent))
            public = dict(CANDIDATE=parent, candidate_proof=lambda: qualified,
                          C=dict(source_binary_hashes=lambda _h: {'qualified': True},
                                 source_evidence=lambda _h: {'qualified': True}))
            c['configure'](public)
            q8 = root / 'q8.gguf'
            q8.write_bytes(b'q8-fixture')
            inherited = runpy.run_path(str(D['Q']['BASE'] / 'contracts.py'))['adapter_code_hashes']()
            real_load = c['load']
            def load(path):
                return inherited if Path(path).name == 'adapter-code-hashes-before.json' else real_load(path)
            real_sha = c['sha']
            def sha(path):
                path = Path(path)
                name = str(path.relative_to(HERE)) if path.is_relative_to(HERE) else None
                return real_sha(root / name) if name in D['BUILD_FILES'] else real_sha(path)
            with mock.patch.dict(S['proof'].__globals__, HERE=root, PARENT=parent, CANDIDATE=candidate), \
                 mock.patch.dict(S['proof'].__globals__['_BASE']['proof'].__globals__,
                                 HERE=root, PARENT=parent, CANDIDATE=candidate), \
                 mock.patch.dict(D['Q']['bind'].__globals__, CANDIDATE=q8,
                                 CANDIDATE_SIZE=q8.stat().st_size, CANDIDATE_SHA256=real_sha(q8)), \
                 mock.patch.dict(c, load=load, sha=sha):
                c['adapter_code_hashes']()
                for wrapper in ('source_binary_hashes', 'source_evidence'):
                    c[wrapper](None)
                    for path in (candidate / 'common/speculative.cpp',
                                 candidate / 'build-sycl/bin/libllama-common.so', q8):
                        original = path.read_bytes()
                        path.write_bytes(b'X' + original[1:])
                        with self.assertRaises(AssertionError, msg=str(path)):
                            c[wrapper](None)
                        path.write_bytes(original)
                    extra = candidate / 'unapproved-source'
                    extra.write_text('drift')
                    with self.assertRaises(AssertionError):
                        c[wrapper](None)
                    extra.unlink()
                    c[wrapper](None)

    def test_original_first_package_thirteen_cpu_tests_are_inherited_unchanged(self):
        result = subprocess.run([sys.executable, '-B', str(FIRST / 'test-profile.py')],
                                cwd=FIRST, check=True, capture_output=True, text=True)
        self.assertRegex(result.stdout + result.stderr, r'Ran 13 tests')


class BindingTests(unittest.TestCase):
    def test_private_namespace_rebound_without_qualified_ancestry_rebind(self):
        runtime, analysis, contract = D['bind']()
        self.assertIsNotNone(runtime)
        self.assertIsNotNone(analysis)
        self.assertEqual(contract['ROOT'], D['ROOT'])
        self.assertEqual(contract['OUT'], D['OUT'])
        self.assertEqual(contract['PREFIX'], D['PREFIX'])
        self.assertEqual(D['ROOT'], Path('/home/mike/b70-evals/20261019-lilicorr-b70-sycl-profile'))
        self.assertEqual(D['S']['PARENT'], Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2'))
        self.assertIn('../20261019-lilicorr-b70-profile/run-profile.py', D['OWN_FILES'])
        self.assertIn('../20261019-lilicorr-b70-profile/analyze-profile.py', D['OWN_FILES'])
        self.assertEqual(D['S']['SOURCE_SCOPE'], [
            'common/speculative.cpp', 'ggml/src/ggml-sycl/ggml-sycl.cpp',
            'src/llama-context.cpp'])


if __name__ == '__main__':
    (HERE / '.scratch').mkdir(exist_ok=True)
    unittest.main(verbosity=2)
