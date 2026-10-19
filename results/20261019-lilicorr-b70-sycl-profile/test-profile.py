#!/usr/bin/env python3
"""CPU-only tests for the second diagnostic; no Docker, GPU, network, or remote mutation."""
import gzip
import hashlib
import json
import re
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-profile.py'))
S = D['S']
A = runpy.run_path(str(HERE / 'analyze-profile.py'))


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
            'common/speculative.cpp', 'ggml/src/ggml-sycl/ggml-sycl.cpp', 'src/llama-context.cpp'])
        self.assertEqual(S['CHANGED_BINARIES'], [
            'libggml-sycl.so.0', 'libllama-common.so', 'libllama.so.0.7.0'])
    def test_retained_log_accepts_raw_or_gzip_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'profile log\\n'
            (root / 'server.stderr.log').write_bytes(payload)
            self.assertEqual(saved_stderr(root), payload.decode())
            (root / 'server.stderr.log.gz').write_bytes(gzip.compress(payload))
            self.assertEqual(saved_stderr(root), payload.decode())
            (root / 'server.stderr.log').write_bytes(b'different\\n')
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
        self.assertNotRegex(patched, r'LLAMA_LILICORR_SYCL_PROFILE\\n')
        added = '\n'.join(re.findall(
            r'^[ \\t]*// LILICORR_SYCL_PROFILE_ONLY_BEGIN\\n(.*?)^[ \\t]*// LILICORR_SYCL_PROFILE_ONLY_END\\n',
            patched, flags=re.M | re.S))
        self.assertNotRegex(added, r'\.wait\s*\(|queue\.wait|sycl::event|memcpy\s*\(')
        self.assertLess(patched.index('bool ok = ggml_sycl_compute_forward'),
                        patched.index('stage=op'))
        self.assertEqual(original, strip_blocks(patched, 'LILICORR_SYCL_PROFILE_ONLY'))




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
        return [dict(stage='ubatch', line=4, begin_us=100, build_alloc_us=2,
                     inputs_us=3, submit_us=48, rows=2, gtype=1, nodes=5)]

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
        self.assertEqual(len(report['operator_totals']), 2)
        self.assertEqual(len(report['operator_totals'][0]['raw_refs']), 1)

    def test_parser_rejects_overlap_missing_coverage_and_negative_duration(self):
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

    def test_gate_disabled_is_not_claimed_as_coverage(self):
        rows = self.lines()
        disabled_gate = rows[0].replace('enabled=1 marker=1', 'enabled=0 marker=0').replace(
            'marker_name=lilicorr.out_head', 'marker_name=-')
        disabled_graph = rows[3].replace(
            'enabled=1 marker=1 covered_nodes=3 uncovered_nodes=2 skipped_nodes=2 spans=2 first_index=0 last_index=3 begin_us=100 end_us=150 core_wall_us=50 event_wall_us=20 uncounted_wall_us=30 log_wall_us=4 covered_ranges=0-1,3 uncovered_ranges=2,4 marker_name=lilicorr.out_head',
            'enabled=0 marker=0 covered_nodes=0 uncovered_nodes=5 skipped_nodes=0 spans=0 first_index=-1 last_index=-1 begin_us=0 end_us=0 core_wall_us=0 event_wall_us=0 uncounted_wall_us=0 log_wall_us=0 covered_ranges=- uncovered_ranges=all marker_name=-')
        with self.assertRaises(AssertionError):
            A['request_sycl']({'lines': [disabled_gate, disabled_graph], 'first_line': 1},
                              {'events': self.phase_events()})


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
            'common/speculative.cpp', 'ggml/src/ggml-sycl/ggml-sycl.cpp', 'src/llama-context.cpp'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
