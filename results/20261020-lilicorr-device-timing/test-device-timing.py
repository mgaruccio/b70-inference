#!/usr/bin/env python3
"""Bounded CPU/source checks only. Never builds llama/SYCL, installs, or accesses a GPU/remote."""
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
SCRATCH = HERE / '.scratch'
PINS = json.loads((HERE / 'source-pins.json').read_text())
CONTEXT = WORKTREE / 'results/20261020-lilicorr-context-profile'
IQ2 = WORKTREE / 'results/20261020-lilicorr-iq2-batch4'
SNAPSHOT = Path(os.environ.get('LILICORR_SOURCE_ROOT', PINS['immutable_snapshot'])).resolve()
PREFIX = 'ggml/src/ggml-sycl/'
HEADER = PREFIX + 'lilicorr-device-timing.hpp'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(args, cwd, check=True):
    assert cwd.resolve().is_relative_to(WORKTREE)
    result = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True,
                            env={**os.environ, 'TMPDIR': str(cwd)}, timeout=25)
    if check and result.returncode:
        raise AssertionError(f'{args}\nexit={result.returncode}\n{result.stdout}{result.stderr}')
    return result


def apply(root, patch, reverse=False):
    flags = ['--reverse'] if reverse else []
    run(['git', 'apply', '--check', *flags, patch], root)
    run(['git', 'apply', '--whitespace=error', *flags, patch], root)


def check_map(root, expected):
    actual = {name: sha((root / name).read_bytes()) for name in expected}
    assert actual == expected, {n: (expected[n], actual[n]) for n in expected if expected[n] != actual[n]}


def parent(root):
    for name in PINS['parent']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in ('common/speculative.cpp', 'src/llama-context.cpp'):
            data = gzip.decompress((CONTEXT / 'fixtures' / (name + '.gz')).read_bytes())
        else:
            data = (SNAPSHOT / name).read_bytes()
        path.write_bytes(data)
    check_map(root, PINS['parent'])


def strip_diagnostic(text):
    text = re.sub(r'^[ \t]*// LILICORR_DEVICE_ONLY_BEGIN\n.*?^[ \t]*// LILICORR_DEVICE_ONLY_END\n',
                  '', text, flags=re.M | re.S)
    return re.sub(r'/\* LILICORR_DEVICE_INLINE_BEGIN \*/.*?/\* LILICORR_DEVICE_INLINE_END \*/', '', text)


class DeviceTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        cls.root = Path(cls.temp.name)
        cls.seed, cls.candidate = cls.root / 'seed', cls.root / 'candidate'
        for name, expected in {**PINS['ancestry_files'], **PINS['reused_fixtures']}.items():
            assert sha((WORKTREE / name).read_bytes()) == expected, name
        assert sha((HERE / 'device-timing.patch').read_bytes()) == PINS['patch_sha256']
        parent(cls.seed)
        assert (cls.seed / 'UPSTREAM_COMMIT').read_text().strip() == PINS['upstream_commit']
        # Direct application to the qualified parent is deliberately NOT the supported ancestry.
        assert run(['git', 'apply', '--check', HERE / 'device-timing.patch'], cls.seed, check=False).returncode
        apply(cls.seed, CONTEXT / 'lilicorr-profile.patch')
        check_map(cls.seed, PINS['context_seed'])
        apply(cls.seed, IQ2 / 'iq2-batch4.patch')
        check_map(cls.seed, PINS['composed_seed'])
        shutil.copytree(cls.seed, cls.candidate)
        apply(cls.candidate, HERE / 'device-timing.patch')
        check_map(cls.candidate, PINS['diagnostic'])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_ancestry_pins_exact_composition_reverse_and_wrong_source(self):
        tree = self.root / 'reverse'
        shutil.copytree(self.candidate, tree)
        apply(tree, HERE / 'device-timing.patch', reverse=True)
        self.assertFalse((tree / HEADER).exists())
        check_map(tree, PINS['composed_seed'])
        apply(tree, IQ2 / 'iq2-batch4.patch', reverse=True)
        check_map(tree, PINS['context_seed'])
        apply(tree, CONTEXT / 'lilicorr-profile.patch', reverse=True)
        check_map(tree, PINS['parent'])
        # The two existing source deltas are structurally disjoint, in either composition order.
        apply(tree, IQ2 / 'iq2-batch4.patch')
        apply(tree, CONTEXT / 'lilicorr-profile.patch')
        check_map(tree, PINS['composed_seed'])
        (tree / (PREFIX + 'quantize.hpp')).write_text('wrong-source\n')
        with self.assertRaises(AssertionError):
            check_map(tree, PINS['composed_seed'])

    def test_exact_source_and_arithmetic_geometry_preservation(self):
        changed = []
        for name in PINS['composed_seed']:
            before = (self.seed / name).read_text()
            after = (self.candidate / name).read_text()
            self.assertEqual(strip_diagnostic(after), before, name)
            if before != after:
                changed.append(name)
        self.assertEqual(set(changed), {'common/speculative.cpp'} | {PREFIX + n for n in ('common.hpp', 'dpct/helper.hpp', 'ggml-sycl.cpp', 'mmvq.cpp', 'quantize.hpp')})
        self.assertEqual(set(PINS['diagnostic']) - set(PINS['composed_seed']), {HEADER})
        # This compares the entire composed IQ2/Q8 source, not a model of its K-loop or geometry.
        for name in ('src/llama-context.cpp', PREFIX + 'mmvq.hpp',
                     PREFIX + 'vecdotq.hpp', PREFIX + 'quants.hpp', PREFIX + 'presets.hpp'):
            self.assertEqual((self.seed / name).read_bytes(), (self.candidate / name).read_bytes())

    def test_actual_launches_and_only_existing_drain_queries(self):
        backend = (self.candidate / (PREFIX + 'ggml-sycl.cpp')).read_text()
        header = (self.candidate / HEADER).read_text()
        quant = (self.candidate / (PREFIX + 'quantize.hpp')).read_text()
        mmvq = (self.candidate / (PREFIX + 'mmvq.cpp')).read_text()
        self.assertEqual(quant.count('device_submit.finish('), 1)
        self.assertEqual(mmvq.count('device_submit.finish('), 1)
        self.assertIn('ncols_dst >= 2 && ncols_dst <= 8, false, stream', mmvq)
        self.assertIn('ky >= 2 && ky <= 8 && std::is_same<quantize_f<1>, quantize_and_reorder_q8_1_soa<1>>::value', quant)
        self.assertEqual(backend.count('lilicorr_device_after_wait('), 1)
        sync = backend[backend.index('static void ggml_backend_sycl_synchronize('):backend.index('static bool ggml_sycl_is_view_or_noop(')]
        self.assertLess(sync.index('SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));'), sync.index('lilicorr_device_after_wait('))
        self.assertNotRegex(header, r'(\.|->)\s*wait\s*\(|submit_barrier|sycl::queue\s*\(')
        before_drain = header[:header.index('inline void lilicorr_device_after_wait(')]
        self.assertNotIn('get_profiling_info<', before_drain)
        self.assertNotIn('get_info<', before_drain)
        self.assertEqual(header.count('get_profiling_info<'), 2)
        self.assertIn('limit = 2048', header)
        self.assertIn('std::array<char, GGML_MAX_NAME> name, weight;', header)
        self.assertIn('pending->records.clear()', header)
        self.assertIn('other_device_work=unmeasured', header)

    def test_old_host_and_phase_schemas_unchanged(self):
        for name in ('common/speculative.cpp', 'src/llama-context.cpp', PREFIX + 'ggml-sycl.cpp'):
            # Strict previous consumers need no schema relaxation or field changes.
            def schemas(root):
                return re.findall(r'"LLAMA_LILICORR_(?:SYCL_)?PROFILE .*', (root / name).read_text())
            self.assertEqual(schemas(self.candidate), schemas(self.seed))
        self.assertNotIn('LLAMA_LILICORR_PROFILE v=', (self.candidate / HEADER).read_text())

    def test_process_components_actual_chunk_host_mocks(self):
        common = (self.candidate / 'common/speculative.cpp').read_text()
        start = common.index('                // gather target features per extract layer;')
        end = common.index('            }\n        }\n\n        // LILICORR_PROFILE_ONLY_BEGIN', start)
        chunk = common[start:end]
        helper_start = common.index('struct lilicorr_process_components {')
        helper_end = common.index('// LILICORR_DEVICE_ONLY_END', helper_start)
        gate_start = common.index('static bool lilicorr_profile_enabled(')
        gate_end = common.index('// LILICORR_PROFILE_ONLY_END', gate_start)
        source = (HERE / 'fixtures/process-driver.cpp').read_text()
        for label, code in [('PROFILE_GATE', common[gate_start:gate_end]), ('COMPONENT_HELPER', common[helper_start:helper_end]), ('PROCESS_CHUNK', chunk)]:
            self.assertEqual(source.count('// EXTRACTED_' + label), 1)
            source = source.replace('// EXTRACTED_' + label, code)
        added = '\n'.join(re.findall(r'// LILICORR_DEVICE_ONLY_BEGIN\n(.*?)// LILICORR_DEVICE_ONLY_END', common, re.S))
        self.assertNotRegex(added, r'\bllama_\w+\s*\(|\bmemcpy\s*\(|\bwait\s*\(|\bsynchronize\s*\(')
        self.assertEqual(chunk.count('llama_get_embeddings_layer_inp('), 1)
        self.assertEqual(chunk.count('llama_get_embeddings_nextn('), 1)
        self.assertEqual(chunk.count('const int32_t rc = llama_process('), 1)
        cpp, exe = self.root / 'process-driver.cpp', self.root / 'mock-process'
        cpp.write_text(source)
        command = ['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(exe)]
        print('COMMAND:', ' '.join(command), flush=True)
        run(command, self.root)
        result = run([exe], self.root)
        print('COMMAND:', exe, '\n' + result.stdout, flush=True)
        self.assertIn('PASS extracted process chunk', result.stdout)

    def test_fc46_route_evidence_and_host_only_markers(self):
        backend = (self.candidate / (PREFIX + 'ggml-sycl.cpp')).read_text()
        header = (self.candidate / HEADER).read_text()
        selected = backend[backend.index('static bool bolt_route_selected_weight('):backend.index('static const char * bolt_route_name(')]
        self.assertNotIn('fc.weight', selected)  # original fulltrace did NOT record this weight
        self.assertEqual(backend.count('device_submit.finish_library();'), 4)
        for route in ('LIBRARY_DNNL_F16', 'LIBRARY_MKL_F16', 'LIBRARY_DNNL_F32', 'LIBRARY_MKL_F32'):
            self.assertEqual(backend.count('src1_ncols, "' + route + '");'), 1)
        self.assertIn('(!library_route_ || lilicorr_device_active->fc46)', header)
        self.assertIn('!event ? "device_unavailable_library"', header)
        self.assertIn('record.library_route ? "library_submit" : "event"', header)
        self.assertNotIn('sycl_interop::execute', backend)
        self.assertIn('"host_only_library_call"', header)

    def test_extracted_cpp_with_mocked_sycl_no_sdk(self):
        backend = (self.candidate / (PREFIX + 'ggml-sycl.cpp')).read_text()
        quant = (self.candidate / (PREFIX + 'quantize.hpp')).read_text()
        mmvq = (self.candidate / (PREFIX + 'mmvq.cpp')).read_text()
        helper = (self.candidate / (PREFIX + 'dpct/helper.hpp')).read_text()
        extracts = {
            'DIAGNOSTIC_HEADER': '#include "lilicorr-device-timing.hpp"',
            'SYNCHRONIZE': backend[backend.index('static void ggml_backend_sycl_synchronize('):backend.index('static bool ggml_sycl_is_view_or_noop(')],
            'SCOPE_GATE': backend[backend.index('static bool lilicorr_sycl_profile_requested('):backend.index('static const char * lilicorr_sycl_profile_name(')],
            'QUANTIZER_LAUNCH': quant[quant.index('template <template <int> typename quantize_f>'):],
            'MMVQ_LAUNCH': mmvq[mmvq.rindex('template <int ncols_dst>', 0, mmvq.index('static void reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(')):mmvq.index('static void reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(')],
            'QUEUE_INIT': helper[helper.index('      void init_queues() {'):helper.index('      /// Caller should acquire resource', helper.index('      void init_queues() {'))],
            'GRAPH_SCOPE': re.search(r'    lilicorr_device_scope device_scope\(.*?;\n', backend, re.S).group(),
        }
        for route in ('LIBRARY_DNNL_F16', 'LIBRARY_MKL_F16', 'LIBRARY_DNNL_F32', 'LIBRARY_MKL_F32'):
            start = backend.index('lilicorr_device_submit device_submit(src1_ncols == 46', backend.index('bolt_route_record("' + route + '"'))
            end = backend.index('device_submit.finish_library();', start) + len('device_submit.finish_library();')
            extracts[route] = backend[start:end]
        source = (HERE / 'fixtures/event-driver.cpp').read_text()
        for label, code in extracts.items():
            self.assertEqual(source.count('// EXTRACTED_' + label), 1)
            source = source.replace('// EXTRACTED_' + label, code)
        cpp = self.root / 'event-driver.cpp'
        cpp.write_text(source)
        for macro in (False, True):
            with self.subTest(DPCT_PROFILING_ENABLED=macro):
                exe = self.root / ('mock-dpct' if macro else 'mock-default')
                command = ['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-Wno-attributes', '-pthread',
                           '-I' + str(self.candidate / PREFIX), str(cpp), '-o', str(exe)]
                if macro:
                    command.append('-DDPCT_PROFILING_ENABLED')
                print('COMMAND:', ' '.join(command), flush=True)
                run(command, self.root)
                result = run([exe], self.root)
                print('COMMAND:', exe, '\n' + result.stdout, flush=True)
                self.assertIn('ALL CPU MOCK CHECKS PASSED', result.stdout)
                failed_wait = run([exe, 'wait-error'], self.root, check=False)
                print('COMMAND:', exe, 'wait-error; expected exit=1; observed=', failed_wait.returncode,
                      '\n' + failed_wait.stdout + failed_wait.stderr, flush=True)
                self.assertEqual(failed_wait.returncode, 1)
                self.assertIn('original-wait-error', failed_wait.stderr)
                self.assertIn('PASS original wait failure: no profiling query', failed_wait.stdout)


if __name__ == '__main__':
    print('Development CPU/source checks; no performance comparison or device/compiler proof.', flush=True)
    print('Environment:', platform.platform(), sys.version.replace('\n', ' '), flush=True)
    print(run(['c++', '--version'], HERE).stdout.splitlines()[0], flush=True)
    unittest.main(verbosity=2)
