#!/usr/bin/env python3
"""CPU preparation tests only: no Docker/GPU/network/service/installation commands."""
import copy
import gzip
import hashlib
import json
import os
import re
import runpy
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-performance.py'))
S = D['S']
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'
CXX = os.environ.get('CXX', 'clang++')
SANITIZER_ENV = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')


def temp():
    return tempfile.TemporaryDirectory(prefix='.cpu-', dir=HERE)


def materialize(root):
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(HERE / 'fixtures/support.tar.gz') as stream:
        for member in stream:
            assert member.isfile() and '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
            dest = root / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(stream.extractfile(member).read())
    for name in S['SOURCE_SCOPE']:
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.decompress((HERE / 'fixtures' / (name + '.gz')).read_bytes()))
    for name in S['load'](HERE / 'source-hashes-f32.json')['cpu_reference_fixtures']:
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.decompress((HERE / 'fixtures-f32' / (name + '.gz')).read_bytes()))
    (root / 'UPSTREAM_COMMIT').write_text(S['load'](HERE / 'source-hashes-f32.json')['upstream_commit'] + '\n')


def patch(root, *flags):
    result = subprocess.run(['patch', '--fuzz=0', '--batch', '-p1', *flags,
                             '-i', str(HERE / 'device-embedding-f32.patch')], cwd=root,
                            check=True, capture_output=True, text=True)
    assert not re.search(r'fuzz|offset|failed', (result.stdout + result.stderr).lower())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def function(text, start):
    index = text.index(start)
    return text[index:text.index('\n}\n', index) + 3]


def build_driver(root):
    """Compile extracted production/control code with the actual native C converter."""
    S['check_fixtures']()
    materialize(root); patch(root, '--forward')
    context = (root / 'src/llama-context.cpp').read_text()
    header = (root / 'src/llama-context.h').read_text()
    params = (root / 'src/llama-cparams.h').read_text()
    core = (root / 'ggml/src/ggml.c').read_text()
    quants = (root / 'ggml/src/ggml-quants.c').read_text()
    cpu_ops = (root / 'ggml/src/ggml-cpu/ops.cpp').read_text()
    parts = dict(
        INIT=function(context, 'void llama_context::init_lilicorr_device_embedding()'),
        DESTRUCTOR=function(context, 'llama_context::~llama_context()'),
        OWNERS=header[header.index('    void init_lilicorr_device_embedding();'):
                      header.index('    ggml_backend_buffer_ptr lilicorr_embd_buf;') + len('    ggml_backend_buffer_ptr lilicorr_embd_buf;')],
        CPARAMS_POINTER=next(line for line in params.splitlines() if 'struct ggml_tensor * lilicorr_tok_embd' in line),
        NEW2D=function(core, 'struct ggml_tensor * ggml_new_tensor_2d('),
        CPU_GET_ROWS=function(cpu_ops, 'static void ggml_compute_forward_get_rows_q('),
    )
    driver = (HERE / 'fixtures-f32/embedding-driver.cpp').read_text()
    for key, value in parts.items():
        marker = '// EXTRACTED_' + key
        assert driver.count(marker) == 1
        driver = driver.replace(marker, value)
    # Keep native initializer entries and function bodies verbatim. No substitute
    # half/quantization formula: ggml-impl.h supplies the pinned native half helper.
    entries = re.findall(r'^    \[GGML_TYPE_(?:F32|Q8_0)\] = \{\n.*?^    \},', core, re.M | re.S)
    assert len(entries) == 2
    native = '#include "ggml-quants.h"\n#include "ggml-impl.h"\n#include <math.h>\n'
    native += function(quants, 'void quantize_row_q8_0_ref(')
    native += function(quants, 'void dequantize_row_q8_0(')
    native += 'static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n' + '\n'.join(entries) + '\n};\n'
    for start in ('const struct ggml_type_traits * ggml_get_type_traits(', 'const char * ggml_type_name(',
                  'size_t ggml_type_size(', 'int64_t ggml_nelements(', 'int64_t ggml_nrows('):
        native += function(core, start)
    cpp, cfile, obj, exe = (root / name for name in ('driver.cpp', 'native-reference.c', 'native-reference.o', 'driver'))
    cpp.write_text(driver); cfile.write_text(native)
    flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-Wall', '-Wextra', '-Werror',
             '-I' + str(root / 'ggml/include'), '-isystem', str(root / 'ggml/src')]
    # Native GGML intentionally casts block_q8_0* callbacks to ggml_to_float_t's
    # void* signature. Omit function-type metadata ONLY in this extracted C TU
    # (as in the real unsanitized GGML library); keep all other ASan/UBSan checks.
    for command in ([os.environ.get('CC', 'clang'), '-std=c11', *flags, '-fno-sanitize=function', '-c', str(cfile), '-o', str(obj)],
                    [CXX, '-std=c++17', *flags, str(cpp), str(obj), '-lm', '-o', str(exe)]):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, (command, result.stdout, result.stderr)
    return exe


class SourceTests(unittest.TestCase):
    def test_exact_forward_reverse_whole_tree_and_private_headers(self):
        S['check_fixtures']()
        old_pins = S['load'](HERE / 'source-hashes.json')
        new_pins = S['load'](HERE / 'source-hashes-f32.json')
        self.assertEqual(old_pins['parent'], new_pins['parent'])
        self.assertEqual(old_pins['fixtures'], new_pins['fixtures'])
        self.assertEqual(old_pins['support_archive_sha256'], new_pins['support_archive_sha256'])
        self.assertEqual(old_pins['support'], new_pins['support'])
        for name in S['SOURCE_SCOPE'][1:]:
            self.assertEqual(old_pins['candidate'][name], new_pins['candidate'][name])
        with temp() as directory:
            root = Path(directory); parent, candidate = root / 'parent', root / 'candidate'
            materialize(parent); materialize(candidate)
            original = S['source_map'](parent)
            patch(candidate, '--dry-run', '--forward'); patch(candidate, '--forward')
            proof = S['check_sources'](parent, candidate)
            self.assertEqual(S['SOURCE_SCOPE'], proof['changed_relative_to_parent'])
            # Unlisted source, private header, deletion, and new-file drift all fail.
            for relative in ('src/llama-context.h', 'src/llama-cparams.h', 'src/llama-model.h'):
                path = candidate / relative; data = path.read_bytes(); path.write_bytes(data + b'\n')
                with self.assertRaises(AssertionError): S['check_sources'](parent, candidate)
                path.write_bytes(data)
            path = candidate / 'src/new-private-header.h'; path.write_text('unexpected')
            with self.assertRaises(AssertionError): S['check_sources'](parent, candidate)
            path.unlink()
            path = candidate / 'src/llama-model.h'; data = path.read_bytes(); path.unlink()
            with self.assertRaises(AssertionError): S['check_sources'](parent, candidate)
            path.write_bytes(data)
            patch(candidate, '--dry-run', '--reverse'); patch(candidate, '--reverse')
            self.assertEqual(original, S['source_map'](candidate))
            self.assertEqual(original, S['source_map'](parent))

    def test_actual_translation_units_syntax_and_no_math_delta(self):
        with temp() as directory:
            root = Path(directory); materialize(root)
            before = (root / 'src/models/dflash.cpp').read_text()
            patch(root, '--forward')
            after = (root / 'src/models/dflash.cpp').read_text()
            old = '    // tok_embd from the target model (shared via ctx_other)\n    auto * tok_embd = model.tok_embd;'
            new = ('    // Both noise and lattice candidate GET_ROWS use the context-owned copy, if active.\n'
                   '    // Otherwise retain the original shared target table; neither model is mutated.\n'
                   '    auto * tok_embd = model.tok_embd ? model.tok_embd : cparams.lilicorr_tok_embd;')
            self.assertEqual(after, before.replace(old, new, 1))
            self.assertIn('ggml_get_rows(ctx0, tok_embd, inp_tokens)', after)
            self.assertIn('ggml_get_rows(ctx0, tok_embd, ggml_reshape_1d(ctx0, cand_run, nodes * n_blocks))', after)
            self.assertIn('build_lilicorr_scores(*this, model, tok_embd, inp_anchor)', after)
            context = (root / 'src/llama-context.cpp').read_text()
            legacy = root / 'q8-reference'; materialize(legacy)
            result = subprocess.run(['patch', '--fuzz=0', '--batch', '-p1', '--forward',
                                     '-i', str(HERE / 'device-embedding.patch')], cwd=legacy,
                                    check=True, capture_output=True, text=True)
            self.assertNotRegex(result.stdout + result.stderr, r'fuzz|offset|failed')
            q8_context = (legacy / 'src/llama-context.cpp').read_text()
            self.assertEqual(S['load'](HERE / 'source-hashes.json')['candidate']['src/llama-context.cpp'],
                             hashlib.sha256(q8_context.encode()).hexdigest())
            signature = 'void llama_context::init_lilicorr_device_embedding()'
            self.assertEqual(context, q8_context.replace(function(q8_context, signature),
                                                        function(context, signature), 1))  # Only initializer/log changed.
            self.assertLess(context.index('    init_lilicorr_device_embedding();'), context.index('    // init the memory module'))
            header = (root / 'src/llama-context.h').read_text()
            self.assertLess(header.index('ggml_context_ptr lilicorr_embd_ctx'), header.index('ggml_backend_buffer_ptr lilicorr_embd_buf'))
            self.assertLess(header.index('ggml_backend_buffer_ptr lilicorr_embd_buf'), header.index('ggml_backend_sched_ptr sched'))
            for file in ('src/llama-context.cpp', 'src/models/dflash.cpp'):
                command = [CXX, '-std=c++17', '-fsyntax-only',
                           *['-I' + str(root / p) for p in ('src', 'include', 'ggml/include', 'ggml/src')], str(root / file)]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            print('full patched llama-context.cpp + dflash.cpp syntax passed against exact original headers')

    def test_extracted_cpp_guards_copy_fence_ownership_and_failures(self):
        with temp() as directory:
            exe = build_driver(Path(directory))
            result = subprocess.run([str(exe)], capture_output=True, text=True, env=SANITIZER_ENV)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            print(result.stdout.strip())

    def test_native_cpu_reference_f32_chunk_boundaries_and_edge_values(self):
        with temp() as directory:
            exe = build_driver(Path(directory))
            result = subprocess.run([str(exe), 'numeric'], capture_output=True, text=True, env=SANITIZER_ENV)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            print(result.stdout.strip())

    def test_pinned_native_layout_and_default_queue_fence_assumptions(self):
        with tarfile.open(HERE / 'fixtures/support.tar.gz') as stream:
            text = lambda name: stream.extractfile(name).read().decode()
            sycl = text('ggml/src/ggml-sycl/ggml-sycl.cpp')
            getrows = text('ggml/src/ggml-sycl/getrows.cpp')
            queues = text('ggml/src/ggml-sycl/dpct/helper.hpp')
        init = function(sycl, 'ggml_backend_sycl_buffer_init_tensor(')
        self.assertIn('new ggml_tensor_extra_gpu{}', init)
        self.assertIn('case GGML_TYPE_Q8_0:', init)
        setter = function(sycl, 'static void ggml_backend_sycl_buffer_set_tensor(')
        getter = function(sycl, 'static void ggml_backend_sycl_buffer_get_tensor(')
        self.assertIn('ctx->staging.events[slot] = stream->memcpy', setter)
        self.assertNotIn('reorder_qw', setter)
        self.assertIn('default_queue()', getter)
        self.assertIn('.wait()', getter)
        self.assertIn('default_queue() { return in_order_queue(); }', queues)
        self.assertIn('get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>', getrows)
        self.assertNotIn('case GGML_TYPE_F32:', init)  # F32 receives no Q8 reorder extras.
        f32 = getrows[getrows.index('case GGML_TYPE_F32:'):getrows.index('case GGML_TYPE_I32:')]
        self.assertIn('(const float *)dst->src[0]->data', f32)
        self.assertIn('(float *)dst->data', f32)
        self.assertIn('dst_row[i00] = src0_row[i00];', function(getrows, 'static void k_get_rows_float('))
        self.assertNotIn('reorder', getrows)
        reorder = function(sycl, 'static bool should_reorder_tensor(ggml_backend_sycl_context&')
        self.assertIn('dst->op == GGML_OP_MUL_MAT', reorder)
        self.assertIn('size >= ((size_t)4 * MEM_SIZE_1G)', sycl)
        self.assertIn('drop_host_staging();', sycl)
        self.assertIn('staging.events[i].wait_and_throw();', sycl)

    def test_fresh_whole_source_and_each_of_six_binary_proofs(self):
        with temp() as directory:
            root = Path(directory); parent, candidate = root / 'parent', root / 'candidate'
            materialize(parent); materialize(candidate); patch(candidate, '--forward')
            for tree in (parent, candidate):
                for name in S['ARTIFACTS']:
                    path = tree / 'build-sycl/bin' / name; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(name + (' changed' if tree == candidate and name in S['CHANGED_BINARIES'] else ' original'))
            (root / 'source-hashes-f32.json').write_bytes((HERE / 'source-hashes-f32.json').read_bytes())
            attempt = root / 'build-attempt-r3'; attempt.mkdir()
            (attempt / 'exit-code.txt').write_text('0\n')
            (attempt / 'post-build-verification.log').write_text('LILICORR_PROFILE_BUILD_PASSED\n')
            (attempt / 'runtime-precheck.log').write_text('LILICORR_PROFILE_DEPENDENCIES_PASSED\n')
            (attempt / 'build.log').write_text('LILICORR_DEVICE_EMBEDDING_MODULE_R2_PASSED\n')
            save(attempt / 'source-diff.json', dict(changed_relative_to_parent=S['SOURCE_SCOPE']))
            save(attempt / 'parent-binaries.json', S['binaries'](parent))
            save(attempt / 'candidate-binaries.json', S['binaries'](candidate))
            original = dict(source=S['source_map'](parent), binaries=S['binaries'](parent))
            calls = []
            def qualified():
                calls.append(1)
                self.assertEqual(original, dict(source=S['source_map'](parent), binaries=S['binaries'](parent)))
                return original
            p = dict(CANDIDATE=parent, candidate_proof=qualified)
            with mock.patch.dict(S['_G'], HERE=root, PARENT=parent, CANDIDATE=candidate):
                S['proof'](p); S['proof'](p)
                self.assertEqual(2, len(calls))
                for name in S['ARTIFACTS']:
                    for tree in (parent, candidate):
                        path = tree / 'build-sycl/bin' / name; data = path.read_bytes(); path.write_bytes(data + b'drift')
                        with self.assertRaises(AssertionError): S['proof'](p)
                        path.write_bytes(data)
                # Even a refreshed candidate artifact record cannot authorize a second delta.
                for name in set(S['ARTIFACTS']) - set(S['CHANGED_BINARIES']):
                    path = candidate / 'build-sycl/bin' / name; data = path.read_bytes()
                    path.write_bytes(data + b'drift')
                    save(attempt / 'candidate-binaries.json', S['binaries'](candidate))
                    with self.assertRaises(AssertionError): S['proof'](p)
                    path.write_bytes(data)
                save(attempt / 'candidate-binaries.json', S['binaries'](candidate))
                path = candidate / 'src/llama-context.h'; data = path.read_bytes(); path.write_bytes(data + b'drift')
                with self.assertRaises(AssertionError): S['proof'](p)
                path.write_bytes(data)
                (attempt / 'runtime-precheck.log').write_text('WRONG_MARKER')
                with self.assertRaises(AssertionError): S['proof'](p)

    def test_build_is_original_seed_incremental_cpu_only(self):
        path = HERE / 'build-candidate.sh'
        subprocess.run(['bash', '-n', str(path)], check=True)
        script = path.read_text()
        self.assertNotRegex(script, r'--device|/dev/dri|--privileged|docker cp|ssh ')
        self.assertIn('SEED=$PARENT', script)
        self.assertIn('cp -a --reflink=auto "$SEED" "$CANDIDATE"', script)
        self.assertIn('UNEXPECTED_COMPILE_SCOPE', script)
        self.assertIn('object src/CMakeFiles/llama.dir/', script)
        self.assertNotIn('seed-incremental.patch', script)
        self.assertNotIn('cmake -S', script)
        for name in S['ARTIFACTS']: self.assertIn(name, script)
        for name in ('LILICORR_PROFILE_BUILD_PASSED', 'LILICORR_PROFILE_DEPENDENCIES_PASSED'):
            self.assertIn(name, script)
            self.assertIn(name, (S['FIRST'] / 'source-proof.py').read_text())
        self.assertIn('aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0', script)


class RunnerTests(unittest.TestCase):
    def bound(self, mode):
        runtime, analysis, c = D['bind'](mode)
        p = dict(OUT=QUALIFIED, C={}, CANDIDATE=S['PARENT'],
                 make_create=lambda cell, _h: S['load'](QUALIFIED / cell / 'create-argv.json'))
        c['configure'](p)
        return runtime, analysis, c, p

    def test_actual_private_function_globals_and_clean_abba_bindings(self):
        contracts = []
        for mode in ('baseline', 'candidate'):
            runtime, analysis, c, p = self.bound(mode); contracts.append(c)
            before = copy.deepcopy(p)
            run_name = 'candidate-r2' if mode == 'candidate' else 'baseline-r1'
            self.assertEqual(c['OUT'], D['ROOT'] / run_name)
            self.assertEqual(c['PREFIX'], 'bolt-lilicorr-device-embedding-' + run_name + '-')
            self.assertIs(c['configure'].__globals__, c)
            self.assertIs(analysis['analyze'].__globals__['C'], c)
            for method in ('pid', 'verify_restored', '__init__'):
                self.assertIs(getattr(runtime['Runtime'], method).__globals__['C'], c)
            self.assertIs(runtime['capture'].__globals__['C'], c)
            for cell in c['CELLS']:
                argv = c['make_create'](cell, None)
                base = p['make_create'](c['base_cell'](cell), None)
                self.assertEqual(c['server_argv'](base), c['server_argv'](argv))
                active = mode == 'candidate' and 'W3' in cell
                self.assertIn(D['source_bind'](S['CANDIDATE'] if active else S['PARENT']), argv)
                self.assertNotIn(D['source_bind'](S['PARENT'] if active else S['CANDIDATE']), argv)
                self.assertEqual('W3' in cell, D['Q']['candidate_draft_bind']() in argv)
                self.assertNotIn('LLAMA_LILICORR_PROFILE', ' '.join(argv))
                self.assertNotIn('W7', cell)
            self.assertEqual(p, before)
        self.assertIsNot(contracts[0], contracts[1])
        self.assertNotEqual(contracts[0]['OUT'], contracts[1]['OUT'])
        for name in D['OWN_FILES']: self.assertTrue((HERE / name).is_file(), name)

    def test_actual_container_identity_env_mounts_and_no_rewriting(self):
        for mode in ('baseline', 'candidate'):
            _rt, _a, c, _p = self.bound(mode)
            for cell in c['CELLS']:
                record = S['load'](c['qualified_dir'](c['role_for'](cell)) / 'container-before.json')
                record.update(Name='/' + c['name'](cell), Id='b' * 64)
                record['Config']['Hostname'] = 'b' * 12
                record['Config']['Cmd'] = c['make_create'](cell, None)[-3:]
                if c['role_for'](cell) == 'W3':
                    substitutions = {D['Q']['old_draft_bind'](): D['Q']['candidate_draft_bind']()}
                    if mode == 'candidate': substitutions[D['source_bind'](S['PARENT'])] = D['source_bind'](S['CANDIDATE'])
                    record['HostConfig']['Binds'] = [substitutions.get(x, x) for x in record['HostConfig']['Binds']]
                    record['Mounts'] = D['Q']['_expected_mounts'](record, cell)
                    if mode == 'candidate':
                        next(m for m in record['Mounts'] if m['Destination'] == '/src')['Source'] = str(S['CANDIDATE'])
                before = copy.deepcopy(record)
                self.assertTrue(c['check_container'](record, cell, None))
                self.assertEqual(before, record)
                for corruption in ('env', 'source', 'write', 'id'):
                    bad = copy.deepcopy(record)
                    if corruption == 'env': bad['Config']['Env'].append('LLAMA_LILICORR_PROFILE=1')
                    if corruption == 'source': next(m for m in bad['Mounts'] if m['Destination'] == '/src')['Source'] = '/wrong'
                    if corruption == 'write': next(m for m in bad['Mounts'] if m['Destination'] == '/src')['RW'] = True
                    if corruption == 'id': bad['Config']['Hostname'] = 'wrong'
                    with self.assertRaises(AssertionError): c['check_container'](bad, cell, None)

    def test_placement_fails_closed_and_acceptance_is_exact(self):
        good = ('lilicorr_device_embedding: active backend=SYCL0 buffer=SYCL0 source_type=q8_0 '
                'source_bytes=570425344 type=f32 tensor_bytes=2147483648 buffer_bytes=2147483648\n')
        self.assertTrue(D['check_placement'](good, True)['active'])
        self.assertFalse(D['check_placement']('', False)['active'])
        for bad in ('', good + good, 'lilicorr_device_embedding: inactive reason=oom',
                    good.replace('SYCL0', 'CPU'), good.replace('570425344', '1'),
                    good.replace('tensor_bytes=2147483648', 'tensor_bytes=570425344'),
                    good.replace('type=f32', 'type=q8_0'), good.replace('source_type=q8_0', 'source_type=f32'),
                    good.replace('buffer_bytes=2147483648', 'buffer_bytes=3221225473'),
                    good.replace('buffer_bytes=2147483648', 'buffer_bytes=2147483647'),
                    'lilicorr_device_embedding: active backend=SYCL0 buffer=SYCL0 type=q8_0 tensor_bytes=570425344 buffer_bytes=570425344'):
            with self.assertRaises(AssertionError): D['check_placement'](bad, True)
        with self.assertRaises(AssertionError): D['check_placement'](good, False)
        for point, counts in [('short', (129, 84)), ('long', (133, 82))]:
            row = dict(cell='VSPF-W3a', point=point, timings=dict(zip(('draft_n', 'draft_n_accepted'), counts)))
            self.assertIs(row, D['check_acceptance'](row))
            row['timings']['draft_n_accepted'] -= 1
            with self.assertRaises(AssertionError): D['check_acceptance'](row)
        failed_q8 = dict(cell='VSPF-W3a', point='short', timings=dict(draft_n=126, draft_n_accepted=85))
        with self.assertRaises(AssertionError): D['check_acceptance'](failed_q8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
