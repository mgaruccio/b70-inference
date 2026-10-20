#!/usr/bin/env python3
"""CPU preparation only. No Docker, GPU, network, installation, or live-proof claim."""
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
SYCL = HERE.parent / '20261019-lilicorr-b70-sycl-profile'
SEED = HERE.parent / '20261019-lilicorr-b70-decoder-profile'
D = runpy.run_path(str(HERE / 'run-profile.py'))
S = D['S']
A = runpy.run_path(str(HERE / 'analyze-profile.py'))
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'


def fixture():
    return json.loads(gzip.decompress((HERE / 'fixtures/combined-startup-measured.json.gz').read_bytes()))


def materialize(root):
    for name in S['SOURCE_SCOPE']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.decompress((HERE / 'fixtures' / (name + '.gz')).read_bytes()))
    (root / 'UPSTREAM_COMMIT').write_text(S['load'](HERE / 'source-hashes.json')['upstream_commit'] + '\n')


def patch(root, name, *flags):
    result = subprocess.run(['patch', '--fuzz=0', '--batch', '-p1', *flags, '-i', str(name)],
                            cwd=root, check=True, capture_output=True, text=True)
    assert not re.search(r'fuzz|offset|failed', (result.stdout + result.stderr).lower())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PreparationTests(unittest.TestCase):
    def test_full_incremental_exact_pins_reverse_and_no_ordinary_source_change(self):
        S['check_fixtures']()
        pins = S['load'](HERE / 'source-hashes.json')
        with tempfile.TemporaryDirectory() as directory:
            full, seed = Path(directory) / 'full', Path(directory) / 'seed'
            materialize(full); materialize(seed)
            patch(seed, SEED / 'lilicorr-profile.patch', '--forward')
            self.assertEqual(pins['seed'], {n: S['sha'](seed / n) for n in S['SOURCE_SCOPE']})
            for tree, name in ((full, 'lilicorr-profile.patch'), (seed, 'seed-incremental.patch')):
                patch(tree, HERE / name, '--dry-run', '--forward')
                patch(tree, HERE / name, '--forward')
            for name in S['SOURCE_SCOPE']:
                data = (full / name).read_bytes()
                self.assertEqual(data, (seed / name).read_bytes())
                self.assertEqual(pins['candidate'][name], hashlib.sha256(data).hexdigest())
                marker = 'LILICORR_SYCL_PROFILE_ONLY' if name.startswith('ggml/') else 'LILICORR_PROFILE_ONLY'
                plain = re.sub(r'^[ \t]*// ' + marker + r'_BEGIN\n.*?^[ \t]*// ' + marker +
                               r'_END\n', '', data.decode(), flags=re.M | re.S)
                self.assertEqual(gzip.decompress((HERE / 'fixtures' / (name + '.gz')).read_bytes()), plain.encode())
            for tree, name in ((full, 'lilicorr-profile.patch'), (seed, 'seed-incremental.patch')):
                patch(tree, HERE / name, '--dry-run', '--reverse')
                patch(tree, HERE / name, '--reverse')
            self.assertEqual(pins['seed'], {n: S['sha'](seed / n) for n in S['SOURCE_SCOPE']})
            patch(seed, SEED / 'lilicorr-profile.patch', '--reverse')
            self.assertEqual(S['source_map'](full), S['source_map'](seed))

    def test_scope_hook_and_all_existing_fusion_accounting_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); materialize(root)
            patch(root, HERE / 'lilicorr-profile.patch', '--forward')
            sycl = (root / 'ggml/src/ggml-sycl/ggml-sycl.cpp').read_text()
            context = (root / 'src/llama-context.cpp').read_text()
        self.assertNotIn('lilicorr_sycl_profile_graph_marker', sycl)
        self.assertNotIn('lilicorr_sycl_profile_marker(', sycl)
        self.assertIn('lilicorr_sycl_profile_context && lilicorr_sycl_profile_requested()', sycl)
        self.assertIn('return (void *)set_lilicorr_profile_scope;', sycl)
        self.assertIn('const char * marker_name = "lilicorr.context";', sycl)
        self.assertIn('return ggml_time_us();', sycl)
        self.assertNotRegex(sycl, r'std::chrono|steady_clock')
        self.assertLess(sycl.index('bool ok = ggml_sycl_compute_forward'), sycl.index('stage=op'))
        added = '\n'.join(re.findall(r'// LILICORR_SYCL_PROFILE_ONLY_BEGIN\n(.*?)// LILICORR_SYCL_PROFILE_ONLY_END', sycl, re.S))
        self.assertNotRegex(added, r'\.wait\s*\(|queue\.wait|sycl::event|memcpy\s*\(')
        original_patch = (SEED / 'lilicorr-profile.patch').read_text()
        self.assertEqual(re.findall(r'profile_scope.consume\(.*', original_patch),
                         re.findall(r'profile_scope.consume\(.*', (HERE / 'lilicorr-profile.patch').read_text()))
        self.assertRegex(context, r'profile_scope\(backends, model.hparams.lilicorr_candidate_topk > 0\);\n'
                         r'    // LILICORR_PROFILE_ONLY_END\n'
                         r'    auto status = ggml_backend_sched_graph_compute_async\(sched.get\(\), gf\);\n'
                         r'    // LILICORR_PROFILE_ONLY_BEGIN\n    profile_scope.restore\(\);')

    def test_actual_extracted_cpp_guard_and_graph_compute_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); materialize(root)
            patch(root, HERE / 'lilicorr-profile.patch', '--forward')
            sycl = (root / 'ggml/src/ggml-sycl/ggml-sycl.cpp').read_text()
            ctx = (root / 'src/llama-context.cpp').read_text()
            backend = sycl[sycl.index('static thread_local bool lilicorr_sycl_profile_context'):sycl.index('static const char * lilicorr_sycl_profile_name')]
            helper = ctx[ctx.index('static thread_local unsigned lilicorr_context_profile_depth'):ctx.index('// LILICORR_PROFILE_ONLY_END\nggml_status llama_context::graph_compute')]
            compute = ctx[ctx.index('ggml_status llama_context::graph_compute('):ctx.index('bool llama_context::sched_copy_experts(')]
            source = (HERE / 'fixtures/scope-guard-driver.cpp').read_text()
            for label, code in [('BACKEND_SCOPE', backend), ('CONTEXT_SCOPE', helper), ('GRAPH_COMPUTE', compute)]:
                self.assertEqual(1, source.count('// EXTRACTED_' + label))
                source = source.replace('// EXTRACTED_' + label, code)
            cpp, exe = root / 'scope.cpp', root / 'scope-test'
            cpp.write_text(source)
            result = subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-pthread', str(cpp), '-o', str(exe)],
                                    capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_build_contract_markers_and_no_gpu_or_host_install(self):
        script = (HERE / 'build-candidate.sh').read_text()
        subprocess.run(['bash', '-n', str(HERE / 'build-candidate.sh')], check=True)
        self.assertNotRegex(script, r'--device|/dev/dri|--privileged|docker cp|ssh ')
        self.assertIn('cp -a --reflink=auto "$SEED" "$CANDIDATE"', script)
        for key in ('ROOT', 'PARENT', 'SEED', 'CANDIDATE'):
            self.assertIn(key + '=' + str(S[key]), script)
        for marker in ('LILICORR_PROFILE_BUILD_PASSED', 'LILICORR_PROFILE_DEPENDENCIES_PASSED'):
            self.assertIn(marker, script)
            self.assertIn(marker, (FIRST / 'source-proof.py').read_text())
        for text in ('4194304 + 1048576', 'CMAKE_HOME_DIRECTORY:INTERNAL=/src',
                     'seed-built-before.json', 'seed-built-after.json',
                     'candidate-source-before.json', 'candidate-source-after.json',
                     'aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0'):
            self.assertIn(text, script)

    def test_original_eighteen_test_sycl_suite_unchanged(self):
        result = subprocess.run([sys.executable, '-B', str(SYCL / 'test-profile.py')],
                                check=True, capture_output=True, text=True)
        self.assertRegex(result.stdout + result.stderr, r'Ran 18 tests')
        print(result.stdout + result.stderr)


class CoverageTests(unittest.TestCase):
    def test_first_public_job_scope_probe_rejects_head_only_and_startup_only(self):
        lines = next(item for item in fixture()['requests'] if item['phase'] == 'measured')['lines']
        proof = D['check_first_job_scope']('\n'.join(lines))
        self.assertTrue(proof['passed'])
        self.assertGreater(proof['decoder_projection_records'], 0)
        head_only = [line for line in lines if A['MARKER'] not in line
                     or not (A['event'](line)['stage'] == 'op'
                             and A['event'](line)['src0_name'] == 'blk.0.attn_q.weight')]
        with self.assertRaisesRegex(AssertionError, 'no measured decoder projection'):
            D['check_first_job_scope']('\n'.join(head_only))
        with self.assertRaisesRegex(AssertionError, 'no current public request'):
            D['check_first_job_scope']('\n'.join(line for line in lines if A['MARKER'] in line))

    def test_all_synthetic_requests_inside_real_boundaries_and_full_scope(self):
        f = fixture(); g = A['BASE']['requests'].__globals__
        for item in f['requests']:
            lines = item['lines']
            start = [i for i, l in enumerate(lines) if g['START'].search(l)]
            end = [i for i, l in enumerate(lines) if g['END'].search(l)]
            self.assertEqual((1, 1), (len(start), len(end)))
            self.assertTrue(all(start[0] < i < end[0] for i, l in enumerate(lines) if A['MARKER'] in l))
            if item['phase'] != 'measured':
                continue
            result = A['request_sycl'](item, item['phase_request'])
            scope = result['context_scope']
            count = 43 if item['point'] == 'short' else 45
            self.assertEqual(count, scope['decoder_projection_count'])
            self.assertEqual(count - 1, len(scope['body_rounds']))
            self.assertEqual('mul_mat_qkv_fused', scope['first_round']['projection_group'])
            self.assertEqual(825, scope['first_round']['observed_backend_nodes'])
            self.assertEqual(1, scope['first_round']['uninstrumented_scheduler_nodes'])
            self.assertEqual(0, result['scheduler_coverage']['uncovered_ubatches'])

    def test_head_only_and_memory_metadata_are_not_decoder_proof(self):
        item = next(i for i in fixture()['requests'] if i['phase'] == 'measured')
        old = copy.deepcopy(item)
        head_ids = {b['graph'] for b in A['_graph_bundles'](A['parse_sycl'](old['lines']))
                    if b['summary']['graph_nodes'] == 167}
        old['lines'] = [l for l in old['lines'] if A['MARKER'] not in l or A['event'](l)['graph'] in head_ids]
        with self.assertRaises(AssertionError):
            A['request_sycl'](old, old['phase_request'])
        for before, after in [('first_op=MUL_MAT', 'first_op=CONT'),
                              ('src0_type=bf16', 'src0_type=f32'),
                              ('blk.0.attn_q.weight', 'blk.0.attn_q.weight (view)')]:
            bad = copy.deepcopy(item)
            index = next(i for i, l in enumerate(bad['lines']) if 'src0_name=blk.0.attn_q.weight ' in l)
            bad['lines'][index] = bad['lines'][index].replace(before, after)
            with self.assertRaisesRegex(AssertionError, 'exactly once per round'):
                A['request_sycl'](bad, bad['phase_request'])

    def test_missing_injection_decoder_and_round_rejected(self):
        item = next(i for i in fixture()['requests'] if i['phase'] == 'measured')
        bundles = A['_graph_bundles'](A['parse_sycl'](item['lines'], item['first_line']))
        for nodes in (59, 658):
            bundle = next(b for b in bundles if b['summary']['graph_nodes'] == nodes)
            bad = copy.deepcopy(item)
            bad['lines'] = [l for l in bad['lines'] if A['MARKER'] not in l or A['event'](l)['graph'] != bundle['graph']]
            with self.assertRaises(AssertionError):
                A['request_sycl'](bad, bad['phase_request'])
        # Removing both the round's backend evidence AND phase owner must not evade the count gate.
        bundle = next(b for b in bundles if b['summary']['graph_nodes'] == 658)
        owner = A['_phase_parent'](bundle, item['phase_request']['events'])
        ids = {b['graph'] for b in bundles if A['_phase_parent'](b, item['phase_request']['events'])['ubatch_line'] == owner['ubatch_line']}
        bad = copy.deepcopy(item)
        bad['lines'] = [l for l in bad['lines'] if A['MARKER'] not in l or A['event'](l)['graph'] not in ids]
        bad['phase_request']['events'] = [e for e in bad['phase_request']['events'] if e['line'] not in (owner['ubatch_line'], owner['parent_line'])]
        with self.assertRaisesRegex(AssertionError, 'measured draft round'):
            A['request_sycl'](bad, bad['phase_request'])

    def _full_logs(self, out, c):
        f = fixture()
        baseline = A['RetainedPath'](FIRST / 'profile-r1/VSPF-W3a/server.stderr.log').read_text()
        requests, _ = A['BASE']['requests'](baseline)
        for old, item in zip(requests[25:], f['requests']):
            block = '\n'.join(old['lines'])
            self.assertEqual(1, baseline.count(block))
            baseline = baseline.replace(block, '\n'.join(item['lines']), 1)
        # Public W3 context records are permitted, but are not measured-round proof.
        public = '\n'.join(f['startup_lines'])
        baseline = baseline.replace(requests[0]['lines'][-1], public + '\n' + requests[0]['lines'][-1], 1)
        report = dict(status='analyzed', errors=[], requests=[], startup_events={})
        for cell in c['CELLS']:
            (out / cell).mkdir()
            text = baseline if 'W3' in cell else A['RetainedPath'](FIRST / 'profile-r1' / cell / 'server.stderr.log').read_text()
            (out / cell / 'server.stderr.log').write_text(text)
            (out / cell / 'server.stdout.log').write_text('')
            if 'W3' in cell:
                parsed, _ = A['BASE']['requests'](text)
                for request, item in zip(parsed[25:], f['requests']):
                    report['requests'].append(dict(cell=cell, phase=item['phase'], point=item['point'],
                        ordinal=item['ordinal'], events=request['events'], counters=item['phase_request']['counters']))
        return report

    def test_full_analyzer_public_gzip_baseline_and_missing_request(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); c = D['bind']()[2]
            phase_report = self._full_logs(out, c)
            with mock.patch.dict(A['BASE'], {'analyze': lambda *_: copy.deepcopy(phase_report)}):
                result = A['analyze'](out, c)
                self.assertEqual('analyzed', result['status'], result['errors'])
                self.assertEqual(12, len(result['sycl_diagnostic']['graphs']))
                w3 = next(x for x in result['sycl_diagnostic']['public'] if 'W3' in x['cell'])
                self.assertEqual(1, w3['requests'][0]['enabled_graph_count'])
                for name in ('server.stderr.log', 'server.stdout.log'):
                    path = out / 'VSPF-W3a' / name
                    path.with_name(name + '.gz').write_bytes(gzip.compress(path.read_bytes()))
                    path.unlink()
                result = A['analyze'](out, c)
                self.assertEqual('analyzed', result['status'], result['errors'])
                raw = out / 'VSPF-W3a/server.stderr.log'
                raw.write_text('mismatch')
                result = A['analyze'](out, c)
                self.assertEqual('failed', result['status'])
                self.assertIn('representations differ', str(result['errors']))
                raw.unlink()
                measured_path = out / 'VSPF-W3b/server.stderr.log'
                original = measured_path.read_text()
                requests, _ = A['BASE']['requests'](original)
                block = '\n'.join(requests[26]['lines'])
                clean = '\n'.join(l for l in requests[26]['lines'] if A['MARKER'] not in l)
                measured_path.write_text(original.replace(block, clean, 1))
                failed = A['analyze'](out, c)
                self.assertEqual('failed', failed['status'])
                self.assertIn('missing SYCL diagnostic events', str(failed['errors']))
                measured_path.write_text(original)
                baseline = out / 'VSPF-A1/server.stderr.log'
                baseline.write_text('\n'.join(fixture()['startup_lines']) + '\n' + baseline.read_text())
                self.assertEqual('failed', A['analyze'](out, c)['status'])

    def test_duplicate_decoder_projection_is_not_extra_coverage(self):
        item = next(i for i in fixture()['requests'] if i['phase'] == 'measured')
        index = next(i for i, l in enumerate(item['lines']) if 'src0_name=blk.0.attn_q.weight ' in l)
        second = item['lines'][index + 1]
        self.assertIn('src0_name=Qcur ', second)
        item['lines'][index + 1] = second.replace('first_op=ADD ', 'first_op=MUL_MAT ').replace(
            'src0_name=Qcur ', 'src0_name=blk.0.attn_q.weight ').replace('src0_type=Q8_0 ', 'src0_type=bf16 ')
        # The fixture's second source type is inherited Q8_0; prove this really became a second bf16 projection.
        self.assertEqual('bf16', A['event'](item['lines'][index + 1])['src0_type'])
        with self.assertRaisesRegex(AssertionError, 'exactly once per round'):
            A['request_sycl'](item, item['phase_request'])

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.analysis, self.c = D['bind']()
        self.p = dict(OUT=QUALIFIED, C={}, CANDIDATE=S['PARENT'],
                      make_create=lambda cell, _h: S['load'](QUALIFIED / cell / 'create-argv.json'))
        self.c['configure'](self.p)

    def test_actual_paths_source_pid_model_and_private_bindings(self):
        before = copy.deepcopy(self.p)
        self.assertEqual(self.c['ROOT'], Path('/home/mike/b70-evals/20261020-lilicorr-context-profile'))
        self.assertEqual(self.c['OUT'], self.c['ROOT'] / 'profile-r1')
        self.assertEqual(self.c['PREFIX'], 'bolt-lilicorr-context-profile-r1-')
        self.assertNotIn('_FIRST_GLOBALS', D['_FIRST_GLOBALS'])
        self.assertNotIn('_BASE_GLOBALS', S['_BASE_GLOBALS'])
        for cell in self.c['CELLS']:
            argv = self.c['make_create'](cell, None)
            if 'W3' in cell:
                self.assertIn(str(S['CANDIDATE']) + ':/src:ro', argv)
                self.assertNotIn(str(S['PARENT']) + ':/src:ro', argv)
                self.assertIn('LLAMA_LILICORR_PROFILE=1', argv)
            else:
                self.assertNotIn('LLAMA_LILICORR_PROFILE=1', argv)
        self.assertEqual(before, self.p)
        record = dict(pid=1, origin='/proc/1/environ', profile_environment_origin='/proc/1/environ',
                      profile_environment=[['LLAMA_LILICORR_PROFILE', '1']])
        self.assertTrue(D['_FIRST_RUN']['check_profile_pid'](record, 'VSPF-W3a'))
        record['profile_environment'][0][1] = '0'
        with self.assertRaises(AssertionError):
            D['_FIRST_RUN']['check_profile_pid'](record, 'VSPF-W3a')
        for path in D['OWN_FILES']:
            self.assertTrue((HERE / path).is_file(), path)
        fresh = runpy.run_path(str(HERE.parent / '20261019-lilicorr-q8-drafter/run-performance.py'))
        for key in ('CANDIDATE', 'CANDIDATE_SIZE', 'CANDIDATE_SHA256'):
            self.assertEqual(fresh[key], D['Q'][key])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q8.gguf'; path.write_bytes(b'q8-fixture')
            size, sha = path.stat().st_size, S['sha'](path)
            D['Q']['candidate_file_proof'](path, expected_size=size, expected_sha256=sha)
            path.write_bytes(b'q8-drift!!')
            with self.assertRaises(AssertionError):
                D['Q']['candidate_file_proof'](path, expected_size=size, expected_sha256=sha)

    def test_actual_new_container_identity_mounts_and_environment(self):
        source_bind = D['_FIRST_RUN']['source_bind']
        for cell in self.c['CELLS']:
            record = S['load'](self.c['qualified_dir'](self.c['role_for'](cell)) / 'container-before.json')
            record.update(Name='/' + self.c['name'](cell), Id='b' * 64)
            record['Config']['Hostname'] = 'b' * 12
            record['Config']['Cmd'] = self.c['make_create'](cell, None)[-3:]
            if self.c['role_for'](cell) == 'W3':
                record['Config']['Env'].append(D['PROFILE_ENV'] + '=1')
                substitutions = {D['Q']['old_draft_bind'](): D['Q']['candidate_draft_bind'](),
                                 source_bind(S['PARENT']): source_bind(S['CANDIDATE'])}
                record['HostConfig']['Binds'] = [substitutions.get(x, x) for x in record['HostConfig']['Binds']]
                record['Mounts'] = D['Q']['_expected_mounts'](record, cell)
                next(m for m in record['Mounts'] if m['Destination'] == '/src')['Source'] = str(S['CANDIDATE'])
            before = copy.deepcopy(record)
            self.assertTrue(self.c['check_container'](record, cell, None))
            self.assertEqual(before, record)
            if self.c['role_for'](cell) == 'W3':
                for field in ('mount', 'environment'):
                    bad = copy.deepcopy(record)
                    if field == 'mount':
                        next(m for m in bad['Mounts'] if m['Destination'] == '/src')['Source'] = str(S['PARENT'])
                    else:
                        bad['Config']['Env'].append(D['PROFILE_ENV'] + '=0')
                    with self.assertRaises(AssertionError):
                        self.c['check_container'](bad, cell, None)

    def test_fresh_complete_source_and_six_binary_proof_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent, candidate = root / 'parent', root / 'candidate'
            materialize(parent); materialize(candidate)
            patch(candidate, HERE / 'lilicorr-profile.patch', '--forward')
            for tree in (parent, candidate):
                (tree / 'build-sycl/bin').mkdir(parents=True)
                (tree / 'untouched.txt').write_text('protected source')
                for name in S['ARTIFACTS']:
                    (tree / 'build-sycl/bin' / name).write_text(name + (' candidate' if tree == candidate and name in S['CHANGED_BINARIES'] else ' parent'))
            for name in ('source-hashes.json', 'lilicorr-profile.patch'):
                (root / name).write_bytes((HERE / name).read_bytes())
            (root / 'build-attempt-r1').mkdir()
            (root / 'build-attempt-r1/exit-code.txt').write_text('0\n')
            (root / 'post-build-verification.log').write_text('LILICORR_PROFILE_BUILD_PASSED\n')
            (root / 'runtime-precheck.log').write_text('LILICORR_PROFILE_DEPENDENCIES_PASSED\n')
            save(root / 'source-diff.json', dict(changed_relative_to_parent=S['SOURCE_SCOPE']))
            save(root / 'parent-binaries.json', S['binaries'](parent))
            save(root / 'candidate-binaries.json', S['binaries'](candidate))
            calls = []
            qualified = dict(source=S['source_map'](parent), binaries=S['binaries'](parent))
            def proof():
                calls.append(1)
                self.assertEqual(qualified['source'], S['source_map'](parent))
                self.assertEqual(qualified['binaries'], S['binaries'](parent))
                return qualified
            p = dict(CANDIDATE=parent, candidate_proof=proof)
            with mock.patch.dict(S['_BASE_GLOBALS'], HERE=root, PARENT=parent, CANDIDATE=candidate):
                S['proof'](p); S['proof'](p)
                self.assertEqual(2, len(calls))
                for path in (candidate / 'src/llama-context.cpp', candidate / 'untouched.txt',
                             candidate / 'build-sycl/bin/libggml-sycl.so.0',
                             parent / 'build-sycl/bin/llama-server'):
                    original = path.read_bytes(); path.write_bytes(original + b'drift')
                    with self.assertRaises(AssertionError):
                        S['proof'](p)
                    path.write_bytes(original)
                    S['proof'](p)
                self.assertGreaterEqual(len(calls), 10)
                # Exercise the NEW runner's actual wrappers, not just the proof in isolation.
                public = dict(CANDIDATE=parent, candidate_proof=proof,
                              C=dict(source_binary_hashes=lambda _: {'qualified': True},
                                     source_evidence=lambda _: {'qualified': True}))
                self.c['configure'](public)
                model = root / 'q8.gguf'; model.write_bytes(b'q8-fixture')
                for name in D['BUILD_FILES']:
                    path = root / name
                    if not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('CPU fixture only')
                inherited = runpy.run_path(str(D['Q']['BASE'] / 'contracts.py'))['adapter_code_hashes']()
                real_load, real_sha = self.c['load'], self.c['sha']
                def load(path):
                    return inherited if Path(path).name == 'adapter-code-hashes-before.json' else real_load(path)
                def sha(path):
                    path = Path(path)
                    name = str(path.relative_to(HERE)) if path.is_relative_to(HERE) else None
                    return real_sha(root / name) if name in D['BUILD_FILES'] else real_sha(path)
                with mock.patch.dict(D['Q']['bind'].__globals__, CANDIDATE=model,
                                     CANDIDATE_SIZE=model.stat().st_size, CANDIDATE_SHA256=real_sha(model)), \
                     mock.patch.dict(self.c, load=load, sha=sha):
                    hashes = self.c['adapter_code_hashes']()
                    self.assertEqual(set(str(HERE / n) for n in (*D['OWN_FILES'], *D['BUILD_FILES'])),
                                     set(hashes['profile_adapter']))
                    for wrapper in ('source_binary_hashes', 'source_evidence'):
                        self.assertEqual(str(candidate), self.c[wrapper](None)['profile']['path'])
                        for path in (candidate / 'src/llama-context.cpp',
                                     candidate / 'build-sycl/bin/libggml-sycl.so.0', model):
                            original = path.read_bytes(); path.write_bytes(b'X' + original[1:])
                            with self.assertRaises(AssertionError, msg=wrapper + ': ' + str(path)):
                                self.c[wrapper](None)
                            path.write_bytes(original)
                        self.c[wrapper](None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
