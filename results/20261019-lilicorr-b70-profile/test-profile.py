#!/usr/bin/env python3
"""CPU fixtures only: no Docker, GPU, network, builds, or remote mutation."""
import copy
import gzip
import hashlib
import json
import re
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-profile.py'))
S = D['S']
A = runpy.run_path(str(HERE / 'analyze-profile.py'))
QUALIFIED = HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PatchTests(unittest.TestCase):
    def test_zero_fuzz_forward_reverse_and_insertion_only_operation_order(self):
        pins = S['load'](HERE / 'source-hashes.json')
        with tempfile.TemporaryDirectory(dir=HERE / '.scratch') as temporary:
            root = Path(temporary)
            originals = {}
            for name in S['SOURCE_SCOPE']:
                data = gzip.decompress((HERE / 'fixtures' / (name + '.gz')).read_bytes())
                self.assertEqual(pins['parent'][name], hashlib.sha256(data).hexdigest())
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                originals[name] = data
            for flags in (['--dry-run', '--forward'], ['--forward']):
                result = subprocess.run(['patch', '--fuzz=0', '--batch', '-p1', *flags,
                                         '-i', str(HERE / 'lilicorr-profile.patch')],
                                        cwd=root, check=True, capture_output=True, text=True)
                self.assertNotRegex(result.stdout.lower(), r'fuzz|offset|failed')
            for name, original in originals.items():
                data = (root / name).read_text()
                self.assertEqual(pins['candidate'][name], hashlib.sha256(data.encode()).hexdigest())
                blocks = re.findall(r'^[ \t]*// LILICORR_PROFILE_ONLY_BEGIN\n(.*?)^[ \t]*// LILICORR_PROFILE_ONLY_END\n',
                                    data, flags=re.M | re.S)
                stripped = re.sub(r'^[ \t]*// LILICORR_PROFILE_ONLY_BEGIN\n.*?^[ \t]*// LILICORR_PROFILE_ONLY_END\n',
                                  '', data, flags=re.M | re.S)
                self.assertEqual(original, stripped.encode())
                added = '\n'.join(blocks)
                self.assertNotRegex(added, r'\b(?:llama_process|llama_get_embeddings\w*|ggml_backend_\w+|memcpy|exp)\s*\(')
                self.assertIn('std::strcmp(', added)
                self.assertIn('"1") == 0', added)
                # All clock reads are conditional, including those inside guarded log blocks.
                for line in added.splitlines():
                    if 'ggml_time_us()' in line and '? ggml_time_us() : 0' not in line:
                        self.assertRegex(line, r'^\s+const int64_t profile_(?:t[235]|end) = ggml_time_us\(\);$')
                self.assertEqual(added.count('LOG_INF(') + added.count('LLAMA_LOG_INFO('),
                                 added.count('if (profile) {'))
            result = subprocess.run(['patch', '--fuzz=0', '--batch', '--reverse', '-p1',
                                     '-i', str(HERE / 'lilicorr-profile.patch')], cwd=root,
                                    check=True, capture_output=True, text=True)
            self.assertNotRegex(result.stdout.lower(), r'fuzz|offset|failed')
            for name, data in originals.items():
                self.assertEqual(data, (root / name).read_bytes())

    def test_build_is_cpu_only_and_exact_source_scope(self):
        script = (HERE / 'build-candidate.sh').read_text()
        self.assertNotRegex(script, r'--device|/dev/dri|--privileged')
        self.assertIn('CMAKE_HOME_DIRECTORY:INTERNAL=/src', script)
        self.assertIn('--fuzz=0', script)
        subprocess.run(['bash', '-n', str(HERE / 'build-candidate.sh')], check=True)
        self.assertEqual(['common/speculative.cpp', 'src/llama-context.cpp'], S['SOURCE_SCOPE'])


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.analysis, self.c = D['bind']()
        self.public = dict(OUT=QUALIFIED, C={}, CANDIDATE=S['PARENT'],
                           make_create=lambda cell, _h: S['load'](QUALIFIED / cell / 'create-argv.json'))
        self.c['configure'](self.public)

    def fixture(self, cell):
        record = S['load'](self.c['qualified_dir'](self.c['role_for'](cell)) / 'container-before.json')
        record.update(Name='/' + self.c['name'](cell), Id='b' * 64)
        record['Config']['Hostname'] = 'b' * 12
        record['Config']['Cmd'] = self.c['make_create'](cell, None)[-3:]
        if self.c['role_for'](cell) == 'W3':
            record['Config']['Env'].append(D['PROFILE_ENV'] + '=1')
            substitutions = {D['Q']['old_draft_bind'](): D['Q']['candidate_draft_bind'](),
                             D['source_bind'](S['PARENT']): D['source_bind'](S['CANDIDATE'])}
            record['HostConfig']['Binds'] = [substitutions.get(x, x) for x in record['HostConfig']['Binds']]
            record['Mounts'] = D['Q']['_expected_mounts'](record, cell)
            next(m for m in record['Mounts'] if m['Destination'] == '/src')['Source'] = str(S['CANDIDATE'])
        return record

    def test_exact_A_and_W3_delta_and_shared_private_namespace(self):
        before = copy.deepcopy({k: v for k, v in self.public.items() if k != 'make_create'})
        for cell in self.c['CELLS']:
            base = self.public['make_create'](self.c['base_cell'](cell), None)
            argv = self.c['make_create'](cell, None)
            self.assertEqual(base[-3:], argv[-3:])
            self.assertEqual(self.c['server_argv'](base), self.c['server_argv'](argv))
            reverse = list(argv)
            if self.c['role_for'](cell) == 'W3':
                self.assertEqual(reverse[-6:-4], ['-e', D['PROFILE_ENV'] + '=1'])
                del reverse[-6:-4]
                reverse[reverse.index(D['source_bind'](S['CANDIDATE']))] = D['source_bind'](S['PARENT'])
                reverse[reverse.index(D['Q']['candidate_draft_bind']())] = D['Q']['old_draft_bind']()
            else:
                self.assertFalse(any(D['PROFILE_ENV'] in item for item in reverse))
            reverse[reverse.index('--name') + 1] = base[base.index('--name') + 1]
            self.assertEqual(base, reverse)
        self.assertEqual(before, {k: v for k, v in self.public.items() if k != 'make_create'})
        for fn in (self.runtime['Runtime'].begin, self.runtime['Runtime'].end,
                   self.runtime['Runtime'].restore, self.runtime['Runtime'].verify_restored,
                   self.runtime['_BOOT_K']['capture'], self.analysis['analyze']):
            self.assertIs(fn.__globals__['C'], self.c)
        self.assertIsNot(D['Q']['bind']()[2], self.c)
        self.assertEqual(self.c['CELLS'], ('VSPF-A1', 'VSPF-W3a', 'VSPF-W3b', 'VSPF-A2'))

    def test_actual_inspect_is_not_mutated_and_drift_rejected(self):
        for cell in self.c['CELLS']:
            record = self.fixture(cell)
            before = copy.deepcopy(record)
            self.assertTrue(self.c['check_container'](record, cell, None))
            self.assertEqual(before, record)
        pristine = self.fixture('VSPF-W3a')
        changes = [lambda r: r['Config']['Env'].append(D['PROFILE_ENV'] + '=0'),
                   lambda r: r['Config'].__setitem__('Cmd', ['bash', '-lc', 'bad']),
                   lambda r: r.__setitem__('Image', 'bad'),
                   lambda r: next(m for m in r['Mounts'] if m['Destination'] == '/src').__setitem__('RW', True),
                   lambda r: next(m for m in r['Mounts'] if m['Destination'] == '/src').__setitem__('Source', str(S['PARENT'])),
                   lambda r: r['HostConfig']['Binds'].append(D['source_bind'](S['PARENT']))]
        for change in changes:
            record = copy.deepcopy(pristine)
            change(record)
            with self.assertRaises(AssertionError):
                self.c['check_container'](record, 'VSPF-W3a', None)
        with self.assertRaises(AssertionError):
            D['replace_source_bind'](['docker', 'create'])

    def test_real_pid_extra_opt_in_is_required_without_sanitization(self):
        record = dict(pid=1, origin='/proc/1/environ', profile_environment_origin='/proc/1/environ',
                      profile_environment=[[D['PROFILE_ENV'], '1']])
        self.assertTrue(D['check_profile_pid'](record, 'VSPF-W3a'))
        for pairs in ([], [[D['PROFILE_ENV'], '0']], [[D['PROFILE_ENV'], '01']],
                      [[D['PROFILE_ENV'], '1'], [D['PROFILE_ENV'], '1']]):
            bad = dict(record, profile_environment=pairs)
            with self.assertRaises(AssertionError):
                D['check_profile_pid'](bad, 'VSPF-W3a')
        with self.assertRaises(AssertionError):
            D['check_profile_pid'](record, 'VSPF-A1')
        self.assertTrue(D['check_profile_pid'](dict(record, profile_environment=[]), 'VSPF-A1'))


class ProofTests(unittest.TestCase):
    def test_every_before_after_wrapper_rehashes_source_and_binaries_after_code_capture(self):
        runtime, analysis, c = D['bind']()
        with tempfile.TemporaryDirectory(dir=HERE / '.scratch') as directory:
            root = Path(directory)
            parent, candidate = root / 'parent', root / 'candidate'
            for tree in (parent, candidate):
                tree.mkdir()
                (tree / 'UPSTREAM_COMMIT').write_text(S['load'](HERE / 'source-hashes.json')['upstream_commit'] + '\n')
                for name in S['SOURCE_SCOPE']:
                    path = tree / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(gzip.decompress((HERE / 'fixtures' / (name + '.gz')).read_bytes()))
                for name in S['ARTIFACTS']:
                    path = tree / 'build-sycl/bin' / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((name + ('profile' if tree == candidate and name in S['CHANGED_BINARIES'] else '')).encode())
            subprocess.run(['patch', '--fuzz=0', '--batch', '-p1', '-i', str(HERE / 'lilicorr-profile.patch')],
                           cwd=candidate, check=True, capture_output=True)
            for name in ('source-hashes.json', 'lilicorr-profile.patch'):
                (root / name).write_bytes((HERE / name).read_bytes())
            save(root / 'parent-binaries.json', S['binaries'](parent))
            save(root / 'candidate-binaries.json', S['binaries'](candidate))
            save(root / 'source-diff.json', dict(changed_relative_to_parent=S['SOURCE_SCOPE']))
            (root / 'build-attempt-r1').mkdir()
            (root / 'build-attempt-r1/exit-code.txt').write_text('0\n')
            (root / 'runtime-precheck.log').write_text('LILICORR_PROFILE_DEPENDENCIES_PASSED')
            (root / 'post-build-verification.log').write_text('LILICORR_PROFILE_BUILD_PASSED')
            save(root / 'build-attempt-r1/image.json', {'fixture': True})
            qualified = dict(source=S['source_map'](parent), binaries=S['binaries'](parent))
            p = dict(CANDIDATE=parent, candidate_proof=lambda: qualified,
                     C=dict(source_binary_hashes=lambda _h: {'qualified': True},
                            source_evidence=lambda _h: {'qualified': True}))
            c['configure'](p)
            q8 = root / 'q8.gguf'
            q8.write_bytes(b'q8-fixture')
            inherited = runpy.run_path(str(D['Q']['BASE'] / 'contracts.py'))['adapter_code_hashes']()
            real_load = c['load']
            def load(path):
                return inherited if Path(path).name == 'adapter-code-hashes-before.json' else real_load(path)
            def sha(path):
                name = str(Path(path).relative_to(HERE)) if Path(path).is_relative_to(HERE) else None
                return S['sha'](root / name) if name in D['BUILD_FILES'] else S['sha'](path)
            with mock.patch.dict(S['proof'].__globals__, HERE=root, PARENT=parent, CANDIDATE=candidate), \
                 mock.patch.dict(D['Q']['bind'].__globals__, CANDIDATE=q8, CANDIDATE_SIZE=q8.stat().st_size,
                                 CANDIDATE_SHA256=S['sha'](q8)), \
                 mock.patch.dict(c, load=load, sha=sha):
                c['adapter_code_hashes']()  # Earlier proof capture must never hide later drift.
                for wrapper in ('source_binary_hashes', 'source_evidence'):
                    c[wrapper](None)
                    for path in (candidate / 'common/speculative.cpp', candidate / 'build-sycl/bin/libllama-common.so', q8):
                        original = path.read_bytes()
                        path.write_bytes(b'X' + original[1:])  # SAME LENGTH, different content.
                        with self.assertRaises(AssertionError, msg=str(path)):
                            c[wrapper](None)
                        path.write_bytes(original)
                    extra = candidate / 'unapproved-source'
                    extra.write_text('drift')
                    with self.assertRaises(AssertionError):
                        c[wrapper](None)
                    extra.unlink()
                    c[wrapper](None)


def marker(stage, **values):
    return A['MARKER'] + ' '.join(f'{key}={value}' for key, value in dict(v=1, stage=stage, **values).items())


def synthetic_events(base=100, proposed=3):
    # Intentionally generated fixtures, not measured evidence.
    return [
        marker('ubatch', rows=4, gtype=0, nodes=90, reused=0, rc=0, begin_us=base+10, end_us=base+17,
               build_alloc_us=2, inputs_us=1, submit_us=4),
        marker('process_chunk', seq=0, batch_rows=4, seq_rows=4, offset=0, rows=4, pos0=0, pos1=3,
               projected=1, begin_us=base+2, end_us=base+27, gather_us=5, prepare_us=2, submit_us=10,
               projected_wait_us=6, ring_us=2),
        marker('process_batch', rows=4, begin_us=base+1, end_us=base+30),
        marker('ubatch', rows=8, gtype=0, nodes=300, reused=1, rc=0, begin_us=base+45, end_us=base+58,
               build_alloc_us=3, inputs_us=2, submit_us=8),
        marker('draft_batch', rows=8, rc=0, begin_us=base+40, end_us=base+60, prepare_us=4, submit_us=16),
        marker('draft_select', seq=0, rows=8, pos0=4, selected=3, limit=proposed, begin_us=base+65, end_us=base+75,
               lattice_wait_us=8, walk_us=2),
    ]


class AnalyzerTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(A['event'](line), line=i+1) for i, line in enumerate(synthetic_events())]
        req = dict(task=7, first_line=1, last_line=10, events=rows, prompts=[55])
        previous = dict(calls={'g': 2}, generated={'tokens': 6}, accepted={'tokens': 4})
        stats = dict(calls={'g': 3}, generated={'tokens': 9}, accepted={'tokens': 6})
        return req, previous, stats, dict(prompt=55, proposed=3, accepted=2)

    def test_nested_costs_and_graph_attribution(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        result = A['request_profile'](*args)
        self.assertEqual(args, before)
        self.assertEqual(10, result['totals']['process_chunk']['submit_us'])
        self.assertEqual(16, result['totals']['draft_batch']['submit_us'])
        self.assertEqual(12, result['totals']['ubatch']['submit_us'])
        self.assertEqual(1, result['before_first_draft']['process_batch']['calls'])
        self.assertEqual(1, result['from_first_draft']['draft_select']['calls'])
        self.assertEqual({'process_chunk', 'draft_batch'}, {r['parent_stage'] for r in result['graphs']})

    def test_missing_duplicate_and_ambiguous_events_fail_closed(self):
        for index in range(6):
            args = self.fixture()
            del args[0]['events'][index]
            with self.assertRaises(AssertionError):
                A['request_profile'](*args)
        args = self.fixture()
        args[0]['events'].append(copy.deepcopy(args[0]['events'][0]))
        with self.assertRaises(AssertionError):
            A['request_profile'](*args)
        for key in ('prompt', 'proposed', 'accepted'):
            args = self.fixture()
            args[3][key] += 1
            with self.assertRaises(AssertionError):
                A['request_profile'](*args)

    def test_strict_marker_parser(self):
        good = synthetic_events()[4]
        for bad in (good.replace('v=1', 'v=2'), good + ' rc=0', good.replace('rows=8', 'rows=3'),
                    good.replace('rc=0', 'rc=1'), good.replace('submit_us=16', 'submit_us=NaN'),
                    good.replace('submit_us=16', 'submit_us=-1'), good.replace('submit_us=16', 'submit_us=15'),
                    good.replace('stage=draft_batch', 'stage=unknown'), good.replace(' prepare_us=4', '')):
            with self.assertRaises((AssertionError, KeyError)):
                A['event'](bad)

    def test_real_Q8_log_order_and_missing_profile_refused(self):
        for cell in ('VSPF-A1', 'VSPF-W3a', 'VSPF-W3b', 'VSPF-A2'):
            path = D['Q']['HERE'] / 'timing-r2' / cell / 'server.stderr.log.gz'
            text = gzip.decompress(path.read_bytes()).decode()
            reqs, startup = A['requests'](text)
            self.assertEqual(33, len(reqs))
            self.assertEqual([[55]] * 4 + [[8270]] * 4, [r['prompts'] for r in reqs[-8:]])
            self.assertEqual([], startup)
            with self.assertRaises(AssertionError):
                A['requests'](text[:text.rfind('stop processing:')])
            with self.assertRaises(AssertionError):
                A['requests'](text + '\n' + synthetic_events()[0])
            if 'W3' in cell:
                c = D['bind']()[2]
                stats = [c['log_duration_stats']('\n'.join(req['lines']))['records'][0] for req in reqs]
                with self.assertRaises(AssertionError):
                    A['request_profile'](reqs[25], stats[24], stats[25], A['EXPECTED']['short'])

    def test_full_analyzer_on_explicitly_synthetic_profile_plus_saved_Q8_requests(self):
        c = D['bind']()[2]
        old = D['Q']['HERE'] / 'timing-r2'
        with tempfile.TemporaryDirectory(dir=HERE / '.scratch') as directory:
            out = Path(directory)
            save(out / 'analysis-r1/analysis.json', S['load'](old / 'analysis-r1/analysis.json'))
            save(out / 'analysis-live-restoration.json', {'passed': True, 'fixture_only': True})
            for cell in c['CELLS']:
                (out / cell).mkdir()
                text = gzip.decompress((old / cell / 'server.stderr.log.gz').read_bytes()).decode()
                reqs, _ = A['requests'](text)
                if c['role_for'](cell) == 'W3':
                    for index, job in enumerate([j for j in c['schedule']() if j['cell'] == cell], 25):
                        count = 43 if job['point'] == 'short' else 45
                        events = []
                        for call in range(count):
                            limit = 1 if job['point'] == 'long' and call == count - 1 else 3
                            events.extend(synthetic_events(base=index * 100000 + call * 100, proposed=limit))
                        boundary = reqs[index]['lines'][-1]
                        self.assertEqual(1, text.count(boundary))
                        text = text.replace(boundary, '\n'.join(events) + '\n' + boundary)
                        stem = Path(cell) / job['point'] / f"{job['phase']}-{job['ordinal']:02}.result.json"
                        save(out / stem, S['load'](old / stem))
                (out / cell / 'server.stderr.log').write_text(text)
                (out / cell / 'server.stdout.log').write_text('')
            result = A['analyze'](out, c)
            self.assertEqual('analyzed', result['status'], result['errors'])
            self.assertEqual(16, len(result['requests']))
            long = next(r for r in result['requests'] if r['point'] == 'long')
            self.assertEqual(45, long['totals']['draft_select']['calls'])
            selected = [r for r in long['events'] if r['stage'] == 'draft_select']
            self.assertEqual(135, sum(r['selected'] for r in selected))
            self.assertEqual(1, selected[-1]['limit'])
            stem = out / 'VSPF-W3a/long/measured-02.result.json'
            bad = S['load'](stem)
            bad['timings']['draft_n'] = 135
            save(stem, bad)
            failed = A['analyze'](out, c)
            self.assertEqual('failed', failed['status'])
            self.assertTrue(failed['errors'])

    def test_analyzer_missing_inherited_evidence_has_no_success(self):
        with tempfile.TemporaryDirectory(dir=HERE / '.scratch') as directory:
            c = D['bind']()[2]
            result = A['analyze'](Path(directory), c)
            self.assertEqual('failed', result['status'])
            self.assertTrue(result['errors'])
            self.assertFalse(result['speed_comparison_permitted'])


if __name__ == '__main__':
    (HERE / '.scratch').mkdir(exist_ok=True)
    unittest.main(verbosity=2)
