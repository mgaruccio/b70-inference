#!/usr/bin/env python3
"""Remote CPU-only regression tests. No Docker launch, API calls, or model load."""
import array
import ast
import copy
import json
import runpy
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

assert socket.gethostname() == 'inference-host'
M = runpy.run_path('run-trace.py')
A = runpy.run_path('analyze-trace.py')
OLD = runpy.run_path(str(M['WIDTH1'] / 'run-controls.py'))
NATIVE = runpy.run_path(str(M['WIDTH1'] / 'analyze-controls.py'))
BASE = json.loads((OLD['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
IMAGE = 'intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04'


def fixture(directory, speculative=False, diverge=False, chunked_prompt=False):
    directory.mkdir()
    tokens = list(range(1000, 1128))
    tokens[115] = 4618 if diverge else 6833
    vectors = array.array('f', [-1] * 8192)
    vectors[6833], vectors[4618] = 10, 9
    for i in range(128):
        (directory / f'row-{i:06d}').write_bytes(vectors.tobytes())
    rows = []
    def log(event, **kwargs):
        fields = []
        for key, value in kwargs.items():
            if isinstance(value, list):
                value = ','.join(map(str, value)) or '-'
            fields.append(f'{key}={value}')
        rows.append('2026-10-17T00:00:00Z 0.0.1 I BOLT_VERIFY ' + event + ' ' + ' '.join(fields))
    def batch(ids, pos, outputs):
        log('batch', ctx='0xabc', off=0, n=len(ids), tokens=ids, pos=pos,
            outputs=outputs, seq=','.join('0' for _ in ids))
        log('decoded', ctx='0xabc', ret=0)
    def sample(i, idx):
        log('sample', ctx='0xabc', smpl='0xdef', idx=idx, selected=tokens[i], vocab=8192,
            status='cpu', file=f'row-{i:06d}')
    def emit(i):
        log('emit', ctx='0xabc', slot=0, ordinal=i + 1, token=tokens[i])
    log('remove', ctx='0xabc', slot=0, kind='prompt', p0=0, p1=-1, rollback=0)
    if chunked_prompt:
        for start, end in ((0, 5), (5, 51), (51, 55)):
            if start:
                log('remove', ctx='0xabc', slot=0, kind='prompt', p0=start, p1=-1, rollback=0)
            batch(list(range(start, end)), list(range(start, end)),
                  [int(p == 54) for p in range(start, end)])
        first_row = 3
    else:
        batch(list(range(55)), list(range(55)), [0] * 54 + [1])
        first_row = 54
    sample(0, first_row)
    log('single', ctx='0xabc', smpl='0xdef', slot=0, row=first_row, id=tokens[0])
    emit(0)
    i = 1
    while i < 128:
        if not speculative or i == 127:
            batch([tokens[i - 1]], [54 + i], [1])
            sample(i, 0)
            log('single', ctx='0xabc', smpl='0xdef', slot=0, row=0, id=tokens[i])
            emit(i)
            i += 1
            continue
        rejected = i == 1
        draft = 777 if rejected else tokens[i]
        batch([tokens[i - 1], draft], [54 + i, 55 + i], [1, 1])
        sample(i, 0)
        emitted = [tokens[i]]
        if not rejected:
            sample(i + 1, 1)
            emitted.append(tokens[i + 1])
        log('verify', ctx='0xabc', smpl='0xdef', slot=0, rows=[0, 1], drafts=[draft],
            emitted=emitted, accepted=len(emitted) - 1, rollback=int(rejected), replay=0, rejection=0, synth=0)
        log('remove', ctx='0xabc', slot=0, kind='verify', p0=54 + i + len(emitted), p1=-1, rollback=int(rejected))
        for j in range(len(emitted)):
            emit(i + j)
        i += len(emitted)
    return '\n'.join(rows) + '\n', tokens


class Trace(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=M['ROOT'] / 'worker-cpu')
        self.root = Path(self.temp.name)
        self.log, self.tokens = fixture(self.root / 'S0')
        self.events = A['records'](self.log)

    def tearDown(self):
        self.temp.cleanup()

    def aligned(self, directory='S0', events=None, tokens=None):
        result = A['align'](events or self.events, tokens or self.tokens, self.root / directory)
        result.update(observer_passed=True, capture_path=str(self.root / directory))
        return result

    def test_syntax_and_patch_scope(self):
        for path in Path('.').glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
        subprocess.run(['bash', '-n', 'build-candidate.sh'], check=True)
        patch = Path('verify-trace.patch').resolve()
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=M['PARENT'], check=True)
        names = [line[6:] for line in patch.read_text().splitlines() if line.startswith('+++ b/')]
        self.assertEqual(names, M['SOURCE_SCOPE'])
        self.assertNotIn('llama_get_logits_ith(ctx, idx)', '\n'.join(
            l for l in patch.read_text().splitlines() if l.startswith('+')))

    def test_missing_c0_changes_only_five_controller_constants(self):
        continuation = runpy.run_path('continue-c0.py')
        original = ast.parse((M['ROOT'] / 'run-trace.py').read_text())
        generated = ast.parse(continuation['controller_source']())
        self.assertEqual(len(original.body), len(generated.body))
        changed = {}
        for before, after in zip(original.body, generated.body):
            if ast.dump(before) != ast.dump(after):
                self.assertIsInstance(before, ast.Assign)
                self.assertIsInstance(after, ast.Assign)
                self.assertEqual(len(after.targets), 1)
                self.assertEqual(ast.dump(before.targets[0]), ast.dump(after.targets[0]))
                changed[after.targets[0].id] = after.value
        self.assertEqual(set(changed), {'ROOT', 'WIDTH1', 'PARENT_RESULTS', 'ORDER', 'PREFIX'})
        self.assertEqual(ast.literal_eval(changed['ORDER']), ('C0',))
        self.assertEqual(ast.literal_eval(changed['PREFIX']), 'bolt-lilicorr-verify-trace-c0-r1-')
        for key, expected in (('ROOT', M['ROOT'] / 'continuation-c0'),
                              ('WIDTH1', M['WIDTH1']), ('PARENT_RESULTS', M['PARENT_RESULTS'])):
            self.assertEqual(changed[key].args[0].value, str(expected))

    def test_exact128_rows_real_positions_and_raw_argmax(self):
        result = self.aligned()
        self.assertEqual(len(result['samples']), 128)
        self.assertEqual(result['samples'][0]['batch_row'], 54)
        self.assertEqual(result['samples'][115]['position'], 169)
        self.assertEqual(result['samples'][115]['raw_argmax'], 6833)
        self.assertEqual(result['samples'][0]['selected'], 1000)  # raw != post-sampler is allowed

    def test_prompt_chunk_tail_boundaries(self):
        log, tokens = fixture(self.root / 'chunked', chunked_prompt=True)
        events = A['records'](log)
        result = self.aligned('chunked', events, tokens)
        self.assertEqual([b['n'] for b in result['batches'][:3]], [5, 46, 4])
        self.assertEqual(result['prompt_ids'], list(range(55)))
        self.assertEqual(result['samples'][0]['batch_row'], 3)
        self.assertEqual([r['p0'] for r in result['removals']], [0, 5, 51])
        second = [i for i, e in enumerate(events) if e['event'] == 'remove'][1]
        for key, value in (('p0', 4), ('p0', 6), ('p1', 9), ('rollback', 1),
                           ('ctx', '0xbad'), ('slot', 1)):
            bad = copy.deepcopy(events)
            bad[second][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                self.aligned('chunked', bad, tokens)
        bad = copy.deepcopy(events)
        boundary = copy.deepcopy(bad[second])
        boundary['p0'] = 55
        first_emit = next(i for i, e in enumerate(bad) if e['event'] == 'emit')
        bad.insert(first_emit + 1, boundary)
        with self.assertRaises(AssertionError):
            self.aligned('chunked', bad, tokens)

    def test_greedy_verification_acceptance_and_actual_removal(self):
        log, tokens = fixture(self.root / 'C0', speculative=True)
        result = self.aligned('C0', A['records'](log), tokens)
        rejected = [r for r in result['removals'] if r['rollback']]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['removed_batch_positions'], [56])
        self.assertEqual(result['samples'][3]['batch_row'], 1)

    def test_framing_errors_fail_closed(self):
        for bad in ('BOLT_VERIFY error op=write_close errno=28\n', 'BOLT_VERIFY\n',
                    'BOLT_VERIFY decoded ctx=0x1 ret=0 ret=0\n',
                    'BOLT_VERIFY decoded ctx=0x1 ret=0 extra=3\n',
                    'BOLT_VERIFY decoded ctx=0x1 ret=0 BOLT_VERIFY sample\n'):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                A['records'](bad)
        self.assertEqual(A['records']('unrelated\n' + self.log), self.events)

    def test_capture_path_bounds_symlinks_and_sizes(self):
        sample = next(e for e in self.events if e['event'] == 'sample')
        for name in ('../row-000000', '/tmp/row-000000', 'row-00000', 'row-000000/extra'):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                A['raw_bytes'](self.root / 'S0', dict(sample, file=name))
        link = self.root / 'S0/row-999999'
        link.symlink_to(self.root / 'S0/row-000000')
        with self.assertRaises(AssertionError):
            A['raw_bytes'](self.root / 'S0', dict(sample, file=link.name))
        path = self.root / 'S0/row-000000'
        for size in (0, 32767, 32769):
            path.write_bytes(b'\0' * size)
            with self.assertRaises(AssertionError):
                A['raw_bytes'](self.root / 'S0', sample)
        for n in (0, -1, 1_000_001):
            with self.assertRaises(AssertionError):
                A['raw_bytes'](self.root / 'S0', dict(sample, vocab=n))

    def test_backend_uncaptured_and_duplicate_block_attribution(self):
        for status in ('backend', 'uncaptured'):
            events = copy.deepcopy(self.events)
            next(e for e in events if e['event'] == 'sample')['status'] = status
            with self.assertRaises(AssertionError):
                self.aligned(events=events)
        events = copy.deepcopy(self.events)
        samples = [e for e in events if e['event'] == 'sample']
        samples[1]['file'] = samples[0]['file']
        with self.assertRaises(AssertionError):
            self.aligned(events=events)

    def test_alignment_failures(self):
        for event, key, value in (('sample', 'idx', 0), ('sample', 'selected', 999),
                                  ('sample', 'ctx', '0x999'), ('decoded', 'ret', -1),
                                  ('emit', 'ordinal', 2), ('single', 'row', 0)):
            events = copy.deepcopy(self.events)
            next(e for e in events if e['event'] == event)[key] = value
            with self.subTest(event=event, key=key), self.assertRaises(AssertionError):
                self.aligned(events=events)
        with self.assertRaises(AssertionError):
            self.aligned(tokens=self.tokens[:-1])

    def test_batch_history_and_output_flag_failures(self):
        for key, value in (('tokens', [999]), ('pos', [123]), ('outputs', [0])):
            events = copy.deepcopy(self.events)
            [e for e in events if e['event'] == 'batch'][1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.aligned(events=events)

    def test_negative_row_index_resolves_actual_last_output(self):
        events = copy.deepcopy(self.events)
        next(e for e in events if e['event'] == 'sample')['idx'] = -1
        next(e for e in events if e['event'] == 'single')['row'] = -1
        self.assertEqual(self.aligned(events=events)['samples'][0]['batch_row'], 54)

    def test_rollback_and_verifier_guards(self):
        log, tokens = fixture(self.root / 'C0', speculative=True)
        for kind, key, value in (('verify', 'accepted', 1), ('verify', 'rows', [1, 0]),
                                  ('verify', 'replay', 1), ('remove', 'p0', 100)):
            events = A['records'](log)
            target = next(e for e in events if e['event'] == kind and (kind != 'remove' or e['kind'] == 'verify'))
            target[key] = value
            with self.subTest(kind=kind, key=key), self.assertRaises(AssertionError):
                self.aligned('C0', events, tokens)

    def test_first_difference_and_history_cutoff(self):
        log, tokens = fixture(self.root / 'C0', speculative=True, diverge=True)
        left, right = self.aligned(), self.aligned('C0', A['records'](log), tokens)
        file = self.root / 'C0/row-000116'  # ONLY after the differing output116; must not compare
        file.write_bytes(b'\0' * 32768)
        result = A['compare'](left, right)
        self.assertEqual(result['compared_through_ordinal'], 116)
        self.assertIsNone(result['first_score_difference'])
        path = self.root / 'C0/row-000115'
        data = bytearray(path.read_bytes()); data[40] ^= 1; path.write_bytes(data)
        result = A['compare'](left, right)
        self.assertEqual(result['first_score_difference']['ordinal'], 116)
        self.assertEqual(result['first_score_difference']['first_differing_float_id'], 10)
        self.assertTrue(result['first_score_difference']['C0_rollback_preceded'])
        path = self.root / 'C0/row-000000'
        data = bytearray(path.read_bytes()); data[80] ^= 1; path.write_bytes(data)
        result = A['compare'](left, right)
        self.assertEqual(result['first_score_difference']['ordinal'], 1)
        self.assertFalse(result['first_score_difference']['C0_rollback_preceded'])
        self.assertFalse(result['near_token116'][-1]['identical_prior_history'])

    def test_observer_rejection(self):
        ref = M['WIDTH1'] / 'run-r1/A0'
        ordinary = json.loads((ref / 'repeat-00.result.json').read_text())
        diagnostic = json.loads((ref / 'diagnostic.response.json').read_text())
        A['observer'](ordinary, diagnostic, ordinary, diagnostic)
        changed = copy.deepcopy(diagnostic)
        changed['__verbose']['tokens'][0] += 1
        with self.assertRaises(AssertionError):
            A['observer'](ordinary, changed, ordinary, diagnostic)
        changed = copy.deepcopy(diagnostic)
        changed['choices'][0]['message']['content'] += '!'
        with self.assertRaises(AssertionError):
            A['observer'](ordinary, changed, ordinary, diagnostic)
        with self.assertRaises(AssertionError):
            A['observer'](dict(ordinary, text='wrong'), diagnostic, ordinary, diagnostic)
        left = self.aligned()
        with self.assertRaises(AssertionError):
            A['compare'](dict(left, observer_passed=False), left)

    def test_parameters_mounts_frozen_payload_and_order(self):
        before = copy.deepcopy(BASE)
        self.assertEqual(M['ORDER'], ('S0', 'C0'))
        self.assertEqual(len(M['ARTIFACTS']), 5)
        self.assertIn('libllama-common.so', M['ARTIFACTS'])
        for label in M['ORDER']:
            capture = M['ROOT'] / 'run-r1' / label / 'capture'
            argv = M['make_create'](BASE, label, IMAGE, capture, OLD)
            self.assertIn(str(M['CANDIDATE']) + ':/src:ro', argv)
            self.assertIn(str(capture) + ':/capture', argv)
            for key, value in M['ENV']:
                self.assertEqual(argv.count(key + '=' + value), 1)
            self.assertIn('bash /native-outprod.sh;', argv[-1])
            self.assertEqual(argv[-1].rsplit('exec ', 1)[1], BASE[-1].rsplit('exec ', 1)[1] + ' ' + M['shlex'].join(M['spec_for_arm'](label)))
        self.assertEqual(before, BASE)
        self.assertEqual(M['spec_for_arm']('S0')[3:6], ['1', '--spec-draft-n-min', '2'])
        self.assertEqual(M['spec_for_arm']('C0')[3:6], ['1', '--spec-draft-n-min', '0'])
        with self.assertRaises(AssertionError):
            M['make_create'](BASE, 'S0', IMAGE, Path('/tmp/escape'), OLD)
        payload = json.loads((M['WIDTH1'] / 'run-r1/frozen-payload.json').read_text())
        diag = json.loads((M['WIDTH1'] / 'run-r1/C0/diagnostic.request.json').read_text())
        self.assertEqual(OLD['diagnostic_payload'](payload), diag)
        self.assertFalse(diag['backend_sampling'])

    def test_native_and_startup_guards_on_saved_real_log(self):
        log = (M['WIDTH1'] / 'run-r1/C0/native-and-startup.log').read_text().replace(OLD['SERVER_MARKER'], M['MARKER'])
        M['check_startup'](log, 'C0', OLD, NATIVE)
        suppressed = log.replace('n_max=1, n_min=0,', 'n_max=1, n_min=2,')
        M['check_startup'](suppressed, 'S0', OLD, NATIVE)
        for broken in (log.replace('BOLT_LILICORR_OUTPROD_NATIVE_PASSED', 'missing'),
                       log.replace('n_max=1, n_min=0,', 'n_max=4, n_min=0,'),
                       log + '\nBOLT_VERIFY error op=mkstemp errno=28\n'):
            with self.assertRaises(AssertionError):
                M['check_startup'](broken, 'C0', OLD, NATIVE)

    def test_restoration_identity_config_all_mounts_and_hostconfig(self):
        original = dict(Id='original-id', State=dict(Running=True), Config=dict(Env=['x=1']),
                        HostConfig=dict(PortBindings={'a': 'b'}), Mounts=[dict(Destination='/src', Source='/parent', RW=False)])
        M['check_restored'](original, copy.deepcopy(original))
        for key, value in (('Id', 'replacement'), ('State', dict(Running=False)),
                           ('Config', {}), ('HostConfig', {}), ('Mounts', [])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                M['check_restored'](original, dict(original, **{key: value}))
        tree = ast.parse(Path('run-trace.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        self.assertTrue(any(isinstance(n, ast.Try) and n.finalbody and any(
            isinstance(inner, ast.Try) and inner.finalbody for inner in ast.walk(ast.Module(body=n.finalbody, type_ignores=[])))
            for n in ast.walk(main)), 'restoration must remain in nested finally')

    def test_stop_flushes_logs_before_post_snapshot_and_alignment(self):
        source = Path('run-trace.py').read_text()
        saved = source.index("save(cell / 'diagnostic.response.json', body)")
        validated = source.index("old['check_diagnostic_result'](body)", saved)
        stopped = source.index("command('docker', 'stop', '-t', '90', name)", validated)
        post = source.index("post = command('docker', 'logs', '--timestamps', name)", saved)
        aligned = source.index("checked = analyze['analyze_arm'](cell, label, old)", saved)
        self.assertLess(validated, stopped)
        self.assertLess(stopped, post)
        self.assertLess(post, aligned)

    def test_log_window_framing(self):
        self.assertEqual(M['diagnostic_window']('prefix\n', 'prefix\n' + self.log), self.log)
        with self.assertRaises(AssertionError):
            M['diagnostic_window']('prefix\n', 'other\n' + self.log)


if __name__ == '__main__':
    unittest.main(verbosity=2)
