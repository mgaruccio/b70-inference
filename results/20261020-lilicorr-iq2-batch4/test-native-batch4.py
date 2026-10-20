#!/usr/bin/env python3
"""CPU-only route qualification tests; no native run, build, or evidence mutation."""
import os
import re
import runpy
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
QUALIFIER = runpy.run_path(str(HERE / 'native-batch4.py'))
FROZEN = runpy.run_path(str(QUALIFIER['FROZEN_PATH']))
FROZEN_TEST = runpy.run_path(str(QUALIFIER['FROZEN_PATH'].with_name('test-native.py')))
BATCH = QUALIFIER['BATCH_ROUTE']


def _route_row(line):
    assert line.startswith('BOLT_ROUTE ')
    fields = [part.split('=', 1) for part in line.split('BOLT_ROUTE ', 1)[1].split()]
    row = {}
    for key, value in fields:
        row[key] = int(value) if key in FROZEN['OLD']['ROUTE_INTS'] else value
    match = re.fullmatch(r'(.+)-(wide|s[0-8])', row['dst_name'])
    assert match, row
    return row, FROZEN['parse_id'](match[1]), match[2]


def _serialize(row):
    return 'BOLT_ROUTE ' + ' '.join(f'{key}={value}' for key, value in sorted(row.items()))


def _rewrite_routes(log, predicate, update=None, remove=False, duplicate=False):
    changed = 0
    output = []
    for line in log.splitlines():
        if line.startswith('BOLT_ROUTE '):
            row, key, col = _route_row(line)
            if changed == 0 and predicate(row, key, col):
                changed += 1
                if remove:
                    continue
                replacement = dict(row)
                if update:
                    replacement.update(update(dict(row), key, col))
                output.append(_serialize(replacement))
                if duplicate:
                    output.append(_serialize(replacement))
                continue
        output.append(line)
    assert changed == 1, changed
    return '\n'.join(output) + ('\n' if log.endswith('\n') else '')


def _eligible_keys():
    return {
        key for key in FROZEN['cases']('iq2')
        if key[2] in (2, 3, 4) and key[4] == 0 and FROZEN['exact'](key)
    }


def _batch_parent(row, key):
    kind, p, n, mark, layout, annotation = key
    k, m, used = FROZEN['dimensions'](key)
    weight_row = (k // 256) * 66
    parent = dict(row)
    parent.update({
        'route': BATCH,
        'src0_type': 'iq2_xxs', 'src1_type': 'f32', 'dst_type': 'f32',
        'aux_weight': '-', 'aux_type': '-',
        'weight': f'blk.1.ffn_{"up" if p == 0 else "down"}_exps.weight',
        'op': 'MUL_MAT_ID', 'op_id': 30,
        'ne0': k, 'ne1': m, 'ne2': 16, 'ne3': 1,
        'nb0': 66, 'nb1': weight_row, 'nb2': weight_row * m,
        'nb3': weight_row * m * 16,
        'src1_ne0': k, 'src1_ne1': used, 'src1_ne2': n, 'src1_ne3': 1,
        'src1_nb0': 4, 'src1_nb1': 4 * k, 'src1_nb2': 4 * k * used,
        'src1_nb3': 4 * k * used * n,
        'dst_ne0': m, 'dst_ne1': 9, 'dst_ne2': n, 'dst_ne3': 1,
        'dst_nb0': 4, 'dst_nb1': 4 * m, 'dst_nb2': 4 * m * 9,
        'dst_nb3': 4 * m * 9 * n,
        'ids_ne0': 9, 'ids_ne1': n, 'ids_ne2': 1, 'ids_ne3': 1,
        'ids_nb0': 4, 'ids_nb1': 64, 'ids_nb2': 64 * n, 'ids_nb3': 64 * n,
        'reorder': 0, 'expert': -1, 'result': -1,
        'src1_prec': 0, 'operand_prec': -1,
        'accum_prec': 10 if mark else 0, 'dynamic_prec': 0, 'device': 0,
        'dst_name': FROZEN['identity'](key) + '-wide',
    })
    return parent


def synthetic_batch_fixture():
    """Copy the frozen synthetic IQ2 fixture, changing only eligible wide rows."""
    base = FROZEN_TEST['synthetic']('iq2')
    eligible = _eligible_keys()
    seen = set()
    output = []
    for line in base.splitlines():
        if line.startswith('BOLT_ROUTE '):
            row, key, col = _route_row(line)
            if col == 'wide' and key in eligible:
                if key not in seen:
                    output.append(_serialize(_batch_parent(row, key)))
                    seen.add(key)
                continue
        output.append(line)
    assert seen == eligible, (seen, eligible)
    return '\n'.join(output) + '\n'


def _old_log(kind):
    filename = f'ready-verify-singletons-{kind}.log'
    explicit = os.environ.get(f'IQ2_BATCH4_OLD_{kind.upper()}_LOG')
    roots = [Path(explicit).parent] if explicit else []
    roots.extend((
        Path(os.environ.get(
            'IQ2_BATCH4_OLD_RESULTS',
            '/home/mike/code/b70-inference/results/'
            '20261019-nemotron35-bolt-lilicorr-verify-singletons')) / 'qualification-r1' / 'VSB-W3',
        Path('/home/mike/code/b70-inference/results/'
             '20261019-nemotron35-bolt-lilicorr-verify-singletons/qualification-r1/VSB-A'),
    ))
    for root in roots:
        candidate = Path(explicit) if explicit else root / filename
        if candidate.is_file():
            return candidate
    raise AssertionError(f'missing required read-only old native evidence: {filename}')


class BatchRouteTests(unittest.TestCase):
    def test_real_old_q8_and_iq2_logs_stay_frozen(self):
        # These are read-only lead artifacts.  The frozen namespace still
        # proves every old Q8/legacy-IQ2 byte and fallback assertion unchanged;
        # the strict qualifier rejects old IQ2 logs because their eligible
        # parent records do not contain the new physical route.
        for kind in ('q8', 'iq2'):
            path = _old_log(kind)
            text = path.read_bytes().decode('utf-8')
            with self.subTest(kind=kind, path=path):
                expected = FROZEN['check'](text, kind)
                self.assertEqual(FROZEN_TEST['M']['check'](text, kind), expected)
                self.assertEqual(expected['cases'], 82 if kind == 'q8' else 34)
                self.assertEqual(expected['byte_exact_cases'], 54 if kind == 'q8' else 16)
                if kind == 'q8':
                    self.assertEqual(QUALIFIER['check'](text, kind), expected)
                else:
                    with self.assertRaises(AssertionError):
                        QUALIFIER['check'](text, kind)

    def test_eligible_wide_iq2_uses_one_truthful_parent(self):
        log = synthetic_batch_fixture()
        result = QUALIFIER['check'](log, 'iq2')
        self.assertEqual(result['cases'], 34)
        self.assertEqual(result['byte_exact_cases'], 16)

        rows = FROZEN['route_records'](log)
        grouped = {}
        for row in rows:
            _, key, col = _route_row(_serialize(row))
            grouped.setdefault((key, col), []).append(row)
        for key in _eligible_keys():
            with self.subTest(key=key):
                parent = grouped[(key, 'wide')]
                self.assertEqual(len(parent), 1)
                self.assertEqual(parent[0]['route'], BATCH)
                self.assertEqual(parent[0]['dst_name'], FROZEN['identity'](key) + '-wide')

    def test_parent_metadata_mutations_reject(self):
        log = synthetic_batch_fixture()
        key = ('iq2', 0, 4, 1, 0, 1)
        mutations = {
            'wrong parent dimension': {'src1_ne2': 3},
            'wrong weight stride': {'nb1': 1},
            'wrong parent stride': {'src1_nb3': 1},
            'wrong ids stride': {'ids_nb1': 36},
            'wrong source precision': {'src1_prec': 1},
            'wrong accumulation precision': {'accum_prec': 0},
            'wrong source tag': {'src0_type': 'q8_0'},
            'wrong route tag': {'route': 'MOE_SINGLE_MMVQ_PLAIN_Q8_1'},
        }
        for name, changes in mutations.items():
            bad = _rewrite_routes(
                log,
                lambda row, candidate, col: candidate == key and col == 'wide' and row['route'] == BATCH,
                lambda row, candidate, col, changes=changes: changes,
            )
            with self.subTest(name=name), self.assertRaises(AssertionError):
                QUALIFIER['check'](bad, 'iq2')

    def test_parent_route_count_and_presence_reject(self):
        log = synthetic_batch_fixture()
        key = ('iq2', 0, 4, 1, 0, 1)
        target = lambda row, candidate, col: candidate == key and col == 'wide' and row['route'] == BATCH
        missing = _rewrite_routes(log, target, remove=True)
        extra = _rewrite_routes(log, target, duplicate=True)
        old_fixture = FROZEN_TEST['synthetic']('iq2')
        for name, bad in (('missing', missing), ('extra', extra), ('absent from old fixture', old_fixture)):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                QUALIFIER['check'](bad, 'iq2')

    def test_batch_route_on_wrong_eligibility_rejects(self):
        log = synthetic_batch_fixture()
        cases = [
            ('single reference', ('iq2', 0, 4, 1, 0, 1), 's0', 'MOE_SINGLE_MMVQ_PLAIN_Q8_1'),
            ('unmarked', ('iq2', 0, 4, 0, 0, 1), 'wide', 'MOE_GROUPED_GEMM_ATTEMPT'),
            ('negative annotation a0', ('iq2', 0, 4, 1, 0, 0), 'wide', 'MOE_GROUPED_GEMM_ATTEMPT'),
            ('negative annotation a2', ('iq2', 0, 4, 1, 0, 2), 'wide', 'MOE_GROUPED_GEMM_ATTEMPT'),
            ('negative layout l1', ('iq2', 0, 4, 1, 1, 1), 'wide', 'MOE_PER_EXPERT_FALLBACK'),
            ('negative layout l2', ('iq2', 0, 4, 1, 2, 1), 'wide', 'MOE_GROUPED_GEMM_ATTEMPT'),
            ('wider', ('iq2', 0, 5, 1, 0, 1), 'wide', 'MOE_SINGLE_MMVQ_PLAIN_Q8_1'),
        ]
        for name, key, col, original_route in cases:
            bad = _rewrite_routes(
                log,
                lambda row, candidate, route_col, key=key, col=col, original_route=original_route: (
                    candidate == key and route_col == col and row['route'] == original_route),
                lambda row, candidate, route_col: {'route': BATCH},
            )
            with self.subTest(name=name), self.assertRaises(AssertionError):
                QUALIFIER['check'](bad, 'iq2')

        q8 = FROZEN_TEST['synthetic']('q8')
        q8 = _rewrite_routes(
            q8,
            lambda row, key, col: key == ('q8', 0, 4, 1, 0, 1) and col == 'wide',
            lambda row, key, col: {'route': BATCH},
        )
        with self.assertRaises(AssertionError):
            QUALIFIER['check'](q8, 'q8')

    def test_original_required_byte_mismatch_still_rejects(self):
        log = synthetic_batch_fixture()
        pattern = r'(BOLT_SINGLETON case=iq2-p0-n2-m0-l0 bytes=\d+ required=1 mismatches=)0$'
        bad, count = re.subn(pattern, r'\g<1>1', log, count=1, flags=re.MULTILINE)
        self.assertEqual(count, 1)
        with self.assertRaises(AssertionError):
            QUALIFIER['check'](bad, 'iq2')

    def test_cli_accepts_original_kind_log_shape(self):
        # Exercise the same public --kind LOG shape as the frozen checker
        # without rewriting or staging the evidence file.
        for kind in ('q8', 'iq2'):
            path = _old_log(kind)
            with self.subTest(kind=kind):
                args = ['--kind', kind, str(path)]
                if kind == 'q8':
                    self.assertEqual(QUALIFIER['main'](args), 0)
                else:
                    with self.assertRaises(AssertionError):
                        QUALIFIER['main'](args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
