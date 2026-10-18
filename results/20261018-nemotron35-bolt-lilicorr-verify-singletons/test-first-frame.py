#!/usr/bin/env python3
"""Remote CPU regressions for real-frame parsing; synthetic data is NOT runtime proof."""
import array
import copy
import json
import os
import runpy
import socket
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ['BOLT_VERIFY_SINGLETONS_CPU_SCRATCH']).resolve()
assert socket.gethostname() == 'inference-host'
assert SCRATCH.is_relative_to(HERE / 'worker-scratch')


def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise AssertionError('frame CPU fixtures prohibit process/native/API execution: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        assert not os.fsdecode(args[0]).startswith(('/proc/', '/sys/', '/dev/dri')), args[0]


sys.addaudithook(audit)
F = runpy.run_path(str(HERE / 'analyze-first-frame.py'))
P = runpy.run_path(str(F['PARSER']))
N = runpy.run_path(str(F['NATIVE']))
FIX = runpy.run_path(str(F['NATIVE'].with_name('test-trace.py')))


def _set_tensor(row, prefix, shape, strides):
    for i, value in enumerate(shape):
        row[f'{prefix}ne{i}'] = value
    for i, value in enumerate(strides):
        row[f'{prefix}nb{i}'] = value


def synthetic_special_row(weight, n, marked, context='0xabc'):
    """Construct a clearly synthetic full route record, never runtime evidence."""
    spec = F['SPECIAL_SPECS'][weight]
    if marked:
        route = spec['route']
        accum = 10
    elif spec['family'] == 'q8':
        route = 'DMMV_ESIMD' if n == 1 else 'MMVQ_REORDERED'
        accum = 0
    else:
        route = ('MOE_SINGLE_MMVQ_PLAIN_Q8_1' if n == 1
                 else 'MOE_GROUPED_GEMM_ATTEMPT')
        accum = 0
    row = dict(
        route=route, weight=weight, aux_weight='-', op=spec['op'], op_id=spec['op_id'],
        dst_name='synthetic-' + weight.replace('.', '-'),
        src0_type='q8_0' if spec['family'] == 'q8' else 'iq2_xxs',
        aux_type='-', src1_type='f32', dst_type='f32',
        reorder=1 if spec['family'] == 'q8' else 0, src1_prec=0,
        operand_prec=-1, accum_prec=accum, dynamic_prec=0, device=0,
        backend_ctx=context, expert=-1, result=-1,
    )
    _set_tensor(row, '', spec['weight_shape'], spec['weight_strides'])
    if spec['family'] == 'q8':
        src1_ne0, dst_ne0 = spec['src1_ne0'], spec['dst_ne0']
        _set_tensor(row, 'src1_',
                    (src1_ne0, n, 1, 1),
                    (4, spec['src1_nb1'], spec['src1_nb1'] * n,
                     spec['src1_nb1'] * n))
        _set_tensor(row, 'dst_',
                    (dst_ne0, n, 1, 1),
                    (4, spec['dst_nb1'], spec['dst_nb1'] * n,
                     spec['dst_nb1'] * n))
    else:
        _set_tensor(row, 'src1_',
                    (spec['src1_ne0'], spec['src1_ne1'], n, 1),
                    (4, spec['src1_nb1'], spec['src1_nb1'] * spec['src1_ne1'],
                     spec['src1_nb1'] * spec['src1_ne1'] * n))
        _set_tensor(row, 'dst_',
                    (spec['dst_ne0'], spec['dst_ne1'], n, 1),
                    (4, spec['dst_nb1'], spec['dst_nb1'] * spec['dst_ne1'],
                     spec['dst_nb1'] * spec['dst_ne1'] * n))
        _set_tensor(row, 'ids_', (9, n, 1, 1), (4, 960, 960 * n, 960 * n))
    if spec['family'] == 'q8':
        _set_tensor(row, 'ids_', (0, 0, 0, 0), (0, 0, 0, 0))
    return row


def frame_log(n, start, outputs, marked, tokens=None, special_marked=None):
    tokens = list(range(start, start + n)) if tokens is None else tokens
    special_marked = marked if special_marked is None else special_marked
    csv = lambda values: ','.join(map(str, values))
    lines = [f'BOLT_VERIFY batch ctx=0xabc off=0 n={n} tokens={csv(tokens)} '
             f'pos={csv(range(start, start + n))} outputs={csv(outputs)} seq={csv([0] * n)}']
    seed = FIX['rows_for'](('blk.1.ffn_gate_inp.weight', 240, 0, n),
                           'router_prefill' if not marked else 'router_columns8_r3')[0]
    rows = [dict(seed, weight=weight) for weight in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight')]
    for weight in F['ROUTERS']:
        new = FIX['rows_for']((weight, N['ROUTER_WIDTHS'][weight], 0, n),
                              'router_columns8_r3' if marked else 'router_prefill')
        rows.extend(dict(row, dst_name='ffn_moe_logits-' + weight.split('.')[1]) for row in new)
    rows.extend(synthetic_special_row(weight, n, special_marked)
                for weight in F['SPECIAL_ORDER'])
    if any(outputs):
        rows.append(dict(seed, weight='output.weight'))
    lines += ['BOLT_ROUTE ' + ' '.join(f'{key}={value}' for key, value in sorted(row.items()))
              for row in rows]
    lines.append('BOLT_VERIFY decoded ctx=0xabc ret=0')
    return '\n'.join(lines) + '\n'


def frame(n=4, start=55, outputs=None, marked=True, special_marked=None):
    return P['route_frames'](frame_log(
        n, start, [1] * n if outputs is None else outputs, marked,
        special_marked=special_marked))[0]


class FirstFrameFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=SCRATCH))

    def fixture(self, columns):
        directory = self.tmp / str(columns)
        (directory / 'capture').mkdir(parents=True)
        (directory / 'public/short').mkdir(parents=True)
        (directory / 'short').mkdir()
        log = ''.join(frame_log(n, start, outputs, False, special_marked=False)
                      for n, start, outputs in ((5, 0, [0] * 5),
                                                 (46, 5, [0] * 46),
                                                 (4, 51, [0, 0, 0, 1])))
        log += 'BOLT_VERIFY sample ctx=0xabc smpl=0xdef idx=-1 selected=500 vocab=8192 status=cpu file=row-AAA001\n'
        log += frame_log(columns, 55, [1] * columns, columns == 4,
                         list(range(500, 500 + columns)), special_marked=columns == 4)
        log += 'BOLT_VERIFY sample ctx=0xabc smpl=0xdef idx=0 selected=501 vocab=8192 status=cpu file=row-AAA002\n'
        (directory / 'short-diagnostic-log-fresh.stdout.log').write_text('')
        (directory / 'short-diagnostic-log-fresh.stderr.log').write_text(log)
        (directory / 'public/short/diagnostic.prompt-comparison.json').write_text(
            json.dumps({'actual_ids': list(range(55))}))
        (directory / 'short/diagnostic.diagnostic-view.json').write_text(
            json.dumps({'ids': list(range(500, 628))}))
        for i, selected in enumerate((500, 501), 1):
            values = array.array('f', [0.0] * 8192)
            values[selected] = 1.0
            (directory / 'capture' / f'row-AAA00{i}').write_bytes(values.tobytes())
        return directory

    def test_full_frames_prefix_api_ids_and_raw_f32(self):
        for n in (1, 4):
            report, data = F['first_frame'](P, N, self.fixture(n), n)
            self.assertEqual([500, 501], report['first_emitted_ids'])
            self.assertEqual(list(range(55)), report['prompt_ids'])
            self.assertEqual([5, 46, 4], [f['batch']['n'] for f in report['prefill_frames']])
            self.assertEqual(10 if n == 4 else 0, report['router_accum_prec'])
            self.assertEqual(10 if n == 4 else 0, report['special_accum_prec'])
            self.assertEqual(n, report['router_operation_pairs_per_weight'])
            self.assertTrue(all(len(raw) == 8192 * 4 for raw in data.values()))

    def test_singleton_intervention_is_complete_and_strict(self):
        parsed = frame()
        self.assertTrue(F['check_router_frame'](N, parsed, True))
        for weight in F['SPECIAL_ORDER']:
            bad = copy.deepcopy(parsed)
            bad['routes'] = [row for row in bad['routes'] if row['weight'] != weight]
            with self.assertRaises(AssertionError):
                F['check_router_frame'](N, bad, True)
        mutations = (
            lambda rows: rows[0].__setitem__('backend_ctx', '0xddd'),
            lambda rows: next(row for row in rows if row['weight'] in F['SPECIAL_WEIGHTS']).__setitem__('src0_type', 'f16'),
            lambda rows: next(row for row in rows if row['weight'] == F['SPECIAL_ORDER'][0]).__setitem__('route', 'MMVQ_REORDERED'),
            lambda rows: next(row for row in rows if row['weight'] == F['SPECIAL_ORDER'][1]).__setitem__('accum_prec', 0),
            lambda rows: next(row for row in rows if row['weight'] == F['SPECIAL_ORDER'][2]).__setitem__('ne1', 1023),
        )
        for mutate in mutations:
            bad = copy.deepcopy(parsed)
            mutate(bad['routes'])
            with self.assertRaises(AssertionError):
                F['check_router_frame'](N, bad, True)
        bad = copy.deepcopy(parsed)
        extra = next(row for row in bad['routes'] if row['weight'] == 'blk.1.ffn_latent_down.weight')
        extra = copy.deepcopy(extra)
        extra['weight'] = 'blk.8.ffn_latent_down.weight'
        bad['routes'].append(extra)
        with self.assertRaises(AssertionError):
            F['check_router_frame'](N, bad, True)
        bad = copy.deepcopy(parsed)
        duplicate = next(row for row in bad['routes'] if row['weight'] == 'blk.1.ffn_up_exps.weight')
        bad['routes'].append(copy.deepcopy(duplicate))
        with self.assertRaises(AssertionError):
            F['check_router_frame'](N, bad, True)
        bad = copy.deepcopy(parsed)
        special = [i for i, row in enumerate(bad['routes']) if row['weight'] in F['SPECIAL_WEIGHTS']]
        bad['routes'][special[0]], bad['routes'][special[-1]] = (
            bad['routes'][special[-1]], bad['routes'][special[0]])
        with self.assertRaises(AssertionError):
            F['check_router_frame'](N, bad, True)

    def test_parent_w3_routes_fail_new_intervention_contract(self):
        old_parent = frame(marked=True, special_marked=False)
        with self.assertRaises(AssertionError):
            F['check_router_frame'](N, old_parent, True)

    def test_singleton_views_retain_outer_strides(self):
        parsed = frame()
        for row in parsed['routes']:
            if row['weight'] in ('blk.1.ffn_latent_down.weight',
                                 'blk.1.ffn_latent_up.weight'):
                row['src1_ne1'] = row['dst_ne1'] = 1
            elif row['weight'] in ('blk.1.ffn_up_exps.weight',
                                   'blk.1.ffn_down_exps.weight'):
                row['src1_ne2'] = row['dst_ne2'] = row['ids_ne1'] = 1
        self.assertTrue(F['check_router_frame'](N, parsed, True))

    def test_split_is_ordered_singleton_pairs_not_outer_batch_width(self):
        parsed = frame()
        original = copy.deepcopy(parsed)
        report = F['check_router_frame'](N, parsed, True)
        rows = [r for r in report['routes'] if r['weight'] in F['ROUTERS']]
        self.assertEqual(40, len(rows))
        self.assertEqual({1}, {r['src1_ne1'] for r in rows})
        self.assertEqual({65536}, {r['src1_nb2'] for r in rows})
        self.assertEqual(original, parsed)
        mutations = (
            lambda rows: rows[2].__setitem__('accum_prec', 0),
            lambda rows: rows[3].__setitem__('route', 'LIBRARY_DNNL_F32'),
            lambda rows: rows.__setitem__(slice(2, 4), rows[2:4][::-1]),
            lambda rows: rows.__delitem__(slice(2, 4)),
            lambda rows: rows[2].__setitem__('src1_ne1', 4),
            lambda rows: rows[2].__setitem__('backend_ctx', '0xddd'),
        )
        for mutate in mutations:
            bad = copy.deepcopy(parsed)
            mutate(bad['routes'])
            with self.assertRaises(AssertionError):
                F['check_router_frame'](N, bad, True)
        with self.assertRaises(AssertionError):
            F['check_router_frame'](N, frame(marked=False), True)

    def test_all_five_target_routers_required_not_drafter_or_native_names(self):
        for weight in F['ROUTERS']:
            bad = frame()
            bad['routes'] = [r for r in bad['routes'] if r['weight'] != weight]
            with self.assertRaises(AssertionError):
                F['check_router_frame'](N, bad, True)
        bad = frame()
        for row in bad['routes']:
            if row['weight'] in F['ROUTERS']:
                row['dst_name'] = 'router_columns8-m240-l0-n4'
        with self.assertRaises(AssertionError):
            F['check_router_frame'](N, bad, True)
        bad = frame()
        bad['routes'][0]['backend_ctx'] = '0xddd'
        with self.assertRaises(AssertionError):
            F['check_router_frame'](N, bad, True)

    def test_unmarked_prefill_chunks_and_complete55_are_mandatory(self):
        frames = [frame(n, start, outputs, False, special_marked=False)
                  for n, start, outputs in ((5, 0, [0] * 5),
                                             (46, 5, [0] * 46),
                                             (4, 51, [0, 0, 0, 1]))]
        result = F['check_prompt_frames'](N, frames, list(range(55)), '0xabc')
        self.assertTrue(all(f['router_accum_prec'] == 0 for f in result))
        self.assertTrue(all(f['special_accum_prec'] == 0 for f in result))
        for mutate in (
            lambda fs: fs.pop(0),
            lambda fs: fs[0]['batch']['tokens'].__setitem__(0, 999),
            lambda fs: fs[1]['batch']['pos'].__setitem__(0, 0),
            lambda fs: fs[2]['batch']['outputs'].__setitem__(0, 1),
            lambda fs: fs[2]['routes'][2].__setitem__('accum_prec', 10),
            lambda fs: fs[0]['routes'].__setitem__(
                2, dict(fs[0]['routes'][2], route='DMMV_ESIMD')),
        ):
            bad = copy.deepcopy(frames)
            mutate(bad)
            with self.assertRaises(AssertionError):
                F['check_prompt_frames'](N, bad, list(range(55)), '0xabc')

    def test_first_frame_positions_all_outputs_and_actual_api_ids(self):
        directory = self.fixture(4)
        path = directory / 'short-diagnostic-log-fresh.stderr.log'
        original = path.read_text()
        for bad in (original.replace('pos=55,56,57,58', 'pos=55,56,57,59'),
                    original.replace('outputs=1,1,1,1', 'outputs=0,0,0,1'),
                    original.replace('selected=501', 'selected=502')):
            path.write_text(bad)
            with self.assertRaises(AssertionError):
                F['first_frame'](P, N, directory, 4)
        path.write_text(original)
        self.assertTrue(F['first_frame'](P, N, directory, 4))

    def test_raw_byte_comparison_not_probability_or_numeric_equality(self):
        a, b = array.array('f', [0.0, 1.0]), array.array('f', [-0.0, 2.0])
        result = F['compare_rows'](a.tobytes(), b.tobytes())
        self.assertFalse(result['equal_bytes'])
        self.assertEqual(2, result['changed_f32_words'])
        self.assertEqual(1.0, result['max_abs_finite_delta'])
        self.assertTrue(F['compare_rows'](a.tobytes(), a.tobytes())['equal_bytes'])


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
