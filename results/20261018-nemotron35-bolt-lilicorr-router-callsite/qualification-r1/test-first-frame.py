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
SCRATCH = Path(os.environ['BOLT_ROUTER_CALLSITE_CPU_SCRATCH']).resolve()
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


def frame_log(n, start, outputs, marked, tokens=None):
    tokens = list(range(start, start + n)) if tokens is None else tokens
    csv = lambda values: ','.join(map(str, values))
    lines = [f'BOLT_VERIFY batch ctx=0xabc off=0 n={n} tokens={csv(tokens)} '
             f'pos={csv(range(start, start + n))} outputs={csv(outputs)} seq={csv([0] * n)}']
    seed = FIX['rows_for'](('blk.1.ffn_gate_inp.weight', 240, 0, n), 'router_prefill')[0]
    rows = [dict(seed, weight=weight) for weight in ('blk.0.ssm_in.weight', 'blk.0.ssm_out.weight')]
    for weight in F['ROUTERS']:
        new = FIX['rows_for']((weight, N['ROUTER_WIDTHS'][weight], 0, n),
                              'router_columns8_r3' if marked else 'router_prefill')
        rows.extend(dict(row, dst_name='ffn_moe_logits-' + weight.split('.')[1]) for row in new)
    if any(outputs):
        rows.append(dict(seed, weight='output.weight'))
    lines += ['BOLT_ROUTE ' + ' '.join(f'{key}={value}' for key, value in sorted(row.items())) for row in rows]
    lines.append('BOLT_VERIFY decoded ctx=0xabc ret=0')
    return '\n'.join(lines) + '\n'


def frame(n=4, start=55, outputs=None, marked=True):
    return P['route_frames'](frame_log(n, start, [1] * n if outputs is None else outputs, marked))[0]


class FirstFrameFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=SCRATCH))

    def fixture(self, columns):
        directory = self.tmp / str(columns)
        (directory / 'capture').mkdir(parents=True)
        (directory / 'public/short').mkdir(parents=True)
        (directory / 'short').mkdir()
        log = ''.join(frame_log(n, start, outputs, False)
                      for n, start, outputs in ((5, 0, [0] * 5), (46, 5, [0] * 46), (4, 51, [0, 0, 0, 1])))
        log += 'BOLT_VERIFY sample ctx=0xabc smpl=0xdef idx=-1 selected=500 vocab=8192 status=cpu file=row-AAA001\n'
        log += frame_log(columns, 55, [1] * columns, columns == 4, list(range(500, 500 + columns)))
        log += 'BOLT_VERIFY sample ctx=0xabc smpl=0xdef idx=0 selected=501 vocab=8192 status=cpu file=row-AAA002\n'
        (directory / 'short-diagnostic-log-fresh.stdout.log').write_text('')
        (directory / 'short-diagnostic-log-fresh.stderr.log').write_text(log)
        (directory / 'public/short/diagnostic.prompt-comparison.json').write_text(json.dumps({'actual_ids': list(range(55))}))
        (directory / 'short/diagnostic.diagnostic-view.json').write_text(json.dumps({'ids': list(range(500, 628))}))
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
            self.assertEqual(n, report['router_operation_pairs_per_weight'])
            self.assertTrue(all(len(raw) == 8192 * 4 for raw in data.values()))

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
        frames = [frame(n, start, outputs, False) for n, start, outputs in
                  ((5, 0, [0] * 5), (46, 5, [0] * 46), (4, 51, [0, 0, 0, 1]))]
        result = F['check_prompt_frames'](N, frames, list(range(55)), '0xabc')
        self.assertTrue(all(f['router_accum_prec'] == 0 for f in result))
        for mutate in (
            lambda fs: fs.pop(0),
            lambda fs: fs[0]['batch']['tokens'].__setitem__(0, 999),
            lambda fs: fs[1]['batch']['pos'].__setitem__(0, 0),
            lambda fs: fs[2]['batch']['outputs'].__setitem__(0, 1),
            lambda fs: fs[2]['routes'][2].__setitem__('accum_prec', 10),
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
