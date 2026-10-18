#!/usr/bin/env python3
"""LEAD/REMOTE CPU tests only. Synthetic parser fixtures are NOT runtime evidence."""
import re
import runpy
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

CASE = Path(__file__).resolve().parent
A = runpy.run_path(str(CASE / 'analyze-trace.py'))
M = runpy.run_path(str(CASE / 'run-trace.py'))


def rows_for(key, tag, force_split=None):
    weight, m, layout, n = key
    split = (m in (240, 304, 368, 432, 496) and layout == 0 and
             weight.endswith('.ffn_gate_inp.weight') and 2 <= n <= 8)
    if force_split is not None:
        split = force_split
    view_n, calls = (1, n) if split else (n, 1)
    stride = 4 * (4096 + (16 if layout == 2 else 0))
    row = {k: 0 for k in A['ROUTE_INTS']}
    row.update({k: '-' for k in A['ROUTE_STRINGS']})
    row.update(weight=weight, op='MUL_MAT', op_id=29, backend_ctx='0xabc', expert=-1, result=-1,
               src0_type='f32', src1_type='f32', dst_type='f32',
               dst_name=(f'router_single2-l{layout}-n{n}' if tag == 'router_single2'
                         else f'router_columns8-m{m}-l{layout}-n{n}'))
    for prefix, shape, strides in (
            ('', (4096, m, 1, 1), (4, 16384, 16384*m, 16384*m)),
            ('src1_', (4096, view_n, 1, 1), (4, stride, stride*n, stride*n)),
            ('dst_', (m, view_n, 1, 1), (4, 4*m, 4*m*n, 4*m*n))):
        row.update({f'{prefix}ne{i}': v for i, v in enumerate(shape)})
        row.update({f'{prefix}nb{i}': v for i, v in enumerate(strides)})
    library = 'LIBRARY_DNNL_F32' if m * view_n * 4096 >= 16777216 else 'LIBRARY_MKL_F32'
    return [dict(row, route=route, operand_prec=prec) for _ in range(calls)
            for route, prec in (('MUL_MAT_SYCL', -1), (library, 10))]


def fixture(tag='router_columns8', mutate=None):
    lines = ['Backend 1/2: SYCL0']
    cases = sorted(A['native_cases'](tag))
    for key in cases:
        rows = rows_for(key, tag)
        if mutate:
            rows = mutate(key, rows)
        lines += ['BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in sorted(row.items())) for row in rows]
        w, m, layout, n = key
        prefix = 'router_columns8=1,' if tag == 'router_columns8' else ''
        lines.append(f'  MUL_MAT({prefix}router_single2=1,weight={w},layout={layout},'
                     f'type_a=f32,type_b=f32,m={m},n={n},k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],'
                     'k_v=0,o=1,src_overlap=0,m_v=0,pad=0): \x1b[1;32mOK\x1b[0m')
    lines += [f'  {len(cases)}/{len(cases)} tests passed', '  Backend SYCL0: OK',
              'Backend 2/2: CPU', '  Skipping', '2/2 backends passed', 'OK']
    return '\n'.join(lines) + '\n'


class NativeParser(unittest.TestCase):
    def check(self, log, tag='router_columns8'):
        return A['check_router_native_output'](log, tag)

    def reject(self, log, tag='router_columns8'):
        with self.assertRaises(AssertionError):
            self.check(log, tag)

    def test_separate_cases_and_counts(self):
        for tag, count, records, splits in [('router_columns8', 62, 404, 35), ('router_single2', 30, 90, 10)]:
            with self.subTest(tag=tag):
                result = self.check(fixture(tag), tag)
                self.assertEqual((result['cases'], result['route_records'], result['split_cases']), (count, records, splits))
                self.assertEqual(result['public_api_fidelity'], 'NOT_TESTED')
                self.assertEqual(result['device_batched_vs_singleton_byte_equality'], 'NOT_TESTED')

    def test_every_positive_requires_all_actual_singleton_routes(self):
        for key in sorted(A['native_cases']('router_columns8')):
            if not A['split_case'](key):
                continue
            for mode in ('missing', 'last-column-missing', 'extra-column', 'unsplit'):
                def mutate(k, rows):
                    if k != key:
                        return rows
                    return {'missing': [], 'last-column-missing': rows[:-2],
                            'extra-column': rows + rows[-2:], 'unsplit': rows_for(k, 'router_columns8', False)}[mode]
                with self.subTest(key=key, mode=mode):
                    self.reject(fixture(mutate=mutate))

    def test_controls_must_remain_unsplit(self):
        for key in sorted(A['native_cases']('router_columns8')):
            if A['split_case'](key) or key[-1] == 1:
                continue
            with self.subTest(key=key):
                self.reject(fixture(mutate=lambda k, rows: rows_for(k, 'router_columns8', True) if k == key else rows))

    def test_old_selector_n3_now_requires_three_columns(self):
        key = ('blk.1.ffn_gate_inp.weight', 240, 0, 3)
        self.reject(fixture('router_single2', lambda k, rows: rows_for(k, 'router_single2', False) if k == key else rows),
                    'router_single2')

    def test_wrong_metadata_or_route_fails(self):
        changes = dict(op_id=30, src0_type='f16', src1_type='f16', dst_type='f16',
                       reorder=1, device=1, src1_prec=10, accum_prec=10, dynamic_prec=1,
                       operand_prec=10, expert=0, result=0, aux_type='f32', ids_ne0=1,
                       src1_ne1=8, dst_ne1=8, src1_nb2=16384, dst_nb2=960, ne0=2048,
                       backend_ctx='0xdef', route='LIBRARY_DNNL_F32')
        key = ('blk.1.ffn_gate_inp.weight', 240, 0, 8)
        for field, value in changes.items():
            def mutate(k, rows):
                if k == key:
                    rows[0][field] = value
                return rows
            with self.subTest(field=field):
                self.reject(fixture(mutate=mutate))

    def test_no_outputs_only_or_normalized_or_failed_input(self):
        raw = fixture()
        variants = [
            '\n'.join(line for line in raw.splitlines() if 'BOLT_ROUTE' not in line),
            raw.replace('62/62', '61/62'), raw.replace('  Skipping', '  Skipped'),
            raw.replace('Backend 1/2: SYCL0', '2026-10-18T00:00:00Z Backend 1/2: SYCL0'),
            raw + 'ERR = 0.1\n', raw + 'unsupported\n', raw + 'BOLT_VERIFY error\n',
            raw.replace('op_id=29', 'op_id=29 op_id=29', 1),
            raw.replace('op_id=29', 'op_id=29 unknown=0', 1),
            raw.replace('BOLT_ROUTE ', 'BOLT_ROUTE BOLT_ROUTE ', 1),
        ]
        for i, log in enumerate(variants):
            with self.subTest(i=i):
                self.reject(log)
        self.reject(fixture('router_single2'))


class SourcePatch(unittest.TestCase):
    def test_exact_consumed_sources_patch_scope_and_preserved_guards(self):
        M['check_consumed_sources'](M['PARENT'], 'parent')
        with tempfile.TemporaryDirectory(prefix='columns8-source-', dir=CASE) as tmp:
            root = Path(tmp)
            for name in M['SOURCE_SCOPE']:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(M['PARENT'] / name, root / name)
            applied = subprocess.run(['patch', '--fuzz=0', '--batch', '--forward', '-p1'], cwd=root,
                                     input=(CASE / 'router-columns8.patch').read_bytes(), capture_output=True, check=True)
            print(applied.stdout.decode(), end='')
            M['check_consumed_sources'](root, 'candidate')
            old = (M['PARENT'] / M['SOURCE_SCOPE'][0]).read_text()
            new = (root / M['SOURCE_SCOPE'][0]).read_text()
            begin = old.index('// Bounded router control:')
            end = old.index('    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);', begin)
            body = old[begin:end]
            # Explicitly enumerate the only permitted runtime deltas. This keeps every
            # device/buffer/layout/alias/reorder/op/precision guard and all other dispatch byte-identical.
            changes = [
                ('either-column preflight', 'all-column preflight', 1),
                ('ggml_tensor (&src1_col)[2], ggml_tensor (&dst_col)[2]',
                 'ggml_tensor (&src1_col)[8], ggml_tensor (&dst_col)[8]', 1),
                ('src1->ne[0] != 4096 || src1->ne[1] != 2 || dst->ne[0] != src0->ne[1] || dst->ne[1] != 2)',
                 'src1->ne[0] != 4096 || src1->ne[1] < 2 || src1->ne[1] > 8 ||\n'
                 '        dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1])', 1),
                ('across the two launches', 'across all column launches', 1),
                ('for (int col = 0; col < 2; ++col)', 'for (int col = 0; col < src1->ne[1]; ++col)', 2),
                ('ggml_tensor src1_col[2], dst_col[2]', 'ggml_tensor src1_col[8], dst_col[8]', 1),
                ('Both descriptors are already validated.', 'All column descriptors are already validated.', 1),
            ]
            for before, after, count in changes:
                self.assertEqual(body.count(before), count)
                body = body.replace(before, after)
            self.assertEqual(new, old[:begin] + body + old[end:])
            old_test = (M['PARENT'] / M['SOURCE_SCOPE'][1]).read_text()
            new_test = (root / M['SOURCE_SCOPE'][1]).read_text()
            start = old_test.index('struct test_mul_mat_router_single2 :')
            stop = old_test.index('\n};', start) + len('\n};')
            self.assertIn(old_test[start:stop], new_test)  # tolerance + discriminating input unchanged
            start = old_test.index('    // Exactly six router cases,')
            stop = old_test.index('    // Experimental NVFP4 reorder lifetime/', start)
            self.assertIn(old_test[start:stop], new_test)  # all old 30 registrations unchanged
            table = new_test.split('router_columns8_profiles[] = {', 1)[1].split('    };', 1)[0]
            profiles = re.findall(r'\{"(blk\.[^"\n]+)",\s*"(blk\.[^"\n]+)",\s*(\d+)\}', table)
            self.assertEqual({(w, int(m)) for w, _, m in profiles}, set(A['ROUTER_WIDTHS'].items()))
            self.assertEqual({(w, int(m)) for _, w, m in profiles}, set(A['UNRELATED_WIDTHS'].items()))
            self.assertIn('for (int64_t n = 1; n <= 9; ++n)', new_test)
            self.assertIn('std::string("router_columns8=1,") + test_mul_mat_router_single2::vars()', new_test)

    def test_native_command_is_only_a_printed_inner_command(self):
        subprocess.run(['bash', '-n'], input=M['NATIVE_COMMAND'].encode(), check=True)
        self.assertIn('for tag in router_single2 router_columns8;', M['NATIVE_COMMAND'])
        self.assertIn('BOLT_VERIFY_CAPTURE_DIR=/capture', M['NATIVE_COMMAND'])
        self.assertIn('-p "^$tag=1,"', M['NATIVE_COMMAND'])
        self.assertNotIn('docker ', M['NATIVE_COMMAND'])


if __name__ == '__main__':
    assert socket.gethostname() == 'inference-host', 'remote CPU execution only'
    unittest.main(verbosity=2)
