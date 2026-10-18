#!/usr/bin/env python3
"""LEAD/REMOTE CPU tests only. Synthetic fixtures are NOT runtime evidence."""
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
    # Independent fixture oracle: only the new family marks columns 3..8.
    marked = tag == 'router_columns8_r3' and 3 <= n <= 8
    split = (m in (240, 304, 368, 432, 496) and layout == 0 and
             weight.endswith('.ffn_gate_inp.weight') and (n == 2 or marked))
    if force_split is not None:
        split = force_split
    view_n, calls = (1, n) if split else (n, 1)
    stride = 4 * (4096 + (16 if layout == 2 else 0))
    row = {k: 0 for k in A['ROUTE_INTS']}
    row.update({k: '-' for k in A['ROUTE_STRINGS']})
    names = {'router_single2': f'router_single2-l{layout}-n{n}',
             'router_columns8_r3': f'router_columns8-m{m}-l{layout}-n{n}',
             'router_prefill': f'router_prefill-m{m}-n{n}'}
    row.update(weight=weight, op='MUL_MAT', op_id=29, backend_ctx='0xabc', expert=-1, result=-1,
               src0_type='f32', src1_type='f32', dst_type='f32',
               accum_prec=10 if marked else 0, dst_name=names[tag])
    for prefix, shape, strides in (
            ('', (4096, m, 1, 1), (4, 16384, 16384*m, 16384*m)),
            ('src1_', (4096, view_n, 1, 1), (4, stride, stride*n, stride*n)),
            ('dst_', (m, view_n, 1, 1), (4, 4*m, 4*m*n, 4*m*n))):
        row.update({f'{prefix}ne{i}': v for i, v in enumerate(shape)})
        row.update({f'{prefix}nb{i}': v for i, v in enumerate(strides)})
    library = 'LIBRARY_DNNL_F32' if m * view_n * 4096 >= 16777216 else 'LIBRARY_MKL_F32'
    return [dict(row, route=route, operand_prec=prec) for _ in range(calls)
            for route, prec in (('MUL_MAT_SYCL', -1), (library, 10))]


def fixture(tag='router_columns8_r3', mutate=None):
    lines = ['Backend 1/2: SYCL0']
    cases = sorted(A['native_cases'](tag))
    for key in cases:
        rows = rows_for(key, tag)
        if mutate:
            rows = mutate(key, rows)
        lines += ['BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in sorted(row.items())) for row in rows]
        w, m, layout, n = key
        prefix = {'router_single2': '', 'router_columns8_r3': 'router_columns8=1,',
                  'router_prefill': 'router_prefill=1,'}[tag]
        lines.append(f'  MUL_MAT({prefix}router_single2=1,weight={w},layout={layout},'
                     f'type_a=f32,type_b=f32,m={m},n={n},k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],'
                     'k_v=0,o=1,src_overlap=0,m_v=0,pad=0): \x1b[1;32mOK\x1b[0m')
    lines += [f'  {len(cases)}/{len(cases)} tests passed', '  Backend SYCL0: \x1b[1;32mOK\x1b[0m',
              'Backend 2/2: CPU', '  Skipping', '2/2 backends passed', 'OK']
    return '\n'.join(lines) + '\n'


class NativeParser(unittest.TestCase):
    def check(self, log, tag='router_columns8_r3'):
        return A['check_router_native_output'](log, tag)

    def reject(self, log, tag='router_columns8_r3'):
        with self.assertRaises(AssertionError):
            self.check(log, tag)

    def test_separate_cases_and_counts(self):
        for tag, count, records, splits in [('router_columns8_r3', 61, 402, 35),
                                          ('router_single2', 30, 70, 5), ('router_prefill', 2, 4, 0)]:
            with self.subTest(tag=tag):
                result = self.check(fixture(tag), tag)
                self.assertEqual((result['cases'], result['route_records'], result['split_cases']), (count, records, splits))
                self.assertEqual(result['public_api_fidelity'], 'NOT_TESTED')
                self.assertEqual(result['device_batched_vs_singleton_byte_equality'], 'NOT_TESTED')

    def test_colored_status_is_exact_and_unique(self):
        colored = '\x1b[1;32mOK\x1b[0m'
        for tag in A['TAGS']:
            raw = fixture(tag)
            line = '  Backend SYCL0: ' + colored
            self.assertEqual(self.check(raw, tag), self.check(raw.replace(colored, 'OK'), tag))
            for status in ('', 'FAIL', '\x1b[1;31mFAIL\x1b[0m', 'OK trailing',
                           '\x1b[2JOK', 'OK\n  Backend SYCL0: OK'):
                with self.subTest(tag=tag, status=status):
                    self.reject(raw.replace(line, '  Backend SYCL0: ' + status), tag)
            self.reject(raw.replace(line + '\n', ''), tag)
            self.reject(raw.replace('): ' + colored, '): ' + colored + ' trailing', 1), tag)

    def test_every_positive_requires_all_actual_singleton_routes(self):
        for key in sorted(A['native_cases']('router_columns8_r3')):
            if not A['split_case'](key):
                continue
            for mode in ('missing', 'last-column-missing', 'extra-column', 'unsplit'):
                def mutate(k, rows):
                    if k != key:
                        return rows
                    return {'missing': [], 'last-column-missing': rows[:-2],
                            'extra-column': rows + rows[-2:],
                            'unsplit': rows_for(k, 'router_columns8_r3', False)}[mode]
                with self.subTest(key=key, mode=mode):
                    self.reject(fixture(mutate=mutate))

    def test_controls_must_remain_unsplit(self):
        for tag in A['TAGS']:
            for key in sorted(A['native_cases'](tag)):
                if A['split_case'](key, tag) or key[-1] == 1:
                    continue
                with self.subTest(tag=tag, key=key):
                    self.reject(fixture(tag, lambda k, rows: rows_for(k, tag, True) if k == key else rows), tag)

    def test_actual_precision_required_including_negative_controls(self):
        for tag in A['TAGS']:
            for key in sorted(A['native_cases'](tag)):
                def mutate(k, rows):
                    if k == key:
                        # Change both halves so pair agreement alone cannot pass.
                        return [dict(r, accum_prec=0 if r['accum_prec'] == 10 else 10) for r in rows]
                    return rows
                with self.subTest(tag=tag, key=key):
                    self.reject(fixture(tag, mutate), tag)

    def test_checkpoint_default_and_marked_verification_are_distinct(self):
        for n in (4, 5):
            key = ('blk.48.ffn_gate_inp.weight', 496, 0, n)
            prefill = rows_for(key, 'router_prefill')
            verification = rows_for(key, 'router_columns8_r3')
            self.assertEqual(len(prefill), 2)
            self.assertEqual({r['accum_prec'] for r in prefill}, {0})
            self.assertEqual(len(verification), 2*n)
            self.assertEqual({r['accum_prec'] for r in verification}, {10})
        for n in range(3, 9):
            key = ('blk.1.ffn_gate_inp.weight', 240, 0, n)
            self.assertEqual(len(rows_for(key, 'router_columns8_r3')), 2*n)
        # The old selector is wholly default-precision, including all five N3s.
        for w, m in A['ROUTER_WIDTHS'].items():
            self.assertEqual(len(rows_for((w, m, 0, 3), 'router_single2')), 2)

    def test_wrong_metadata_or_route_fails(self):
        changes = dict(op_id=30, src0_type='f16', src1_type='f16', dst_type='f16',
                       reorder=1, device=1, src1_prec=10, accum_prec=0, dynamic_prec=1,
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
            raw.replace('61/61', '60/61'), raw.replace('  Skipping', '  Skipped'),
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
        self.reject(fixture('router_prefill'))


class SourcePatch(unittest.TestCase):
    def test_exact_consumed_sources_patch_scope_and_preserved_guards(self):
        M['check_consumed_sources'](M['PARENT'], 'parent')
        with tempfile.TemporaryDirectory(prefix='verify-phase-source-', dir=CASE) as tmp:
            root = Path(tmp)
            for name in M['SOURCE_SCOPE']:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(M['PARENT'] / name, root / name)
            applied = subprocess.run(['patch', '--fuzz=0', '--batch', '--forward', '-p1'], cwd=root,
                                     input=(CASE / 'router-verify-phase.patch').read_bytes(), capture_output=True, check=True)
            print(applied.stdout.decode(), end='')
            self.assertNotRegex(applied.stdout.decode(), r'offset|fuzz')
            M['check_consumed_sources'](root, 'candidate')
            old = (M['PARENT'] / 'ggml/src/ggml-sycl/ggml-sycl.cpp').read_text()
            new = (root / 'ggml/src/ggml-sycl/ggml-sycl.cpp').read_text()
            begin = old.index('// Bounded router control:')
            end = old.index('    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);', begin)
            body = old[begin:end]
            # Only these runtime changes are permitted; preserve every other guard
            # (device/buffer/layout/alias/reorder/op/hint) and all unrelated dispatch.
            changes = [
                ('either-column preflight', 'all-column preflight', 1),
                ('ggml_tensor (&src1_col)[2], ggml_tensor (&dst_col)[2]',
                 'ggml_tensor (&src1_col)[8], ggml_tensor (&dst_col)[8]', 1),
                ('src1->ne[0] != 4096 || src1->ne[1] != 2 || dst->ne[0] != src0->ne[1] || dst->ne[1] != 2)',
                 'src1->ne[0] != 4096 || src1->ne[1] < 2 || src1->ne[1] > 8 ||\n'
                 '        (src1->ne[1] >= 3 && ggml_get_op_params_i32(dst, 0) != GGML_PREC_F32) ||\n'
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
            old_graph = (M['PARENT'] / 'src/llama-graph.cpp').read_text()
            new_graph = (root / 'src/llama-graph.cpp').read_text()
            before = '''        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
'''
            addition = '''        // Bound the F32 router request to small full-output batches. This excludes
        // ordinary checkpoint prefill (zero/one output), not every possible prompt.
        if (arch == LLM_ARCH_NEMOTRON_H_MOE &&
            gate_inp->type == GGML_TYPE_F32 && cur->type == GGML_TYPE_F32 && logits->type == GGML_TYPE_F32 &&
            logits->op == GGML_OP_MUL_MAT && logits->src[0] == gate_inp && logits->src[1] == cur &&
            n_tokens >= 3 && n_tokens <= 8 && n_tokens == this->n_tokens && this->n_outputs == this->n_tokens) {
            GGML_ASSERT(ggml_prec_set_acc(logits, GGML_PREC_F32));
        }
'''
            self.assertEqual(old_graph.count(before), 1)
            self.assertEqual(new_graph, old_graph.replace(before, before + addition))
            old_test = (M['PARENT'] / 'tests/test-backend-ops.cpp').read_text()
            new_test = (root / 'tests/test-backend-ops.cpp').read_text()
            class_start = new_test.index('// Separate selector; inherit the 1e-7 CPU tolerance')
            class_end = new_test.index('// GGML_HINT_SRC0_IS_HADAMARD\nstruct test_mul_mat_hadamard', class_start)
            new_classes = new_test[class_start:class_end]
            reg_start = new_test.index('    // Exactly 62 additional cases,')
            reg_end = new_test.index('    // Experimental NVFP4 reorder lifetime/', reg_start)
            # Entire old test file survives, including all 30 registrations, old
            # default precision, tolerance 1e-7, names and discriminating inputs.
            self.assertEqual(new_test[:class_start] + new_test[class_end:reg_start] + new_test[reg_end:], old_test)
            self.assertEqual(new_classes.count('ggml_prec_set_acc('), 1)
            self.assertIn('if (n >= 3 && n <= 8) {\n            GGML_ASSERT(ggml_prec_set_acc(out, GGML_PREC_F32));\n        }', new_classes)
            self.assertNotIn('ggml_prec_set_acc', new_classes.split('struct test_mul_mat_router_prefill', 1)[1])
            self.assertIn('"router_prefill-m%d-n%d"', new_classes)
            self.assertIn('std::string("router_prefill=1,") + test_mul_mat_router_single2::vars()', new_classes)
            self.assertIn('std::string("router_columns8=1,") + test_mul_mat_router_single2::vars()', new_classes)
            table = new_test.split('router_columns8_profiles[] = {', 1)[1].split('    };', 1)[0]
            profiles = re.findall(r'\{"(blk\.[^"\n]+)",\s*"(blk\.[^"\n]+)",\s*(\d+)\}', table)
            self.assertEqual({(w, int(m)) for w, _, m in profiles}, set(A['ROUTER_WIDTHS'].items()))
            self.assertEqual({(w, int(m)) for _, w, m in profiles}, set(A['UNRELATED_WIDTHS'].items()))
            self.assertIn('for (int64_t n = 1; n <= 9; ++n)', new_test)
            self.assertIn('''    for (int64_t n : {4, 5}) {
        test_cases.emplace_back(new test_mul_mat_router_prefill("blk.48.ffn_gate_inp.weight", n, 0, 496));
    }''', new_test)

    def test_native_command_is_only_a_printed_inner_command(self):
        subprocess.run(['bash', '-n'], input=M['NATIVE_COMMAND'].encode(), check=True)
        self.assertIn('for tag in router_single2 router_columns8_r3 router_prefill;', M['NATIVE_COMMAND'])
        self.assertIn('BOLT_VERIFY_CAPTURE_DIR=/capture', M['NATIVE_COMMAND'])
        self.assertIn('-p "$selector"', M['NATIVE_COMMAND'])
        self.assertNotIn('docker ', M['NATIVE_COMMAND'])
        selector = re.search(r"router_columns8_r3\) selector='([^']+)'", M['NATIVE_COMMAND']).group(1)
        full = A['native_cases']('router_columns8_r3') | {('blk.48.ffn_gate_inp.weight', 496, 0, 9)}
        selected = set()
        for w, m, layout, n in full:
            params = (f'router_columns8=1,router_single2=1,weight={w},layout={layout},'
                      f'type_a=f32,type_b=f32,m={m},n={n},k=4096,')
            if re.search(selector, params):
                selected.add((w, m, layout, n))
        self.assertEqual(selected, A['native_cases']('router_columns8_r3'))

    def test_build_captures_core_library_and_preserves_old_five_pins(self):
        self.assertEqual(len(M['ARTIFACTS']), 6)
        self.assertIn('libllama.so.0.7.0', M['ARTIFACTS'])
        for _, _, pins in M['PROTECTED']:
            self.assertEqual(set(M['ARTIFACTS']) - set(pins), {'libllama.so.0.7.0'})
        self.assertEqual(M['SOURCE_SCOPE'], ['ggml/src/ggml-sycl/ggml-sycl.cpp',
                                           'src/llama-graph.cpp', 'tests/test-backend-ops.cpp'])
        build = (CASE / 'build-candidate.sh').read_text()
        subprocess.run(['bash', '-n', str(CASE / 'build-candidate.sh')], check=True)
        self.assertIn('--target llama-server test-backend-ops', build)
        self.assertIn("m['check_legacy_protection']()", build)
        self.assertIn("'libllama.so.0.7.0'} <= set(changed)", build)


if __name__ == '__main__':
    assert socket.gethostname() == 'inference-host', 'remote CPU execution only'
    unittest.main(verbosity=2)
