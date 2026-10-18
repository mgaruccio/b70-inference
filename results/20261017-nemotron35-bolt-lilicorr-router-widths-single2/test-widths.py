#!/usr/bin/env python3
"""Additional remote CPU regressions; synthetic routes are NOT runtime evidence."""
import ast
import copy
import json
import re
import runpy
import subprocess
import unittest
from pathlib import Path

T = runpy.run_path('test-trace.py')
M, A, OLD = T['M'], T['A'], T['OLD']


class Widths(unittest.TestCase):
    setUp = T['Trace'].setUp
    tearDown = T['Trace'].tearDown
    saved_cell = T['Trace'].saved_cell
    patched_sources = T['Trace'].patched_sources

    def mutate_model(self, weight, change, label='C0'):
        log = T['router_model_log'](label)
        lines = log.splitlines(True)
        indices = [i for i, line in enumerate(lines) if f'weight={weight} ' in line]
        self.assertTrue(indices)
        change(lines, indices)
        self.assertRaises(AssertionError, A['assert_router_model_routes'], ''.join(lines), label)

    def test_actual_inventory_closed_widths_and_representatives(self):
        inventory = json.loads((M['ROOT'].parent / '20261017-nemotron35-bolt-lilicorr-block8-operands/router-width-inventory.json').read_text())
        self.assertEqual(inventory['width_counts'], {'240': 12, '304': 6, '368': 9, '432': 10, '496': 3})
        self.assertEqual(inventory['routers'], 40)
        tensors = {r['name']: r for r in inventory['tensors']}
        self.assertEqual(set(M['ROUTER_WIDTHS'].values()), {240, 304, 368, 432, 496})
        self.assertEqual(M['ROUTER_WIDTHS'], A['ROUTER_WIDTHS'])
        for name, m in M['ROUTER_WIDTHS'].items():
            self.assertEqual(tensors[name]['dimensions'], [4096, m])
            self.assertEqual(tensors[name]['type'], 0)
            self.assertLess(int(name.split('.')[1]), 87)

    def test_actual_native_generator_yields_exact_thirty_profiles(self):
        _, native = self.patched_sources()
        block = native.split('// Exactly six router cases,', 1)[1].split('// Experimental NVFP4 reorder', 1)[0]
        observed = set()
        for weight, n, tail in re.findall(r'new test_mul_mat_router_single2\("([^"]+)", (n|2)([^)]*)\)', block):
            args = [int(x.strip()) for x in tail.split(',') if x.strip()]
            layout, m = (args + [0, 240])[:2] if len(args) == 2 else (args[0] if args else 0, 240)
            self.assertEqual(M['ROUTER_NATIVE_WIDTHS'][weight], m)
            for width in ((1, 2, 3) if n == 'n' else (2,)):
                self.assertNotIn((weight, layout, width), observed)
                observed.add((weight, layout, width))
        self.assertEqual(observed, M['ROUTER_NATIVE_CASES'])
        self.assertEqual(len(observed), 30)
        parent = (M['PARENT'] / M['SOURCE_SCOPE'][1]).read_text()
        # All non-router native cases retain exact source; SSM385/OUT_PROD190/9,
        # IQ2six/latent7/attention13/view6 selectors are independent of this family.
        self.assertEqual(native[:native.index('// Six isolated F32 CPU-reference cases')],
                         parent[:parent.index('// Six isolated F32 CPU-reference cases')])
        self.assertEqual(native[native.index('// Experimental NVFP4 reorder lifetime/scale regressions.'):],
                         parent[parent.index('// Experimental NVFP4 reorder lifetime/scale regressions.'):])
        for command in ('IQ2_NATIVE_COMMAND', 'Q8_NATIVE_COMMAND', 'ATTENTION_NATIVE_COMMAND', 'ATTENTION_VIEW_NATIVE_COMMAND'):
            self.assertEqual(M[command], runpy.run_path(str(T['PARENT_TOP'] / 'run-trace.py'))[command])

    def test_only_four_exact_logging_selectors_added(self):
        patch = Path('router-widths-single2.patch').read_text()
        additions = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
        selected = re.findall(r'std::strcmp\(weight->name, "([^"]+)"\)', additions)
        self.assertEqual(selected, [w for w in A['ROUTER_WIDTHS'] if not w.startswith('blk.1.')])
        for forbidden in ('BOLT_MOE_OPERAND', 'BOLT_BLOCK', 'getenv(', 'wait(', 'submit(', 'std::strncmp'):
            self.assertNotIn(forbidden, additions)
        source, _ = self.patched_sources()
        self.assertNotIn('bolt_moe_before', source)
        parent = (M['PARENT'] / M['SOURCE_SCOPE'][0]).read_text()
        for name in ('ggml_sycl_mul_mat', 'ggml_sycl_op_mul_mat', 'ggml_sycl_router_single2_range'):
            self.assertEqual(T['cpp_body'](source, name), T['cpp_body'](parent, name))
        # Compile the actual selected-weight predicate; synthetic native output
        # alone cannot prove that the real logger will emit each negative case.
        predicate = T['cpp_body'](source, 'bolt_route_selected_weight')
        program = '#include "ggml.h"\n#include <cstring>\n#include <cassert>\n'
        program += 'static bool bolt_route_selected_weight(const ggml_tensor * weight) {' + predicate + '}\n'
        program += 'int main() { ggml_tensor t{}; assert(!bolt_route_selected_weight(nullptr)); assert(!bolt_route_selected_weight(&t));\n'
        for weight in M['ROUTER_NATIVE_WIDTHS']:
            program += 'std::strcpy(t.name, ' + json.dumps(weight) + '); assert(bolt_route_selected_weight(&t));\n'
        for block in (8, 21, 37, 48):
            for suffix in ('unrelated.weight', 'ffn_latent_down.weight', 'ffn_up_exps.weight', 'ffn_gate_inp.weight.extra'):
                program += 'std::strcpy(t.name, ' + json.dumps(f'blk.{block}.{suffix}') + '); assert(!bolt_route_selected_weight(&t));\n'
        program += '}\n'
        cpp, executable = self.root / 'actual-route-selector.cpp', self.root / 'actual-route-selector'
        cpp.write_text(program)
        subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                        '-I' + str(M['PARENT'] / 'ggml/include'), str(cpp), '-o', str(executable)],
                       check=True, capture_output=True, timeout=30)
        subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)
        print('actual compiled route selector: all 10 native weight names selected; 16 off-scope names rejected')

    def test_public_analyze_arm_rejects_whole_extension_missing(self):
        cell = self.saved_cell('S0')
        self.assertTrue(A['analyze_arm'](cell, 'S0', OLD)['s0_parent_invariant_passed'])
        # Real parent capture has no extension; not accepted as candidate evidence.
        parent = A['parent_reference']('S0')[0]
        (cell / 'diagnostic-window.log').write_text((parent / 'diagnostic-window.log').read_text())
        self.assertRaises(AssertionError, A['analyze_arm'], cell, 'S0', OLD)

    def test_missing_each_representative_on_n1_and_n2(self):
        for label in ('S0', 'C0'):
            for weight in A['ROUTER_WIDTHS']:
                with self.subTest(label=label, weight=weight):
                    self.mutate_model(weight, lambda lines, ids: [lines.__setitem__(i, '') for i in ids], label)

    def test_every_representative_exact_operation_cardinality(self):
        for weight in A['ROUTER_WIDTHS']:
            with self.subTest(weight=weight):
                self.mutate_model(weight, lambda lines, ids: lines.__setitem__(ids[0], ''))
                self.mutate_model(weight, lambda lines, ids: lines.__setitem__(ids[0], lines[ids[0]] * 2))
                self.mutate_model(weight, lambda lines, ids: [lines.__setitem__(i, '') for i in ids[2:]])

    def test_every_representative_order_is_strict(self):
        for weight in A['ROUTER_WIDTHS']:
            def swap(lines, ids):
                lines[ids[0]], lines[ids[1]] = lines[ids[1]], lines[ids[0]]
            self.mutate_model(weight, swap)
        log = T['router_model_log']('C0')
        a, b = 'blk.8.ffn_gate_inp.weight', 'blk.21.ffn_gate_inp.weight'
        bad = log.replace(a, '__TEMP__').replace(b, a).replace('__TEMP__', b)
        self.assertRaises(AssertionError, A['assert_router_model_routes'], bad, 'C0')

    def test_every_representative_retained_strides_and_types(self):
        for weight in A['ROUTER_WIDTHS']:
            for key, value in (('src1_nb2', 16384), ('src1_nb3', 16384), ('dst_nb2', 1), ('dst_nb3', 1),
                               ('src1_ne1', 2), ('dst_ne1', 2), ('src0_type', 'f16'), ('reorder', 1)):
                def alter(lines, ids):
                    for i in ids:
                        lines[i] = re.sub(r'\b' + key + r'=[^\s]+', f'{key}={value}', lines[i])
                with self.subTest(weight=weight, key=key):
                    self.mutate_model(weight, alter)

    def test_every_representative_must_have_actual_model_width(self):
        for weight, m in A['ROUTER_WIDTHS'].items():
            wrong = next(v for v in A['ROUTER_WIDTHS'].values() if v != m)
            def alter(lines, ids):
                for i in ids:
                    r = A['route_records'](lines[i])[0]
                    r.update(ne1=wrong, nb2=16384*wrong, nb3=16384*wrong,
                             dst_ne0=wrong, dst_nb1=4*wrong, dst_nb2=8*wrong, dst_nb3=8*wrong)
                    lines[i] = 'BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in r.items()) + '\n'
            self.mutate_model(weight, alter)

    def test_every_representative_foreign_backend_cannot_fill_target(self):
        for weight in A['ROUTER_WIDTHS']:
            def foreign(lines, ids):
                for i in ids:
                    lines[i] = lines[i].replace('backend_ctx=0xabc ', 'backend_ctx=0x999 ')
            self.mutate_model(weight, foreign)
        for label in ('S0', 'C0'):
            log = T['router_model_log'](label)
            records = '\n'.join(T['router_lines'](2, weight='blk.8.ffn_gate_inp.weight', backend='0x999')) + '\n'
            self.assertTrue(A['assert_router_model_routes'](log + records, label)[-1]['other_backend_routes'])

    def test_prefill_fixtures_without_head_and_all_five_library_pairs(self):
        for label in ('S0', 'C0'):
            original = (A['parent_reference'](label)[0] / 'diagnostic-window.log').read_text()
            log = T['router_candidate_log'](original)
            self.assertEqual(A['records'](log), A['records'](original))
            old_rows = A['route_records'](original)
            new_rows = A['route_records'](log)
            extension = set(A['ROUTER_WIDTHS']) - {'blk.1.ffn_gate_inp.weight'}
            self.assertFalse(any(r['weight'] in extension for r in old_rows))
            self.assertEqual([r for r in new_rows if r['weight'] not in extension], old_rows)
            frames = A['route_frames'](log)
            self.assertEqual([f['batch']['n'] for f in frames[:3]], [5, 46, 4])
            for i, frame in enumerate(frames[:3]):
                self.assertEqual(any(r['weight'] == 'output.weight' for r in frame['routes']), i == 2)
                for weight, m in A['ROUTER_WIDTHS'].items():
                    rows = [r for r in frame['routes'] if r['weight'] == weight]
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(rows[1]['route'], 'LIBRARY_DNNL_F32' if i == 1 else 'LIBRARY_MKL_F32')
                    self.assertEqual(rows[1]['ne1'], m)

    def test_each_prefill_representative_wrong_library_or_missing_fails(self):
        original = (A['parent_reference']('S0')[0] / 'diagnostic-window.log').read_text()
        frames = A['route_frames'](T['router_candidate_log'](original))[:3]
        for frame in frames:
            for weight in A['ROUTER_WIDTHS']:
                missing = copy.deepcopy(frame)
                missing['routes'] = [r for r in missing['routes'] if r['weight'] != weight]
                self.assertRaises(AssertionError, A['_assert_prefill_frame'], missing)
                wrong = copy.deepcopy(frame)
                for r in wrong['routes']:
                    if r['weight'] == weight and r['route'] != 'MUL_MAT_SYCL':
                        r['route'] = 'LIBRARY_MKL_F32' if r['route'] == 'LIBRARY_DNNL_F32' else 'LIBRARY_DNNL_F32'
                self.assertRaises(AssertionError, A['_assert_prefill_frame'], wrong)

    def test_public_analyze_arm_requires_each_decode_representative(self):
        cell = self.saved_cell('C0')
        good = (cell / 'diagnostic-window.log').read_text()
        result = A['analyze_arm'](cell, 'C0', OLD)
        frames = result['route_analysis']['router_batches']
        self.assertEqual(len(frames), len([f for f in A['_verify_frames'](good) if f['batch']['pos'][0] >= 55]))
        for weight in A['ROUTER_WIDTHS']:
            line = next(line for line in reversed(good.splitlines(True)) if f'weight={weight} ' in line)
            bad = good[:good.rindex(line)] + good[good.rindex(line)+len(line):]
            (cell / 'diagnostic-window.log').write_text(bad)
            self.assertRaises(AssertionError, A['analyze_arm'], cell, 'C0', OLD)

    def test_native_old_six_and_partial_widths_cannot_pass(self):
        old = runpy.run_path(str(T['PARENT_TOP'] / 'test-trace.py'))['router_native_output']()
        self.assertRaises(AssertionError, M['check_router_native_output'], old)
        good = T['router_native_output']()
        M['check_router_native_output'](good)
        for weight in A['ROUTER_WIDTHS']:
            lines = good.splitlines(True)
            case = next(line for line in lines if f'weight={weight},' in line and 'MUL_MAT(' in line)
            for bad in (good.replace(case, ''), good.replace(case, case.replace('OK', 'SKIPPED')),
                        good.replace(case, case.replace('m=', 'm=1', 1)),
                        ''.join(line for line in lines if f'weight={weight} ' not in line)):
                self.assertRaises(AssertionError, M['check_router_native_output'], bad)
        self.assertIn("30/30 tests passed", M['ROUTER_NATIVE_COMMAND'])

    def test_histories_are_derived_not_frozen_and_all128_match_is_reported(self):
        log, tokens = T['fixture'](self.root / 'alternate-C0', speculative=True, chunked_prompt=True)
        aligned = A['align'](A['records'](log), tokens, self.root / 'alternate-C0')
        accepted = sum(v['accepted'] for v in aligned['verifications'])
        proposed = sum(len(v['drafts']) for v in aligned['verifications'])
        self.assertNotEqual((accepted, proposed), (60, 67))
        self.assertEqual(len(aligned['samples']), 128)
        aligned.update(within_arm_text_passed=True, capture_path=str(self.root / 'alternate-C0'), prior_reference={})
        report = A['compare'](aligned, copy.deepcopy(aligned))
        self.assertIsNone(report['first_output_difference_ordinal'])
        self.assertEqual(report['compared_through_ordinal'], 128)
        self.assertTrue(all(row['equal'] for row in report['comparisons']))
        self.assertFalse(report['timing_permitted'])
        self.assertFalse(report['promotion_permitted'])
        for path in ('run-trace.py', 'analyze-trace.py'):
            self.assertNotIn('BOLT_MOE_OPERAND', Path(path).read_text())
            self.assertNotIn('operand_analysis', Path(path).read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
