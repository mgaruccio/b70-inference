#!/usr/bin/env python3
"""Focused ownership/argument tests; no model or GPU dependencies."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bolt_cell', Path(__file__).with_name('run-bolt-cell-ssm-prefill-r1.py'))
cell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell)


class BoltCell(unittest.TestCase):
    def test_prefill_gate_keeps_baseline_and_full_array_checks(self):
        gate = cell.CANDIDATES[cell.PREFILL_SSM]
        self.assertEqual(gate.count("grep -Fx '  56/56 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  20/20 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  82/82 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  34/34 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  1/1 tests passed'"), 1)
        self.assertEqual(gate.count('test "$count" -eq 34'), 4)
        self.assertEqual(gate.count('test "$count" -eq 68'), 2)
        self.assertIn('test "$count" -eq 23', gate)
        self.assertEqual(gate.count('BOLT_SSM_PREFILL_MISMATCH'), 3)
        self.assertIn('GGML_SYCL_ENABLE_VMM=0', gate)
        self.assertTrue(gate.endswith('echo BOLT_SSM_PREFILL_NUMERICAL_PASSED; '))

    def test_fixed_candidate_native_gate(self):
        gate = cell.CANDIDATES[cell.SSM]
        self.assertEqual(gate.count("grep -Fx '  56/56 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  26/26 tests passed'"), 2)
        self.assertIn('test "$count" -eq 0', gate)
        self.assertIn('test "$count" -eq 11', gate)
        self.assertTrue(gate.endswith('echo SSM_CACHE_NUMERICAL_PASSED; '))

    def test_fixed_moe_native_gate(self):
        gate = cell.CANDIDATES[cell.MOE]
        self.assertEqual(gate.count("grep -Fx '  56/56 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  20/20 tests passed'"), 2)
        self.assertEqual(gate.count('MOE_REDUCE_9 full-array: mismatches=0 max_abs=0 '), 2)
        self.assertEqual(gate.count('test "$count" -eq 20'), 2)
        self.assertIn('test "$count" -eq 0', gate)
        self.assertIn('test "$count" -eq 8', gate)
        self.assertTrue(gate.endswith('echo MOE_REDUCE_NUMERICAL_PASSED; '))

    def test_combined_gate_keeps_both_suites_and_one_legacy(self):
        gate = cell.CANDIDATES[cell.COMBINED]
        self.assertEqual(gate.count("grep -Fx '  56/56 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  26/26 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  20/20 tests passed'"), 2)
        self.assertIn('echo SSM_CACHE_NUMERICAL_PASSED', gate)
        self.assertTrue(gate.endswith('echo MOE_REDUCE_NUMERICAL_PASSED; '))

    def test_fixed_post_native_gate(self):
        gate = cell.CANDIDATES[cell.POST]
        self.assertEqual(gate.count("grep -Fx '  56/56 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  37/37 tests passed'"), 2)
        self.assertEqual(gate.count('MAMBA_POST full-array: mismatches=0 max_abs=0 '), 2)
        self.assertEqual(gate.count('test "$count" -eq 37'), 2)
        self.assertIn('test "$count" -eq 0', gate)
        self.assertIn('test "$count" -eq 16', gate)
        self.assertTrue(gate.endswith('echo MAMBA_POST_NUMERICAL_PASSED; '))

    def test_moe_post_gate_keeps_both_suites_and_one_legacy(self):
        gate = cell.CANDIDATES[cell.MOE_POST]
        self.assertEqual(gate.count("grep -Fx '  56/56 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  20/20 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  37/37 tests passed'"), 2)
        self.assertIn('echo MOE_REDUCE_NUMERICAL_PASSED', gate)
        self.assertTrue(gate.endswith('echo MAMBA_POST_NUMERICAL_PASSED; '))

    def test_router_gate_keeps_moe_and_exact_router_checks(self):
        gate = cell.CANDIDATES[cell.ROUTER]
        self.assertEqual(gate.count("grep -Fx '  56/56 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  20/20 tests passed'"), 2)
        self.assertEqual(gate.count("grep -Fx '  82/82 tests passed'"), 2)
        self.assertEqual(gate.count('test "$count" -eq 82'), 4)
        self.assertEqual(gate.count('test "$count" -eq 5'), 2)
        self.assertEqual(gate.count('BOLT_ROUTER consumer bytes: mismatches=0 '), 2)
        self.assertEqual(gate.count('test "$count" -eq 2;'), 2)
        self.assertIn('test "$count" -eq 67', gate)
        self.assertTrue(gate.endswith('echo BOLT_ROUTER_NUMERICAL_PASSED; '))

    def test_empty_gpu_needs_no_stop(self):
        with patch.object(cell, 'command', side_effect=['', '']) as cmd:
            cell.replace_owned_container()
        self.assertEqual([c.args[:2] for c in cmd.call_args_list], [('docker', 'ps'), ('docker', 'ps')])

    def test_foreign_gpu_owner_is_not_touched(self):
        with patch.object(cell, 'command', return_value='foreign') as cmd:
            with self.assertRaises(AssertionError):
                cell.replace_owned_container()
        self.assertEqual(cmd.call_count, 1)

    def test_same_name_without_label_is_not_touched(self):
        with patch.object(cell, 'command', side_effect=[cell.NAME, cell.NAME, 'wrong']) as cmd:
            with self.assertRaises(AssertionError):
                cell.replace_owned_container()
        self.assertFalse(any(c.args[:2] == ('docker', 'rm') for c in cmd.call_args_list))

    def test_owned_bolt_removed_before_next_create(self):
        with patch.object(cell, 'command', side_effect=[cell.NAME, cell.NAME, 'bolt-fusion', cell.NAME, '']) as cmd:
            cell.replace_owned_container()
        self.assertEqual(cmd.call_args_list[3].args, ('docker', 'rm', '-f', cell.NAME))
        self.assertTrue(all(c.args[1] not in ('run', 'start') for c in cmd.call_args_list))

    def test_failed_removal_blocks_next_create(self):
        with patch.object(cell, 'command', side_effect=[cell.NAME, cell.NAME, 'bolt-fusion', '', cell.NAME]):
            with self.assertRaises(AssertionError):
                cell.replace_owned_container()

    def test_debug_measure_rejected_before_host_access(self):
        with patch.object(sys, 'argv', ['run-bolt-cell.py', '--name', 'bad', '--measure', '--debug']), \
             patch.object(cell, 'hardware') as hardware:
            with self.assertRaisesRegex(AssertionError, 'Debug traces'):
                cell.main()
            hardware.assert_not_called()

    def test_unknown_candidate_rejected_before_host_access(self):
        with patch.object(sys, 'argv', ['run-bolt-cell.py', '--name', 'bad', '--source', '/tmp/unapproved']), \
             patch.object(cell, 'hardware') as hardware:
            with self.assertRaisesRegex(AssertionError, 'Unregistered'):
                cell.main()
            hardware.assert_not_called()


if __name__ == '__main__':
    unittest.main()
