#!/usr/bin/env python3
"""Remote CPU-only controller/analyzer contract tests; no Docker or inference."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import socket
import tempfile
import unittest


assert socket.gethostname() == 'inference-host'
M = runpy.run_path('run-controls.py')
A = runpy.run_path('analyze-controls.py')
BASE = json.loads((M['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
IMAGE = 'intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04'
PAYLOAD = json.loads((M['PROFILE'] / 'run-r1/baseline/bench/measured-00.request.json').read_text())
DIAGNOSTIC = json.loads((M['ROOT'].parent / '20261017-nemotron35-bolt-fidelity-controls' /
                         'run-r1/A0/diagnostic.response.json').read_text())
DETERMINISTIC = M['ROOT'].parent / '20261017-nemotron35-bolt-lilicorr-deterministic'
ACTIVE_C = json.loads((DETERMINISTIC / 'run-r1/C1/warmup.result.json').read_text())
DFLASH_STARTUP_SNIPPET = (
    "I llama_context: n_rs_seq              = 4\n"
    "I common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'\n"
    "I common_speculative_impl_draft_dflash: - n_max=4, n_min=0, p_min=0.00\n"
    "I common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=990, n_extract=6, sample_from_anchor=true\n"
    "I srv    load_model: speculative decoding context initialized\n"
    "I llama_context: n_rs_seq              = 0\n"
)


class Controls(unittest.TestCase):
    def test_syntax(self):
        for name in ('run-controls.py', 'analyze-controls.py', 'test-controls.py'):
            ast.parse(Path(name).read_text())

    def test_order_and_arm_namespace(self):
        self.assertEqual(M['ORDER'], ('A0', 'B0', 'C0', 'C1', 'B1', 'A1'))
        self.assertEqual([label[0] for label in M['ORDER']], list('ABCCBA'))
        self.assertEqual([label[1] for label in M['ORDER']], list('000111'))
        self.assertEqual(M['PREFIX'], 'bolt-lilicorr-suppressed-r1-')
        self.assertEqual(M['SERVER_MARKER'], 'BOLT_LILICORR_SUPPRESSED_SERVER_START')

    def test_all_names_unique_and_base_unchanged(self):
        before = copy.deepcopy(BASE)
        argv = [M['make_create'](BASE, label, IMAGE) for label in M['ORDER']]
        names = [row[row.index('--name') + 1] for row in argv]
        self.assertEqual(len(set(names)), 6)
        self.assertEqual(BASE[BASE.index('--name') + 1], 'nemotron-bolt-probe')
        self.assertEqual(BASE, before)

    def test_exact_sources_mounts_environment_and_command(self):
        base_server = BASE[-1].rsplit('exec ', 1)[1]
        for label in M['ORDER']:
            argv = M['make_create'](BASE, label, IMAGE)
            source = M['BASELINE'] if label[0] == 'A' else M['CANDIDATE']
            self.assertIn(str(source) + ':/src:ro', argv)
            self.assertNotIn(str(M['ACCEPTED']) + ':/src:ro', argv)
            for key, value in M['SERVER_ENV']:
                self.assertEqual(argv.count(f'{key}={value}'), 1)
            self.assertIn('unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; echo ' + M['SERVER_MARKER'] + '; exec ',
                          argv[-1])
            self.assertNotIn('GGML_SYCL_DEBUG=1', argv[:-1])
            server = argv[-1].rsplit('exec ', 1)[1]
            if label[0] in ('B', 'C'):
                expected_server = base_server + ' ' + shlex.join(M['spec_for_arm'](label[0]))
            else:
                expected_server = base_server
            self.assertEqual(server, expected_server)
            if label[0] == 'A':
                self.assertNotIn(':/native-outprod.sh:ro', ' '.join(argv))
                self.assertNotIn(':/draft:ro', ' '.join(argv))
            else:
                self.assertIn(str(M['PRIOR'] / 'native-outprod.sh') + ':/native-outprod.sh:ro', argv)
                self.assertIn('bash /native-outprod.sh;', argv[-1])
            if label[0] in ('B', 'C'):
                self.assertIn(str(M['DRAFT'] / 'draft') + ':/draft:ro', argv)
            else:
                self.assertNotIn(':/draft:ro', ' '.join(argv))

    def test_spec_and_accepted_baseline_receipts(self):
        self.assertEqual(M['spec_for_arm']('C'), [
            '--spec-type', 'draft-dflash', '--spec-draft-n-max', '4', '--spec-draft-n-min', '0',
            '--spec-draft-p-min', '0', '--spec-draft-sampling', 'greedy',
            '--no-spec-draft-backend-sampling', '--spec-draft-ngl', '999',
            '--spec-draft-model', '/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf',
        ])
        self.assertEqual(M['spec_for_arm']('B')[:5], M['spec_for_arm']('C')[:5])
        self.assertEqual(M['spec_for_arm']('B')[4], '--spec-draft-n-min')
        self.assertEqual(M['spec_for_arm']('B')[5], '5')
        self.assertEqual(M['spec_for_arm']('C')[5], '0')
        self.assertEqual(M['BASELINE_EXPECTED']['libggml-sycl.so.0'],
                         '95733d9042844ad751e82b78d8dccb6326605afdd7ddf0f9c5bb2bc40e82bf58')
        self.assertEqual(M['BASELINE_EXPECTED']['libllama-server-impl.so'],
                         'ae1d0d772b97676ef711605cb52adfcd2529517bfed43671b6e8496482cf566b')
        self.assertEqual(M['ACCEPTED_IMPL'],
                         '67ae7c8673110b52d11924677ad662141dc93f57d7c21d67a3f0d3e4489e09ad')

    def test_four_artifact_hash_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'build-sycl/bin'
            root.mkdir(parents=True)
            expected = {}
            for name in M['ARTIFACTS']:
                data = ('fixture-' + name).encode()
                (root / name).write_bytes(data)
                expected[name] = hashlib.sha256(data).hexdigest()
            self.assertEqual(M['binaries'](root.parents[1]), expected)
            self.assertEqual(set(expected), {
                'llama-server', 'libggml-sycl.so.0', 'test-backend-ops', 'libllama-server-impl.so'})

    def test_diagnostic_payload_and_saved_token_logprob_fixture(self):
        before = copy.deepcopy(PAYLOAD)
        diagnostic = M['diagnostic_payload'](PAYLOAD)
        changed = {key for key in PAYLOAD.keys() | diagnostic.keys()
                   if PAYLOAD.get(key) != diagnostic.get(key)}
        self.assertEqual(changed, {'stream', 'stream_options', 'verbose', 'return_tokens',
                                   'logprobs', 'top_logprobs', 'backend_sampling'})
        self.assertEqual(PAYLOAD, before)
        self.assertFalse(diagnostic['stream'])
        self.assertFalse(diagnostic['backend_sampling'])
        self.assertEqual(M['check_diagnostic_result'](DIAGNOSTIC), DIAGNOSTIC['__verbose']['tokens'])

    def test_stream_count_cache_finish_contract_fixture(self):
        result = json.loads((M['ROOT'].parent / '20261017-nemotron35-bolt-fidelity-controls' /
                             'run-r1/A0/repeat-00.result.json').read_text())
        self.assertEqual(len(M['check_stream_result'](result)), len(result['text']))
        broken = copy.deepcopy(result)
        broken['timings']['cache_n'] = 1
        with self.assertRaises(AssertionError):
            M['check_stream_result'](broken)

    def test_marker_and_debug_contract(self):
        M['check_model_log']('GGML_SYCL_ENABLE_DNN: 1\n', 'A')
        M['check_model_log']("GGML_SYCL_ENABLE_DNN: 1\nadding speculative implementation 'draft-dflash'\n", 'B')
        M['check_model_log']("GGML_SYCL_ENABLE_DNN: 1\nadding speculative implementation 'draft-dflash'\n", 'C')
        with self.assertRaises(AssertionError):
            M['check_model_log']('GGML_SYCL_ENABLE_DNN: 1\ncommon_debug_cb_eval: x\n', 'A')
        with self.assertRaises(AssertionError):
            M['check_model_log']('GGML_SYCL_ENABLE_DNN: 0\n', 'A')

    def test_startup_spec_guards(self):
        baseline_log = 'I llama_context: n_rs_seq              = 0\n'
        self.assertEqual(M['check_startup_spec'](baseline_log, 'A'), {'target_n_rs_seq': 0})
        active = DFLASH_STARTUP_SNIPPET
        self.assertEqual(M['check_startup_spec'](active, 'C')['n_min'], 0)
        suppressed = active.replace('n_min=0', 'n_min=5')
        self.assertEqual(M['check_startup_spec'](suppressed, 'B')['n_min'], 5)
        with self.assertRaises(AssertionError):
            M['check_startup_spec'](active.replace('n_rs_seq              = 4', 'n_rs_seq              = 0'), 'C')

    def test_native_gate_parser_preserves_native_and_outprod_contracts(self):
        native = ''.join(f'  {n}/{n} tests passed\n' for n in A['EXPECTED_NATIVE'] for _ in range(2))
        log = native + 'BOLT_SSM_PREFILL_NUMERICAL_PASSED\n' + M['SERVER_MARKER']
        self.assertEqual(A['parse_native_gate'](log, 'A0'), {'native385_passed': True})
        focused = ''.join(f'  {n}/{n} tests passed\n' for n, _ in A['EXPECTED_OUTPROD'])
        focused += 'not supported (expected)\n' * 6 + '(expected)\n' * 3
        log = native + 'BOLT_SSM_PREFILL_NUMERICAL_PASSED\n' + focused + \
            'BOLT_LILICORR_OUTPROD_NATIVE_PASSED\n' + M['SERVER_MARKER']
        self.assertEqual(A['parse_native_gate'](log, 'B0')['focused_numerical_passed'], 190)

    def test_speculation_window_b_zero_and_c_active(self):
        zero_counters = [{'accepted': 0, 'proposed': 0}] * 5
        b_diag = A['assess_speculation_window']('', 'B', zero_counters)
        self.assertTrue(b_diag['drafter_enabled'])
        self.assertTrue(b_diag['proposal_suppression'])
        self.assertEqual(b_diag['log_events'], 0)

        self.assertGreater(ACTIVE_C['timings']['draft_n'], 0)
        active_counters = [{'accepted': 0, 'proposed': 0},
                           {'accepted': 86, 'proposed': 83}] + zero_counters[2:]
        window = 'accepted  2/ 4 draft tokens\naccepted 84/ 79 draft tokens\n'
        c_diag = A['assess_speculation_window'](window, 'C', active_counters)
        self.assertTrue(c_diag['drafter_enabled'])
        self.assertFalse(c_diag['proposal_suppression'])
        self.assertGreater(c_diag['log_events'], 0)

        with self.assertRaises(AssertionError):
            A['assess_speculation_window'](window, 'B', active_counters)
        with self.assertRaises(AssertionError):
            A['assess_speculation_window']('', 'C', zero_counters)

    def test_fidelity_decision_uses_only_six_ordinary_responses(self):
        stable = {label: ['ordinary'] * 3 for label in M['ORDER']}
        decision = A['assess_fidelity'](stable)
        self.assertTrue(decision['fidelity_passed'])
        self.assertEqual(decision['within_arm']['A']['responses'], 6)
        self.assertEqual(decision['cross_arm']['AB']['pairs'], 36)

        changed = copy.deepcopy(stable)
        changed['B1'][0] = 'ordinary-B-different'
        rejected = A['assess_fidelity'](changed)
        self.assertFalse(rejected['fidelity_passed'])
        self.assertFalse(rejected['arm_stability']['B'])
        mismatches = [row for row in rejected['cross_arm']['AB']['comparisons'] if not row['equal']]
        self.assertTrue(mismatches)
        self.assertEqual(mismatches[0]['first_difference_index'], len('ordinary'))

    def test_diagnostic_comparisons_are_separate(self):
        diagnostics = {label: [1, 2, 3] for label in M['ORDER']}
        diagnostics['C1'] = [1, 2, 9]
        comparisons = A['diagnostic_token_comparisons'](diagnostics)
        self.assertTrue(any(row['left'] == 'C0' and row['right'] == 'C1' and
                            row['first_difference_token_ordinal'] == 3 for row in comparisons))
        self.assertNotIn('fidelity_passed', comparisons[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
