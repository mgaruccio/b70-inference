"""Executed by prepare.py on inference-host; adapts the 70-test computational suite.
The old source fixtures and observer suites remain read-only.
"""
tests = (BASE / 'test-trace.py').read_text()
tests = once(tests, "A = runpy.run_path('analyze-trace.py')", "A = runpy.run_path('analyze-trace.py')\nPARENT_TOP = M['PARENT_RESULTS']\nPARENT_ANALYSIS = runpy.run_path(str(PARENT_TOP / 'analyze-trace.py'))")
tests = tests.replace('router-single2.patch', 'router-widths-single2.patch')
# Generic route metadata without changing inherited non-router fixture generators.
tests = change_function(tests, 'router_lines', lambda s: s.replace(
    "native=False):", "native=False):\n    m = M['ROUTER_NATIVE_WIDTHS'][weight]").replace(
    'ne1=240', 'ne1=m').replace('nb2=3932160, nb3=3932160', 'nb2=16384*m, nb3=16384*m').replace(
    'dst_ne0=240', 'dst_ne0=m').replace('dst_nb1=960, dst_nb2=960*n, dst_nb3=960*n', 'dst_nb1=4*m, dst_nb2=4*m*n, dst_nb3=4*m*n').replace(
    "('LIBRARY_MKL_F32', 10)", "('LIBRARY_DNNL_F32' if n == 46 else 'LIBRARY_MKL_F32', 10)"))
tests = change_function(tests, 'router_native_output', lambda s: s.replace(
    'm=240,n={n}', "m={M['ROUTER_NATIVE_WIDTHS'][weight]},n={n}").replace(
    "f'm={M['ROUTER_NATIVE_WIDTHS'][weight]},n={n},k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],'",
    'f"m={M[\'ROUTER_NATIVE_WIDTHS\'][weight]},n={n},k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],"').replace('6/6 tests passed', '30/30 tests passed'))
tests = replace_function(tests, 'router_candidate_log', '''def router_candidate_log(log):
    """CPU fixture ONLY: synthesize four new selectors on real parent FIFO frames.
    The parent has no four-width records. Preserve all existing records and VERIFY
    history; insert after block7's last route even in prefills without output.weight.
    This is not runtime evidence or an input to the production analyzer.
    """
    frames = PARENT_ANALYSIS['route_frames'](log)
    contexts = {f['target_backend_ctx'] for f in frames}
    assert len(contexts) == 1
    target = contexts.pop()
    extension = {w: m for w, m in A['ROUTER_WIDTHS'].items() if w != 'blk.1.ffn_gate_inp.weight'}
    lines, inserted = [], 0
    for line in log.splitlines():
        lines.append(line)
        if 'BOLT_ROUTE ' not in line:
            continue
        r = A['route_records'](line)[0]
        if r['backend_ctx'] != target:
            continue
        assert r['weight'] not in extension, 'fixture applied twice or wrong reference'
        if r['weight'] == 'blk.7.attn_output.weight' and r['route'] != 'MUL_MAT_SYCL':
            n = A['_route_original_width'](r)
            for weight in extension:
                lines.extend(router_lines(n, weight=weight, backend=target))
            inserted += 1
    assert inserted == len(frames), (inserted, len(frames))
    return '\\n'.join(lines) + '\\n'
''')
tests = change_function(tests, 'router_model_log', lambda s: once(s,
    "            lines.append('BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in route.items()))",
    "            if route['weight'] == 'output.weight':\n"
    "                for weight in A['ROUTER_WIDTHS']:\n"
    "                    if weight != 'blk.1.ffn_gate_inp.weight':\n"
    "                        lines.extend(router_lines(batch['n'], weight=weight))\n"
    "            lines.append('BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in route.items()))"))
# Only real reference reads move to the asymmetric mapping. No old result is edited.
tests = tests.replace("M['PARENT_RESULTS'] / 'run-r1' / label", "A['parent_reference'](label)[0]")
tests = tests.replace("M['ROOT'] / 'run-r1/S0/diagnostic-window.log'", "PARENT_TOP / 'run-r1/S0/diagnostic-window.log'")
tests = tests.replace("source = M['ROOT'] / 'run-r1/S0'", "source = PARENT_TOP / 'run-r1/S0'")
tests = tests.replace("parent_cell = parent_results / 'run-r1' / label", "parent_cell = parent_results / ('run-r1/S0' if label == 'S0' else 'continuation-c0/run-r1/C0')")
tests = change_function(tests, 'test_parameters_mounts_frozen_payload_and_order', lambda s: once(s,
    "'bolt-lilicorr-router-single2-r1-'", "'bolt-lilicorr-router-widths-single2-r1-'"))
tests = change_function(tests, 'test_route_gate_with_synthetic_runtime_metadata', lambda s: once(s,
    "json.loads((cell / 'alignment.json').read_text())", "json.loads(A['parent_reference'](label)[1].read_text())"))
# Fake reference alignments follow the same asymmetric map, including immutable failed S0.
tests = tests.replace("(parent_cell / 'alignment.json')", "reference_alignment(parent_cell)")
tests = once(tests, '\ndef cpp_body(', '''
def reference_alignment(cell):
    return cell.parent.parent / 'reanalysis-S0.json' if cell.name == 'S0' else cell / 'alignment.json'


def cpp_body(''')
tests = once(tests, '    def patched_sources(self):', '''    def saved_cell(self, label, suffix='candidate'):
        source = A['parent_reference'](label)[0]
        cell = self.root / (suffix + '-' + label)
        cell.mkdir()
        for name in ('ordinary.result.json', 'warmup.result.json', 'diagnostic.response.json',
                     'diagnostic.request.json', 'ordinary.request.json', 'warmup.request.json'):
            shutil.copy2(source / name, cell / name)
        shutil.copytree(source / 'capture', cell / 'capture')
        (cell / 'diagnostic-window.log').write_text(router_candidate_log((source / 'diagnostic-window.log').read_text()))
        return cell

    def patched_sources(self):''')
tests = replace_function(tests, 'test_syntax_and_patch_scope', '''    def test_syntax_and_patch_scope(self):
        for path in Path('.').glob('*.py'):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                names = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
                self.assertEqual(len(names), len(set(names)), (path, names))
        subprocess.run(['bash', '-n', 'build-candidate.sh'], check=True)
        patchfile = Path('router-widths-single2.patch').resolve()
        self.assertEqual([s[6:] for s in patchfile.read_text().splitlines() if s.startswith('+++ b/')], M['SOURCE_SCOPE'])
        source, native = self.patched_sources()
        expected = json.loads(Path('source-proof.json').read_text())['candidate_source']
        import hashlib
        for name, text in zip(M['SOURCE_SCOPE'], (source, native)):
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), expected[name])
        directory = self.root / 'patched-source'
        subprocess.run(['git', 'apply', '--check', '--reverse', str(patchfile)], cwd=directory, check=True)
        subprocess.run(['patch', '--batch', '--reverse', '--fuzz=0', '-p1'], cwd=directory,
                       input=patchfile.read_bytes(), check=True, capture_output=True)
        for name in M['SOURCE_SCOPE']:
            self.assertEqual((directory / name).read_bytes(), (M['PARENT'] / name).read_bytes())
        current = ast.parse(Path('analyze-trace.py').read_text())
        parent = ast.parse((PARENT_TOP / 'analyze-trace.py').read_text())
        for name in ('_verify_frames', '_route_stream', '_route_original_width', '_target_group_width',
                     'assert_frozen_prefill_batches', 'route_frames', '_assert_target_frame_width', 'select_route_batches'):
            a = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b), name)
''')
tests = change_function(tests, 'test_capture_reserve_and_parent_results', lambda s: s.replace(
    "M['PARENT_RESULTS'].parts[-2:], ('20261017-nemotron35-bolt-lilicorr-iq2-single2', 'pair-r2')",
    "M['PARENT_RESULTS'].name, '20261017-nemotron35-bolt-lilicorr-router-single2'"))
tests = replace_function(tests, 'test_analyze_arm_saved_parent_capture_wiring', '''    def test_analyze_arm_saved_parent_capture_wiring(self):
        for label in ('S0', 'C0'):
            cell = self.saved_cell(label)
            result = A['analyze_arm'](cell, label, OLD)
            controller = ast.parse(Path('run-trace.py').read_text())
            consumed = {n.slice.value for n in ast.walk(controller)
                        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                        and n.value.id == 'checked' and isinstance(n.slice, ast.Constant)
                        and isinstance(n.slice.value, str)}
            self.assertTrue(consumed <= result.keys())
            self.assertEqual(result['s0_parent_invariant_passed'], label == 'S0')
            self.assertEqual(result['c0_parent_report_only'], label == 'C0')
            self.assertEqual(result['parent_comparison']['raw_vector_count'], 128)
            self.assertFalse(result['parent_comparison']['reference_equivalence_required'])
''')
tests = replace_function(tests, 'test_analyze_arm_parent_gates_through_full_wiring', '''    def test_analyze_arm_parent_gates_through_full_wiring(self):
        for label in ('S0', 'C0'):
            cell = self.saved_cell(label)
            ordinary = json.loads((cell / 'ordinary.result.json').read_text())
            diagnostic = json.loads((cell / 'diagnostic.response.json').read_text())
            real = A['analyze_arm'](cell, label, OLD)
            report = copy.deepcopy(real['parent_comparison'])
            for key in ('prompt_ids_match', 'emitted_ids_match', 'ordinary_text_matches_parent',
                        'diagnostic_text_matches_parent', 'diagnostic_token_ids_match_parent',
                        'actual_batch_inputs_match', 'selected_positions_rows_match'):
                broken = copy.deepcopy(report)
                broken[key] = False
                with patch.dict(A['analyze_arm'].__globals__, report_parent_comparison=lambda *args: broken):
                    if label == 'S0':
                        self.assertRaises(AssertionError, A['analyze_arm'], cell, label, OLD)
                    else:
                        # Report-only C0 does not enforce equality (real report still requires same prompt).
                        self.assertTrue(A['analyze_arm'](cell, label, OLD)['c0_parent_report_only'])
            changed = copy.deepcopy(real)
            changed['prompt_ids'][0] ^= 1
            self.assertRaisesRegex(AssertionError, 'prompt IDs differ', A['report_parent_comparison'], changed, label, ordinary, diagnostic)
''')
tests = replace_function(tests, 'test_iq2_model_two_views_through_real_analyze_arm', '''    def test_iq2_model_two_views_through_real_analyze_arm(self):
        for label in ('S0', 'C0'):
            cell = self.saved_cell(label)
            result = A['analyze_arm'](cell, label, OLD)
            log = (cell / 'diagnostic-window.log').read_text()
            line = next(line for line in reversed(log.splitlines(True))
                        if 'weight=blk.1.ffn_up_exps.weight ' in line and 'route=MOE_SINGLE_MMVQ_PLAIN_Q8_1' in line)
            bad = log.replace(line, line.replace('src1_ne2=1 ', 'src1_ne2=2 '), 1)
            (cell / 'diagnostic-window.log').write_text(bad)
            self.assertRaises(AssertionError, A['analyze_arm'], cell, label, OLD)
''')
# Keep the historical FIFO-shift regression, synthesizing only the required new routes.
tests = change_function(tests, 'test_saved_real_router_fifo_order_and_position70_shift', lambda s: s.replace(
    'log = log_path.read_text()', 'log = router_candidate_log(log_path.read_text())').replace(
    "['routes']), 15)", "['routes']), 23)"))
tests = change_function(tests, 'test_router_fifo_rejects_dropped_extra_bad_width_and_reordered_groups', lambda s: s.replace(
    "log = (PARENT_TOP / 'run-r1/S0/diagnostic-window.log').read_text()",
    "log = router_candidate_log((PARENT_TOP / 'run-r1/S0/diagnostic-window.log').read_text())"))
tests = replace_function(tests, 'test_missing_c0_continuation_preserves_saved_s0_and_controller_shape', '''    def test_asymmetric_parent_mapping_preserves_saved_failed_s0(self):
        self.assertEqual(A['parent_reference']('S0'), (PARENT_TOP / 'run-r1/S0', PARENT_TOP / 'reanalysis-S0.json'))
        cell, alignment = A['parent_reference']('C0')
        self.assertEqual(cell, PARENT_TOP / 'continuation-c0/run-r1/C0')
        self.assertEqual(alignment, cell / 'alignment.json')
        self.assertEqual(json.loads((PARENT_TOP / 'run-r1/summary.json').read_text())['status'], 'failed')
        self.assertEqual(M['ORDER'], ('S0', 'C0'))
''')
tests = replace_function(tests, 'test_original_controller_remains_full_pair', '''    def test_original_controller_remains_full_pair(self):
        self.assertEqual(M['ORDER'], ('S0', 'C0'))
        self.assertIn('BOLT_VERIFY_PAIR_COMPLETED_NO_TIMING', Path('run-trace.py').read_text())
        for label in ('S0', 'C0'):
            cell = self.saved_cell(label)
            result = A['analyze_arm'](cell, label, OLD)
            self.assertEqual(result['s0_parent_invariant_passed'], label == 'S0')
            self.assertEqual(result['c0_parent_report_only'], label == 'C0')
            log = (cell / 'diagnostic-window.log').read_text()
            last = next(line for line in reversed(log.splitlines(True)) if 'weight=blk.1.ffn_gate_inp.weight ' in line)
            (cell / 'diagnostic-window.log').write_text(log[:log.rindex(last)] + log[log.rindex(last)+len(last):])
            self.assertRaises(AssertionError, A['analyze_arm'], cell, label, OLD)
''')
tests = replace_function(tests, 'test_router_patch_only_adds_control_and_isolated_numerical_fixtures', '''    def test_router_patch_only_extends_guard_selectors_and_same_native_family(self):
        source, native = self.patched_sources()
        parent = (M['PARENT'] / M['SOURCE_SCOPE'][0]).read_text()
        for name in ('ggml_sycl_router_single2_range', 'ggml_sycl_mul_mat', 'ggml_sycl_op_mul_mat',
                     'bolt_route_record', 'ggml_sycl_compute_forward', 'ggml_sycl_mul_mat_id'):
            self.assertEqual(cpp_body(source, name), cpp_body(parent, name), name)
        body = cpp_body(source, 'ggml_sycl_router_single2_views')
        before = cpp_body(parent, 'ggml_sycl_router_single2_views')
        # Everything after the first guard is byte-identical: suffix, ownership,
        # layout, range, aliases, preflight of BOTH columns, metadata and returns.
        self.assertEqual(body.split('    constexpr char suffix[]', 1)[1], before.split('    constexpr char suffix[]', 1)[1])
        for pure in ('ggml_sycl_router_single2_range', 'ggml_sycl_router_single2_views'):
            for effect in ('ctx.stream(', 'ctx.pool(', 'opt_for_reorder(', 'malloc(', 'submit(', 'gemm(', 'ggml_sycl_mul_mat('):
                self.assertNotIn(effect, cpp_body(source, pure))
        parent_native = (M['PARENT'] / M['SOURCE_SCOPE'][1]).read_text()
        start, end = '// Six isolated F32 CPU-reference cases', '// GGML_HINT_SRC0_IS_HADAMARD'
        fixture = native.split(start, 1)[1].split(end, 1)[0]
        parent_fixture = parent_native.split(start, 1)[1].split(end, 1)[0]
        self.assertEqual(fixture.replace('int layout = 0, int64_t m = 240', 'int layout = 0').replace(
            'GGML_TYPE_F32, GGML_TYPE_F32, m, n, 4096', 'GGML_TYPE_F32, GGML_TYPE_F32, 240, n, 4096'), parent_fixture)
        self.assertIn('double max_nmse_err() override { return 1e-7; }', fixture)
        self.assertIn('test_case::initialize_tensors(ctx);', fixture)
        self.assertEqual(len(M['ROUTER_NATIVE_CASES']), 30)
        inherited = parent_native.split('// Exactly six router cases,', 1)[1].split('// Experimental NVFP4 reorder', 1)[0].strip()
        self.assertIn(inherited, native)
''')
# Increase only router-family exact counts. Other native-six fixtures remain unchanged.
tests = change_function(tests, 'test_router_native_numeric_and_actual_operation_gates', lambda s: s.replace(
    'len(native), 14', 'len(native), 70').replace('# seven operations, not fourteen', '# 35 operations, two records each').replace(
    '6/6 tests passed', '30/30 tests passed').replace('5/6 tests passed', '29/30 tests passed'))
# Compile actual patched helper/dispatcher, not a parallel eligibility implementation.
def expand_spy(s):
    s = once(s, '    ggml_backend_sycl_context ctx{0}; expected_ctx = &ctx;',
        '    int total_rejected = 0;\n    for (int m : {240,304,368,432,496}) {\n    calls.clear();\n    ggml_backend_sycl_context ctx{0}; expected_ctx = &ctx;')
    s = s.replace('4096*240+32', '4096*m+32').replace('dv(240*3+32', 'dv(m*3+32')
    s = s.replace('tensor(a,4096,240,', 'tensor(a,4096,m,').replace('tensor(d,240,2,', 'tensor(d,m,2,')
    s = s.replace('(char *)d.data+960', '(char *)d.data+4*m').replace('r<240', 'r<m').replace('16+c*240+r', '16+c*m+r')
    s = s.replace('dv.begin()+256,dv.begin()+256', 'dv.begin()+16+m,dv.begin()+16+m').replace('size_t i=496', 'size_t i=16+2*m')
    s = once(s, '    reset(); ggml_tensor_extra_gpu plain{{false}};', '''    assert(rejected == 85); // retain every inherited exclusion, including late preflight
    for (int unsupported : {0,1,239,241,303,305,367,369,431,433,495,497,512}) {
        reject([&] { tensor(a,4096,unsupported,av.data()+16,&ab); tensor(d,unsupported,2,dv.data()+16,&db); });
    }
    for (int mismatch : {240,304,368,432,496}) if (mismatch != m) {
        reject([&] { tensor(d,mismatch,2,dv.data()+16,&db); });
    }
    total_rejected += rejected;
    reset(); ggml_tensor_extra_gpu plain{{false}};''')
    s = once(s, '    printf("router preflight: %d exclusions;', '    printf("router width=%d preflight: %d exclusions;')
    s = once(s, 'OK\\n", rejected);\n}', 'OK\\n", m, rejected);\n    }\n    assert(total_rejected == 510);\n    printf("all five widths: 510 exclusions OK\\n");\n}')
    return s

tests = change_function(tests, 'test_router_compiled_actual_preflight_offsets_metadata_and_no_partial_launch', expand_spy)
ast.parse(tests)
save('test-trace.py', tests)
# The separately readable extension is also part of the full CPU command.
save('test-widths.py', Path(__file__).with_name('test-widths.py').read_text())
original_methods = [n.name for n in ast.walk(ast.parse((BASE / 'test-trace.py').read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
methods = [n.name for n in ast.walk(ast.parse(tests)) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
assert len(methods) == len(original_methods) == 70
print('RETAINED_70_COMPUTATIONAL_TESTS; 85 exclusions per width + 17 new negatives = 510 total', flush=True)
