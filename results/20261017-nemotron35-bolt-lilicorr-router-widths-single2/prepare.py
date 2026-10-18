#!/usr/bin/env python3
"""Remote CPU-only preparation, confined to a fresh worker-scratch attempt.
No build, native executable, GPU, API, service, or accepted-runtime mutation.
"""
import ast
import difflib
import hashlib
import json
import re
import runpy
import socket
import subprocess
import sys
from pathlib import Path

assert socket.gethostname() == 'inference-host'
ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-widths-single2')
BASE = ROOT.with_name('20261017-nemotron35-bolt-lilicorr-router-single2')
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1')
assert len(sys.argv) == 2 and re.fullmatch(r'prepare-r[1-9][0-9]*', sys.argv[1])
OUT = ROOT / 'worker-scratch' / sys.argv[1]
OUT.mkdir()
SCOPE = ('ggml/src/ggml-sycl/ggml-sycl.cpp', 'tests/test-backend-ops.cpp')
SOURCE_HASHES = ('430c67f15a938601611cc5e73a3d25b07fb4fe35b688782833a875ebd522c920',
                 'f694ba3793ff6cd511fae0b3edfe92ef1037d5236a925ac940d484df5e2e5672')
BINARY_HASHES = dict(zip(('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
                         'test-backend-ops', 'libllama-common.so'), (
    '22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d',
    '43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede',
    'b2b3b7950c471bb5658f8474eb1866157fd3311a183e07d03a665842a8cad3d9',
    '5f43858b1cbd1083ced719a8700f93c30a0e68e94f307541f0bee9e43f5e8389',
    '934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812')))
REPS = {1: 240, 8: 304, 21: 432, 37: 368, 48: 496}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, before, after):
    assert text.count(before) == 1, (before[:160], text.count(before))
    return text.replace(before, after, 1)


def function(text, name):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(nodes) == 1, (name, len(nodes))
    node = nodes[0]
    return ''.join(text.splitlines(True)[node.lineno - 1:node.end_lineno])


def change_function(text, name, transform):
    before = function(text, name)
    return once(text, before, transform(before))


def replace_function(text, name, replacement):
    return change_function(text, name, lambda _: replacement.strip('\n') + '\n')


def save(name, text):
    path = OUT / name
    assert OUT in path.resolve().parents
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


for name, expected in zip(SCOPE, SOURCE_HASHES):
    assert sha(PARENT / name) == expected, name
for name, expected in BINARY_HASHES.items():
    assert sha(PARENT / 'build-sycl/bin' / name) == expected, name
print('PINNED_PARENT_SOURCE_AND_FIVE_BINARIES_OK', flush=True)
original = [(PARENT / name).read_text() for name in SCOPE]
sycl, native = original
sycl = once(sycl,
    'src0->ne[0] != 4096 || src0->ne[1] != 240 ||\n'
    '        src1->ne[0] != 4096 || src1->ne[1] != 2 || dst->ne[0] != 240 || dst->ne[1] != 2)',
    'src0->ne[0] != 4096 ||\n'
    '        (src0->ne[1] != 240 && src0->ne[1] != 304 && src0->ne[1] != 368 &&\n'
    '         src0->ne[1] != 432 && src0->ne[1] != 496) ||\n'
    '        src1->ne[0] != 4096 || src1->ne[1] != 2 || dst->ne[0] != src0->ne[1] || dst->ne[1] != 2)')
selector = '           std::strcmp(weight->name, "output.weight") == 0;'
sycl = once(sycl, selector, ''.join(
    f'           std::strcmp(weight->name, "blk.{block}.ffn_gate_inp.weight") == 0 ||\n'
    for block in REPS if block != 1) + selector)
native = once(native,
    'test_mul_mat_router_single2(const char * weight, int64_t n, int layout = 0) :\n'
    '        test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 240, n, 4096, {1, 1}, {1, 1}),',
    'test_mul_mat_router_single2(const char * weight, int64_t n, int layout = 0, int64_t m = 240) :\n'
    '        test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, m, n, 4096, {1, 1}, {1, 1}),')
# Preserve all six original 240 instantiations verbatim; extend the same family.
end = '    test_cases.emplace_back(new test_mul_mat_router_single2("blk.1.ffn_gate_inp.weight", 2, 2));\n'
extension = '\n    // Six identical profiles per additional actual-model width: 30 router cases total.\n'
for block, width in REPS.items():
    if block == 1:
        continue
    weight = f'blk.{block}.ffn_gate_inp.weight'
    extension += ('    for (int64_t n : {1, 2, 3}) {\n'
                  f'        test_cases.emplace_back(new test_mul_mat_router_single2("{weight}", n, 0, {width}));\n'
                  '    }\n'
                  f'    test_cases.emplace_back(new test_mul_mat_router_single2("blk.1.width{width}.unrelated.weight", 2, 0, {width}));\n'
                  f'    test_cases.emplace_back(new test_mul_mat_router_single2("{weight}", 2, 1, {width}));\n'
                  f'    test_cases.emplace_back(new test_mul_mat_router_single2("{weight}", 2, 2, {width}));\n')
native = once(native, end, end + extension)
candidate = (sycl, native)
patch_text = ''.join(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
    fromfile='a/' + name, tofile='b/' + name)) for name, before, after in zip(SCOPE, original, candidate))
save('router-widths-single2.patch', patch_text)
# Two independent strict readers plus actual forward/reverse content checks.
source = OUT / 'apply-source'
for name, text in zip(SCOPE, original):
    save('apply-source/' + name, text)
commands = [
    ['git', 'apply', '--check', '--verbose', str(OUT / 'router-widths-single2.patch')],
    ['patch', '--dry-run', '--batch', '--forward', '--fuzz=0', '-p1'],
    ['patch', '--batch', '--forward', '--fuzz=0', '-p1'],
]
for command in commands:
    print('COMMAND', command, flush=True)
    subprocess.run(command, cwd=source, input=patch_text.encode(), check=True)
assert [(source / n).read_text() for n in SCOPE] == list(candidate)
for command in (['git', 'apply', '--check', '--reverse', '--verbose', str(OUT / 'router-widths-single2.patch')],
                ['patch', '--batch', '--reverse', '--fuzz=0', '-p1']):
    print('COMMAND', command, flush=True)
    subprocess.run(command, cwd=source, input=patch_text.encode(), check=True)
assert [(source / n).read_text() for n in SCOPE] == original
print('FULL_PATCH_FORWARD_AND_REVERSE_EXACT_OK', flush=True)
save('source-proof.json', json.dumps(dict(parent_source=dict(zip(SCOPE, SOURCE_HASHES)),
    parent_binaries=BINARY_HASHES, candidate_source={n: hashlib.sha256(s.encode()).hexdigest() for n, s in zip(SCOPE, candidate)},
    patch_sha256=hashlib.sha256(patch_text.encode()).hexdigest(),
    apply_and_reverse_exact=True, readers=['git apply --check', 'patch --fuzz=0'],
    representatives=REPS, native_profiles_per_width=6, native_total=30,
    native_execution_pending=True, gpu_execution_pending=True), indent=2) + '\n')

run = (BASE / 'run-trace.py').read_text()
analysis = (BASE / 'analyze-trace.py').read_text()
build = (BASE / 'build-candidate.sh').read_text()
for filename, text in [('run-trace.py', run), ('analyze-trace.py', analysis), ('build-candidate.sh', build)]:
    # Original paths distinguish current parent from new candidate; never use an observer runtime.
    text = text.replace('20261017-nemotron35-bolt-lilicorr-router-single2', ROOT.name)
    text = text.replace('20261017-nemotron35-bolt-lilicorr-iq2-single2\' / \'pair-r2', BASE.name + "'")
    # The previous replacement includes the closing quote already; normalize assignment directly.
    if filename.endswith('.py'):
        text = re.sub(r'^PARENT_RESULTS = .*$', "PARENT_RESULTS = ROOT.parent / '" + BASE.name + "'", text, flags=re.M)
    text = text.replace('llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1',
                        'llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1')
    text = text.replace('llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-single2-r1', PARENT.name)
    text = text.replace('bolt-lilicorr-router-single2-r1-', 'bolt-lilicorr-router-widths-single2-r1-')
    text = text.replace('router-single2.patch', 'router-widths-single2.patch')
    if filename == 'run-trace.py':
        run = text
    elif filename == 'analyze-trace.py':
        analysis = text
    else:
        build = text
run = run.replace('959874f59e890957c1a4ec1488be1c322bdc7eeba675b7b2c9e0cdaa2ead66db', BINARY_HASHES['libggml-sycl.so.0'])
run = run.replace('c5592fd927626a468c1464cf4f42ee2de5894a0d55f6b1ded023809f5ffd8db9', BINARY_HASHES['test-backend-ops'])
widths = {f'blk.{block}.ffn_gate_inp.weight': width for block, width in REPS.items()}
width_decl = 'ROUTER_WIDTHS = ' + repr(widths) + '\n'
analysis = once(analysis, '\ndef load(path):', '\n' + width_decl + '\ndef load(path):')
run = once(run, 'ROUTER_NATIVE_CASES = {\n', width_decl +
    '# Unrelated numerical controls retain the already-selected blk.1 prefix.\n'
    'ROUTER_UNRELATED_WIDTHS = {("blk.1.unrelated.weight" if m == 240 else f"blk.1.width{m}.unrelated.weight"): m for m in ROUTER_WIDTHS.values()}\n'
    'ROUTER_NATIVE_WIDTHS = dict(ROUTER_WIDTHS, **ROUTER_UNRELATED_WIDTHS)\n'
    'ROUTER_NATIVE_CASES = {\n')
a = run.index('ROUTER_NATIVE_CASES = {\n')
b = run.index('ROUTER_NATIVE_MARKER', a)
run = run[:a] + '''ROUTER_NATIVE_CASES = {(weight, 0, n) for weight in ROUTER_WIDTHS for n in (1, 2, 3)} | {
    (weight, layout, 2) for weight in ROUTER_WIDTHS for layout in (1, 2)
} | {(weight, 0, 2) for weight in ROUTER_UNRELATED_WIDTHS}
assert len(ROUTER_NATIVE_CASES) == 30
''' + run[b:]
def native_gate(text):
    text = text.replace("[('6', '6')]", "[('30', '30')]")
    text = text.replace('m=240,n=([123])', 'm=(240|304|368|432|496),n=([123])')
    text = text.replace('for w, layout, n in cases', 'for w, layout, m, n in cases')
    text = text.replace('len(keys) == 6', 'len(keys) == 30')
    text = once(text, "    keys = ", "    assert all(ROUTER_NATIVE_WIDTHS.get(w) == int(m) for w, layout, m, n in cases)\n    keys = ")
    text = text.replace("(routes, n, split, layout)", "(routes, n, split, layout, m=ROUTER_NATIVE_WIDTHS[weight])")
    return text
run = change_function(run, 'check_router_native_output', native_gate)
run = once(run, '"-p \'^router_single2=1,\' 2>&1 | tee /tmp/router-single2-native.log; "',
    '"-p \'^router_single2=1,\' 2>&1 | tee /tmp/router-single2-native.log; "\n'
    '    "grep -Fqx \'  30/30 tests passed\' /tmp/router-single2-native.log; "\n'
    '    "count=$(grep -Fc \'  30/30 tests passed\' /tmp/router-single2-native.log || true); test \\"$count\\" -eq 1; "'.replace('\\"', '\\"'))
# Keep the legacy independent-FIFO parser unmodified. Only its exact route schema grows.
analysis = change_function(analysis, '_expected_target_weights', lambda s: once(s, '    if output:\n',
    "    for weight in ROUTER_WIDTHS:\n        if weight != 'blk.1.ffn_gate_inp.weight':\n            weights.extend([weight] * (2 * router_pairs))\n    if output:\n"))
analysis = change_function(analysis, '_assert_prefill_frame', lambda s: once(s,
    "    router = [route for route in routes if route['weight'] == 'blk.1.ffn_gate_inp.weight']\n"
    "    assert_router_operations(router, n, split=False,\n"
    "                             inner_route='LIBRARY_DNNL_F32' if n == 46 else 'LIBRARY_MKL_F32')",
    "    for weight, m in ROUTER_WIDTHS.items():\n"
    "        router = [route for route in routes if route['weight'] == weight]\n"
    "        assert_router_operations(router, n, split=False, m=m,\n"
    "                                 inner_route='LIBRARY_DNNL_F32' if n == 46 else 'LIBRARY_MKL_F32')"))
analysis = change_function(analysis, 'assert_router_operations', lambda s: s.replace(
    "inner_route='LIBRARY_MKL_F32'):", "inner_route='LIBRARY_MKL_F32', m=240):").replace(
    '    calls = ', '    assert m in (240, 304, 368, 432, 496)\n    calls = ').replace(
    '(4096, 240, 1, 1), (4, 16384, 3932160, 3932160)', '(4096, m, 1, 1), (4, 16384, 16384*m, 16384*m)').replace(
    '(240, view_n, 1, 1), (4, 960, 960*n, 960*n)', '(m, view_n, 1, 1), (4, 4*m, 4*m*n, 4*m*n)'))
analysis = change_function(analysis, 'assert_router_model_routes', lambda s: once(s,
    "        routers = [route for route in frame['routes']\n"
    "                   if route['weight'] == 'blk.1.ffn_gate_inp.weight']\n"
    "        assert routers, 'missing actual target router route'\n"
    "        assert_router_operations(routers, n, split=(label == 'C0' and n == 2))",
    "        routers = []\n"
    "        for weight, m in ROUTER_WIDTHS.items():\n"
    "            group = [route for route in frame['routes'] if route['weight'] == weight]\n"
    "            assert group, ('missing actual representative router route', weight)\n"
    "            assert_router_operations(group, n, split=(label == 'C0' and n == 2), m=m)\n"
    "            routers.extend(group)"))
parent_ref = '''def parent_reference(label):
    # Preserve the failed S0 cell: use its separately validated alignment.
    assert label in ('S0', 'C0')
    if label == 'S0':
        return PARENT_RESULTS / 'run-r1/S0', PARENT_RESULTS / 'reanalysis-S0.json'
    cell = PARENT_RESULTS / 'continuation-c0/run-r1/C0'
    return cell, cell / 'alignment.json'


'''
analysis = once(analysis, 'def report_parent_comparison(', parent_ref + 'def report_parent_comparison(')
analysis = change_function(analysis, 'report_parent_comparison', lambda s: s.replace(
    "parent_cell = PARENT_RESULTS / 'run-r1' / label\n    parent = load(parent_cell / 'alignment.json')",
    "parent_cell, parent_alignment = parent_reference(label)\n    parent = load(parent_alignment)").replace(
    "parent_alignment=str(parent_cell / 'alignment.json')", 'parent_alignment=str(parent_alignment)').replace(
    'IQ2 computational parent', 'router-single2 computational parent'))
analysis = analysis.replace('S0 must match the IQ2 computational parent', 'S0 must match the router-single2 computational parent')
# Source hashes and full-hunk validation before copying the computational parent.
build = build.replace('057ab53bd77e1cb943c110a9c414b6b4ac17fabfef9cbde3ed0db32bfdf849ed', SOURCE_HASHES[0])
build = build.replace('4f43e69e156744b6ac0d263cd8a1b159780827eb15873c7e4acbbd3bfb9ec027', SOURCE_HASHES[1])
build = once(build, '(cd "$PARENT" && patch', '(cd "$PARENT" && git apply --check --verbose "$R/router-widths-single2.patch")\n(cd "$PARENT" && patch')
build = once(build, '# Reuse parent configuration/cache;', '(cd "$D" && git apply --reverse --check --verbose "$R/router-widths-single2.patch")\n# Reuse parent configuration/cache;')
for name, text in [('run-trace.py', run), ('analyze-trace.py', analysis), ('build-candidate.sh', build)]:
    if name.endswith('.py'):
        ast.parse(text, filename=name)
    save(name, text)
subprocess.run(['bash', '-n', str(OUT / 'build-candidate.sh')], check=True)
# Test adaptation is separate only to keep this preparation script reviewable.
namespace = dict(globals())
exec(compile(Path(__file__).with_name('prepare-tests.py').read_text(), 'prepare-tests.py', 'exec'), namespace)
print('PREPARED', OUT, flush=True)
