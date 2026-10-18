#!/usr/bin/env python3
"""Development-only F32 router singleton-pair control; inference-host only."""
import re
import hashlib
import json
import runpy
import shlex
import shutil
import signal
import socket
import subprocess
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2')
WIDTH1 = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1'
PARENT_RESULTS = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-iq2-single2' / 'pair-r2'
ACCEPTED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-single2-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1')
ORDER = ('S0', 'C0')
PREFIX = 'bolt-lilicorr-router-single2-r1-'
MARKER = 'BOLT_VERIFY_SERVER_START'
CAPTURE_ENV = 'BOLT_VERIFY_CAPTURE_DIR'
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so')
SOURCE_SCOPE = ['ggml/src/ggml-sycl/ggml-sycl.cpp', 'tests/test-backend-ops.cpp']
ENV = (('BOLT_TRACE_TENSORS', '0'), ('ONEDNN_VERBOSE', '0'), ('GGML_SYCL_ENABLE_DNN', '1'),
       ('GGML_SYCL_ENABLE_OPT', '1'), ('GGML_SYCL_ENABLE_VMM', '1'), (CAPTURE_ENV, '/capture'))
PARENT_EXPECTED = {
    'libggml-sycl.so.0': '959874f59e890957c1a4ec1488be1c322bdc7eeba675b7b2c9e0cdaa2ead66db',
    'test-backend-ops': 'c5592fd927626a468c1464cf4f42ee2de5894a0d55f6b1ded023809f5ffd8db9',
    'libllama-server-impl.so': '43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede',
    'libllama-common.so': '934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812',
    'llama-server': '22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d',
}
ORIGINAL_ID = '7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e'
CAPTURE_RESERVE_BYTES = 4 * 1024**3


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binaries(source):
    return {name: hashlib.sha256((source / 'build-sycl/bin' / name).read_bytes()).hexdigest()
            for name in ARTIFACTS}


def spec_for_arm(label):
    assert label in ORDER
    return ['--spec-type', 'draft-dflash', '--spec-draft-n-max', '1',
            '--spec-draft-n-min', '2' if label == 'S0' else '0', '--spec-draft-p-min', '0',
            '--spec-draft-sampling', 'greedy', '--no-spec-draft-backend-sampling',
            '--spec-draft-ngl', '999', '--spec-draft-model',
            '/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf']

def legacy_native_prefix(prefix):
    # Only the dispatch contract changes; all numerical/state/canary checks remain.
    old = ("count=$(grep -Fc 'call ggml_sycl_ssm_scan_prefill:' "
           '/tmp/ssm-prefill-fusion1.log || true); test "$count" -eq 23')
    assert prefix.count(old) == 1, 'unexpected inherited prefill dispatch guard'
    return prefix.replace(old, old.removesuffix('23') + '0', 1)

IQ2_NATIVE_MARKER = 'BOLT_VERIFY_IQ2_NATIVE_PASSED'
IQ2_NATIVE_CASES = {(b, m, n, k) for b, m, k in ((1, 2048, 1024), (0, 1024, 2048)) for n in (1, 2, 3)}


def check_iq2_width3_routes(routes, analysis):
    """Unmodified width3: grouped attempt/result, or its per-expert MMVQ fallback."""
    common, tensor = analysis['assert_iq2_route_common'], analysis['assert_route_tensor']
    assert len(routes) >= 2
    m, k, b = analysis['IQ2_PROJECTIONS'][routes[0]['weight']]
    row = k // 256 * 66
    for route in routes:
        common(route)
        assert route['weight'] == routes[0]['weight'], route
    for route, name in zip(routes[:2], ('MOE_GROUPED_GEMM_ATTEMPT', 'MOE_GROUPED_GEMM_RESULT')):
        assert route['route'] == name and route['expert'] == -1, route
        tensor(route, '', (k, m, 16, 1), (66, row, row * m, row * m * 16))
        tensor(route, 'src1_', (k, b, 3, 1), (4, k * 4, k * b * 4, k * b * 3 * 4))
        tensor(route, 'dst_', (m, 9, 3, 1), (4, m * 4, m * 9 * 4, m * 9 * 3 * 4))
        tensor(route, 'ids_', (9, 3, 1, 1), (4, 64, 192, 192))
    assert routes[0]['result'] == -1 and routes[1]['result'] in (0, 1)
    if routes[1]['result'] == 1:
        assert len(routes) == 2
        return
    assert len(routes) >= 4 and len(routes) % 2 == 0
    experts, rows = set(), 0
    for marker, actual in zip(routes[2::2], routes[3::2]):
        assert marker['route'] == 'MOE_PER_EXPERT_FALLBACK' and marker['result'] == -1, marker
        assert 0 <= marker['expert'] < 16 and marker['expert'] not in experts, marker
        experts.add(marker['expert'])
        n = marker['src1_ne1']
        assert 1 <= n <= 3, marker
        rows += n
        assert actual['route'] == 'MMVQ_PLAIN' and actual['expert'] == -1 and actual['result'] == -1, actual
        for route in (marker, actual):
            tensor(route, '', (k, m, 1, 1), (66, row, row * m, row * m))
            tensor(route, 'src1_', (k, n, 1, 1), (4, k * 4, k * n * 4, k * n * 4))
            tensor(route, 'dst_', (m, n, 1, 1), (4, m * 4, m * n * 4, m * n * 4))
        tensor(marker, 'ids_', (9, 3, 1, 1), (4, 64, 192, 192))
        tensor(actual, 'ids_', (0, 0, 0, 0), (0, 0, 0, 0))
    assert rows == 27, rows


def check_iq2_native_output(log):
    log = re.sub(r'(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z ', '', log)
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    summaries = re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log)
    assert summaries == [('6', '6')], summaries
    cases = re.findall(
        r'(?m)^\s*MUL_MAT_ID\(type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,'
        r'b=(\d+),m=(\d+),n=(\d+),k=(\d+),amax=1(?:\.0+)?,m_v=0,ids_offset=0\): OK\s*$', log)
    keys = [tuple(map(int, case)) for case in cases]
    assert len(keys) == 6 and set(keys) == IQ2_NATIVE_CASES, keys
    assert re.findall(r'(?m)^Backend \d+/\d+: (\S+)\s*$', log) == ['SYCL0', 'CPU']
    assert re.findall(r'(?m)^\s*Backend SYCL0: (\S+)\s*$', log) == ['OK']
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', log) == [('2', '2')]
    cleaned = re.sub(r'(?m)^Backend 2/2: CPU\n[ \t]+Skipping\n', '', log)
    assert not re.search(r'(?i)not supported|skipp|unsupported|unexpected|err\s*=|failed', cleaned)
    analysis = runpy.run_path(str(Path(__file__).with_name('analyze-trace.py')))
    routes = analysis['route_records'](log)
    grouped = {}
    previous = None
    for route in routes:
        analysis['assert_iq2_route_common'](route)
        weight = route['weight']
        if route['route'] == 'MOE_SINGLE_MMVQ_PLAIN_Q8_1':
            assert route['dst_nb2'] > 0 and route['dst_nb3'] % route['dst_nb2'] == 0, route
            n = route['dst_nb3'] // route['dst_nb2']
            analysis['assert_iq2_singleton_route'](route, n, 16)
        elif route['route'] == 'MMVQ_PLAIN':
            assert previous and previous['route'] == 'MOE_PER_EXPERT_FALLBACK' and previous['weight'] == weight, route
            n = 3
        else:
            assert route['route'] in ('MOE_GROUPED_GEMM_ATTEMPT', 'MOE_GROUPED_GEMM_RESULT', 'MOE_PER_EXPERT_FALLBACK'), route
            assert route['ids_ne1'] == 3, route
            n = 3
        grouped.setdefault((weight, n), []).append(route)
        previous = route
    expected = {(weight, n) for weight in analysis['IQ2_PROJECTIONS'] for n in (1, 2, 3)}
    assert set(grouped) == expected, ('missing actual IQ2 native routes', set(grouped))
    assert len({route['backend_ctx'] for route in routes}) == 1
    for (weight, n), group in grouped.items():
        if n <= 2:
            assert len(group) == n, (weight, n, group)
        else:
            check_iq2_width3_routes(group, analysis)


def check_iq2_native_gate(log):
    before = 'BOLT_LILICORR_OUTPROD_NATIVE_PASSED'
    assert log.count(IQ2_NATIVE_MARKER) == 1 and log.count(before) == 1
    assert log.index(before) < log.index(IQ2_NATIVE_MARKER)
    if Q8_NATIVE_MARKER in log:
        assert log.index(IQ2_NATIVE_MARKER) < log.index(Q8_NATIVE_MARKER)
    focused = log.split(before, 1)[1].split(IQ2_NATIVE_MARKER, 1)[0]
    check_iq2_native_output(focused)


IQ2_NATIVE_COMMAND = (
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT_ID "
    "-p 'type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,' 2>&1 | "
    "tee /tmp/iq2-xxs-native.log; "
    "grep -Fqx '  6/6 tests passed' /tmp/iq2-xxs-native.log; "
    "count=$(grep -Fc '  6/6 tests passed' /tmp/iq2-xxs-native.log || true); test \"$count\" -eq 1; "
    "count=$(grep -Ec '^  MUL_MAT_ID\\(type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,.*\\):.*OK' /tmp/iq2-xxs-native.log || true); test \"$count\" -eq 6; "
    "if grep -Eiq 'not supported|skipping large tensors|skipped|unsupported|UNEXPECTED|ERR =|failed' "
    "/tmp/iq2-xxs-native.log; then exit 1; fi; "
    "python3 -B -c " + shlex.quote(
        "import pathlib,runpy; "
        "runpy.run_path('/iq2-single2-gate/run-trace.py')['check_iq2_native_output']("
        "pathlib.Path('/tmp/iq2-xxs-native.log').read_text())") + "; "
    "echo BOLT_VERIFY_IQ2_NATIVE_PASSED; "
)


# Exactly the seven named CPU-reference fixtures added by latent-dmmv2.patch.
Q8_NATIVE_CASES = {
    (weight, m, n, k): ('DMMV_ESIMD' if n <= 2 else 'MMVQ_REORDERED')
    for weight, m, k in (('blk.1.ffn_latent_down.weight', 1024, 4096),
                         ('blk.1.ffn_latent_up.weight', 4096, 1024))
    for n in (1, 2, 3)
}
Q8_NATIVE_CASES[('output.weight', 1024, 2, 4096)] = 'MMVQ_REORDERED'
Q8_NATIVE_MARKER = 'BOLT_VERIFY_Q8_LATENT_NATIVE_PASSED'


def check_q8_native_output(log):
    # Docker's independent server.log has timestamps; tee's native log does not.
    log = re.sub(r'(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z ', '', log)
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    summaries = re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log)
    assert summaries == [('7', '7')], summaries
    cases = re.findall(
        r'(?m)^\s*MUL_MAT\(latent_dmmv2=1,weight=([^,\n]+),type_a=q8_0,type_b=f32,'
        r'm=(\d+),n=(\d+),k=(\d+),[^\n]*\): OK\s*$', log)
    keys = [(weight, int(m), int(n), int(k)) for weight, m, n, k in cases]
    assert len(keys) == 7 and set(keys) == set(Q8_NATIVE_CASES), keys
    assert re.findall(r'(?m)^Backend \d+/\d+: (\S+)\s*$', log) == ['SYCL0', 'CPU']
    assert re.findall(r'(?m)^\s*Backend SYCL0: (\S+)\s*$', log) == ['OK']
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', log) == [('2', '2')]
    # This filter skip is benign; numerical skips and any other backend skip are not.
    cleaned = re.sub(r'(?m)^Backend 2/2: CPU\n[ \t]+Skipping\n', '', log)
    assert not re.search(r'(?i)not supported|skipp|unsupported|unexpected|err\s*=|failed', cleaned)
    parse = runpy.run_path(str(Path(__file__).with_name('analyze-trace.py')))['route_records']
    routes = parse(log)
    seen = set()
    for route in routes:
        key = (route['weight'], route['ne1'], route['src1_ne1'], route['ne0'])
        assert key in Q8_NATIVE_CASES, route
        _, m, n, k = key
        assert route['route'] == Q8_NATIVE_CASES[key] and route['reorder'] == 1, route
        assert route['op'] == 'MUL_MAT' and route['src0_type'] == 'q8_0', route
        assert route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
        assert route['device'] == 0 and route['expert'] == -1 and route['result'] == -1, route
        assert route['aux_weight'] == '-' and route['aux_type'] == '-', route
        for prefix, shape, strides in (
                ('', (k, m, 1, 1), (34, k // 32 * 34, k // 32 * 34 * m, k // 32 * 34 * m)),
                ('src1_', (k, n, 1, 1), (4, k * 4, k * n * 4, k * n * 4)),
                ('dst_', (m, n, 1, 1), (4, m * 4, m * n * 4, m * n * 4))):
            assert tuple(route[f'{prefix}ne{i}'] for i in range(4)) == shape, route
            assert tuple(route[f'{prefix}nb{i}'] for i in range(4)) == strides, route
        assert all(route[f'ids_{field}{i}'] == 0 for field in ('ne', 'nb') for i in range(4)), route
        seen.add(key)
    assert seen == set(Q8_NATIVE_CASES), ('missing actual native routes', seen)
    assert len({route['backend_ctx'] for route in routes}) == 1


def check_q8_native_gate(log):
    assert log.count(Q8_NATIVE_MARKER) == 1
    assert log.count('BOLT_VERIFY_IQ2_NATIVE_PASSED') == 1
    focused = log.split('BOLT_VERIFY_IQ2_NATIVE_PASSED', 1)[1].split(Q8_NATIVE_MARKER, 1)[0]
    check_q8_native_output(focused)



# Exactly thirteen inherited attention CPU-reference fixtures.
ATTENTION_NATIVE_CASES = {
    **{
        (weight, m, n, k): ('DMMV_ESIMD' if n <= 2 else 'MMVQ_REORDERED')
        for weight, m, k in (
            ('blk.7.attn_q.weight', 4096, 4096),
            ('blk.7.attn_k.weight', 256, 4096),
            ('blk.7.attn_v.weight', 256, 4096),
            ('blk.7.attn_output.weight', 4096, 4096),
        )
        for n in (1, 2, 3)
    },
    ('blk.7.unrelated.weight', 256, 2, 4096): 'MMVQ_REORDERED',
}
ATTENTION_NATIVE_MARKER = 'BOLT_VERIFY_Q8_ATTENTION_NATIVE_PASSED'


# Exactly six named F32-view CPU-reference fixtures.
ATTENTION_VIEW_NATIVE_CASES = {
    ('blk.7.attn_output.weight', 4096, 1, 4096): 'DMMV_ESIMD',
    ('blk.7.attn_output.weight', 4096, 2, 4096): 'DMMV_ESIMD',
    ('blk.7.attn_output.weight', 4096, 3, 4096): 'MMVQ_REORDERED',
    ('blk.7.prefix.attn_output.weight', 4096, 2, 4096): 'MMVQ_REORDERED',
    ('blk.7.offset.attn_output.weight', 4096, 2, 4096): 'MMVQ_REORDERED',
    ('blk.7.unrelated.weight', 4096, 2, 4096): 'MMVQ_REORDERED',
}
ATTENTION_VIEW_NATIVE_MARKER = 'BOLT_VERIFY_Q8_ATTENTION_VIEW_NATIVE_PASSED'


def check_attention_native_output(log):
    log = re.sub(r'(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z ', '', log)
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    summaries = [row for row in re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log) if row == ('13', '13')]
    assert summaries == [('13', '13')], re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log)
    cases = re.findall(
        r'(?m)^\s*MUL_MAT\(attention_dmmv2=1,weight=([^,\n]+),type_a=q8_0,type_b=f32,'
        r'm=(\d+),n=(\d+),k=(\d+),[^\n]*\): OK\s*$', log)
    keys = [(weight, int(m), int(n), int(k)) for weight, m, n, k in cases]
    assert len(keys) == 13 and set(keys) == set(ATTENTION_NATIVE_CASES), keys
    assert re.findall(r'(?m)^Backend \d+/\d+: (\S+)\s*$', log) == ['SYCL0', 'CPU']
    assert re.findall(r'(?m)^\s*Backend SYCL0: (\S+)\s*$', log) == ['OK']
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', log) == [('2', '2')]
    cleaned = re.sub(r'(?m)^Backend 2/2: CPU\n[ \t]+Skipping\n', '', log)
    assert not re.search(r'(?i)not supported|skipp|unsupported|unexpected|err\s*=|failed', cleaned)
    parse = runpy.run_path(str(Path(__file__).with_name('analyze-trace.py')))['route_records']
    routes = parse(log)
    seen = set()
    for route in routes:
        key = (route['weight'], route['ne1'], route['src1_ne1'], route['ne0'])
        assert key in ATTENTION_NATIVE_CASES, route
        _, m, n, k = key
        assert route['route'] == ATTENTION_NATIVE_CASES[key] and route['reorder'] == 1, route
        assert route['operand_prec'] == -1, route
        assert route['op'] == 'MUL_MAT' and route['src0_type'] == 'q8_0', route
        assert route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
        assert route['device'] == 0 and route['expert'] == -1 and route['result'] == -1, route
        assert route['aux_weight'] == '-' and route['aux_type'] == '-', route
        for prefix, shape, strides in (
                ('', (k, m, 1, 1), (34, k // 32 * 34, k // 32 * 34 * m, k // 32 * 34 * m)),
                ('src1_', (k, n, 1, 1), (4, k * 4, k * n * 4, k * n * 4)),
                ('dst_', (m, n, 1, 1), (4, m * 4, m * n * 4, m * n * 4))):
            assert tuple(route[f'{prefix}ne{i}'] for i in range(4)) == shape, route
            assert tuple(route[f'{prefix}nb{i}'] for i in range(4)) == strides, route
        assert all(route[f'ids_{field}{i}'] == 0 for field in ('ne', 'nb') for i in range(4)), route
        seen.add(key)
    assert seen == set(ATTENTION_NATIVE_CASES), ('missing actual native routes', seen)
    assert len({route['backend_ctx'] for route in routes}) == 1


def check_attention_native_gate(log):
    assert log.count(ATTENTION_NATIVE_MARKER) == 1
    assert log.count(Q8_NATIVE_MARKER) == 1
    focused = log.split(Q8_NATIVE_MARKER, 1)[1].split(ATTENTION_NATIVE_MARKER, 1)[0]
    check_attention_native_output(focused)


# Exactly six view cases are run after the thirteen-case attention gate.
def check_attention_view_native_output(log):
    log = re.sub(r'(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z ', '', log)
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    summaries = [row for row in re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log) if row == ('6', '6')]
    assert summaries == [('6', '6')], re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log)
    cases = re.findall(
        r'(?m)^\s*MUL_MAT\(attention_view_dmmv2=1,weight=([^,\n]+),type_a=q8_0,type_b=f32,'
        r'm=(\d+),n=(\d+),k=(\d+),[^\n]*\): OK\s*$', log)
    keys = [(weight, int(m), int(n), int(k)) for weight, m, n, k in cases]
    assert len(keys) == 6 and set(keys) == set(ATTENTION_VIEW_NATIVE_CASES), keys
    assert re.findall(r'(?m)^Backend \d+/\d+: (\S+)\s*$', log) == ['SYCL0', 'CPU']
    assert re.findall(r'(?m)^\s*Backend SYCL0: (\S+)\s*$', log) == ['OK']
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', log) == [('2', '2')]
    cleaned = re.sub(r'(?m)^Backend 2/2: CPU\n[ \t]+Skipping\n', '', log)
    assert not re.search(r'(?i)not supported|skipp|unsupported|unexpected|err\s*=|failed', cleaned)
    parse = runpy.run_path(str(Path(__file__).with_name('analyze-trace.py')))['route_records']
    routes = parse(log)
    seen = set()
    for route in routes:
        key = (route['weight'], route['ne1'], route['src1_ne1'], route['ne0'])
        assert key in ATTENTION_VIEW_NATIVE_CASES, route
        _, m, n, k = key
        assert route['route'] == ATTENTION_VIEW_NATIVE_CASES[key] and route['reorder'] == 1, route
        assert route['operand_prec'] == -1, route
        assert route['op'] == 'MUL_MAT' and route['src0_type'] == 'q8_0', route
        assert route['src1_type'] == 'f32' and route['dst_type'] == 'f32', route
        assert route['device'] == 0 and route['expert'] == -1 and route['result'] == -1, route
        assert route['aux_weight'] == '-' and route['aux_type'] == '-', route
        for prefix, shape, strides in (
                ('', (k, m, 1, 1), (34, k // 32 * 34, k // 32 * 34 * m, k // 32 * 34 * m)),
                ('src1_', (k, n, 1, 1), (4, k * 4, k * n * 4, k * n * 4)),
                ('dst_', (m, n, 1, 1), (4, m * 4, m * n * 4, m * n * 4))):
            assert tuple(route[f'{prefix}ne{i}'] for i in range(4)) == shape, route
            assert tuple(route[f'{prefix}nb{i}'] for i in range(4)) == strides, route
        assert all(route[f'ids_{field}{i}'] == 0 for field in ('ne', 'nb') for i in range(4)), route
        seen.add(key)
    assert seen == set(ATTENTION_VIEW_NATIVE_CASES), ('missing actual native routes', seen)
    assert len({route['backend_ctx'] for route in routes}) == 1


def check_attention_view_native_gate(log):
    assert log.count(ATTENTION_VIEW_NATIVE_MARKER) == 1
    assert log.count(ATTENTION_NATIVE_MARKER) == 1
    focused = log.split(ATTENTION_NATIVE_MARKER, 1)[1].split(ATTENTION_VIEW_NATIVE_MARKER, 1)[0]
    check_attention_view_native_output(focused)


ATTENTION_VIEW_NATIVE_COMMAND = (
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT "
    "-p '^attention_view_dmmv2=1,' 2>&1 | tee /tmp/q8-attention-view-dmmv2-native.log; "
    "grep -Fqx '  6/6 tests passed' /tmp/q8-attention-view-dmmv2-native.log; "
    "count=$(grep -Fc '  6/6 tests passed' /tmp/q8-attention-view-dmmv2-native.log || true); test \"$count\" -eq 1; "
    "python3 -B -c " + shlex.quote(
        "import pathlib,runpy; "
        "runpy.run_path('/iq2-single2-gate/run-trace.py')['check_attention_view_native_output']("
        "pathlib.Path('/tmp/q8-attention-view-dmmv2-native.log').read_text())") + "; "
    "echo " + ATTENTION_VIEW_NATIVE_MARKER + "; "
)


ATTENTION_NATIVE_COMMAND = (
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT "
    "-p '^attention_dmmv2=1,' 2>&1 | tee /tmp/q8-attention-dmmv2-native.log; "
    "grep -Fqx '  13/13 tests passed' /tmp/q8-attention-dmmv2-native.log; "
    "count=$(grep -Fc '  13/13 tests passed' /tmp/q8-attention-dmmv2-native.log || true); test \"$count\" -eq 1; "
    "python3 -B -c " + shlex.quote(
        "import pathlib,runpy; "
        "runpy.run_path('/iq2-single2-gate/run-trace.py')['check_attention_native_output']("
        "pathlib.Path('/tmp/q8-attention-dmmv2-native.log').read_text())") + "; "
    "echo " + ATTENTION_NATIVE_MARKER + "; "
)

Q8_NATIVE_COMMAND = (
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT "
    "-p '^latent_dmmv2=1,' 2>&1 | tee /tmp/q8-latent-dmmv2-native.log; "
    "grep -Fqx '  7/7 tests passed' /tmp/q8-latent-dmmv2-native.log; "
    "count=$(grep -Fc '  7/7 tests passed' /tmp/q8-latent-dmmv2-native.log || true); test \"$count\" -eq 1; "
    "python3 -B -c " + shlex.quote(
        "import pathlib,runpy; "
        "runpy.run_path('/iq2-single2-gate/run-trace.py')['check_q8_native_output']("
        "pathlib.Path('/tmp/q8-latent-dmmv2-native.log').read_text())") + "; "
    "echo " + Q8_NATIVE_MARKER + "; "
)


ROUTER_NATIVE_CASES = {
    ('blk.1.ffn_gate_inp.weight', 0, n) for n in (1, 2, 3)
} | {('blk.1.unrelated.weight', 0, 2), ('blk.1.ffn_gate_inp.weight', 1, 2), ('blk.1.ffn_gate_inp.weight', 2, 2)}
ROUTER_NATIVE_MARKER = 'BOLT_VERIFY_ROUTER_NATIVE_PASSED'


def check_router_native_output(log):
    log = re.sub(r'(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z ', '', log)
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    assert re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log) == [('6', '6')]
    cases = re.findall(
        r'(?m)^\s*MUL_MAT\(router_single2=1,weight=([^,\n]+),layout=([012]),type_a=f32,type_b=f32,'
        r'm=240,n=([123]),k=4096,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],'
        r'k_v=0,o=1,src_overlap=0,m_v=0,pad=0\): OK\s*$', log)
    keys = [(w, int(layout), int(n)) for w, layout, n in cases]
    assert len(keys) == 6 and set(keys) == ROUTER_NATIVE_CASES, keys
    assert re.findall(r'(?m)^Backend \d+/\d+: (\S+)\s*$', log) == ['SYCL0', 'CPU']
    assert re.findall(r'(?m)^\s*Backend SYCL0: (\S+)\s*$', log) == ['OK']
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', log) == [('2', '2')]
    cleaned = re.sub(r'(?m)^Backend 2/2: CPU\n[ \t]+Skipping\n', '', log)
    assert not re.search(r'(?i)not supported|skipp|unsupported|unexpected|err\s*=|failed', cleaned)
    analysis = runpy.run_path(str(Path(__file__).with_name('analyze-trace.py')))
    grouped = {}
    for route in analysis['route_records'](log):
        identity = re.fullmatch(r'router_single2-l([012])-n([123])', route['dst_name'])
        assert identity, route
        key = (route['weight'], *map(int, identity.groups()))
        assert key in ROUTER_NATIVE_CASES, key
        grouped.setdefault(key, []).append(route)
    assert set(grouped) == ROUTER_NATIVE_CASES, ('missing actual native router cases', set(grouped))
    assert len({r['backend_ctx'] for group in grouped.values() for r in group}) == 1
    for (weight, layout, n), routes in grouped.items():
        split = weight.endswith('.ffn_gate_inp.weight') and n == 2 and layout == 0
        analysis['assert_router_operations'](routes, n, split, layout)


def check_router_native_gate(log):
    assert log.count(ROUTER_NATIVE_MARKER) == log.count(ATTENTION_VIEW_NATIVE_MARKER) == 1
    assert log.index(ATTENTION_VIEW_NATIVE_MARKER) < log.index(ROUTER_NATIVE_MARKER) < log.index(MARKER)
    check_router_native_output(log.split(ATTENTION_VIEW_NATIVE_MARKER, 1)[1].split(ROUTER_NATIVE_MARKER, 1)[0])


ROUTER_NATIVE_COMMAND = (
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT "
    "-p '^router_single2=1,' 2>&1 | tee /tmp/router-single2-native.log; "
    "python3 -B -c " + shlex.quote(
        "import pathlib,runpy; "
        "runpy.run_path('/iq2-single2-gate/run-trace.py')['check_router_native_output']("
        "pathlib.Path('/tmp/router-single2-native.log').read_text())") + "; "
    "echo " + ROUTER_NATIVE_MARKER + "; "
)


def make_create(base, label, image, capture, old):
    assert label in ORDER
    assert capture == ROOT / 'run-r1' / label / 'capture'
    argv = list(base)
    assert argv[argv.index('--name') + 1] == old['ORIGINAL']
    argv[argv.index('--name') + 1] = PREFIX + label.lower()
    mount = str(ACCEPTED) + ':/src:ro'
    assert argv.count(mount) == 1
    argv[argv.index(mount)] = str(CANDIDATE) + ':/src:ro'
    assert all(not any(a.startswith(key + '=') for a in argv) for key, _ in ENV)
    extra = [part for key, value in ENV for part in ('-e', key + '=' + value)]
    extra += ['-v', str(capture) + ':/capture', '-v', str(old['DRAFT'] / 'draft') + ':/draft:ro',
              '-v', str(old['PRIOR'] / 'native-outprod.sh') + ':/native-outprod.sh:ro']
    for script in ('run-trace.py', 'analyze-trace.py'):
        extra += ['-v', str(ROOT / script) + ':/iq2-single2-gate/' + script + ':ro']
    argv[argv.index(image):argv.index(image)] = extra
    prefix, server = argv[-1].rsplit('exec ', 1)
    assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in prefix and '--spec-type' not in server
    prefix = legacy_native_prefix(prefix)
    for flag, value in (('--ctx-size', '65536'), ('--parallel', '1'), ('--batch-size', '1024'),
                        ('--ubatch-size', '1024'), ('--cache-type-k', 'q8_0'), ('--cache-type-v', 'q8_0')):
        words = shlex.split(server)
        assert words.count(flag) == 1 and words[words.index(flag) + 1] == value
    for key in ('GGML_SYCL_ENABLE_ESIMD=1', 'GGML_SYCL_ENABLE_FUSION=1', 'GGML_SYCL_ENABLE_GRAPH=0'):
        assert argv.count(key) == 1
    assert '--cache-type-r' not in server  # pinned server default is F32; checked again at startup
    argv[-1] = (prefix + 'bash /native-outprod.sh; ' + IQ2_NATIVE_COMMAND + Q8_NATIVE_COMMAND + ATTENTION_NATIVE_COMMAND
                + ATTENTION_VIEW_NATIVE_COMMAND + ROUTER_NATIVE_COMMAND + 'unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; echo '
                + MARKER + '; exec ' + server + ' ' + shlex.join(spec_for_arm(label)))
    return argv


def check_startup(log, label, old, native):
    assert label in ORDER and MARKER in log
    native['parse_native_gate'](log, 'C0')  # BOTH arms require native385 and focused190/support9
    check_iq2_native_gate(log)
    check_q8_native_gate(log)
    check_attention_native_gate(log)
    check_attention_view_native_gate(log)
    check_router_native_gate(log)
    assert 'call ggml_sycl_ssm_scan_prefill:' not in log.split(MARKER, 1)[0], 'custom prefill ran in legacy-only native gate'
    model = log.split(MARKER, 1)[1]
    old['check_model_log'](model, 'C')
    values = [int(x.group(1)) for x in old['N_RS_SEQ_RE'].finditer(model)]
    assert values[:2] == [1, 0], values
    params = old['DFLASH_PARAMS_RE'].search(model)
    assert params and params.groups() == ('1', '2' if label == 'S0' else '0')
    assert old['DFLASH_BLOCK_RE'].search(model).group(1) == '8'
    spec = model.split('common_speculative_impl_draft_dflash', 1)[1].split('speculative decoding context', 1)[0]
    assert 'clamp' not in spec.lower()
    assert "R (f32)" in model and "S (f32)" in model, 'F32 recurrent state not reported'
    for key in ('n_outputs_max', 'n_outputs_max_per_seq'):
        assert re.search(r'llama_context: ' + key + r'\s*=\s*(\d+)', model).group(1) == '2'
    for key, value in (('DNN', 1), ('FUSION', 1), ('ESIMD', 1), ('OPT', 1), ('VMM', 1), ('GRAPH', 0)):
        assert f'GGML_SYCL_ENABLE_{key}: {value}' in model
    assert 'BOLT_VERIFY error' not in log



def check_restored(original, live):
    assert live['State']['Running'] and live['Id'] == original['Id']
    assert live['Config'] == original['Config'] and live['HostConfig'] == original['HostConfig']
    assert sorted(live['Mounts'], key=lambda m: m['Destination']) == sorted(
        original['Mounts'], key=lambda m: m['Destination'])


def diagnostic_window(before, after):
    assert after.startswith(before), 'Docker log prefix changed; cannot delimit diagnostic'
    window = after[len(before):]
    assert 'BOLT_VERIFY sample ' in window, 'no diagnostic capture records'
    return window


def source_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] != 'build-sycl'}


def check_protected_sources():
    expected = load(ROOT / 'protected-source-hashes.json')
    assert source_hashes(ACCEPTED) == expected['accepted']
    assert source_hashes(PARENT) == expected['parent']
    actual = source_hashes(CANDIDATE)
    changed = sorted(k for k in actual.keys() | expected['parent'].keys()
                     if actual.get(k) != expected['parent'].get(k))
    assert changed == SOURCE_SCOPE, changed


def main():
    assert socket.gethostname() == 'inference-host'
    old = runpy.run_path(str(WIDTH1 / 'run-controls.py'))
    native = runpy.run_path(str(WIDTH1 / 'analyze-controls.py'))
    helper = runpy.run_path(str(old['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(old['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(old['OLD'] / 'probe-api.py'))
    analyze = runpy.run_path(str(ROOT / 'analyze-trace.py'))
    command, ready = helper['command'], helper['ready']
    assert 'BOLT_VERIFY_BUILD_PASSED' in (ROOT / 'post-build-verification.log').read_text()
    assert 'BOLT_VERIFY_DEPENDENCIES_PASSED' in (ROOT / 'runtime-precheck.log').read_text()
    assert load(ROOT / 'source-diff.json')['changed_relative_to_parent'] == SOURCE_SCOPE
    hashes = {name: load(ROOT / (name + '-binaries.json')) for name in ('accepted', 'parent', 'candidate')}
    sources = dict(accepted=ACCEPTED, parent=PARENT, candidate=CANDIDATE)
    assert all(set(hashes[name]) == set(ARTIFACTS) and binaries(p) == hashes[name] for name, p in sources.items())
    assert hashes['parent'] == PARENT_EXPECTED
    check_protected_sources()
    unchanged = ['llama-server', 'libllama-server-impl.so', 'libllama-common.so']
    assert all(hashes['parent'][n] == hashes['candidate'][n] for n in unchanged)
    assert hashes['parent']['libggml-sycl.so.0'] != hashes['candidate']['libggml-sycl.so.0']
    assert hashes['parent']['test-backend-ops'] != hashes['candidate']['test-backend-ops']
    draft = load(old['DRAFT'] / 'draft-verified.json')
    assert draft['pass'] and draft['bytes'] == old['DRAFT_BYTES'] and draft['sha256'] == old['DRAFT_SHA256']
    draft_path = Path(draft['path'])
    assert draft_path.stat().st_size == old['DRAFT_BYTES']
    assert hashlib.sha256(draft_path.read_bytes()).hexdigest() == old['DRAFT_SHA256']
    assert shutil.disk_usage(ROOT).free >= CAPTURE_RESERVE_BYTES
    before_hw = runner['hardware']()
    original = json.loads(command('docker', 'inspect', old['ORIGINAL']))[0]
    assert original['Id'] == ORIGINAL_ID
    assert original['State']['Running'] and original['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in original['Mounts'] if m['Destination'] == '/src') == str(ACCEPTED)
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {old['ORIGINAL']}
    names = [PREFIX + label.lower() for label in ORDER]
    assert not set(names).intersection(command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines())
    base = load(old['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json')
    assert base[-1] == original['Config']['Cmd'][-1]
    payload = load(WIDTH1 / 'run-r1/frozen-payload.json')
    assert payload == load(old['PROFILE'] / 'run-r1/baseline/bench/measured-00.request.json')
    diag = load(WIDTH1 / 'run-r1/A0/diagnostic.request.json')
    assert diag == load(WIDTH1 / 'run-r1/C0/diagnostic.request.json') == old['diagnostic_payload'](payload)
    assert diag['backend_sampling'] is False and diag['max_tokens'] == 128
    out = ROOT / 'run-r1'
    out.mkdir(exist_ok=False)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    readme = ROOT / 'README.md'
    assert readme.is_file(), 'declaration requires README.md at eval root'
    (out / 'declaration.md').write_bytes(readme.read_bytes())
    save(out / 'original-container.json', original)
    save(out / 'hardware-before.json', before_hw)
    save(out / 'frozen-payload.json', payload)
    for name, value in hashes.items():
        save(out / (name + '-binaries.json'), value)
    summary = dict(status='prepared', order=ORDER, tier='development-diagnostic',
                   timing_permitted=False, promotion_permitted=False, phases={}, restored=False,
                   reference_equivalence_required=False)
    created, stopped = [], False

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped = True
        command('docker', 'stop', '-t', '90', old['ORIGINAL'])
        for label, name in zip(ORDER, names):
            cell = out / label
            cell.mkdir()
            capture = cell / 'capture'
            capture.mkdir(mode=0o755)  # frozen container runs as root; files themselves are 0644
            assert shutil.disk_usage(capture).free >= CAPTURE_RESERVE_BYTES
            argv = make_create(base, label, runner['IMAGE'], capture, old)
            save(cell / 'create-argv.json', argv)
            created.append((name, cell))  # include ambiguous create/start failures in cleanup
            command(*argv, timeout=60)
            command('docker', 'start', name)
            data = summary['phases'][label] = dict(status='started')
            data['models'] = ready(api, name)
            startup = command('docker', 'logs', name)
            (cell / 'native-and-startup.log').write_text(startup)
            check_startup(startup, label, old, native)
            quality_log = command('python3', '-B', str(old['OLD'] / 'probe-api.py'), '--out',
                                  str(cell / 'quality'), '--mode', 'quality', timeout=1200)
            (cell / 'quality.log').write_text(quality_log)
            quality = load(cell / 'quality/summary.json')
            assert quality['pass'] and len(quality['checks']) == 20 and all(x['pass'] for x in quality['checks'])
            for stem in ('warmup', 'ordinary'):
                result = api['chat'](old['URL'], payload['messages'][0]['content'], cell / stem,
                                     count=128, stream=True, ignore_eos=True)
                assert load(cell / (stem + '.request.json')) == payload
                old['check_stream_result'](result)
            save(cell / 'diagnostic.request.json', diag)
            pre = command('docker', 'logs', '--timestamps', name)
            (cell / 'pre-diagnostic.log').write_text(pre)
            body = api['request'](old['URL'], '/v1/chat/completions', diag)
            save(cell / 'diagnostic.response.json', body)
            old['check_diagnostic_result'](body)
            command('docker', 'stop', '-t', '90', name)
            post = command('docker', 'logs', '--timestamps', name)
            (cell / 'through-diagnostic.log').write_text(post)
            (cell / 'diagnostic-window.log').write_text(diagnostic_window(pre, post))
            assert 'BOLT_VERIFY error' not in post
            # Fail this arm before starting the next on any within-arm/capture/alignment failure.
            checked = analyze['analyze_arm'](cell, label, old)
            save(cell / 'alignment.json', checked)
            data.update(status='completed', within_arm_text_passed=True,
                        cpu_rows=len(checked['samples']), **checked['prior_reference'])
            data['s0_parent_invariant_passed'] = checked['s0_parent_invariant_passed']
            data['c0_parent_report_only'] = checked['c0_parent_report_only']
            save(cell / 'summary.json', data)
            assert all(binaries(p) == hashes[k] for k, p in sources.items())
            assert runner['hardware']() == before_hw
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        if hasattr(exc, 'read'):
            (out / 'http-error.txt').write_bytes(exc.read())
        raise
    finally:
        cleanup_errors = []
        try:
            for name, cell in created:
                for action in ('stop', 'logs', 'inspect'):
                    try:
                        if action == 'stop':
                            command('docker', 'stop', '-t', '90', name)
                        elif action == 'logs':
                            (cell / 'server.log').write_text(command('docker', 'logs', '--timestamps', name))
                        else:
                            save(cell / 'container.json', json.loads(command('docker', 'inspect', name))[0])
                    except BaseException as exc:
                        cleanup_errors.append(dict(container=name, action=action, error=repr(exc)))
        finally:
            try:
                if stopped:
                    assert not set(names).intersection(command('docker', 'ps', '--format', '{{.Names}}').splitlines())
                    command('docker', 'start', old['ORIGINAL'])
                    summary['restored_models'] = ready(api, old['ORIGINAL'])
                    smoke = api['chat'](old['URL'], 'Reply with exactly this word and nothing else: READY',
                                        out / 'restored-smoke', count=32)
                    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
                    save(out / 'restored-smoke.result.json', smoke)
                    live = json.loads(command('docker', 'inspect', old['ORIGINAL']))[0]
                    save(out / 'restored-container.json', live)
                    check_restored(original, live)
                    assert api['request'](old['URL'], '/health')['status'] == 'ok'
                    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {old['ORIGINAL']}
                assert all(binaries(p) == hashes[k] for k, p in sources.items())
                check_protected_sources()
                after_hw = runner['hardware']()
                save(out / 'hardware-after.json', after_hw)
                assert after_hw == before_hw
                summary['restored'] = True
                if cleanup_errors:
                    summary['cleanup_errors'] = cleanup_errors
                    raise RuntimeError('cleanup evidence incomplete')
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                save(out / 'summary.json', summary)
    print('BOLT_VERIFY_PAIR_COMPLETED_NO_TIMING', flush=True)


if __name__ == '__main__':
    main()
