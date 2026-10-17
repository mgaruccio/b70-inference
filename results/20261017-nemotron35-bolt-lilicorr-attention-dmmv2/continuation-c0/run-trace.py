#!/usr/bin/env python3
"""Development-only attention projection DMMV control; inference-host only."""
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

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-attention-dmmv2/continuation-c0')
WIDTH1 = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1')
PARENT_RESULTS = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-attention-route')
ACCEPTED = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-attention-route-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-attention-dmmv2-r1')
ORDER = ('C0',)
PREFIX = 'bolt-lilicorr-attention-dmmv2-c0-r1-'
MARKER = 'BOLT_VERIFY_SERVER_START'
CAPTURE_ENV = 'BOLT_VERIFY_CAPTURE_DIR'
ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
             'test-backend-ops', 'libllama-common.so')
SOURCE_SCOPE = ['ggml/src/ggml-sycl/ggml-sycl.cpp', 'tests/test-backend-ops.cpp']
ENV = (('BOLT_TRACE_TENSORS', '0'), ('ONEDNN_VERBOSE', '0'), ('GGML_SYCL_ENABLE_DNN', '1'),
       ('GGML_SYCL_ENABLE_OPT', '1'), ('GGML_SYCL_ENABLE_VMM', '1'), (CAPTURE_ENV, '/capture'))
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

def check_iq2_native_gate(log):
    log = re.sub(r'(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z ', '', log)
    marker = 'BOLT_VERIFY_IQ2_NATIVE_PASSED'
    assert log.count(marker) == 1
    assert 'BOLT_LILICORR_OUTPROD_NATIVE_PASSED' in log
    focused = log.split('BOLT_LILICORR_OUTPROD_NATIVE_PASSED', 1)[1].split(marker, 1)[0]
    focused = re.sub(r'\x1b\[[0-9;]*m', '', focused)
    rows = re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', focused)
    assert rows == [('6', '6')], rows
    cases = re.findall(r'(?m)^\s*MUL_MAT_ID\(type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,[^\n]*\): OK\s*$', focused)
    assert len(cases) == 6, len(cases)
    # The backend filter intentionally skips CPU as a test target; CPU remains the numerical reference.
    focused = re.sub(r'(?m)^Backend \d+/\d+: CPU\n[ \t]+Skipping\n', '', focused)
    assert not re.search(r'(?i)not supported|skipp|unsupported|unexpected|err\s*=|failed', focused)


IQ2_NATIVE_COMMAND = (
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT_ID "
    "-p 'type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,' 2>&1 | "
    "tee /tmp/iq2-xxs-native.log; "
    "grep -Fqx '  6/6 tests passed' /tmp/iq2-xxs-native.log; "
    "count=$(grep -Fc '  6/6 tests passed' /tmp/iq2-xxs-native.log || true); test \"$count\" -eq 1; "
    "count=$(grep -Ec '^  MUL_MAT_ID\\(type_a=iq2_xxs,type_b=f32,n_mats=16,n_used=9,.*\\):.*OK' /tmp/iq2-xxs-native.log || true); test \"$count\" -eq 6; "
    "if grep -Eiq 'not supported|skipping large tensors|skipped|unsupported|UNEXPECTED|ERR =|failed' "
    "/tmp/iq2-xxs-native.log; then exit 1; fi; "
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



# Exactly thirteen named CPU-reference fixtures added by attention-dmmv2.patch.
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


ATTENTION_NATIVE_COMMAND = (
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT "
    "-p '^attention_dmmv2=1,' 2>&1 | tee /tmp/q8-attention-dmmv2-native.log; "
    "grep -Fqx '  13/13 tests passed' /tmp/q8-attention-dmmv2-native.log; "
    "count=$(grep -Fc '  13/13 tests passed' /tmp/q8-attention-dmmv2-native.log || true); test \"$count\" -eq 1; "
    "python3 -B -c " + shlex.quote(
        "import pathlib,runpy; "
        "runpy.run_path('/attention-dmmv2-gate/run-trace.py')['check_attention_native_output']("
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
        "runpy.run_path('/attention-dmmv2-gate/run-trace.py')['check_q8_native_output']("
        "pathlib.Path('/tmp/q8-latent-dmmv2-native.log').read_text())") + "; "
    "echo " + Q8_NATIVE_MARKER + "; "
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
        extra += ['-v', str(ROOT / script) + ':/attention-dmmv2-gate/' + script + ':ro']
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
                + 'unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; echo '
                + MARKER + '; exec ' + server + ' ' + shlex.join(spec_for_arm(label)))
    return argv


def check_startup(log, label, old, native):
    assert label in ORDER and MARKER in log
    native['parse_native_gate'](log, 'C0')  # BOTH arms require native385 and focused190/support9
    check_iq2_native_gate(log)
    check_q8_native_gate(log)
    check_attention_native_gate(log)
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
                assert all(binaries(p) == hashes[k] for k, p in sources.items())
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
