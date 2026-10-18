#!/usr/bin/env python3
"""Diagnostics-off behavioral contracts; import is read-only and never launches work."""
import hashlib
import inspect
import json
import re
import runpy
import shlex
import shutil
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off')
LONG = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1-long-fidelity'
L = runpy.run_path(str(LONG / 'contracts.py'))
N, P, OLD = L['N'], L['P'], L['OLD']
REFERENCE, NODRAFT, CANDIDATE = L['REFERENCE'], L['NODRAFT'], L['CANDIDATE']
load, save, sha = L['load'], L['save'], L['sha']
ARMS = ('A0', 'C0')
PREFIX = 'bolt-lilicorr-width1-diagnostics-off-r1-'
FILES = ('contracts.py', 'run-off.py', 'analyze-off.py', 'test-off.py', 'cpu-checks.sh', 'README.md')
DIAG_KEYS = ('BOLT_VERIFY_CAPTURE_DIR', 'LLAMA_TRACE', 'BOLT_TRACE_TENSORS', 'GGML_SYCL_DEBUG')
OLD_BOUNDARY = 'unset GGML_SYCL_DEBUG; export LLAMA_TRACE=1; echo ' + P['MARKER'] + '; exec '
OFF_BOUNDARY = ('unset BOLT_VERIFY_CAPTURE_DIR LLAMA_TRACE; export GGML_SYCL_DEBUG=0; echo '
                + P['MARKER'] + '; exec ')
NATIVE_ON = 'export BOLT_VERIFY_CAPTURE_DIR=/tmp; '
RESERVE, ALLOWANCE = 4 * 1024**3, 1024**3
FORBIDDEN = ('BOLT_VERIFY', 'BOLT_ROUTE', 'BOLT_TENSOR_TRACE_ENABLED',
             'BOLT_PROJECTION_REPEAT', 'common_debug_cb_eval', 'LLAMA_TRACE =', 'onednn_verbose')


def code_hashes():
    return dict(L['code_hashes'](), **{str(ROOT / name): sha(ROOT / name) for name in FILES})


def source_contracts():
    pins = L['source_contracts']()
    sampling = (CANDIDATE / 'common/sampling.cpp').read_text()
    server = (CANDIDATE / 'tools/server/server-context.cpp').read_text()
    sycl = (CANDIDATE / 'ggml/src/ggml-sycl/ggml-sycl.cpp').read_text()
    assert 'trace_dir && trace_dir[0]' in sampling
    assert 'gsmpl->set_logits(ctx, idx, trace ? &raw_logits : nullptr)' in sampling
    assert 'trace ? bolt_verify_write(trace_dir, raw_logits, trace_vocab)' in sampling
    assert 'return dir && dir[0];' in server and 'return dir && dir[0];' in sycl
    assert 'if (!bolt_route_enabled() || !bolt_route_selected_weight(weight))' in sycl
    assert 'g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0)' in sycl
    assert 'std::strcmp(env, "1") == 0' in server and 'if (bolt_tensor_trace_enabled())' in server
    assert 'result.prob         = 1.0f; // set later' in server and '// TODO: set result.probs' in server
    return dict(source_pins=pins, behavioral_off_not_zero_overhead=True,
                speculative_probability_placeholder_not_fidelity=True)


def traced_argv(label):
    assert label in ARMS
    if label == 'A0':
        old = load(REFERENCE / 'run-r1/S0/create-argv.json')
        argv = N['make_create'](old, NODRAFT / 'run-r1/A0/capture')
        assert argv == load(NODRAFT / 'run-r1/A0/create-argv.json')
        return argv
    return load(REFERENCE / 'run-r1/C0/create-argv.json')


def make_create(label):
    """Only diagnostic boundary, capture env/mount, and container name change."""
    argv = list(traced_argv(label))
    argv[argv.index('--name') + 1] = PREFIX + label.lower()
    capture = (NODRAFT / 'run-r1/A0/capture' if label == 'A0'
               else REFERENCE / 'run-r1/C0/capture')
    for flag, value in (('-e', 'BOLT_VERIFY_CAPTURE_DIR=/capture'), ('-v', str(capture) + ':/capture')):
        assert argv.count(value) == 1
        i = argv.index(value)
        assert argv[i - 1] == flag
        del argv[i - 1:i + 1]
    assert argv[-1].count(OLD_BOUNDARY) == 1
    native, server = argv[-1].split(OLD_BOUNDARY)
    assert native.startswith('set -eo pipefail; ')
    # Native numerical/route commands are byte-for-byte unchanged. Their route
    # parsers NEED the nonempty gate; backend tests never run the llama sampler.
    argv[-1] = NATIVE_ON + native + OFF_BOUNDARY + server
    assert not any('/capture' in word for word in argv)
    assert argv[-3:-1] == ['bash', '-lc']
    assert shlex.split(server)[0] == '/src/build-sycl/bin/llama-server'
    assert argv.count('BOLT_TRACE_TENSORS=0') == 1
    return argv


def server_argv(label):
    return shlex.split(make_create(label)[-1].rsplit('; exec ', 1)[1])


def pid_evidence(environ, cmdline, exe, rows):
    """Redact IN the container. Never return or persist its full environment."""
    selected = []
    for item in environ.split(b'\0'):
        key, sep, value = item.partition(b'=')
        if key.decode('ascii', errors='replace') in DIAG_KEYS:
            selected.append([key.decode('ascii'), value.decode('ascii')])
    return dict(pid=1, origin='/proc/1/environ', diagnostics=selected,
                argv=[v.decode() for v in cmdline.split(b'\0') if v], exe=exe,
                scoped_tmp_rows=rows)


PID_PROBE = ("import json,os; from pathlib import Path\nDIAG_KEYS=" + repr(DIAG_KEYS) + '\n'
             + inspect.getsource(pid_evidence) + '\n'
             + "print(json.dumps(pid_evidence(Path('/proc/1/environ').read_bytes(), "
             "Path('/proc/1/cmdline').read_bytes(), os.readlink('/proc/1/exe'), "
             "sorted(p.name for p in Path('/tmp').glob('row-??????')))))")


def check_pid(record, label):
    assert record['pid'] == 1 and record['origin'] == '/proc/1/environ'
    pairs = record['diagnostics']
    assert all(len(p) == 2 and p[0] in DIAG_KEYS for p in pairs)
    env = dict(pairs)
    assert len(env) == len(pairs), 'duplicate diagnostic environment key'
    assert 'BOLT_VERIFY_CAPTURE_DIR' not in env, 'even capture=0 enables hooks'
    assert 'LLAMA_TRACE' not in env
    assert env.get('BOLT_TRACE_TENSORS') == '0'
    assert env.get('GGML_SYCL_DEBUG') == '0'
    assert record['exe'] == '/src/build-sycl/bin/llama-server'
    assert record['argv'] == server_argv(label), 'PID1 must be the exec-ed serving process'
    assert record['scoped_tmp_rows'] == [], 'unexpected raw sampler files in fresh container /tmp'
    return True


def check_off_log(log):
    assert log.count(P['MARKER']) == 1
    native, serving = log.split(P['MARKER'])
    assert 'BOLT_ROUTE route=' in native, 'native route evidence must remain enabled'
    for marker in FORBIDDEN:
        assert marker not in serving, ('serving diagnostic marker', marker)
    assert 'GGML_SYCL_DEBUG: 0' in serving
    for marker in ('GGML_ASSERT', 'SYCL exception'):
        assert marker not in log
    return True


def check_startup(log, label):
    result = L['check_startup'](log, label)
    check_off_log(log)
    return result


def frozen_jobs():
    old = L['frozen_jobs']()
    short, diagnostic = N['frozen_requests']()
    assert short == load(NODRAFT / 'run-r1/frozen-payload.json')
    jobs = old[:21]
    jobs += [dict(stem='short/' + name, payload=payload, historical_prompt_tokens=55)
             for name, payload in (('warmup', short), ('ordinary', short), ('diagnostic', diagnostic))]
    jobs += [dict(j, stem='long/' + j['stem']) for j in old[21:]]
    assert len(jobs) == 27 and sum(j['payload']['max_tokens'] for j in jobs) == 6016
    assert len({j['stem'] for j in jobs}) == 27
    return jobs


def reference_stem(label, stem):
    assert label in ARMS
    if stem.startswith('short/'):
        cell = NODRAFT / 'run-r1/A0' if label == 'A0' else REFERENCE / 'run-r1/C0'
        return cell / stem.split('/')[1]
    return LONG / 'run-r1' / label / stem.removeprefix('long/')


def extra_references(label, stem):
    if stem.startswith('quality/'):
        return [REFERENCE / 'run-r1' / ('S0' if label == 'A0' else 'C0') / stem]
    return []


def reference_prompt(label, job):
    stem = job['stem']
    if stem.startswith('short/'):
        alignment = load(reference_stem(label, 'short/diagnostic').parent / 'alignment.json')
        ids = alignment['prompt_ids']
    else:
        old_jobs = L['frozen_jobs']()
        index = next(i for i, j in enumerate(old_jobs) if j['stem'] == stem.removeprefix('long/'))
        ids = L['check_public'](job['payload'], LONG / 'run-r1' / label / 'public' / f'{index:02}')
        if stem.startswith('long/'):
            alignment = load(LONG / 'run-r1' / label / 'alignment.json')
            assert ids == alignment['prompt_ids']
    if stem.startswith(('short/', 'long/')):
        assert all(row['vocab'] == 131072 for row in alignment['samples'])
        assert len(alignment['samples']) == 128
    assert ids and all(type(t) is int and 0 <= t < 131072 for t in ids)
    if 'historical_prompt_tokens' in job:
        assert len(ids) == job['historical_prompt_tokens']
    return ids


def reference_hashes(jobs):
    paths = set()
    for label in ARMS:
        for job in jobs:
            stem = job['stem']
            ref = reference_stem(label, stem)
            for base in [ref] + extra_references(label, stem):
                request = base.with_suffix('.request.json')
                assert load(request) == job['payload'], (label, stem, 'reference request drift')
                paths.add(request)
                paths.add(base.with_suffix('.response.json' if stem.endswith('/diagnostic') else '.result.json'))
            reference_prompt(label, job)
        paths.add(reference_stem(label, 'short/diagnostic').parent / 'alignment.json')
        paths.add(LONG / 'run-r1' / label / 'alignment.json')
        paths.update((LONG / 'run-r1' / label / 'public').glob('*.json'))
    paths.update(L['FIXTURES'] / name for name in L['FIXTURE_HASHES'])
    paths.add(NODRAFT / 'run-r1/frozen-payload.json')
    for label in ARMS:
        paths.add((NODRAFT / 'run-r1/A0' if label == 'A0' else REFERENCE / 'run-r1/C0') / 'create-argv.json')
    return {str(p): sha(p) for p in sorted(paths)}


def check_tested_code():
    tested = load(ROOT / 'preparation/tested-code.json')
    assert tested['tests_passed'] is True
    assert tested['hashes'] == code_hashes(), 'helper changed since successful CPU test'
    assert tested['references'] == reference_hashes(frozen_jobs()), 'immutable fixture drift'
    return tested


def capacity(path, jobs, free=None):
    caps = [j['payload']['max_tokens'] for j in jobs]
    assert len(caps) <= 27 and all(type(c) is int and 0 < c <= 256 for c in caps)
    available = shutil.disk_usage(path).free if free is None else free
    return dict(remaining_requests=len(caps), remaining_output_caps=sum(caps),
                reserve=RESERVE, logs_http_allowance=ALLOWANCE, required_free=RESERVE + ALLOWANCE,
                free_bytes=available, fits=available >= RESERVE + ALLOWANCE,
                serving_raw_capture_permitted=False)


def result_view(result):
    return {key: result[key] for key in ('text', 'finish_reason', 'usage')}


def diagnostic_view(body, prompt):
    # Primary content[*].id is the emission. Top candidates/probabilities are NOT
    # evidence: the speculative server path still has prob=1.0f / TODO set probs.
    assert len(body['choices']) == 1
    choice = body['choices'][0]
    rows = choice['logprobs']['content']
    ids = [r['id'] for r in rows]
    assert len(ids) == 128 and all(type(t) is int and 0 <= t < 131072 for t in ids)
    assert ids == body['__verbose']['tokens']
    pieces = [r['token'] for r in rows]
    assert all(isinstance(p, str) for p in pieces)
    text = choice['message']['content']
    assert ''.join(pieces) == text
    if all(isinstance(r.get('bytes'), list) for r in rows):
        assert bytes(b for r in rows for b in r['bytes']).decode('utf-8') == text
    assert body['usage']['prompt_tokens'] == len(prompt)
    assert body['usage']['completion_tokens'] == 128 and choice['finish_reason'] == 'length'
    assert body['timings']['cache_n'] == 0
    counts = {k: body.get('timings', {})[k] for k in ('draft_n', 'draft_n_accepted') if k in body.get('timings', {})}
    return dict(ids=ids, pieces=pieces, text=text, finish_reason=choice['finish_reason'],
                usage=body['usage'], proposal_acceptance_counts=counts,
                id_source='choices[0].logprobs.content[*].id', probability_fidelity_claimed=False)


def id_comparison(left, right):
    first = next((i + 1 for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    if first is None and len(left) != len(right):
        first = min(len(left), len(right)) + 1
    return dict(equal=left == right, left_ids=left, right_ids=right,
                first_difference_ordinal=first)
