#!/usr/bin/env python3
"""Case-local contracts. Importing this file never runs inference or native tests."""
import hashlib
import json
import re
import runpy
import shlex
import shutil
import stat
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity')
REFERENCE = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-router-widths-single2'
NODRAFT = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-nodraft-fidelity'
N = runpy.run_path(str(NODRAFT / 'run-a0.py'))
P, OLD = N['P'], N['OLD']
A = runpy.run_path(str(REFERENCE / 'analyze-trace.py'))
NA = runpy.run_path(str(NODRAFT / 'analyze-a0.py'))
load, save = N['load'], N['save']
CANDIDATE = N['CANDIDATE']
PREFIX = 'bolt-lilicorr-width1-long-fidelity-r1-'
ARMS = ('A0', 'C0')
FIXTURES = OLD['PROFILE'] / 'run-r1/baseline/long'
FIXTURE_HASHES = {
    'anchor.request.json': '761909ae4d8795c5afbc79d58ad1c0f893744f1077cdb8a6a6d54b736ddeab05',
    'measured-00.request.json': '1ec54785c5c74f45a6ea40d5a3f292a3ed41bc03b6f309d931adfb031bdb497e',
}
SOURCE_PINS = {
    'common/sampling.cpp': '672a18d2dbe346131ef63c91299d37c889c385783edcd56f8626c64ca4f89433',
    'common/speculative.cpp': 'cb76b5ea42f20cc36705b808232ad30b16310a6cb74406294f751382b21b45b8',
    'tools/server/server-context.cpp': '04b02f4fa079e29c27919b575568928fdd09842eb0e21aef80b465eab5f43997',
    'tools/server/server-common.cpp': 'fcb05757dd6a25d349024fd57df9b05c7ff77902cd17d1b70b9f41ac54f6c45c',
    'ggml/src/ggml-sycl/ggml-sycl.cpp': '22ca5fdf535a86ecff94dac10e5b56a683f92f6629610d0ff86b76234b62dd89',
    'ggml/src/ggml-sycl/fused-gemm.cpp': 'b325bb19635e4cdc80132d103bea9c12f42bcfe3102392cb1e6ed9499648a90e',
    'ggml/src/ggml-sycl/fused-gemm.hpp': '182e667b5dbfa83fe330f05af43e0e9baaeaa069a31cd48a2975fb89a034f897',
}
RESERVE = 4 * 1024**3
LOG_ALLOWANCE = 1024**3
ROW_BYTES = 131072 * 4
LOCAL_FILES = ('contracts.py', 'run-long.py', 'analyze-long.py', 'test-long.py', 'cpu-checks.sh', 'README.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_contracts():
    actual = {name: sha(CANDIDATE / name) for name in SOURCE_PINS}
    assert actual == SOURCE_PINS, 'pinned endpoint/sampling/dispatch source changed'
    return actual


def code_hashes():
    # Include executed/imported legacy code, not their raw captures or binaries.
    paths = {ROOT / name for name in LOCAL_FILES}
    for directory in (REFERENCE, NODRAFT, P['WIDTH1'], OLD['OLD'], OLD['GROUPED'], OLD['PRIOR']):
        paths.update(directory.glob('*.py'))
        paths.update(directory.glob('*.sh'))
    return {str(path): sha(path) for path in sorted(paths)}


def check_tested_code():
    tested = load(ROOT / 'preparation/tested-code.json')
    assert tested['tests_passed'] is True
    assert tested['hashes'] == code_hashes(), 'controller/helper changed since CPU tests'
    return tested


def frozen_jobs():
    assert {name: sha(FIXTURES / name) for name in FIXTURE_HASHES} == FIXTURE_HASHES
    anchor = load(FIXTURES / 'anchor.request.json')
    long = load(FIXTURES / 'measured-00.request.json')
    for payload in (anchor, long):
        assert payload['max_tokens'] == 128 and payload['temperature'] == 0 and payload['seed'] == 42
        assert payload['cache_prompt'] is False
        assert payload['chat_template_kwargs'] == {'enable_thinking': False}
        assert len(payload['messages']) == 1 and payload['messages'][0]['role'] == 'user'
    assert anchor['stream'] is False and anchor['ignore_eos'] is False
    assert long['stream'] is True and long['ignore_eos'] is True
    assert long['stream_options'] == {'include_usage': True}
    summary = load(REFERENCE / 'run-r1/C0/quality/summary.json')
    names = [row['name'] for row in summary['checks']]
    assert len(names) == len(set(names)) == 20
    jobs = []
    for name in names:
        payload = load(REFERENCE / 'run-r1/C0/quality' / (name + '.request.json'))
        assert payload == load(REFERENCE / 'run-r1/S0/quality' / (name + '.request.json'))
        assert payload['max_tokens'] == 256 and payload['stream'] is False
        assert payload['temperature'] == 0 and payload['seed'] == 42
        assert payload['cache_prompt'] is False and payload['ignore_eos'] is False
        assert payload['chat_template_kwargs'] == {'enable_thinking': False}
        jobs.append(dict(stem='quality/' + name, payload=payload))
    jobs += [dict(stem='anchor', payload=anchor, historical_prompt_tokens=8238),
             dict(stem='warmup', payload=long, historical_prompt_tokens=8270),
             dict(stem='ordinary', payload=long, historical_prompt_tokens=8270),
             dict(stem='diagnostic', payload=OLD['diagnostic_payload'](long), historical_prompt_tokens=8270)]
    assert sum(j['payload']['max_tokens'] for j in jobs) == 5632
    return jobs


def make_create(label, capture):
    assert label in ARMS and capture == ROOT / 'run-r1' / label / 'capture'
    old_label = 'S0' if label == 'A0' else 'C0'
    original = load(REFERENCE / 'run-r1' / old_label / 'create-argv.json')
    if label == 'A0':
        # The old helper performs the entire no-draft deletion and checks native mounts.
        argv = N['make_create'](original, NODRAFT / 'run-r1/A0/capture')
        old_name, old_capture = N['NAME'], NODRAFT / 'run-r1/A0/capture'
    else:
        argv = list(original)
        old_name, old_capture = P['PREFIX'] + 'c0', REFERENCE / 'run-r1/C0/capture'
        assert argv[-1].endswith(' ' + shlex.join(P['spec_for_arm']('C0')))
    assert argv[argv.index('--name') + 1] == old_name
    argv[argv.index('--name') + 1] = PREFIX + label.lower()
    assert argv.count(str(old_capture) + ':/capture') == 1
    argv[argv.index(str(old_capture) + ':/capture')] = str(capture) + ':/capture'
    # No native helper, environment, runtime, port, launch prefix, or shape changes.
    return argv


def check_startup(log, label):
    if label == 'A0':
        return N['check_startup'](log)
    assert label == 'C0' and log.count(P['MARKER']) == 1
    native = runpy.run_path(str(P['WIDTH1'] / 'analyze-controls.py'))
    P['check_startup'](log, 'C0', OLD, native)
    return dict(label='C0', target_n_rs_seq=1, draft_n_rs_seq=0,
                n_outputs_max=2, n_outputs_max_per_seq=2, speculative_implementation=True)


def budget(jobs):
    caps = [j['payload']['max_tokens'] for j in jobs]
    assert all(type(cap) is int and 0 < cap <= 256 for cap in caps)
    # Every target sampler call appends exactly one accepted token. Each processed
    # token increments n_gen; get_n_draft_max also clamps to n_remaining-1.
    # Budget a whole terminal width-1 verification (2 rows) beyond the cap anyway.
    # Grammar retry writes no second file; LiLiCorr drafting samples no raw logits.
    # n_rs_seq=1 covers the maximum rollback, so there is no checkpoint replay.
    rows = sum(cap + 2 for cap in caps)
    return dict(remaining_requests=len(caps), remaining_output_caps=sum(caps),
                raw_row_bound=rows, row_bytes=ROW_BYTES, raw_byte_bound=rows * ROW_BYTES,
                logs_and_evidence_allowance=LOG_ALLOWANCE, reserve=RESERVE,
                required_free=rows * ROW_BYTES + LOG_ALLOWANCE + RESERVE)


def capacity(path, jobs, free=None):
    record = budget(jobs)
    record['free_bytes'] = shutil.disk_usage(path).free if free is None else free
    record['fits'] = record['free_bytes'] >= record['required_free']
    return record


def capture_inventory(path):
    count = size = allocated = 0
    assert path.is_dir() and not path.is_symlink()
    for p in path.iterdir():
        info = p.lstat()
        assert re.fullmatch(r'row-[a-zA-Z0-9]{6}', p.name), p
        assert stat.S_ISREG(info.st_mode) and info.st_size == ROW_BYTES, p
        count += 1
        size += info.st_size
        allocated += info.st_blocks * 512
    return dict(files=count, bytes=size, allocated_bytes=allocated)


def apply_request(payload):
    # Preserve every chat input option. Explicitly spell the parser default;
    # do not render user content with /tokenize directly or a private substitute.
    assert payload['chat_template_kwargs'] == {'enable_thinking': False}
    assert not payload.get('continue_final_message', False)
    assert payload['messages'][-1]['role'] == 'user'
    result = dict(payload, add_generation_prompt=True)
    return result


def tokenize_request(rendered):
    assert set(rendered) == {'prompt'} and isinstance(rendered['prompt'], str)
    assert rendered['prompt']
    return dict(content=rendered['prompt'], add_special=True, parse_special=True, with_pieces=False)


def token_ids(response):
    assert set(response) == {'tokens'}
    tokens = response['tokens']
    assert tokens and all(type(t) is int and 0 <= t < 131072 for t in tokens)
    assert len(tokens) + 256 < 65536
    return tokens


def public_tokenize(payload, stem, request):
    template = apply_request(payload)
    save(stem.with_suffix('.template.request.json'), template)
    rendered = request(OLD['URL'], '/apply-template', template)
    save(stem.with_suffix('.template.response.json'), rendered)
    tokenize = tokenize_request(rendered)
    save(stem.with_suffix('.tokenize.request.json'), tokenize)
    response = request(OLD['URL'], '/tokenize', tokenize)
    save(stem.with_suffix('.tokenize.response.json'), response)
    return token_ids(response)


def check_public(payload, stem):
    assert load(stem.with_suffix('.template.request.json')) == apply_request(payload)
    rendered = load(stem.with_suffix('.template.response.json'))
    assert load(stem.with_suffix('.tokenize.request.json')) == tokenize_request(rendered)
    return token_ids(load(stem.with_suffix('.tokenize.response.json')))


def functional_quality(name, result):
    # Exactly probe-api.py's existing quality criteria (including permissive code).
    fixed = {'literal': 'READY', 'arithmetic': '323', 'literal-red': 'RED',
             'literal-cedar': 'cedar', 'literal-code': 'X7-Q92', 'sort-numbers': '1,2,5,9',
             'json-lookup': '42', 'extract': '13'}
    fixed.update({f'arithmetic-{i:02}': str(n) for i, n in enumerate((95, 616, 1081, 12, 1369, 55, -17, 16, 800, 1000))})
    text = result['text']
    if name == 'code':
        ok = 'def add(' in text and 'return' in text and '+' in text
    elif name == 'json':
        ok = ''.join(text.split()) == '{"ready":true,"count":3}'
    else:
        ok = text.strip() == fixed[name]
    return bool(ok and result['finish_reason'] == 'stop' and
                result['usage'].get('prompt_tokens', 0) > 0 and result['usage'].get('completion_tokens', 0) > 0)


def functional_anchor(result):
    # Exact probe-api.py long-anchor criterion; no new completion-count or text rule.
    return (result['text'].strip() == 'BOLT-7319-CEDAR' and result['finish_reason'] == 'stop'
            and result['usage']['prompt_tokens'] >= 8192)
