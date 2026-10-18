#!/usr/bin/env python3
"""LEAD ONLY: one same-width suppression control; analysis after restoration."""
import argparse
import hashlib
import json
import math
import re
import runpy
import socket
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width3-suppression')
OLD = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-width-sweep'
# Frozen at 74ebc785; fail before importing a changed controller/contract.
for filename, digest in (
    ('contracts.py', '80547ac6ca7689e259e9a9e70f3d0f7dc0fbb8248e763e99f7bbc0d6606a877f'),
    ('controller.py', 'e8f4a537b355b5315a08fdb410682f4ac5aef13878a723f51ca4c7812bd931d8'),
):
    assert hashlib.sha256((OLD / filename).read_bytes()).hexdigest() == digest, filename
K = runpy.run_path(str(OLD / 'controller.py'))
C = K['C']  # Same exception classes as inherited run_job; never mutate these globals.
CELL = 'w3-suppressed'
NAME = C['PREFIX'] + CELL
OUT = ROOT / 'run-r1'
FILES = ('probe.py', 'test-suppression.py', 'cpu-checks.sh', 'README.md')
STEMS = ('short/ordinary', 'short/diagnostic', 'long/ordinary', 'long/diagnostic')


def jobs(h):
    selected = [job for job in C['qualification_jobs'](h) if job['stem'] in STEMS]
    assert tuple(job['stem'] for job in selected) == STEMS
    return selected


def make_create(h):
    base = C['make_create']('W3', h)
    argv = list(base)
    argv[argv.index('--name') + 1] = NAME
    native, server = base[-1].rsplit('; exec ', 1)
    matches = list(re.finditer(r'--spec-draft-n-min\s+(\d+)(?=\s|$)', server))
    assert len(matches) == 1 and matches[0].group(1) == '0'
    start, end = matches[0].span(1)
    argv[-1] = native + '; exec ' + server[:start] + '4' + server[end:]
    assert argv[-1].rsplit('; exec ', 1)[0] == native
    C['diagnostics_off_server'](argv[-1])
    assert all('/capture' not in word for word in argv)
    assert [d['index'] for d in C['argv_diff'](base, argv)] == [argv.index('--name') + 1, len(argv) - 1]
    return argv


def check_pid(record, h):
    # The old allowlist, unchanged except for the expected suppression argv.
    assert record['pid'] == 1 and record['origin'] == '/proc/1/environ'
    pairs = record['diagnostics']
    assert all(len(pair) == 2 and pair[0] in C['DIAGNOSTIC_KEYS'] for pair in pairs)
    env = dict(pairs)
    assert len(env) == len(pairs)
    assert 'BOLT_VERIFY_CAPTURE_DIR' not in env and 'LLAMA_TRACE' not in env
    assert env.get('BOLT_TRACE_TENSORS') == '0' and env.get('GGML_SYCL_DEBUG') == '0'
    assert record['exe'] == '/src/build-sycl/bin/llama-server'
    assert record['argv'] == C['server_argv'](make_create(h))
    assert record['scoped_tmp_rows'] == []


def check_startup(log, h):
    # Narrow copy of check_width_startup: min=4, never normalize live evidence.
    off, p, old = h['off'], h['off']['P'], h['off']['OLD']
    assert p['MARKER'] in log and log.count(p['MARKER']) == 1
    off['check_off_log'](log)
    native = runpy.run_path(str(p['WIDTH1'] / 'analyze-controls.py'))
    native['parse_native_gate'](log, 'C0')
    for name in ('iq2', 'q8', 'attention', 'attention_view', 'router'):
        p['check_' + name + '_native_gate'](log)
    assert 'call ggml_sycl_ssm_scan_prefill:' not in log.split(p['MARKER'], 1)[0]
    model = log.split(p['MARKER'], 1)[1]
    old['check_model_log'](model, 'C')
    values = [int(match.group(1)) for match in old['N_RS_SEQ_RE'].finditer(model)]
    assert values[:2] == [3, 0], values
    params = old['DFLASH_PARAMS_RE'].search(model)
    assert params and params.groups() == ('3', '4'), params
    block = old['DFLASH_BLOCK_RE'].search(model)
    assert block and block.group(1) == '8', block
    spec = model.split('common_speculative_impl_draft_dflash', 1)[1]
    spec = spec.split('speculative decoding context', 1)[0]
    assert 'clamp' not in spec.lower(), 'unexpected speculative parameter clamp'
    assert 'R (f32)' in model and 'S (f32)' in model
    for key in ('n_outputs_max', 'n_outputs_max_per_seq'):
        match = re.search(r'llama_context: ' + key + r'\s*=\s*(\d+)', model)
        assert match and match.group(1) == '4', (key, match)
    for key, value in (('DNN', 1), ('FUSION', 1), ('ESIMD', 1), ('OPT', 1), ('VMM', 1), ('GRAPH', 0)):
        assert f'GGML_SYCL_ENABLE_{key}: {value}' in model
    assert 'BOLT_VERIFY error' not in log
    return dict(width=3, n_max=3, n_min=4, block_size=8, target_n_rs_seq=3,
                draft_n_rs_seq=0, n_outputs_max=4, n_outputs_max_per_seq=4,
                evidence='unaltered startup; shared native/model/off-log gates')


def check_names(existing):
    assert NAME not in set(existing), ('control already exists; no reuse/replacement', NAME)
    return dict(planned_cells=[CELL], planned_names=[NAME], collisions=[])


def freeze(h):
    hashes = C['code_hashes'](h)
    hashes.update({str(ROOT / name): C['sha'](ROOT / name) for name in FILES})
    references = C['reference_hashes'](h)
    paths = [C['QUALIFY_DIR'] / name for name in ('summary.json', 'qualification.json')]
    paths += [C['QUALIFY_DIR'] / 'W3' / name for name in (
        'native-and-startup.stdout.log', 'native-and-startup.stderr.log', 'pid-before.json')]
    for cell in ('A', 'W3'):
        for job in jobs(h):
            stem = C['QUALIFY_DIR'] / cell / job['stem']
            paths += [C['artifact'](stem, suffix) for suffix in (
                '.request.json', '.request-body', '.http-request.json', '.http.json', '.http-body')]
    references.update({str(path): C['sha'](path) for path in paths})
    prior = C['load'](C['QUALIFY_DIR'] / 'qualification.json')
    assert prior['status'] == 'passed' and prior['independently_restored'] and prior['restored']
    assert prior['reports']['A']['passed'] and not prior['reports']['W3']['passed']
    assert prior['qualified_widths'] == [], 'this is isolation, not a timing phase'
    return dict(hashes=hashes, references=references, source_binaries=C['source_binary_hashes'](h))


def check_tested(h):
    tested = C['load'](ROOT / 'preparation/tested-code.json')
    assert tested['tests_passed'] is True
    assert tested['freeze'] == freeze(h), 'code/reference/source/binary drift since CPU checks'
    return tested


class Runtime(K['Runtime']):
    def preflight(self):
        self.capacity('initial')
        check_tested(self.h)
        self.hashes = C['source_binary_hashes'](self.h)
        self.before_hw = self.h['runner']['hardware']()
        assert self.before_hw == C['load'](self.off['REFERENCE'] / 'run-r1/hardware-before.json')
        frozen_original = C['load'](self.off['ROOT'] / 'run-r1/original-container.json')
        live = self.inspect(self.original_name)
        assert C['same_container'](frozen_original, live) and live['State']['Running']
        self.original, self.original_was_running = live, True
        C['save'](self.out / 'original-container.json', live)
        C['save'](self.out / 'hash-checks-before.json', self.hashes)
        C['save'](self.out / 'hardware-before.json', self.before_hw)
        assert self.h['api']['request'](self.off['OLD']['URL'], '/health')['status'] == 'ok'
        existing = self.command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
        C['save'](self.out / 'planned-containers.json', check_names(existing))
        C['save'](self.out / 'source-contracts.json', C['source_evidence'](self.h))
        env = dict(kernel=self.command('uname', '-a'), os_release=Path('/etc/os-release').read_text(),
                   cpu=self.command('lscpu'), memory=Path('/proc/meminfo').read_text(),
                   driver=self.command('modinfo', 'xe'),
                   source_ids={str(source): (source / 'UPSTREAM_COMMIT').read_text()
                               for source in (self.off['N']['ACCEPTED'], self.off['N']['CANDIDATE'])})
        C['save'](self.out / 'environment.json', env)
        C['save'](self.out / 'original-live.json', live)
        self.observe('run-before', self.original_name)

    def pid(self, cell, phase):
        assert cell == CELL
        record = json.loads(self.command('docker', 'exec', NAME, 'python3', '-B', '-c', C['PID_PROBE']))
        C['save'](self.out / cell / ('pid-' + phase + '.json'), record)
        check_pid(record, self.h)
        maps = self.command('docker', 'exec', NAME, 'cat', '/proc/1/maps')
        (self.out / cell / ('libraries-' + phase + '.txt')).write_text(
            '\n'.join(sorted({line.split()[-1] for line in maps.splitlines() if '.so' in line})) + '\n')
        return record

    def begin(self, cell):
        assert cell == CELL and self.active is None and not self.created
        self.capacity(cell)
        self._stop_original_for_case()
        argv = make_create(self.h)
        cell_dir = self.out / cell
        cell_dir.mkdir()
        C['save'](cell_dir / 'create-argv.json', argv)
        C['save'](cell_dir / 'argv-diff-vs-W3.json', C['argv_diff'](C['make_create']('W3', self.h), argv))
        self.created.append(cell)
        self.command(*argv, timeout=60)
        self.command('docker', 'start', NAME)
        C['save'](cell_dir / 'models.json', self.ready(self.h['api'], NAME))
        startup = self.save_logs(cell, 'native-and-startup', NAME)
        contract = check_startup(startup, self.h)
        contract['pid_argv'] = self.pid(cell, 'before')['argv']
        contract['pid_width_evidence'] = 'actual /proc/1/cmdline'
        C['save'](cell_dir / 'startup.json', contract)
        self.observe(cell + '-ready', NAME)
        C['save'](cell_dir / 'startup-buffers.json', [line for line in startup.splitlines()
                  if any(word in line.lower() for word in ('buffer size', 'kv cache size', 'recurrent'))])
        C['save'](cell_dir / 'container-before.json', self.inspect(NAME))
        self.active = cell


def collect(rt, cell):
    assert cell == CELL
    report = dict(jobs=[], analysis_pending=True, suppression_proven=False)
    try:
        for job in jobs(rt.h):
            # The inherited public input, HTTP, and API checks still fail closed.
            # A valid output mismatch is retained, not raised or retried.
            result = rt.run_job(cell, job, expected_for_compare=False)
            comparison = result['comparison']
            equal = comparison['passed'] if result['kind'] == 'diagnostic' else comparison['equal']
            report['jobs'].append(dict(stem=job['stem'], matched_frozen_a=equal, result=result))
    finally:
        C['save'](rt.out / cell / 'capture.json', report)
        log = rt.save_logs(cell, 'before-stop')
        rt.off['check_off_log'](log)
        C['save'](rt.out / cell / 'before-stop-duration-stats.json', C['log_duration_stats'](log))
    return []


def zero_proposal_proof(before, final, http_counts):
    before_stats, final_stats = C['log_duration_stats'](before), C['log_duration_stats'](final)
    assert final_stats['available'], 'no final cumulative counters: suppression unproven'
    for row in before_stats['records'] + final_stats['records']:
        assert row['generated']['tokens'] == row['accepted']['tokens'] == 0, row
    last = final_stats['records'][-1]
    assert last['calls']['g'] > 0, 'no real draft generation calls'
    assert math.isfinite(last['duration_ms']['g']) and last['duration_ms']['g'] > 0
    for counts in http_counts:
        for key in ('draft_n', 'draft_n_accepted'):
            if key in counts:
                assert type(counts[key]) is int and counts[key] == 0, counts
    return dict(proven=True, before_stop=before_stats, final=final_stats,
                http_counts=http_counts, missing_http_counts_are_not_zero_proof=True,
                timing_interpretation_permitted=False)


def read_view(h, directory, job):
    raw = h['analysis']['read_http'](directory / job['stem'], job['payload'], '/v1/chat/completions')
    prompt = h['off']['reference_prompt']('A0', job)
    if C['qualification_kinds'](job) == 'diagnostic':
        return h['off']['diagnostic_view'](json.loads(raw), prompt)
    result = h['analysis']['normalize'](raw, job['payload'])
    C['check_output'](result, job, prompt_ids=prompt)
    return dict(C['result_view'](result), proposal_acceptance_counts={
        key: result['timings'][key] for key in ('draft_n', 'draft_n_accepted') if key in result['timings']})


def compare(job, actual, expected):
    if C['qualification_kinds'](job) == 'diagnostic':
        return C['diagnostic_compare'](actual, expected)
    return dict(passed=C['result_view'](actual) == C['result_view'](expected))


def analyze(h):
    summary = C['load'](OUT / 'summary.json')
    assert summary['status'] == 'captured' and summary['restored']
    assert summary['order'] == summary['completed'] == [CELL] and not summary['timing_permitted']
    # Fresh live restoration verification precedes any suppression/output claim.
    rt = Runtime(h, OUT, 'suppression')
    rt.verify_restored(OUT / 'independent-restoration.json')
    check_tested(h)
    cell_dir = OUT / CELL
    capture = C['load'](cell_dir / 'capture.json')
    assert tuple(row['stem'] for row in capture['jobs']) == STEMS
    logs = {}
    for phase in ('before-stop', 'server', 'restored-final'):
        logs[phase] = {stream: (cell_dir / (phase + '.' + stream + '.log')).read_bytes()
                       for stream in ('stdout', 'stderr')}
    for stream in ('stdout', 'stderr'):
        assert logs['server'][stream].startswith(logs['before-stop'][stream])
        assert logs['restored-final'][stream].startswith(logs['server'][stream])
    text = {phase: (streams['stdout'] + streams['stderr']).decode('utf-8', errors='replace')
            for phase, streams in logs.items()}
    for log in text.values():
        h['off']['check_off_log'](log)
    check_startup((cell_dir / 'native-and-startup.stdout.log').read_text(errors='replace') +
                  (cell_dir / 'native-and-startup.stderr.log').read_text(errors='replace'), h)
    for phase in ('before', 'after'):
        check_pid(C['load'](cell_dir / ('pid-' + phase + '.json')), h)
    rows, counts = [], []
    for job in jobs(h):
        actual = read_view(h, cell_dir, job)
        expected = dict(frozen_a=read_view(h, h['off']['ROOT'] / 'run-r1/A0', job),
                        fresh_a=read_view(h, C['QUALIFY_DIR'] / 'A', job),
                        active_w3=read_view(h, C['QUALIFY_DIR'] / 'W3', job))
        assert compare(job, expected['fresh_a'], expected['frozen_a'])['passed']
        matches = {key: compare(job, actual, view) for key, view in expected.items()}
        rows.append(dict(stem=job['stem'], actual=actual, expected=expected, comparisons=matches))
        counts.append(actual['proposal_acceptance_counts'])
    proof = zero_proposal_proof(text['before-stop'], text['restored-final'], counts)
    matched = all(row['comparisons']['frozen_a']['passed'] for row in rows)
    C['save'](OUT / 'analysis.json', dict(status='analyzed', independently_restored=True,
              suppression=proof, matched_frozen_a=matched, jobs=rows,
              interpretation=('active verification/rollback path implicated, not proven causal'
                              if matched else 'state/prefill/drafter presence also implicated; not router proof'),
              timing_interpretation_permitted=False, promotion_permitted=False,
              retokenized_output_as_ids_claimed=False, raw_f32_equality_claimed=False))
    print('ANALYZED_AFTER_RESTORE suppression_proven=True matched_frozen_a=' + str(matched), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-suppression', action='store_true')
    action.add_argument('--lead-after-restore', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'inference-host' and Path(__file__).resolve().parent == ROOT
    h = C['helpers']()
    check_tested(h)
    if args.lead_run_suppression:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_bytes((ROOT / 'README.md').read_bytes())
        C['save'](OUT / 'tested-code.json', check_tested(h))
        K['capture'](Runtime(h, OUT, 'suppression'), (CELL,), collect)
        check_tested(h)
        print('CAPTURED_RESTORED analysis_pending=True suppression_unproven', flush=True)
    else:
        analyze(h)


if __name__ == '__main__':
    main()
