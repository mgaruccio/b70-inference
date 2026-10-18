#!/usr/bin/env python3
"""Case-local, bounded three-arm development contracts; no live work on import."""
import hashlib
import json
import math
import re
import runpy
import shlex
import shutil
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing')
OFF = ROOT.with_name('20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off')
SAMPLER = Path('/home/mike/inference/launchers/glimmer-phase0-instrument.py')
ORDER = ('O1', 'A1', 'C1', 'C2', 'A2', 'O2')
POINTS = {'short': 55, 'long': 8270}
PREFIX = 'bolt-lilicorr-width1-dev-timing-r1-'
FILES = ('contracts.py', 'run-timing.py', 'analyze-timing.py', 'test-timing.py', 'cpu-checks.sh', 'README.md')
RESERVE, ALLOWANCE = 4 * 1024**3, 1024**3
ORIGINAL_MARKER = 'BOLT_SSM_PREFILL_NUMERICAL_PASSED'
METRICS = ('predicted_ms', 'predicted_per_second', 'prompt_ms', 'prompt_per_second',
           'ttfc_s', 'elapsed_s', 'end_to_end_output_tps', 'first_content_decode_proxy_tps',
           'draft_n', 'draft_n_accepted')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helpers():
    c = runpy.run_path(str(OFF / 'contracts.py'))
    return dict(off=c, http=runpy.run_path(str(OFF / 'run-off.py')),
                analysis=runpy.run_path(str(OFF / 'analyze-off.py')),
                runner=runpy.run_path(str(c['OLD']['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py')),
                api=runpy.run_path(str(c['OLD']['OLD'] / 'probe-api.py')),
                grouped=runpy.run_path(str(c['OLD']['GROUPED'] / 'run-grouped-diagnostic.py')),
                sampler=runpy.run_path(str(SAMPLER)))


def label(cell):
    assert cell in ORDER
    return {'A': 'A0', 'C': 'C0'}.get(cell[0])


def make_create(cell, off):
    assert label(cell) is not None
    argv = list(off['make_create'](label(cell)))
    argv[argv.index('--name') + 1] = PREFIX + cell.lower()
    return argv


def jobs(off):
    frozen = off['frozen_jobs']()
    by_stem = {j['stem']: j for j in frozen}
    pre = frozen[:21]
    assert len(pre) == 21 and pre[-1]['stem'] == 'anchor'
    points = {p: {kind: by_stem[p + '/' + kind] for kind in ('ordinary', 'diagnostic')}
              for p in POINTS}
    for point, n in POINTS.items():
        payload = points[point]['ordinary']['payload']
        assert payload['temperature'] == 0 and payload['seed'] == 42
        assert payload['max_tokens'] == 128 and payload['ignore_eos'] is True
        assert payload['cache_prompt'] is False and payload['stream'] is True
        assert payload['stream_options'] == {'include_usage': True}
        assert not {'logprobs', 'response_format', 'n_probs', 'post_sampling_probs'} & payload.keys()
        assert points[point]['ordinary']['historical_prompt_tokens'] == n
    return pre, points


def schedule():
    return [dict(cell=cell, point=point, phase=phase, ordinal=i)
            for cell in ORDER for point in POINTS
            for phase, i in [('warmup', 0), ('measured', 0), ('measured', 1), ('measured', 2)]]


def capacity(path=ROOT, free=None):
    available = shutil.disk_usage(path).free if free is None else free
    return dict(free_bytes=available, reserve_bytes=RESERVE, evidence_allowance_bytes=ALLOWANCE,
                required_free=RESERVE + ALLOWANCE, fits=available >= RESERVE + ALLOWANCE,
                raw_capture_permitted=False)


def fresh_logs(before, after):
    """Per-stream stopped/drained prefixes: stderr/stdout merge order is not stable."""
    assert set(before) == set(after) == {'stdout', 'stderr'}
    result = {}
    for key in before:
        assert after[key].startswith(before[key]), ('log prefix changed', key)
        result[key] = after[key][len(before[key]):]
    return result


def original_contract(original, h):
    off, runner = h['off'], h['runner']
    assert original['Id'] == off['P']['ORIGINAL_ID']
    accepted = off['N']['ACCEPTED']
    assert runner['PREFILL_SSM'] == accepted
    cmd = original['Config']['Cmd']
    base = load(off['OLD']['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json')
    assert cmd == ['bash', '-lc', base[-1]]
    prefix, server = cmd[-1].rsplit('exec ', 1)
    assert prefix.startswith('set -eo pipefail; ')
    assert runner['CANDIDATES'][accepted] in prefix
    assert prefix.endswith(ORIGINAL_MARKER + '; ')
    gates = re.findall(r"grep -Fx '  (\d+/\d+ tests passed)'", prefix)
    assert gates, 'derive accepted gate counts from the actual immutable command'
    return dict(server_argv=shlex.split(server), native_counts=gates,
                marker=ORIGINAL_MARKER, command=cmd, native_contract='accepted existing shell gates only')


def check_original_log(log, contract):
    # The exact set -eo pipefail prefix must reach its final marker in THIS restart.
    assert log.count(ORIGINAL_MARKER) == 1
    assert 'BOLT_VERIFY_SERVER_START' not in log and 'BOLT_ROUTE route=' not in log
    native, serving = log.split(ORIGINAL_MARKER)
    for count, n in Counter(contract['native_counts']).items():
        # Native prints a summary, grep echoes it; do not confuse repeated restarts.
        assert len(re.findall(r'(?m)^.*?  ' + re.escape(count) + r'\s*$', native)) == 2 * n, count
    for marker in ('GGML_ASSERT', 'SYCL exception', 'BOLT_SSM_PREFILL_MISMATCH'):
        assert marker not in log
    assert 'llama_context:' in serving
    return dict(passed=True, native_contract=contract, fresh_restart=True)


def view(result):
    return {k: result[k] for k in ('text', 'finish_reason', 'usage')}


def check_result(result, point, reference=None):
    n = POINTS[point]
    assert result['usage']['prompt_tokens'] == n
    assert result['usage']['completion_tokens'] == 128
    assert result['usage']['total_tokens'] == n + 128
    assert result['finish_reason'] == 'length' and result['text'].strip()
    assert result['timings']['cache_n'] == 0 and result['timings']['predicted_n'] == 128
    if reference is not None:
        assert view(result) == view(reference), 'candidate differs from diagnostics-off reference'
    return True


def reference(h, arm, job):
    assert arm in ('A0', 'C0')
    stem = OFF / 'run-r1' / arm / job['stem']
    raw = h['analysis']['read_http'](stem, job['payload'], '/v1/chat/completions')
    return raw, h['analysis']['normalize'](raw, job['payload'])


def timing_metrics(result):
    elapsed, first = result['elapsed_s'], result['ttfc_s']
    assert math.isfinite(elapsed) and elapsed > 0
    assert first is not None and math.isfinite(first) and 0 <= first < elapsed
    n = result['usage']['completion_tokens']
    metrics = {k: result['timings'].get(k) for k in METRICS[:4] + METRICS[-2:]}
    for k in METRICS[:4]:
        assert isinstance(metrics[k], (int, float)) and math.isfinite(metrics[k]) and metrics[k] > 0
    metrics.update(ttfc_s=first, elapsed_s=elapsed, end_to_end_output_tps=n / elapsed,
                   first_content_decode_proxy_tps=(n - 1) / (elapsed - first))
    return metrics


def describe(values):
    """Hyndman-Fan type 7: linear interpolation at (n-1)*p, including n=1."""
    values = sorted(values)
    assert values and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
    def q(p):
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return dict(n=len(values), median=q(.5), q1=q(.25), q3=q(.75), iqr=q(.75) - q(.25))


def aggregate(rows):
    result = {}
    for metric in METRICS:
        values = [r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
        result[metric] = dict(available=len(values), missing=len(rows) - len(values),
                              summary=describe(values) if values else None)
    return result


def telemetry(samples, window_s):
    result = dict(samples=samples, window_duration_s=window_s, sample_span_s=None,
                  counter_caveat='unknown wraps/resets cannot be corrected; no peak VRAM/occupancy/bandwidth',
                  peak_vram_bytes=None, occupancy=None, bandwidth=None)
    dt = samples[-1]['t'] - samples[0]['t'] if len(samples) >= 2 else 0
    result['sample_span_s'] = dt if dt > 0 else None
    for key, name in (('energy1_uj', 'card'), ('energy2_uj', 'pkg')):
        values = [s.get(key) for s in samples]
        reason = None
        if len(values) < 2 or dt <= 0:
            reason = 'unavailable: insufficient samples'
        elif any(v is None for v in values):
            reason = 'unavailable: missing counter samples'
        elif any(b < a for a, b in zip(values, values[1:])):
            reason = 'unavailable: counter reset or wrap (modulus unknown)'
        energy = None if reason else (values[-1] - values[0]) / 1e6
        result[name + '_j'] = energy
        result[name + '_w'] = None if energy is None else energy / dt
        result[name + '_status'] = reason or 'observed counter delta; undetected full wraps not excluded'
    for key in ('freq0_mhz', 'freq1_mhz', 'temp2_mc'):
        values = [s[key] for s in samples if s.get(key) is not None]
        result[key] = dict(summary=describe(values) if values else None, missing=len(samples) - len(values))
    return result


def code_hashes(h):
    paths = {str(ROOT / f): sha(ROOT / f) for f in FILES}
    return dict(h['off']['code_hashes'](), **paths, **{str(SAMPLER): sha(SAMPLER)})


def reference_hashes(h):
    off = h['off']
    paths = set()
    for arm in off['ARMS']:
        for job in off['frozen_jobs']():
            stem = OFF / 'run-r1' / arm / job['stem']
            for suffix in ('.request.json', '.request-body', '.http-request.json', '.http.json', '.http-body'):
                paths.add(Path(str(stem) + suffix))
    paths.update([OFF / 'run-r1/original-container.json', OFF / 'run-r1/analysis-r1/analysis.json',
                  off['OLD']['OLD'] / 'model-verified.json', off['P']['WIDTH1'] / 'run-r1/draft-verified.json',
                  off['REFERENCE'] / 'protected-source-hashes.json', off['REFERENCE'] / 'source-proof.json'])
    return dict(off['reference_hashes'](off['frozen_jobs']()), **{str(p): sha(p) for p in sorted(paths)})


def source_evidence(h):
    """Read-only pinned source evidence; never execute a compiled runtime."""
    c = h['off']
    c['source_contracts']()
    paths = ('tools/server/server-common.cpp', 'tools/server/server-common.h', 'tools/server/server-context.cpp')
    records = {}
    for source in (c['N']['ACCEPTED'], c['CANDIDATE']):
        rows = {}
        for path in paths:
            lines = (source / path).read_text().splitlines()
            selected = [dict(line=i, text=line) for i, line in enumerate(lines, 1)
                        if any(s in line for s in ('t_gen_last', 't_prompt_last', 'predicted_ms',
                                                 'predicted_per_second', 'common_speculative_draft(',
                                                 'common_speculative_accept(', 'update_gen_last('))]
            rows[path] = dict(sha256=sha(source / path), evidence=selected)
        assert 't_gen_last - t_prompt_last' in (source / paths[1]).read_text()
        records[str(source)] = rows
    return records


def check_frozen(h):
    tested = load(ROOT / 'preparation/tested-code.json')
    assert tested['tests_passed'] is True
    assert tested['hashes'] == code_hashes(h), 'executed/helper code drift since CPU tests'
    assert tested['references'] == reference_hashes(h), 'reference drift since CPU tests'
    assert load(OFF / 'run-r1/analysis-r1/analysis.json')['status'] == 'passed'
    return tested
