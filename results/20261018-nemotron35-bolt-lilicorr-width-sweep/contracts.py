#!/usr/bin/env python3
"""Case-local contracts for the bounded LiLiCorr width sweep.

Importing this module is read-only.  The live controllers are lead-only and are
never called by this file or by the CPU fixture suite.
"""
import hashlib
import inspect
import json
import math
import re
import runpy
import shlex
import shutil
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width-sweep')
PRIOR = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1-dev-timing'
OFF = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off'
SAMPLER = Path('/home/mike/inference/launchers/glimmer-phase0-instrument.py')

PREFIX = 'bolt-lilicorr-width-sweep-r1-'
ORIGINAL = 'nemotron-bolt-probe'
QUALIFY_DIR = ROOT / 'qualification-r1'
TIMING_DIR = ROOT / 'timing-r1'
QUALIFY_ORDER = ('A', 'W3', 'W7')
TIMING_ORDER = ('A1', 'W3a', 'W7a', 'W7b', 'W3b', 'A2')
POINTS = ('short', 'long')
WIDTHS = {'W3': 3, 'W7': 7, 'W3a': 3, 'W7a': 7, 'W7b': 7, 'W3b': 3}
LABELS = {'A': 'A0', 'A1': 'A0', 'A2': 'A0',
          'W3': 'C0', 'W7': 'C0', 'W3a': 'C0', 'W7a': 'C0',
          'W7b': 'C0', 'W3b': 'C0'}

RESERVE = 4 * 1024**3
ALLOWANCE = 1 * 1024**3
HTTP_LIMIT = 4 * 1024**2
LOCAL_FILES = (
    'contracts.py', 'controller.py', 'run-qualify.py', 'analyze-qualify.py',
    'run-timing.py', 'analyze-timing.py', 'test-width-sweep.py',
    'cpu-checks.sh', 'README.md', 'preparation-results.md',
)
METRICS = (
    'predicted_ms', 'predicted_per_second', 'prompt_ms', 'prompt_per_second',
    'ttfc_s', 'elapsed_s', 'end_to_end_output_tps',
    'first_content_decode_proxy_tps', 'draft_n', 'draft_n_accepted',
)
DIAGNOSTIC_KEYS = ('BOLT_VERIFY_CAPTURE_DIR', 'LLAMA_TRACE',
                   'BOLT_TRACE_TENSORS', 'GGML_SYCL_DEBUG')


class FidelityFailure(Exception):
    """A valid response differed from the frozen no-draft contract."""


class InfraFailure(Exception):
    """A lifecycle, source, isolation, transport, or API-contract failure."""


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(stem, suffix):
    return Path(str(stem) + suffix)


def helpers():
    """Load frozen helpers without modifying their module globals."""
    off = runpy.run_path(str(OFF / 'contracts.py'))
    return {
        'off': off,
        'http': runpy.run_path(str(OFF / 'run-off.py')),
        'analysis': runpy.run_path(str(OFF / 'analyze-off.py')),
        'runner': runpy.run_path(str(off['OLD']['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py')),
        'api': runpy.run_path(str(off['OLD']['OLD'] / 'probe-api.py')),
        'grouped': runpy.run_path(str(off['OLD']['GROUPED'] / 'run-grouped-diagnostic.py')),
        'sampler': runpy.run_path(str(SAMPLER)),
    }


def cell_label(cell):
    assert cell in LABELS, cell
    return LABELS[cell]


def width_for(cell):
    return WIDTHS.get(cell)


def candidate_cell(cell):
    return cell != 'A' and cell in LABELS


def server_argv(argv):
    assert argv[-3:-1] == ['bash', '-lc']
    command = argv[-1]
    marker = '; exec '
    assert marker in command
    return shlex.split(command.rsplit(marker, 1)[1])


def _replace_width_flag(command, width):
    assert '--draft-max' not in command
    matches = list(re.finditer(r'(--spec-draft-n-max)(\s+)(\d+)(?=\s|$)', command))
    assert len(matches) == 1, 'expected one current --spec-draft-n-max flag'
    assert matches[0].group(3) == '1', 'immutable diagnostics-off candidate is not width1'
    start, end = matches[0].span(3)
    return command[:start] + str(width) + command[end:]


def diagnostics_off_server(command):
    # The diagnostics-off helper deliberately leaves route/native evidence on,
    # but all sampler/capture diagnostics must stay disabled in this case.
    server = command.rsplit('; exec ', 1)[1]
    boundary = command.rsplit('; exec ', 1)[0]
    assert 'BOLT_VERIFY_CAPTURE_DIR' not in server
    assert 'LLAMA_TRACE' not in server
    assert 'BOLT_TENSOR_TRACE' not in server
    assert 'GGML_SYCL_DEBUG=1' not in boundary.rsplit(';', 1)[-1]
    assert 'BOLT_TRACE_TENSORS=1' not in server
    assert 'GGML_SYCL_DEBUG=0' in boundary
    assert '/capture' not in command
    return True


def make_create(cell, h):
    """Derive a launch argv from immutable diagnostics-off argv.

    The only allowed changes are the case-local container name and, for W3/W7,
    the intentional current ``--spec-draft-n-max`` value.  No legacy flag or
    diagnostics switch is introduced here.
    """
    assert cell in LABELS
    off = h['off']
    base = list(off['make_create'](cell_label(cell)))
    argv = list(base)
    name_index = argv.index('--name') + 1
    argv[name_index] = PREFIX + cell.lower()
    if width_for(cell) is not None:
        argv[-1] = _replace_width_flag(argv[-1], width_for(cell))
    diagnostics_off_server(argv[-1])
    assert all('/capture' not in str(word) for word in argv)
    changed = [i for i, (left, right) in enumerate(zip(base, argv)) if left != right]
    allowed = [name_index] + ([len(argv) - 1] if width_for(cell) is not None else [])
    assert changed == allowed, (cell, changed, allowed)
    if width_for(cell) is not None:
        old = _replace_width_flag(base[-1], width_for(cell))
        # The helper's command is otherwise byte-for-byte unchanged.
        assert old == argv[-1]
        assert '--spec-draft-n-min 0' in argv[-1]
        assert '--spec-draft-p-min 0' in argv[-1]
    return argv


def expected_pid_argv(cell, h):
    return server_argv(make_create(cell, h))


def argv_diff(base, derived):
    """Return the complete positional diff, retaining shell command bytes."""
    assert len(base) == len(derived)
    return [{'index': i, 'base': left, 'derived': right}
            for i, (left, right) in enumerate(zip(base, derived)) if left != right]


def frozen_jobs(h):
    jobs = h['off']['frozen_jobs']()
    assert len(jobs) == 27
    by_stem = {job['stem']: job for job in jobs}
    quality = [by_stem['quality/' + name] for name in
               [job['stem'].split('/', 1)[1] for job in jobs if job['stem'].startswith('quality/')]]
    assert len(quality) == 20 and len({j['stem'] for j in quality}) == 20
    selected = {
        'quality': quality,
        'anchor': by_stem['anchor'],
        'short': {
            'ordinary': by_stem['short/ordinary'],
            'diagnostic': by_stem['short/diagnostic'],
        },
        'long': {
            'ordinary': by_stem['long/ordinary'],
            'diagnostic': by_stem['long/diagnostic'],
        },
    }
    for job in quality:
        payload = job['payload']
        assert payload['max_tokens'] == 256 and payload['temperature'] == 0
        assert payload['seed'] == 42 and payload['stream'] is False
        assert payload['ignore_eos'] is False and payload['cache_prompt'] is False
    assert selected['anchor']['payload']['max_tokens'] == 128
    for point in POINTS:
        ordinary = selected[point]['ordinary']['payload']
        diagnostic = selected[point]['diagnostic']['payload']
        assert ordinary['max_tokens'] == diagnostic['max_tokens'] == 128
        assert ordinary['temperature'] == diagnostic['temperature'] == 0
        assert ordinary['seed'] == diagnostic['seed'] == 42
        assert ordinary['cache_prompt'] is False
        assert diagnostic['cache_prompt'] is False
        assert ordinary['stream'] is True
        assert ordinary['stream_options'] == {'include_usage': True}
        assert ordinary['ignore_eos'] is True
        assert diagnostic['stream'] is False
        assert diagnostic['ignore_eos'] is True
    return selected


def qualification_jobs(h):
    selected = frozen_jobs(h)
    return (list(selected['quality']) + [selected['anchor'],
            selected['short']['ordinary'], selected['short']['diagnostic'],
            selected['long']['ordinary'], selected['long']['diagnostic']])


def timing_jobs(h):
    selected = frozen_jobs(h)
    return {point: selected[point]['ordinary'] for point in POINTS}


def qualification_kinds(job):
    stem = job['stem']
    if stem.startswith('quality/'):
        return 'quality'
    if stem == 'anchor':
        return 'anchor'
    if stem.endswith('/diagnostic'):
        return 'diagnostic'
    if stem.endswith('/ordinary'):
        return 'ordinary'
    raise AssertionError(stem)


def schedule():
    return [dict(cell=cell, point=point, phase=phase, ordinal=ordinal)
            for cell in TIMING_ORDER for point in POINTS
            for phase, ordinal in [('warmup', 0), ('measured', 0),
                                   ('measured', 1), ('measured', 2)]]


def eligible_order(widths):
    widths = {int(width) for width in widths}
    assert widths <= {3, 7}
    return tuple(cell for cell in TIMING_ORDER
                 if cell.startswith('A') or width_for(cell) in widths)


def expected_measurements(widths):
    return [job for job in schedule() if job['phase'] == 'measured'
            and job['cell'] in eligible_order(widths)]


def canonical_mounts(mounts):
    """Canonicalize order only; retain every mount object and duplicate."""
    assert isinstance(mounts, list)
    encoded = [json.dumps(mount, sort_keys=True, separators=(',', ':'))
               for mount in mounts]
    return sorted(encoded)


def same_container(expected, actual):
    assert isinstance(expected, dict) and isinstance(actual, dict)
    for key in ('Id', 'Config', 'HostConfig'):
        if expected.get(key) != actual.get(key):
            return False
    return canonical_mounts(expected.get('Mounts', [])) == canonical_mounts(actual.get('Mounts', []))


def result_view(result):
    return {key: result[key] for key in ('text', 'finish_reason', 'usage')}


def first_difference(left, right):
    first = next((i + 1 for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    if first is None and len(left) != len(right):
        first = min(len(left), len(right)) + 1
    return dict(equal=left == right, first_difference_ordinal=first,
                left_length=len(left), right_length=len(right),
                left_ids=list(left), right_ids=list(right))


def diagnostic_compare(actual, expected):
    keys = ('ids', 'pieces', 'text', 'finish_reason', 'usage')
    result = {key: actual.get(key) == expected.get(key) for key in keys}
    result['passed'] = all(result.values())
    result['id_comparison'] = first_difference(actual.get('ids', []), expected.get('ids', []))
    # Counts/probabilities are deliberately absent from equality.
    result['proposal_acceptance_counts_compared'] = False
    result['probabilities_compared'] = False
    result['raw_f32_equality_claimed'] = False
    return result


def check_output(result, job, expected=None, prompt_ids=None):
    kind = qualification_kinds(job)
    usage = result['usage']
    if prompt_ids is not None:
        assert usage['prompt_tokens'] == len(prompt_ids)
    else:
        assert usage['prompt_tokens'] > 0
    assert usage['completion_tokens'] > 0
    assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
    if kind == 'quality':
        assert result['finish_reason'] == 'stop'
    elif kind == 'anchor':
        assert result['finish_reason'] == 'stop'
    else:
        assert usage['completion_tokens'] == 128
        assert result['finish_reason'] == 'length'
        assert usage['total_tokens'] == usage['prompt_tokens'] + 128
        assert result['timings']['cache_n'] == 0
        assert result['timings']['predicted_n'] == 128
    if expected is not None and result_view(result) != result_view(expected):
        raise FidelityFailure('text/finish/usage differs from matched no-draft')
    return True


def check_functional(h, job, result):
    kind = qualification_kinds(job)
    if kind == 'quality':
        return bool(h['off']['L']['functional_quality'](job['stem'].split('/', 1)[1], result))
    if kind == 'anchor':
        return bool(h['off']['L']['functional_anchor'](result))
    return (result['finish_reason'] == 'length' and
            result['usage']['completion_tokens'] == 128 and
            result['timings']['cache_n'] == 0)


def exposed_metrics(result):
    metrics = {key: result.get('timings', {}).get(key) for key in METRICS[:4] + METRICS[-2:]}
    metrics['draft_n'] = result.get('timings', {}).get('draft_n')
    metrics['draft_n_accepted'] = result.get('timings', {}).get('draft_n_accepted')
    return metrics


def timing_metrics(result):
    elapsed, first = result['elapsed_s'], result['ttfc_s']
    assert isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0
    assert first is not None and math.isfinite(first) and 0 <= first < elapsed
    count = result['usage']['completion_tokens']
    values = exposed_metrics(result)
    for key in METRICS[:4]:
        value = values[key]
        assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    values.update(ttfc_s=first, elapsed_s=elapsed,
                  end_to_end_output_tps=count / elapsed,
                  first_content_decode_proxy_tps=(count - 1) / (elapsed - first))
    return values


def describe(values):
    """Hyndman-Fan type 7 quartiles; no significance calculation."""
    values = sorted(values)
    assert values and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values)
    def q(percentile):
        position = (len(values) - 1) * percentile
        low, high = int(position // 1), int(-(-position // 1))
        return values[low] + (values[high] - values[low]) * (position - low)
    return dict(n=len(values), median=q(.5), q1=q(.25), q3=q(.75),
                iqr=q(.75) - q(.25))


def aggregate(rows):
    result = {}
    for metric in METRICS:
        values = [row['metrics'].get(metric) for row in rows
                  if row.get('metrics', {}).get(metric) is not None]
        result[metric] = dict(available=len(values), missing=len(rows) - len(values),
                              summary=describe(values) if values else None)
    return result


def telemetry(samples, window_s):
    result = dict(samples=samples, window_duration_s=window_s, sample_span_s=None,
                  counter_caveat='unknown wraps/resets cannot be corrected; no peak VRAM/occupancy/bandwidth',
                  peak_vram_bytes=None, occupancy=None, bandwidth=None)
    if len(samples) >= 2:
        span = samples[-1]['t'] - samples[0]['t']
        result['sample_span_s'] = span if span > 0 else None
    for key, name in (('energy1_uj', 'card'), ('energy2_uj', 'pkg')):
        values = [sample.get(key) for sample in samples]
        reason = None
        if len(values) < 2 or not result['sample_span_s']:
            reason = 'unavailable: insufficient samples'
        elif any(value is None for value in values):
            reason = 'unavailable: missing counter samples'
        elif any(right < left for left, right in zip(values, values[1:])):
            reason = 'unavailable: counter reset or wrap (modulus unknown)'
        energy = None if reason else (values[-1] - values[0]) / 1e6
        result[name + '_j'] = energy
        result[name + '_w'] = None if energy is None else energy / result['sample_span_s']
        result[name + '_status'] = reason or 'observed counter delta; undetected full wraps not excluded'
    for key in ('freq0_mhz', 'freq1_mhz', 'temp2_mc'):
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        result[key] = dict(summary=describe(values) if values else None,
                           missing=len(samples) - len(values))
    return result


def capacity(path=ROOT, free=None):
    available = shutil.disk_usage(path).free if free is None else free
    return dict(free_bytes=available, reserve_bytes=RESERVE,
                evidence_allowance_bytes=ALLOWANCE, required_free=RESERVE + ALLOWANCE,
                fits=available >= RESERVE + ALLOWANCE, raw_capture_permitted=False)


def fresh_logs(before, after):
    assert set(before) == set(after) == {'stdout', 'stderr'}
    result = {}
    for stream in before:
        assert after[stream].startswith(before[stream]), ('log prefix changed', stream)
        result[stream] = after[stream][len(before[stream]):]
    return result


def pid_evidence(environ, cmdline, exe, rows):
    selected = []
    for item in environ.split(b'\0'):
        key, sep, value = item.partition(b'=')
        if sep and key.decode('ascii', errors='replace') in DIAGNOSTIC_KEYS:
            selected.append([key.decode('ascii'), value.decode('ascii')])
    return dict(pid=1, origin='/proc/1/environ', diagnostics=selected,
                argv=[value.decode() for value in cmdline.split(b'\0') if value],
                exe=exe, scoped_tmp_rows=rows)


PID_PROBE = ("import json,os; from pathlib import Path\nDIAGNOSTIC_KEYS=" + repr(DIAGNOSTIC_KEYS) + '\n'
             + inspect.getsource(pid_evidence) + '\n'
             + "print(json.dumps(pid_evidence(Path('/proc/1/environ').read_bytes(), "
             "Path('/proc/1/cmdline').read_bytes(), os.readlink('/proc/1/exe'), "
             "sorted(p.name for p in Path('/tmp').glob('row-??????')))))")


def check_pid(record, cell, h):
    assert record['pid'] == 1 and record['origin'] == '/proc/1/environ'
    pairs = record['diagnostics']
    assert all(len(pair) == 2 and pair[0] in DIAGNOSTIC_KEYS for pair in pairs)
    env = dict(pairs)
    assert len(env) == len(pairs)
    assert 'BOLT_VERIFY_CAPTURE_DIR' not in env and 'LLAMA_TRACE' not in env
    assert env.get('BOLT_TRACE_TENSORS') == '0'
    assert env.get('GGML_SYCL_DEBUG') == '0'
    assert record['exe'] == '/src/build-sycl/bin/llama-server'
    assert record['argv'] == expected_pid_argv(cell, h)
    assert record['scoped_tmp_rows'] == []
    return True


def source_evidence(h):
    # The imported helper owns the accepted five-binary/source pin set.  This
    # call only reads it; callers save both this result and the exact hash map.
    return h['off']['source_contracts']()


def source_binary_hashes(h):
    return h['off']['N']['check_sources_and_binaries']()


def code_hashes(h):
    paths = {str(ROOT / name): sha(ROOT / name) for name in LOCAL_FILES}
    paths.update(h['off']['code_hashes']())
    paths[str(SAMPLER)] = sha(SAMPLER)
    return paths


def reference_hashes(h):
    # The diagnostics-off helper already hashes all frozen payload/response,
    # prompt, source-proof, and accepted/candidate reference artifacts.
    return h['off']['reference_hashes'](h['off']['frozen_jobs']())


def check_tested(h):
    tested = load(ROOT / 'preparation/tested-code.json')
    assert tested['tests_passed'] is True
    assert tested['hashes'] == code_hashes(h), 'controller/helper changed since CPU checks'
    assert tested['references'] == reference_hashes(h), 'frozen reference drift since CPU checks'
    return tested


def natural_and_length_contract(job, result):
    kind = qualification_kinds(job)
    if kind in ('quality', 'anchor'):
        return result['finish_reason'] == 'stop'
    return (result['finish_reason'] == 'length' and
            result['usage']['completion_tokens'] == 128 and
            result['timings']['cache_n'] == 0)


def check_speculation_counts(view):
    counts = view.get('proposal_acceptance_counts', {})
    draft = counts.get('draft_n')
    accepted = counts.get('draft_n_accepted')
    valid = (isinstance(draft, int) and isinstance(accepted, int) and
             draft > 0 and 0 <= accepted <= draft)
    return dict(exercised=valid, draft_n=draft, draft_n_accepted=accepted,
                rejected_draft_observed=bool(valid and accepted < draft),
                no_synthetic_acceptance=True)


def log_duration_stats(log_text):
    """Extract only duration fields actually emitted in existing server logs."""
    found = []
    known = ('draft_ms', 'draft_duration_ms', 'speculative_ms',
             'speculative_duration_ms', 'acceptance_ms')
    for line_number, line in enumerate(log_text.splitlines(), 1):
        if not any(name in line for name in known):
            continue
        found.append(dict(line=line_number, text=line,
                          fields={name: line.count(name) for name in known if name in line}))
    return dict(available=bool(found), records=found,
                source='server logs only; no diagnostic/profile hook added')
EOF