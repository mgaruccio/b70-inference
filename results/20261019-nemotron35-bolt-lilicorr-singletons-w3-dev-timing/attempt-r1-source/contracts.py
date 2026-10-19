#!/usr/bin/env python3
"""Small, case-local contracts for the W3 development timing adapter.

This module is deliberately inert on import.  The qualified singleton gate and
its width-sweep lifecycle are loaded by the lead-only runner; this file only
contains the new cell names, timing shape, and narrow validation helpers.
"""
import hashlib
import itertools
import json
import math
import re
import shlex
import shutil
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
# The physical destination used by the lead on inference-host.  PACKAGE_ROOT
# is used for read-only sibling imports so CPU fixtures can run from a checkout.
ROOT = Path('/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing')
QUALIFIED_PACKAGE = PACKAGE_ROOT.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons'
WIDTH_SWEEP_PACKAGE = PACKAGE_ROOT.parent / '20261018-nemotron35-bolt-lilicorr-width-sweep'

PREFIX = 'bolt-lilicorr-singletons-w3-dev-timing-r1-'
ORIGINAL = 'nemotron-bolt-probe'
CELLS = ('VSPF-A1', 'VSPF-W3a', 'VSPF-W3b', 'VSPF-A2')
ROLES = {'VSPF-A1': 'A', 'VSPF-W3a': 'W3', 'VSPF-W3b': 'W3', 'VSPF-A2': 'A'}
BASE_CELLS = {'A': 'VSB-A', 'W3': 'VSB-W3'}
QUALIFIED_CELLS = {'A': 'VSB-A', 'W3': 'VSB-W3', 'W7': 'VSB-W7'}
POINTS = ('short', 'long')
METRICS = (
    'predicted_ms', 'predicted_per_second', 'prompt_ms', 'prompt_per_second',
    'ttfc_s', 'elapsed_s', 'end_to_end_output_tps',
    'first_content_decode_proxy_tps', 'draft_n', 'draft_n_accepted',
)
HTTP_LIMIT = 4 * 1024**2
RESERVE = 4 * 1024**3
ALLOWANCE = 1 * 1024**3
ORDER = CELLS
TIMING_ORDER = CELLS
ARMS = ('A', 'W3')
MODE = 'timing'
TIER = 'development'
OUT = ROOT / 'run-r1'
# These are the adapter files whose bytes are recorded before a lead run and
# checked again by the independent analyzer.  Existing qualified/source pins
# are checked through public-gate.py; this is only the adapter's own code.
FILES = ('contracts.py', 'controller.py', 'analyze-timing.py', 'test-timing.py',
         'cpu-checks.sh', 'README.md')


class FidelityFailure(Exception):
    """A valid response differed from the matched no-draft contract."""


class InfraFailure(Exception):
    """A lifecycle, source, transport, or public-boundary failure."""


_P = None
_BASE = None
PID_PROBE = None


def configure(public_gate):
    """Bind the read-only qualified gate namespace for a lead run.

    ``public_gate`` is a runpy namespace, not a modified module.  In
    particular, callers never replace its ROOT, ROLES, or frozen ancestry.
    """
    global _P, _BASE
    assert public_gate is not None
    _P = public_gate
    _BASE = public_gate['C']
    if 'PID_PROBE' in _BASE:
        globals()['PID_PROBE'] = _BASE['PID_PROBE']
    return public_gate


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


def adapter_code_hashes():
    paths = [PACKAGE_ROOT / filename for filename in FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    assert not missing, ('declared adapter files missing', missing)
    return {str(path): sha(path) for path in paths}


def check_adapter_code(expected):
    actual = adapter_code_hashes()
    assert actual == expected, ('adapter code drift', {'expected': expected, 'actual': actual})
    return actual


def _public():
    if _P is None:
        raise RuntimeError('qualified public gate has not been configured')
    return _P


def _base():
    if _BASE is None:
        raise RuntimeError('qualified public gate has not been configured')
    return _BASE


def role_for(cell):
    assert cell in ROLES, cell
    return ROLES[cell]


def base_cell(cell):
    return BASE_CELLS[role_for(cell)]


def qualified_dir(role):
    assert role in QUALIFIED_CELLS
    return _public()['OUT'] / QUALIFIED_CELLS[role]


def name(cell):
    assert cell in CELLS, cell
    return PREFIX + cell.lower()


def planned_names(cells=CELLS):
    assert tuple(cells) == CELLS or set(cells) <= set(CELLS)
    result = [name(cell) for cell in cells]
    assert len(result) == len(set(result))
    return result


def check_name_collisions(existing, *_unused, **_unused_kwargs):
    existing = set(existing)
    planned = planned_names()
    collisions = sorted(set(planned) & existing)
    assert not collisions, ('planned timing container names already exist', collisions)
    return dict(mode='timing', planned_cells=list(CELLS), planned_names=planned,
                collisions=collisions, excluded_widths=[7])


def eligible_order(qualified_widths):
    """Return the fixed ABBA order; W7 is explicitly not eligible here."""
    widths = {int(width) for width in qualified_widths}
    assert widths == {3}, 'this package is W3-only; W7 is excluded'
    return CELLS


def schedule():
    return [dict(cell=cell, point=point, phase=phase, ordinal=ordinal)
            for cell in CELLS for point in POINTS
            for phase, ordinal in (('warmup', 0), ('measured', 0),
                                   ('measured', 1), ('measured', 2))]


def expected_measurements():
    return [job for job in schedule() if job['phase'] == 'measured']


def argv_diff(base, derived):
    """Keep every positional argv difference, including appended arguments."""
    return [dict(index=index, base=left, derived=right)
            for index, (left, right) in enumerate(
                itertools.zip_longest(base, derived)) if left != right]


def server_argv(argv):
    assert argv[-3:-1] == ['bash', '-lc']
    marker = '; exec '
    assert marker in argv[-1]
    return shlex.split(argv[-1].rsplit(marker, 1)[1])


def make_create(cell, h):
    """Use the qualified P launch exactly, changing only the container name."""
    p = _public()
    base = list(p['make_create'](base_cell(cell), h))
    derived = list(base)
    index = derived.index('--name') + 1
    derived[index] = name(cell)
    assert argv_diff(base, derived) == [dict(index=index, base=base[index],
                                             derived=name(cell))]
    # The server process argv is byte-for-byte the qualified source argv.
    assert server_argv(base) == server_argv(derived)
    return derived


def expected_pid_argv(cell, h):
    return server_argv(make_create(cell, h))


def qualification_kinds(job):
    if _BASE is not None:
        return _BASE['qualification_kinds'](job)
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


def qualification_jobs(h):
    """The exact current 25-job P set, never a locally invented prompt set."""
    jobs = _public()['jobs'](h)
    assert len(jobs) == 25
    assert [job['stem'] for job in jobs[-4:]] == [
        'short/ordinary', 'short/diagnostic', 'long/ordinary', 'long/diagnostic']
    return jobs


def cell_label(cell):
    """Compatibility for inherited read-only helpers; no old cell is launched."""
    if cell in CELLS:
        return base_cell(cell)
    return cell


def width_for(cell):
    # The adapter intentionally has no width sweep; W3 is selected by role.
    return None


def check_width_startup(log, cell, h):
    return _public()['check_startup'](log, base_cell(cell), h)


def check_tested(h):
    return _public()['check_tested'](h)

def timing_jobs(h):
    jobs = qualification_jobs(h)
    selected = {job['stem']: job for job in jobs}
    result = {point: selected[point + '/ordinary'] for point in POINTS}
    for point, job in result.items():
        payload = job['payload']
        assert payload['max_tokens'] == 128 and payload['temperature'] == 0
        assert payload['seed'] == 42 and payload['ignore_eos'] is True
        assert payload['cache_prompt'] is False and payload['stream'] is True
        assert payload['stream_options'] == {'include_usage': True}
        assert job['historical_prompt_tokens'] == (55 if point == 'short' else 8270)
    return result


def check_output(result, job, expected=None, prompt_ids=None):
    """Delegate output invariants to the existing width-sweep contract."""
    if _BASE is not None:
        return _BASE['check_output'](result, job, expected, prompt_ids)
    # Small fixture fallback used only when the remote qualified gate is absent.
    usage = result['usage']
    assert usage['completion_tokens'] == 128
    assert usage['total_tokens'] == usage['prompt_tokens'] + 128
    assert result['finish_reason'] == 'length'
    assert result['timings']['cache_n'] == 0
    assert result['timings']['predicted_n'] == 128
    if expected is not None:
        assert result_view(result) == result_view(expected)
    return True

def first_difference(left, right):
    first = next((index + 1 for index, (a, b) in enumerate(zip(left, right))
                  if a != b), None)
    if first is None and len(left) != len(right):
        first = min(len(left), len(right)) + 1
    return dict(equal=left == right, first_difference_ordinal=first,
                left_length=len(left), right_length=len(right),
                left_ids=list(left), right_ids=list(right))


def diagnostic_compare(actual, expected):
    if _BASE is not None:
        return _BASE['diagnostic_compare'](actual, expected)
    keys = ('ids', 'pieces', 'text', 'finish_reason', 'usage')
    result = {key: actual.get(key) == expected.get(key) for key in keys}
    result['passed'] = all(result.values())
    result['id_comparison'] = first_difference(actual.get('ids', []), expected.get('ids', []))
    result['proposal_acceptance_counts_compared'] = False
    result['probabilities_compared'] = False
    result['raw_f32_equality_claimed'] = False
    return result


def check_speculation_counts(view):
    if _BASE is not None:
        return _BASE['check_speculation_counts'](view)
    counts = view.get('proposal_acceptance_counts', {})
    draft, accepted = counts.get('draft_n'), counts.get('draft_n_accepted')
    valid = isinstance(draft, int) and isinstance(accepted, int) and draft > 0 and 0 <= accepted <= draft
    return dict(exercised=valid, draft_n=draft, draft_n_accepted=accepted,
                rejected_draft_observed=bool(valid and accepted < draft),
                no_synthetic_acceptance=True)


def check_functional(h, job, result):
    if _BASE is not None:
        return _BASE['check_functional'](h, job, result)
    return result['finish_reason'] == 'length' and result['usage']['completion_tokens'] == 128


def result_view(result):
    return {key: result[key] for key in ('text', 'finish_reason', 'usage')}


def counter_metrics(result):
    """Expose counters as unavailable when absent; never turn absence into zero."""
    timings = result.get('timings', {})
    draft = timings.get('draft_n')
    accepted = timings.get('draft_n_accepted')
    for value, key in ((draft, 'draft_n'), (accepted, 'draft_n_accepted')):
        if value is not None:
            assert isinstance(value, (int, float)) and math.isfinite(value), key
            assert value >= 0 and int(value) == value, key
    if draft is not None and accepted is not None:
        assert accepted <= draft
    return dict(draft_n=draft, draft_n_accepted=accepted)


def timing_metrics(result):
    elapsed = result['elapsed_s']
    first = result['ttfc_s']
    assert isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0
    assert first is not None and math.isfinite(first) and 0 <= first < elapsed
    timings = result['timings']
    values = {key: timings.get(key) for key in METRICS[:4]}
    for key, value in values.items():
        assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0, key
    count = result['usage']['completion_tokens']
    values.update(counter_metrics(result))
    values.update(
        ttfc_s=first,
        elapsed_s=elapsed,
        end_to_end_output_tps=count / elapsed,
        # This is retained only as the historical first-content proxy.  It is
        # never presented as server decode/ITL speed by the analyzer.
        first_content_decode_proxy_tps=(count - 1) / (elapsed - first),
    )
    return values


def describe(values):
    """Hyndman-Fan type 7 quartiles; no significance calculation."""
    values = sorted(values)
    assert values and all(isinstance(value, (int, float)) and math.isfinite(value)
                          for value in values)

    def quantile(percentile):
        position = (len(values) - 1) * percentile
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return dict(n=len(values), median=quantile(.5), q1=quantile(.25),
                q3=quantile(.75), iqr=quantile(.75) - quantile(.25))


def aggregate(rows):
    result = {}
    for metric in METRICS:
        values = [row.get('metrics', {}).get(metric) for row in rows
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
    for key, name_ in (('energy1_uj', 'card'), ('energy2_uj', 'pkg')):
        values = [sample.get(key) for sample in samples]
        reason = None
        if len(values) < 2 or not result['sample_span_s']:
            reason = 'unavailable: insufficient samples'
        elif any(value is None for value in values):
            reason = 'unavailable: missing counter samples'
        elif any(right < left for left, right in zip(values, values[1:])):
            reason = 'unavailable: counter reset or wrap (modulus unknown)'
        energy = None if reason else (values[-1] - values[0]) / 1e6
        result[name_ + '_j'] = energy
        result[name_ + '_w'] = (None if energy is None
                                 else energy / result['sample_span_s'])
        result[name_ + '_status'] = reason or 'observed counter delta; undetected full wraps not excluded'
    for key in ('freq0_mhz', 'freq1_mhz', 'temp2_mc'):
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        result[key] = dict(summary=describe(values) if values else None,
                           missing=len(samples) - len(values))
    return result


def capacity(path=ROOT, free=None):
    available = shutil.disk_usage(path).free if free is None else free
    return dict(free_bytes=available, reserve_bytes=RESERVE,
                evidence_allowance_bytes=ALLOWANCE,
                required_free=RESERVE + ALLOWANCE,
                fits=available >= RESERVE + ALLOWANCE,
                raw_capture_permitted=False)


def fresh_logs(before, after):
    assert set(before) == set(after) == {'stdout', 'stderr'}
    result = {}
    for stream in before:
        assert after[stream].startswith(before[stream]), ('log prefix changed', stream)
        result[stream] = after[stream][len(before[stream]):]
    return result


def canonical_mounts(mounts):
    assert isinstance(mounts, list)
    return sorted(json.dumps(mount, sort_keys=True, separators=(',', ':'))
                  for mount in mounts)


def same_container(expected, actual):
    assert isinstance(expected, dict) and isinstance(actual, dict)
    for key in ('Id', 'Config', 'HostConfig'):
        if expected.get(key) != actual.get(key):
            return False
    return canonical_mounts(expected.get('Mounts', [])) == canonical_mounts(actual.get('Mounts', []))


def _env_map(values):
    result = {}
    for entry in values:
        key, separator, value = entry.partition('=')
        assert separator and key and key not in result
        result[key] = value
    return result


def check_container(record, cell, h):
    """Validate the actual Docker record against the exact new-name argv.

    The qualified A/W3 container is the immutable reference.  No recorded
    object is normalized or rewritten; only Docker list ordering and the
    already-accepted ``OomKillDisable`` null/false spelling are tolerated.
    """
    p = _public()
    role = role_for(cell)
    base = load(qualified_dir(role) / 'container-before.json')
    assert record['Name'] == '/' + name(cell)
    assert record['Config']['Cmd'] == make_create(cell, h)[-3:]
    assert re.fullmatch(r'[0-9a-f]{64}', record['Id'])
    assert record['Config']['Hostname'] == record['Id'][:12]
    for key in ('Cmd', 'Env', 'Hostname'):
        actual = record['Config'].get(key)
        expected = base['Config'].get(key)
        if key == 'Cmd':
            continue
        if key == 'Env':
            assert _env_map(actual) == _env_map(expected)
        else:
            assert actual == expected
    expected_config = {key: value for key, value in base['Config'].items()
                       if key not in ('Cmd', 'Env', 'Hostname')}
    actual_config = {key: value for key, value in record['Config'].items()
                     if key not in ('Cmd', 'Env', 'Hostname')}
    assert actual_config == expected_config
    expected_host = dict(base['HostConfig'])
    actual_host = dict(record['HostConfig'])
    for host in (expected_host, actual_host):
        value = host.pop('OomKillDisable', None)
        assert value is None or value is False
    assert actual_host == expected_host
    assert canonical_mounts(record['Mounts']) == canonical_mounts(base['Mounts'])
    return True


def check_pid(record, cell, h):
    role = role_for(cell)
    if _BASE is not None:
        return _BASE['check_pid'](record, role, h)
    assert record['argv'] == expected_pid_argv(cell, h)
    return True


def source_binary_hashes(h):
    if _BASE is not None:
        return _BASE['source_binary_hashes'](h)
    raise RuntimeError('source proof requires the qualified gate')


def source_evidence(h):
    if _BASE is not None and 'source_evidence' in _BASE:
        return _BASE['source_evidence'](h)
    raise RuntimeError('source evidence requires the qualified gate')


def log_duration_stats(log_text):
    if _BASE is not None and 'log_duration_stats' in _BASE:
        return _BASE['log_duration_stats'](log_text)
    number = r'(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    pattern = re.compile(
        r'#calls\(b,g,a\)\s*=\s*(\d+)\s+(\d+)\s+(\d+)\s*,\s*'
        r'#gen drafts\s*=\s*(\d+)\s*,\s*#acc drafts\s*=\s*(\d+)\s*,\s*'
        r'#gen tokens\s*=\s*(\d+)\s*,\s*#acc tokens\s*=\s*(\d+)\b.*?'
        r'dur\(b,g,a\)\s*=\s*(' + number + r')\s*,\s*(' + number +
        r')\s*,\s*(' + number + r')\s*ms')
    records = []
    for line_number, line in enumerate(log_text.splitlines(), 1):
        if '#calls(b,g,a)' not in line:
            continue
        match = pattern.search(line)
        assert match, ('unparseable cumulative server statistics', line)
        records.append(dict(line=line_number, text=line, cumulative=True,
                            calls=dict(zip(('b', 'g', 'a'),
                                           (int(value) for value in match.group(1, 2, 3)))),
                            generated=dict(drafts=int(match.group(4)), tokens=int(match.group(6))),
                            accepted=dict(drafts=int(match.group(5)), tokens=int(match.group(7))),
                            duration_ms=dict(zip(('b', 'g', 'a'),
                                                 (float(value) for value in match.group(8, 9, 10))))))
    return dict(available=bool(records), records=records, cumulative=True,
                per_request_durations_derived=False,
                source='cumulative server log records only')
