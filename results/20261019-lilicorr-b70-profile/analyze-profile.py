#!/usr/bin/env python3
"""Small fail-closed reader of the real server's sequential request/log boundaries.

Run only after the inherited independent restored/raw-HTTP analysis. Host wall
intervals are not GPU durations. Ubatch intervals are NESTED in submit intervals.
"""
import re
from collections import Counter
from pathlib import Path

MARKER = 'LLAMA_LILICORR_PROFILE '
DURATIONS = {
    'process_batch': (),
    'process_chunk': ('gather_us', 'prepare_us', 'submit_us', 'projected_wait_us', 'ring_us'),
    'draft_batch': ('prepare_us', 'submit_us'),
    'draft_select': ('lattice_wait_us', 'walk_us'),
    'ubatch': ('build_alloc_us', 'inputs_us', 'submit_us'),
}
EXTRA = {
    'process_batch': (),
    'process_chunk': ('seq', 'batch_rows', 'seq_rows', 'offset', 'pos0', 'pos1', 'projected'),
    'draft_batch': ('rc',),
    'draft_select': ('seq', 'pos0', 'selected', 'limit'),
    'ubatch': ('gtype', 'nodes', 'reused', 'rc'),
}
START = re.compile(r'slot\s+launch_slot_:\s+id\s+(\d+)\s+\| task (\d+) \| processing task, is_child = 0')
END = re.compile(r'slot\s+release:\s+id\s+(\d+)\s+\| task (\d+) \| stop processing:')
PROMPT = re.compile(r'new prompt,.*task\.n_tokens = (\d+)')
EXPECTED = {'short': dict(prompt=55, proposed=129, accepted=84),
            'long': dict(prompt=8270, proposed=133, accepted=82)}


def event(line):
    assert line.count(MARKER) == 1
    fields = [part.split('=') for part in line.split(MARKER)[1].split()]
    assert all(len(part) == 2 for part in fields)
    row = dict(fields)
    assert len(row) == len(fields), 'duplicate profile field'
    stage = row['stage']
    assert stage in DURATIONS, stage
    assert set(row) == {'v', 'stage', 'rows', 'begin_us', 'end_us', *DURATIONS[stage], *EXTRA[stage]}
    for key in row.keys() - {'stage'}:
        assert re.fullmatch(r'-?\d+' if key == 'limit' else r'\d+', row[key]), (key, row[key])
        row[key] = int(row[key])
    assert row['v'] == 1 and row['rows'] > 0
    assert 0 < row['begin_us'] <= row['end_us']
    if DURATIONS[stage]:
        assert sum(row[key] for key in DURATIONS[stage]) == row['end_us'] - row['begin_us']
    if 'seq' in row:
        assert row['seq'] == 0, 'this experiment is single-slot/sequential only'
    if 'rc' in row:
        assert row['rc'] == 0, 'profiled operation failed'
    if stage == 'ubatch':
        assert row['nodes'] > 0 and row['reused'] in (0, 1)
    if stage == 'process_chunk':
        assert row['projected'] == 1
        assert row['batch_rows'] == row['seq_rows'] >= row['offset'] + row['rows']
        assert row['pos1'] - row['pos0'] + 1 == row['rows']
    if stage.startswith('draft_'):
        assert row['rows'] == 8, 'full eight-row LiliCorr block required'
    if stage == 'draft_select':
        assert 1 <= row['selected'] <= 3
    return row


def requests(text):
    """Existing launch/release markers, not injected or sanitized log observations."""
    result, startup, current = [], [], None
    for number, line in enumerate(text.splitlines(), 1):
        start, end = START.search(line), END.search(line)
        if start:
            assert current is None, 'overlapping requests'
            slot, task = map(int, start.groups())
            assert slot == 0 and (not result or task > result[-1]['task'])
            current = dict(task=task, slot=slot, first_line=number, events=[], prompts=[], lines=[])
        if MARKER in line:
            parsed = dict(event(line), line=number)
            if current is None:
                assert not result, 'profile event outside a request after startup'
                startup.append(parsed)
            else:
                current['events'].append(parsed)
        if current is not None:
            current['lines'].append(line)
            prompt = PROMPT.search(line)
            if prompt:
                current['prompts'].append(int(prompt.group(1)))
        if end:
            assert current is not None and tuple(map(int, end.groups())) == (current['slot'], current['task'])
            assert len(current['prompts']) == 1
            current['last_line'] = number
            result.append(current)
            current = None
    assert current is None, 'truncated final request'
    assert len(result) == 33, ('expected 25 public + 8 timing requests', len(result))
    return result, startup


def contains(parent, child):
    return parent['begin_us'] <= child['begin_us'] <= child['end_us'] <= parent['end_us']


def request_profile(request, previous_stats, stats, expected):
    rows = request['events']
    groups = {stage: sorted((r for r in rows if r['stage'] == stage), key=lambda r: r['begin_us'])
              for stage in DURATIONS}
    assert all(groups.values()), 'missing profile stage'
    top = sorted(groups['process_batch'] + groups['draft_batch'] + groups['draft_select'],
                 key=lambda r: r['begin_us'])
    assert all(a['end_us'] <= b['begin_us'] for a, b in zip(top, top[1:])), 'overlapping host intervals'
    chunks = groups['process_chunk']
    for chunk in chunks:
        assert sum(contains(p, chunk) for p in groups['process_batch']) == 1
    for batch in groups['process_batch']:
        children = [r for r in chunks if contains(batch, r)]
        assert children and sum(r['rows'] for r in children) == batch['rows']
        offset = 0
        for child in children:
            assert child['offset'] == offset and child['batch_rows'] == batch['rows']
            offset += child['rows']
    # Each low-level event must be inside EXACTLY one existing llama_process submit.
    submissions = []
    for parent in chunks + groups['draft_batch']:
        prefix = parent.get('gather_us', 0) + parent['prepare_us']
        begin = parent['begin_us'] + prefix
        submissions.append(dict(parent, begin_us=begin, end_us=begin + parent['submit_us']))
    ubatch_sources = {}
    for child in groups['ubatch']:
        owners = [p for p in submissions if contains(p, child)]
        assert len(owners) == 1, 'orphan/ambiguous ubatch'
        ubatch_sources[child['line']] = owners[0]['stage']
    for parent in submissions:
        children = [r for r in groups['ubatch'] if contains(parent, r)]
        assert children and sum(r['rows'] for r in children) == parent['rows'], 'missing ubatch rows'
    drafts, selects = groups['draft_batch'], groups['draft_select']
    calls = stats['calls']['g'] - previous_stats['calls']['g']
    assert calls > 0 and len(drafts) == len(selects) == calls
    for index, (draft, select) in enumerate(zip(drafts, selects)):
        assert draft['end_us'] <= select['begin_us']
        assert index + 1 == len(drafts) or select['end_us'] <= drafts[index + 1]['begin_us']
    assert request['prompts'] == [expected['prompt']]
    assert stats['generated']['tokens'] - previous_stats['generated']['tokens'] == expected['proposed']
    assert stats['accepted']['tokens'] - previous_stats['accepted']['tokens'] == expected['accepted']
    # common_speculative_draft applies this existing cap AFTER the logged lattice walk.
    # Retain raw selected/limit fields; compare the derived post-cap count, not raw size.
    assert sum(min(r['selected'], r['limit']) if r['limit'] > 0 else r['selected']
               for r in selects) == expected['proposed']
    # All host sums below retain overhead. Do not add nested ubatch sums to parent sums.
    def totals(selected):
        return {stage: dict(calls=len(items), rows=sum(r['rows'] for r in items),
                            wall_us=sum(r['end_us'] - r['begin_us'] for r in items),
                            **{key: sum(r[key] for r in items) for key in DURATIONS[stage]})
                for stage in DURATIONS
                for items in [[r for r in selected if r['stage'] == stage]]}

    first_draft = drafts[0]['begin_us']
    assert all(r['end_us'] <= first_draft or r['begin_us'] >= first_draft for r in rows)
    graph_counts = Counter((ubatch_sources[r['line']], r['rows'], r['gtype'], r['nodes'], r['reused'])
                           for r in groups['ubatch'])
    return dict(task=request['task'], first_line=request['first_line'], last_line=request['last_line'],
                totals=totals(rows),
                before_first_draft=totals([r for r in rows if r['end_us'] <= first_draft]),
                from_first_draft=totals([r for r in rows if r['begin_us'] >= first_draft]),
                graphs=[dict(parent_stage=k[0], rows=k[1], gtype=k[2], nodes=k[3], reused=k[4], calls=v)
                        for k, v in sorted(graph_counts.items())],
                events=rows, counters=expected)


def analyze(out, c):
    out = Path(out)
    report = dict(status='failed', tier='development', gpu_time_claim=False,
                  speed_comparison_permitted=False, nested_ubatch_not_additive=True,
                  requests=[], startup_events={}, errors=[])
    try:
        inherited = c['load'](out / 'analysis-r1/analysis.json')
        assert inherited['status'] == 'analyzed' and not inherited['errors']
        assert c['load'](out / 'analysis-live-restoration.json')['passed']
        assert len(inherited['rows']) == 24 and all(r['passed'] for r in inherited['rows'])
        assert sum(len(r['jobs']) for r in inherited['prevalidation'].values()) == 100
        for cell in c['CELLS']:
            directory = out / cell
            stderr = (directory / 'server.stderr.log').read_text()
            stdout = (directory / 'server.stdout.log').read_text()
            assert MARKER not in stdout, 'unexpected logging stream; cannot infer cross-stream order'
            reqs, startup = requests(stderr)
            if c['role_for'](cell) == 'A':
                assert not startup and not any(r['events'] for r in reqs), 'no-draft baseline was profiled'
                continue
            report['startup_events'][cell] = startup
            stats = []
            for req in reqs:
                parsed = c['log_duration_stats']('\n'.join(req['lines']))['records']
                assert len(parsed) == 1, 'missing/duplicate cumulative speculative statistics'
                stats.append(parsed[0])
            schedule = [j for j in c['schedule']() if j['cell'] == cell]
            assert len(schedule) == 8
            for index, job in enumerate(schedule, 25):
                req, exp = reqs[index], EXPECTED[job['point']]
                parsed = request_profile(req, stats[index - 1], stats[index], exp)
                stem = directory / job['point'] / f"{job['phase']}-{job['ordinal']:02}"
                raw_result = c['load'](c['artifact'](stem, '.result.json'))
                assert raw_result['timings']['draft_n'] == exp['proposed']
                assert raw_result['timings']['draft_n_accepted'] == exp['accepted']
                report['requests'].append(dict(job, **parsed))
        assert len(report['requests']) == 16
        report['status'] = 'analyzed'
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        report['errors'].append(repr(exc))
    c['save'](out / 'analysis-r1/profile.json', report)
    return report
