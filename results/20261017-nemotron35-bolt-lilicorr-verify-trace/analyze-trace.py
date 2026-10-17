#!/usr/bin/env python3
"""Fail-closed analysis of this one S0/C0 experiment, not a general trace format."""
import array
import json
import math
import os
import re
import runpy
import socket
import stat
import sys
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-verify-trace')
WIDTH1 = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1'
FIELDS = {
    'batch': 'ctx off n tokens pos outputs seq', 'decoded': 'ctx ret',
    'sample': 'ctx smpl idx selected vocab status file', 'single': 'ctx smpl slot row id',
    'verify': 'ctx smpl slot rows drafts emitted accepted rollback replay rejection synth',
    'emit': 'ctx slot ordinal token', 'remove': 'ctx slot kind p0 p1 rollback',
    'checkpoint': 'ctx slot p0 p1 rollback',
}
LISTS = {'tokens', 'pos', 'outputs', 'rows', 'drafts', 'emitted'}
STRINGS = {'ctx', 'smpl', 'status', 'file', 'kind', 'seq'}


def load(path):
    return json.loads(path.read_text())


def records(log):
    result = []
    for line in log.splitlines():
        if 'BOLT_VERIFY' not in line:
            continue
        assert line.count('BOLT_VERIFY ') == 1, 'bad trace framing'
        tail = line.split('BOLT_VERIFY ', 1)[1]
        words = tail.split()
        assert words and words[0] in FIELDS, ('capture error or unknown record', tail)
        kind, pairs = words[0], words[1:]
        assert all(re.fullmatch(r'[a-z_][a-z_0-9]*=[^\s=]+', p) for p in pairs), tail
        fields = [p.split('=', 1) for p in pairs]
        assert len({k for k, _ in fields}) == len(fields), 'duplicate field'
        assert {k for k, _ in fields} == set(FIELDS[kind].split()), tail
        row = dict(event=kind, event_index=len(result))
        for key, value in fields:
            if key in LISTS:
                assert value == '-' or re.fullmatch(r'-?\d+(,-?\d+)*', value), tail
                row[key] = [] if value == '-' else list(map(int, value.split(',')))
            elif key in STRINGS:
                if key in ('ctx', 'smpl'):
                    assert re.fullmatch(r'0x[0-9a-fA-F]+', value), tail
                row[key] = value
            else:
                assert re.fullmatch(r'-?\d+', value), tail
                row[key] = int(value)
        result.append(row)
    assert result, 'empty diagnostic trace'
    return result


def raw_bytes(directory, sample):
    """Read exactly one regular, bounded, non-symlink mkstemp file."""
    name, vocab = sample['file'], sample['vocab']
    assert sample['status'] == 'cpu', 'uncaptured/backend sample blocks attribution'
    assert re.fullmatch(r'row-[a-zA-Z0-9]{6}', name), 'capture path outside expected basename'
    assert 6833 < vocab <= 1_000_000, 'invalid vocabulary bound'
    directory = directory.resolve(strict=True)
    path = directory / name
    assert path.resolve(strict=True).parent == directory and not path.is_symlink(), 'capture escapes directory'
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        assert stat.S_ISREG(info.st_mode) and info.st_size == 4 * vocab, 'wrong F32 capture size/type'
        data = stream.read(4 * vocab + 1)
    assert len(data) == 4 * vocab
    return data


def score_summary(data):
    assert sys.byteorder == 'little', 'experiment captures are little-endian F32'
    values = array.array('f')
    assert values.itemsize == 4
    values.frombytes(data)
    assert not any(math.isnan(x) for x in values), 'NaN raw logits'
    argmax = max(range(len(values)), key=values.__getitem__)
    def number(value):
        return value if math.isfinite(value) else str(value)
    return dict(raw_argmax=argmax, raw_max=number(values[argmax]),
                score_6833=number(values[6833]), score_4618=number(values[4618]))


def observer(ordinary, diagnostic, reference_ordinary, reference_diagnostic):
    assert ordinary['text'] == reference_ordinary['text'], 'ordinary observer mismatch'
    assert diagnostic['choices'][0]['message']['content'] == reference_diagnostic['choices'][0]['message']['content'], 'diagnostic text observer mismatch'
    assert diagnostic['__verbose']['tokens'] == reference_diagnostic['__verbose']['tokens'], 'diagnostic IDs observer mismatch'


def align(events, tokens, capture, prompt_tokens=55):
    assert len(tokens) == 128 and all(type(t) is int for t in tokens)
    batches, samples, removals, verifications, emissions = [], [], [], [], []
    pending, to_emit, current, last_verify = [], [], None, None
    context = sampler = slot_id = None
    prompt = {}
    files = set()
    for e in events:
        kind = e['event']
        if kind == 'batch':
            assert not pending and not to_emit, 'unmatched sample/emission at next decode'
            assert current is None or current['decoded'], 'decode result missing'
            if context is None:
                context = e['ctx']
            assert e['ctx'] == context and e['off'] == 0, 'unexpected context/sub-batch'
            assert 0 < e['n'] <= 1024
            assert all(len(e[k]) == e['n'] for k in ('tokens', 'pos', 'outputs'))
            seq = e['seq'].split(',')
            assert len(seq) == e['n'] and all(re.fullmatch(r'\d+', s) for s in seq)
            assert len(set(seq)) == 1, 'multi-slot batch outside frozen journey'
            if slot_id is None:
                slot_id = int(seq[0])
            assert int(seq[0]) == slot_id
            assert all(o in (0, 1) for o in e['outputs']) and all(t >= 0 for t in e['tokens'])
            assert e['pos'] == list(range(e['pos'][0], e['pos'][0] + e['n']))
            current = dict(e, decoded=False, batch_index=len(batches))
            batches.append(current)
            for pos, token in zip(e['pos'], e['tokens']):
                if pos < prompt_tokens:
                    assert pos >= 0 and (pos not in prompt or prompt[pos] == token)
                    prompt[pos] = token
            continue
        if kind == 'remove' and context is None:
            assert e['kind'] == 'prompt' and e['rollback'] == 0
            removals.append(e)
            continue
        assert current is not None and e['ctx'] == context, 'unmapped context/sample'
        if 'slot' in e:
            assert e['slot'] == slot_id
        if kind == 'decoded':
            assert not current['decoded'] and e['ret'] == 0, 'failed/repeated decode'
            current['decoded'] = True
        elif kind == 'sample':
            assert current['decoded'] and not to_emit
            if sampler is None:
                sampler = e['smpl']
            assert e['smpl'] == sampler, 'sampler changed within frozen request'
            outputs = [i for i, o in enumerate(current['outputs']) if o]
            idx = e['idx']
            if idx < 0:
                assert -idx <= len(outputs)
                row = outputs[idx]
            else:
                row = idx
            assert row in outputs, 'selected row is not a real output row'
            ordinal = len(samples) + 1
            assert ordinal <= 128 and e['selected'] == tokens[ordinal - 1], 'sample does not align to emission'
            pos = current['pos'][row]
            assert pos == prompt_tokens + ordinal - 2, 'selected row position/history mismatch'
            for p, token in zip(current['pos'][:row + 1], current['tokens'][:row + 1]):
                if p >= prompt_tokens:
                    assert token == tokens[p - prompt_tokens], 'batch draft differs from prior emitted history'
            assert e['file'] not in files, 'duplicate raw capture'
            files.add(e['file'])
            data = raw_bytes(capture, e)
            item = dict(e, ordinal=ordinal, batch_index=current['batch_index'], batch_n=current['n'],
                        batch_row=row, position=pos, input_id=current['tokens'][row], **score_summary(data))
            assert not samples or item['vocab'] == samples[0]['vocab']
            samples.append(item)
            pending.append(item)
        elif kind in ('single', 'verify'):
            assert e['smpl'] == sampler and pending and not to_emit
            if kind == 'single':
                assert len(pending) == 1 and e['row'] == pending[0]['idx'] and e['id'] == pending[0]['selected']
                to_emit = [e['id']]
                last_verify = None
            else:
                assert not e['replay'] and not e['rejection'] and not e['synth'], 'not the greedy verification path'
                assert len(e['drafts']) == 1 and len(e['rows']) == 2
                assert e['rows'] == [i for i, o in enumerate(current['outputs']) if o]
                assert e['rows'][:len(pending)] == [s['idx'] for s in pending]
                assert e['emitted'] == [s['selected'] for s in pending]
                assert e['accepted'] == len(e['emitted']) - 1
                assert e['rollback'] == len(e['drafts']) - e['accepted']
                assert e['emitted'][:e['accepted']] == e['drafts'][:e['accepted']]
                if e['rollback']:
                    assert e['emitted'][-1] != e['drafts'][e['accepted']]
                assert current['tokens'][e['rows'][1]] == e['drafts'][0]
                last_verify = dict(e, first_position=pending[0]['position'], removed=False)
                verifications.append(last_verify)
                to_emit = list(e['emitted'])
            pending = []
        elif kind == 'emit':
            assert to_emit and e['token'] == to_emit.pop(0)
            assert e['ordinal'] == len(emissions) + 1 and e['token'] == tokens[len(emissions)]
            if last_verify is not None:
                assert last_verify['removed'], 'missing real removal boundary'
            emissions.append(e)
        elif kind == 'remove':
            assert e['kind'] == 'verify' and last_verify is not None, 'unexpected history mutation'
            assert not last_verify['removed'] and e['rollback'] == last_verify['rollback']
            assert e['p1'] == -1 and e['p0'] == last_verify['first_position'] + len(last_verify['emitted'])
            removed_positions = [p for p in current['pos'] if p >= e['p0']]
            assert len(removed_positions) == e['rollback']
            removals.append(dict(e, removed_batch_positions=removed_positions))
            last_verify['removed'] = True
        elif kind == 'checkpoint':
            raise AssertionError('checkpoint/replay outside bounded width1 journey; attribution blocked')
    assert not pending and not to_emit and len(samples) == len(emissions) == 128
    assert current and current['decoded']
    assert sorted(prompt) == list(range(prompt_tokens)), 'incomplete actual prompt batch'
    assert all(e['ctx'] == context and e['slot'] == slot_id for e in removals)
    return dict(samples=samples, batches=batches, removals=removals, verifications=verifications,
                emitted_ids=tokens, prompt_ids=[prompt[i] for i in range(prompt_tokens)])


def analyze_arm(cell, label, old):
    assert label in ('S0', 'C0')
    reference = WIDTH1 / 'run-r1' / ('A0' if label == 'S0' else 'C0')
    ordinary, diagnostic = load(cell / 'ordinary.result.json'), load(cell / 'diagnostic.response.json')
    old['check_stream_result'](ordinary)
    old['check_stream_result'](load(cell / 'warmup.result.json'))
    tokens = old['check_diagnostic_result'](diagnostic)
    assert load(cell / 'diagnostic.request.json') == load(reference / 'diagnostic.request.json')
    for stem in ('warmup', 'ordinary'):
        assert load(cell / (stem + '.request.json')) == load(WIDTH1 / 'run-r1/frozen-payload.json')
    observer(ordinary, diagnostic, load(reference / 'repeat-00.result.json'), load(reference / 'diagnostic.response.json'))
    result = align(records((cell / 'diagnostic-window.log').read_text()), tokens, cell / 'capture')
    verifications = result['verifications']
    accepted, proposed = sum(v['accepted'] for v in verifications), sum(len(v['drafts']) for v in verifications)
    assert (accepted, proposed) == (diagnostic['timings'].get('draft_n_accepted', 0), diagnostic['timings'].get('draft_n', 0))
    assert (not verifications) if label == 'S0' else (proposed > 0)
    result.update(observer_passed=True, accepted=accepted, proposed=proposed,
                  rejection_events=sum(v['rollback'] > 0 for v in verifications), capture_path=str(cell / 'capture'))
    return result


def compare(left, right):
    assert left['observer_passed'] and right['observer_passed']
    assert left['prompt_ids'] == right['prompt_ids'], 'actual input prompts differ'
    a, b = left['emitted_ids'], right['emitted_ids']
    assert len(a) == len(b) == len(left['samples']) == len(right['samples']) == 128
    first_output = next((i + 1 for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
    cutoff = first_output or 128
    comparisons, first_score = [], None
    for i in range(cutoff):
        assert a[:i] == b[:i], 'raw-vector comparison beyond identical prior history'
        x, y = left['samples'][i], right['samples'][i]
        assert x['position'] == y['position'] and x['vocab'] == y['vocab']
        xb, yb = raw_bytes(Path(left['capture_path']), x), raw_bytes(Path(right['capture_path']), y)
        equal = xb == yb  # actual bytes, not hashes, top-five probabilities, or rounded scores
        row = dict(ordinal=i + 1, equal=equal, S0_batch=x['batch_index'], C0_batch=y['batch_index'],
                   S0_row=x['batch_row'], C0_row=y['batch_row'], position=x['position'])
        if not equal and first_score is None:
            first_byte = next(j for j in range(len(xb)) if xb[j] != yb[j])
            first_score = dict(row, first_differing_float_id=first_byte // 4,
                               S0=x, C0=y, S0_batch=left['batches'][x['batch_index']],
                               C0_batch=right['batches'][y['batch_index']])
            for label, arm, sample in (('S0', left, x), ('C0', right, y)):
                preceding = [r for r in arm['removals'] if r['rollback'] > 0 and r['event_index'] < sample['event_index']]
                first_score[label + '_rollback_preceded'] = bool(preceding)
                first_score[label + '_preceding_rollbacks'] = preceding
        comparisons.append(row)
    near = [{ 'ordinal': i + 1, 'identical_prior_history': i < cutoff,
              'S0': left['samples'][i], 'C0': right['samples'][i]} for i in range(113, 118)]
    return dict(first_output_difference_ordinal=first_output, compared_through_ordinal=cutoff,
                first_score_difference=first_score, comparisons=comparisons, near_token116=near,
                raw_logits_warning='Raw F32 logits precede EOS masking and all samplers. Raw argmax is not necessarily the emitted ID; API logprobs are not raw logits.',
                attribution_gates_passed=True, timing_permitted=False, promotion_permitted=False)


def main():
    assert socket.gethostname() == 'inference-host'
    control = runpy.run_path(str(ROOT / 'run-trace.py'))
    old = runpy.run_path(str(WIDTH1 / 'run-controls.py'))
    native = runpy.run_path(str(WIDTH1 / 'analyze-controls.py'))
    helper = runpy.run_path(str(old['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(old['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(old['OLD'] / 'probe-api.py'))
    out = ROOT / 'run-r1'
    result = dict(attribution_gates_passed=False)
    try:
        summary = load(out / 'summary.json')
        assert summary['status'] == 'completed' and summary['restored'] and summary['order'] == ['S0', 'C0']
        assert summary['timing_permitted'] is False and summary['promotion_permitted'] is False
        arms = {}
        for label in control['ORDER']:
            cell = out / label
            log = (cell / 'server.log').read_text()
            control['check_startup'](log, label, old, native)
            assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
            assert (cell / 'diagnostic-window.log').read_text() == control['diagnostic_window'](
                (cell / 'pre-diagnostic.log').read_text(), (cell / 'through-diagnostic.log').read_text())
            quality = load(cell / 'quality/summary.json')
            assert quality['pass'] and len(quality['checks']) == 20 and all(r['pass'] for r in quality['checks'])
            arms[label] = analyze_arm(cell, label, old)
            control['save'](cell / 'alignment.json', arms[label])
        command = helper['command']
        live = json.loads(command('docker', 'inspect', old['ORIGINAL']))[0]
        control['check_restored'](load(out / 'original-container.json'), live)
        assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {old['ORIGINAL']}
        for name in ('accepted', 'parent', 'candidate'):
            assert control['binaries'](control[name.upper()]) == load(out / (name + '-binaries.json')) == load(ROOT / (name + '-binaries.json'))
        assert runner['hardware']() == load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
        assert api['request'](old['URL'], '/health')['status'] == 'ok'
        smoke = load(out / 'restored-smoke.result.json')
        assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
        result = compare(arms['S0'], arms['C0'])
        result['restoration_verified'] = True
    except BaseException as exc:
        result.update(attribution_gates_passed=False, error=repr(exc))
        raise
    finally:
        control['save'](out / 'analysis.json', result)
    print(json.dumps({k: result[k] for k in ('first_output_difference_ordinal', 'compared_through_ordinal', 'restoration_verified')}))
    print('BOLT_VERIFY_ANALYZED_NO_TIMING')


if __name__ == '__main__':
    main()
