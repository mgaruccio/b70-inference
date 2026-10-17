#!/usr/bin/env python3
"""Analyze saved one-token draft-width controls and independently verify restoration."""
import datetime
import itertools
import json
from pathlib import Path
import re
import runpy


ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1')
EXPECTED_NATIVE = [56, 56, 20, 20, 82, 82, 34, 34, 1]
EXPECTED_OUTPROD = [(80, 80), (80, 80), (18, 18), (18, 18)] * 2
ACCEPTANCE_RE = re.compile(r'accepted\s+(\d+)\s*/\s*(\d+)\s+draft tokens')


def load(path):
    return json.loads(path.read_text())


def difference(left, right):
    """Return a bounded, position-preserving comparison for text or token IDs."""
    index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                 min(len(left), len(right)))
    return dict(equal=left == right, common_prefix_length=index,
                first_difference_index=None if left == right else index,
                left_next=left[index:index + 32], right_next=right[index:index + 32])


def _labeled_difference(left_label, left, right_label, right):
    row = difference(left, right)
    row.update(left=left_label, right=right_label)
    return row


def assess_fidelity(texts):
    """Assess only ordinary streamed output; diagnostics never affect this result."""
    within_arm = {}
    for arm in 'ABC':
        labels = [arm + '0', arm + '1']
        records = [(label, index, text)
                   for label in labels for index, text in enumerate(texts.get(label, []))]
        comparisons = []
        if records:
            first_label, first_index, first_text = records[0]
            comparisons = [
                _labeled_difference(first_label + '/repeat-' + f'{first_index:02}', first_text,
                                    label + '/repeat-' + f'{index:02}', text)
                for label, index, text in records[1:]
            ]
        launch_stability = {
            label: len(set(texts.get(label, []))) == 1 and len(texts.get(label, [])) == 3
            for label in labels
        }
        within_arm[arm] = dict(
            launches=labels,
            responses=len(records),
            unique_outputs=len({text for _, _, text in records}),
            launch_stability=launch_stability,
            stable_across_six=(len(records) == 6 and len({text for _, _, text in records}) == 1),
            comparisons=comparisons,
        )

    cross_arm = {}
    for left_arm, right_arm in itertools.combinations('ABC', 2):
        left_records = [(label, index, text)
                        for label in (left_arm + '0', left_arm + '1')
                        for index, text in enumerate(texts.get(label, []))]
        right_records = [(label, index, text)
                         for label in (right_arm + '0', right_arm + '1')
                         for index, text in enumerate(texts.get(label, []))]
        comparisons = [
            _labeled_difference(left_label + '/repeat-' + f'{left_index:02}', left,
                                right_label + '/repeat-' + f'{right_index:02}', right)
            for left_label, left_index, left in left_records
            for right_label, right_index, right in right_records
        ]
        cross_arm[left_arm + right_arm] = dict(
            pairs=len(comparisons), exact_matches=sum(row['equal'] for row in comparisons),
            equal=(len(comparisons) == 36 and all(row['equal'] for row in comparisons)),
            comparisons=comparisons,
        )
    arm_stability = {arm: within_arm[arm]['stable_across_six'] for arm in 'ABC'}
    cross_equal = {pair: value['equal'] for pair, value in cross_arm.items()}
    return dict(
        arm_stability=arm_stability,
        cross_arm_equal=cross_equal,
        within_arm=within_arm,
        cross_arm=cross_arm,
        fidelity_passed=(all(arm_stability.values()) and all(cross_equal.values())),
    )


def diagnostic_token_comparisons(diagnostics):
    """Compare diagnostic token IDs separately; never inspect placeholder probabilities."""
    comparisons = []
    for left, right in itertools.combinations(sorted(diagnostics), 2):
        row = difference(diagnostics[left], diagnostics[right])
        row.update(left=left, right=right,
                   first_difference_token_ordinal=(None if row['first_difference_index'] is None
                                                   else row['first_difference_index'] + 1))
        comparisons.append(row)
    return comparisons


def parse_native_gate(log, label):
    assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in log
    native = log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED', 1)[0]
    rows = [tuple(map(int, pair)) for pair in re.findall(r'(\d+)/(\d+) tests passed', native)]
    assert rows == [(count, count) for count in EXPECTED_NATIVE for _ in range(2)], rows
    result = dict(native385_passed=True)
    if label[0] != 'A':
        assert 'BOLT_LILICORR_OUTPROD_NATIVE_PASSED' in log
        focused = log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED', 1)[1].split(
            'BOLT_LILICORR_OUTPROD_NATIVE_PASSED', 1)[0]
        outprod_rows = [tuple(map(int, pair)) for pair in
                        re.findall(r'(\d+)/(\d+) tests passed', focused)]
        assert outprod_rows == EXPECTED_OUTPROD, outprod_rows
        assert len(re.findall(r'not supported.*\(expected\)', focused)) == 6
        assert focused.count('(expected)') == 9
        assert '(UNEXPECTED)' not in focused and 'ERR =' not in focused
        result.update(focused_numerical_passed=190, focused_support_contracts_passed=9)
    return result


def model_phase(log, control, arm):
    assert control['SERVER_MARKER'] in log
    model_log = log.split(control['SERVER_MARKER'], 1)[1]
    control['check_model_log'](model_log, arm)
    control['check_startup_spec'](model_log, arm)
    return model_log


def _window(log, since):
    if not since:
        return log
    try:
        start = datetime.datetime.fromisoformat(since)
    except ValueError:
        return log
    rows = []
    for line in log.splitlines():
        try:
            stamp = datetime.datetime.fromisoformat(line.split()[0].replace('Z', '+00:00'))
        except (IndexError, ValueError):
            continue
        if stamp >= start:
            rows.append(line)
    return '\n'.join(rows) + ('\n' if rows else '')


def _diagnostic_counts(body):
    timings = body['timings']
    return dict(accepted=int(timings.get('draft_n_accepted', 0)),
                proposed=int(timings.get('draft_n', 0)))


def assess_speculation_window(window, arm, counters):
    """Classify draft verification evidence for one observation window."""
    pairs = [tuple(map(int, pair)) for pair in ACCEPTANCE_RE.findall(window)]
    accepted, proposed = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    api_accepted = sum(row['accepted'] for row in counters)
    api_proposed = sum(row['proposed'] for row in counters)
    if arm == 'C':
        assert pairs and proposed > 0, (pairs, counters)
        assert (accepted, proposed) == (api_accepted, api_proposed)
        return dict(
            drafter_enabled=True, api_counts=counters,
            log_acceptance_events=[dict(accepted=a, proposed=b) for a, b in pairs],
            log_events=len(pairs), rejection_events=sum(a < b for a, b in pairs),
            counts_match_api=True, proposal_suppression=False, speed_evidence=False,
        )
    if arm == 'B':
        assert not pairs and all(row == {'accepted': 0, 'proposed': 0} for row in counters)
        return dict(
            drafter_enabled=True, api_counts=counters, log_acceptance_events=[], log_events=0,
            rejection_events=0, counts_match_api=True, proposal_suppression=True,
            speed_evidence=False,
        )
    assert not pairs and all(row == {'accepted': 0, 'proposed': 0} for row in counters)
    return dict(
        drafter_enabled=False, api_counts=counters, log_acceptance_events=[], log_events=0,
        rejection_events=0, counts_match_api=True, proposal_suppression=False, speed_evidence=False,
    )


def main():
    control = runpy.run_path(str(ROOT / 'run-controls.py'))
    helper = runpy.run_path(str(control['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(control['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(control['OLD'] / 'probe-api.py'))
    command, binaries = helper['command'], control['binaries']
    assert command('hostname').strip() == 'inference-host'

    out = ROOT / 'run-r1'
    summary = load(out / 'summary.json')
    assert summary['restored'] and summary['timing_permitted'] is False
    assert summary['promotion_permitted'] is False
    frozen = load(out / 'frozen-payload.json')
    result = dict(
        verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        matrix_status=summary['status'], matrix_error=summary.get('error'),
        timing_permitted=False, promotion_permitted=False,
        diagnostic_probability_warning=(
            'Speculative logprob values are placeholders; diagnostics are token-ID-only '
            'and are not confidence or replay-equivalence evidence.'),
        cells={}, primary_output_groups=[], diagnostic_token_comparisons=[],
        speculation_diagnostics={},
    )
    texts, diagnostics = {}, {}
    for label, data in summary['phases'].items():
        cell = out / label
        log = (cell / 'server.log').read_text()
        item = dict(status=data.get('status', 'incomplete'), arm=data['arm'], launch=data['launch'])
        result['cells'][label] = item
        item.update(parse_native_gate(log, label))
        if (cell / 'quality/summary.json').exists():
            quality = load(cell / 'quality/summary.json')
            assert quality['pass'] and len(quality['checks']) == 20 and all(x['pass'] for x in quality['checks'])
            item.update(quality_passed=20, quality_total=20, quality_gate=True)
        assert item['status'] == 'completed'
        assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
        model_log = model_phase(log, control, label[0])
        item['model_log_checks'] = dict(server_marker=control['SERVER_MARKER'], dnn1=True,
                                        forbidden_debug_records=False)
        item['startup_spec'] = data.get('startup_spec') or control['check_startup_spec'](model_log, label[0])

        warmup = load(cell / 'warmup.result.json')
        responses = [load(cell / f'repeat-{index:02}.result.json') for index in range(3)]
        for stem, body in [('warmup', warmup)] + [
                (f'repeat-{index:02}', body) for index, body in enumerate(responses)]:
            assert load(cell / (stem + '.request.json')) == frozen
            control['check_stream_result'](body)
        texts[label] = [body['text'] for body in responses]

        assert load(cell / 'diagnostic.request.json') == control['diagnostic_payload'](frozen)
        diagnostic = load(cell / 'diagnostic.response.json')
        token_ids = control['check_diagnostic_result'](diagnostic)
        diagnostics[label] = token_ids
        item.update(
            ordinary_unique_outputs=len(set(texts[label])),
            ordinary_same_as_warmup=[text == warmup['text'] for text in texts[label]],
            ordinary_vs_first=[difference(texts[label][0], text) for text in texts[label][1:]],
            diagnostic_token_count=len(token_ids),
            diagnostic_logprob_count=len(diagnostic['choices'][0]['logprobs']['content']),
            diagnostic_logprobs_ignored_for_fidelity=True,
        )

        window = _window(log, data.get('probe_window_since'))
        (cell / 'probe-window.log').write_text(window)
        counters = [_diagnostic_counts(body) for body in [warmup] + responses + [diagnostic]]
        result['speculation_diagnostics'][label] = assess_speculation_window(
            window, label[0], counters)

    decision = assess_fidelity(texts)
    result['fidelity'] = decision
    result['fidelity_passed'] = decision['fidelity_passed']
    result['diagnostic_token_comparisons'] = diagnostic_token_comparisons(diagnostics)
    result['diagnostic_comparisons_are_not_fidelity'] = True
    groups = {}
    for label, values in texts.items():
        for index, text in enumerate(values):
            groups.setdefault(text, []).append(f'{label}/repeat-{index:02}')
    result['primary_output_groups'] = [dict(members=members, text=text)
                                       for text, members in groups.items()]
    assert summary['status'] == 'completed'
    assert set(texts) == set(control['ORDER'])

    # Independent restoration and complete four-artifact checks.
    original = load(out / 'original-container.json')
    live = json.loads(command('docker', 'inspect', control['ORIGINAL']))[0]
    assert live['State']['Running'] and live['Id'] == original['Id']
    assert live['Config'] == original['Config']
    assert sorted(live['Mounts'], key=lambda item: item['Destination']) == sorted(
        original['Mounts'], key=lambda item: item['Destination'])
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {control['ORIGINAL']}
    for label in summary['phases']:
        name = control['PREFIX'] + label.lower()
        assert command('docker', 'inspect', '-f', '{{.State.Running}}', name).strip() == 'false'
    assert binaries(control['ACCEPTED']) == load(out / 'accepted-binaries.json')
    assert binaries(control['BASELINE']) == load(out / 'baseline-binaries.json')
    assert binaries(control['CANDIDATE']) == load(out / 'candidate-binaries.json')
    assert binaries(control['ACCEPTED']) == load(ROOT / 'accepted-binaries.json')
    assert binaries(control['BASELINE']) == load(ROOT / 'baseline-binaries.json')
    assert binaries(control['CANDIDATE']) == load(ROOT / 'candidate-binaries.json')
    hardware = runner['hardware']()
    assert hardware == load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
    health = api['request'](control['URL'], '/health')
    assert health['status'] == 'ok'
    smoke = load(out / 'restored-smoke.result.json')
    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
    result['restoration'] = dict(
        verified=True, original_id=original['Id'], live_id=live['Id'],
        config_identical=True, all_mount_fields_identical=True, binaries_identical=True,
        hardware=hardware, health=health, ready_output=smoke['text'],
        all_experiments_stopped=True,
    )
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('cells', 'diagnostic_token_comparisons', 'primary_output_groups')},
                     indent=2))
    print('BOLT_LILICORR_WIDTH1_FIDELITY_ANALYZED_AND_RESTORATION_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
