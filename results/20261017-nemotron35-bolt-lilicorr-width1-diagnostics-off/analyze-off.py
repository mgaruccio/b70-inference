#!/usr/bin/env python3
"""LEAD ONLY, after restoration: public-output fidelity, never raw/route equivalence."""
import argparse
import json
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
ROOT, OLD, P = C['ROOT'], C['OLD'], C['P']
load, save = C['load'], C['save']


def artifact(stem, suffix):
    return Path(str(stem) + suffix)


def read_http(stem, payload, endpoint):
    assert load(artifact(stem, '.request.json')) == payload
    assert artifact(stem, '.request-body').read_bytes() == json.dumps(payload).encode()
    request = load(artifact(stem, '.http-request.json'))
    assert request['url'] == OLD['URL'] + endpoint and request['method'] == 'POST'
    assert request['headers']['Content-type'] == 'application/json'
    record = load(artifact(stem, '.http.json'))
    assert record['status'] == 200 and record['complete'] and 'error' not in record
    raw = artifact(stem, '.http-body').read_bytes()
    assert record['bytes'] == len(raw)
    return raw


def normalize(raw, payload):
    if not payload['stream']:
        body = json.loads(raw)
        assert 'error' not in body and len(body['choices']) == 1
        choice = body['choices'][0]
        return dict(text=choice['message'].get('content') or '', usage=body['usage'],
                    finish_reason=choice['finish_reason'], timings=body.get('timings', {}))
    chunks, usage, timings, finish, done, count = [], None, {}, None, False, 0
    for line in raw.splitlines():
        if not line.startswith(b'data: '):
            continue
        assert not done, 'SSE data after DONE'
        data = line[6:].strip()
        if data == b'[DONE]':
            done = True
            continue
        body = json.loads(data)
        assert 'error' not in body
        count += 1
        assert len(body.get('choices', [])) <= 1
        for choice in body.get('choices', []):
            assert choice['index'] == 0
            chunks.append(choice.get('delta', {}).get('content') or '')
            if choice.get('finish_reason') is not None:
                assert finish is None, 'repeated stream finish'
                finish = choice['finish_reason']
        usage = body.get('usage') or usage
        timings = body.get('timings') or timings
    assert done and count and usage and finish is not None, 'incomplete stream'
    return dict(text=''.join(chunks), usage=usage, finish_reason=finish, timings=timings)


def check_public(cell, index, job):
    stem = cell / 'public' / f'{index:02}'
    template = C['L']['apply_request'](job['payload'])
    rendered = json.loads(read_http(artifact(stem, '.template'), template, '/apply-template'))
    assert rendered == load(artifact(stem, '.template.response.json'))
    tokenize = C['L']['tokenize_request'](rendered)
    response = json.loads(read_http(artifact(stem, '.tokenize'), tokenize, '/tokenize'))
    assert response == load(artifact(stem, '.tokenize.response.json'))
    return C['L']['token_ids'](response)


def reference_result(stem, diagnostic):
    if diagnostic:
        body = load(artifact(stem, '.response.json'))
        return normalize(json.dumps(body).encode(), dict(stream=False))
    return load(artifact(stem, '.result.json'))


def analyze_responses(cell, label, jobs):
    report = dict(rows=[], errors=[], diagnostics={}, within_arm=[])
    results, prompts = {}, {}
    for index, job in enumerate(jobs):
        stem, payload = job['stem'], job['payload']
        row = dict(stem=stem, actual_ids_claimed=stem.endswith('/diagnostic'))
        report['rows'].append(row)
        try:
            raw = read_http(cell / stem, payload, '/v1/chat/completions')
            result = normalize(raw, payload)
            results[stem] = result
            row['actual'] = C['result_view'](result)
            ids = check_public(cell, index, job)
            prompts[stem] = ids
            reference_ids = C['reference_prompt'](label, job)
            row['prompt_comparison'] = C['id_comparison'](ids, reference_ids)
            row['prompt_usage_equal'] = len(ids) == result['usage']['prompt_tokens']
            row['functional_pass'] = True
            if stem.startswith('quality/'):
                row['functional_pass'] = C['L']['functional_quality'](stem.split('/')[1], result)
            elif stem == 'anchor':
                row['functional_pass'] = C['L']['functional_anchor'](result)
            elif not stem.endswith('/diagnostic'):
                OLD['check_stream_result'](result, prompt_tokens=len(ids))
            references = []
            for ref in [C['reference_stem'](label, stem)] + C['extra_references'](label, stem):
                assert load(artifact(ref, '.request.json')) == payload
                old_result = reference_result(ref, stem.endswith('/diagnostic'))
                references.append(dict(path=str(ref), expected=C['result_view'](old_result),
                                       equal=C['result_view'](result) == C['result_view'](old_result)))
            row['saved_references'] = references
            if stem.endswith('/diagnostic'):
                # Requests, vocab and public/reference prompt IDs are checked first.
                assert row['prompt_comparison']['equal'] and row['prompt_usage_equal']
                actual = C['diagnostic_view'](json.loads(raw), ids)
                ref = C['reference_stem'](label, stem)
                expected = C['diagnostic_view'](load(artifact(ref, '.response.json')), reference_ids)
                report['diagnostics'][stem.split('/')[0]] = dict(
                    actual=actual, reference=expected,
                    comparison=C['id_comparison'](actual['ids'], expected['ids']),
                    pieces_equal=actual['pieces'] == expected['pieces'])
            row['passed'] = (row['prompt_comparison']['equal'] and row['prompt_usage_equal']
                             and row['functional_pass'] and all(r['equal'] for r in references))
        except BaseException as exc:
            row['passed'] = False
            report['errors'].append(dict(stem=stem, error=repr(exc)))
    for length in ('short', 'long'):
        try:
            warm, ordinary, diagnostic = [results[length + '/' + kind] for kind in ('warmup', 'ordinary', 'diagnostic')]
            equal = C['result_view'](warm) == C['result_view'](ordinary) == C['result_view'](diagnostic)
            equal_prompt = prompts[length + '/warmup'] == prompts[length + '/ordinary'] == prompts[length + '/diagnostic']
            report['within_arm'].append(dict(length=length, text_finish_usage_equal=equal,
                                             prompt_ids_equal=equal_prompt))
        except BaseException as exc:
            report['errors'].append(dict(length=length, error=repr(exc)))
    report['passed'] = (not report['errors'] and len(report['rows']) == 27 and all(r['passed'] for r in report['rows'])
                        and len(report['diagnostics']) == 2 and all(
                            d['comparison']['equal'] and d['pieces_equal'] for d in report['diagnostics'].values())
                        and len(report['within_arm']) == 2 and all(
                            r['text_finish_usage_equal'] and r['prompt_ids_equal'] for r in report['within_arm']))
    return report, results, prompts


def verify_live_restoration(out, command, api, hardware):
    original = load(out / 'original-container.json')
    assert original['Id'] == P['ORIGINAL_ID']
    P['check_restored'](original, load(out / 'restored-container.json'))
    smoke = load(out / 'restored-smoke.result.json')
    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
    live = json.loads(command('docker', 'inspect', OLD['ORIGINAL']))[0]
    P['check_restored'](original, live)
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {OLD['ORIGINAL']}
    health = api['request'](OLD['URL'], '/health')
    assert health['status'] == 'ok'
    hw = hardware()
    assert hw == load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
    return dict(container=live, health=health, hardware=hw, independently_queried_live=True)


def check_arm(cell, label):
    argv = C['make_create'](label)
    assert load(cell / 'create-argv.json') == argv
    container = load(cell / 'container.json')
    assert container['Name'] == '/' + C['PREFIX'] + label.lower() and not container['State']['Running']
    assert container['Config']['Cmd'] == argv[-3:]
    assert all(m['Destination'] != '/capture' for m in container['Mounts'])
    assert any(m['Destination'] == '/draft' for m in container['Mounts']) == (label == 'C0')
    for phase in ('before', 'after'):
        C['check_pid'](load(cell / ('serving-pid-' + phase + '.json')), label)
    C['check_startup']((cell / 'server.log').read_text(), label)
    assert not (cell / 'capture').exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-after-restore', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host' and Path(__file__).resolve().parent == ROOT
    out = ROOT / 'run-r1'
    analysis = out / 'analysis-r1'
    analysis.mkdir(exist_ok=False)  # Failure is permanent evidence, never overwritten.
    report = dict(status='failed', errors=[], timing_permitted=False, promotion_permitted=False,
                  raw_vector_claimed=False, fresh_route_equivalence_claimed=False,
                  probability_fidelity_claimed=False, behavioral_off_not_zero_overhead=True)
    try:
        summary = load(out / 'summary.json')
        assert summary['restored'] and load(out / 'restoration-checks.json')['passed']
        assert summary['order'] == list(C['ARMS'])
        assert summary['timing_permitted'] is summary['promotion_permitted'] is False
        tested = C['check_tested_code']()
        assert load(out / 'tested-code.json') == tested
        assert load(out / 'source-contracts.json') == C['source_contracts']()
        assert load(out / 'hash-checks-before.json') == load(out / 'hash-checks-after.json') == C['N']['check_sources_and_binaries']()
        helper = runpy.run_path(str(OLD['GROUPED'] / 'run-grouped-diagnostic.py'))
        runner = runpy.run_path(str(OLD['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
        api = runpy.run_path(str(OLD['OLD'] / 'probe-api.py'))
        live = verify_live_restoration(out, helper['command'], api, runner['hardware'])
        save(analysis / 'live-restoration.json', live)
        jobs = C['frozen_jobs']()
        assert load(out / 'frozen-requests.json') == jobs
        for record in [load(out / 'initial-capacity.json')] + [load(p) for p in out.glob('capacity-*.json')]:
            assert record['fits'] and record['required_free'] == C['RESERVE'] + C['ALLOWANCE']
            assert record['free_bytes'] >= record['required_free']
        responses, prompts = {}, {}
        report['arms'] = {}
        # Normalize both arms independently BEFORE comparing or requiring success.
        for label in C['ARMS']:
            arm, responses[label], prompts[label] = analyze_responses(out / label, label, jobs)
            report['arms'][label] = arm
            try:
                check_arm(out / label, label)
                for suffix in ([''] + ['-public']):
                    for index in range(28 if not suffix else 27):
                        assert (out / f'capacity-{label}{suffix}-{index:02}.json').is_file()
            except BaseException as exc:
                report['errors'].append(dict(arm=label, error=repr(exc)))
        report['cross_arm'] = []
        for job in jobs:
            stem = job['stem']
            try:
                a, b = [responses[label][stem] for label in C['ARMS']]
                row = dict(stem=stem, text_finish_usage_equal=C['result_view'](a) == C['result_view'](b),
                           prompt_ids_equal=prompts['A0'][stem] == prompts['C0'][stem],
                           request_bytes_equal=artifact(out / 'A0' / stem, '.request-body').read_bytes()
                           == artifact(out / 'C0' / stem, '.request-body').read_bytes(),
                           actual_ids_claimed=stem.endswith('/diagnostic'))
                if row['actual_ids_claimed']:
                    length = stem.split('/')[0]
                    left, right = [report['arms'][label]['diagnostics'][length]['actual'] for label in C['ARMS']]
                    row['id_comparison'] = C['id_comparison'](left['ids'], right['ids'])
                    row['pieces_equal'] = left['pieces'] == right['pieces']
                row['passed'] = (row['text_finish_usage_equal'] and row['prompt_ids_equal'] and row['request_bytes_equal']
                                 and (not row['actual_ids_claimed'] or (row['id_comparison']['equal'] and row['pieces_equal'])))
                report['cross_arm'].append(row)
            except BaseException as exc:
                report['errors'].append(dict(stem=stem, error=repr(exc)))
        assert summary['status'] == 'captured' and summary['completed'] == list(C['ARMS'])
        assert not report['errors'] and all(a['passed'] for a in report['arms'].values())
        assert len(report['cross_arm']) == 27 and all(r['passed'] for r in report['cross_arm'])
        report.update(status='passed', restoration_verified=True)
    except BaseException as exc:
        report['errors'].append(dict(error=repr(exc)))
        raise
    finally:
        save(analysis / 'analysis.json', report)
    print('BOLT_OFF_PUBLIC_FIDELITY_PASSED_NO_RAW_OR_ROUTE_CLAIM_NO_TIMING')


if __name__ == '__main__':
    main()
