#!/usr/bin/env python3
"""Host-side real llama.cpp chat/API smoke and development timings (stdlib only)."""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


def request(base, path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def chat(base, prompt, out, *, count=128, stream=False, ignore_eos=False):
    payload = {'model': 'nemotron-bolt', 'messages': [{'role': 'user', 'content': prompt}],
               'temperature': 0, 'seed': 42, 'max_tokens': count, 'stream': stream,
               'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False},
               'ignore_eos': ignore_eos}
    if stream:
        payload['stream_options'] = {'include_usage': True}
    out.with_suffix('.request.json').write_text(json.dumps(payload, indent=2) + '\n')
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    chunks, events, first = [], [], None
    usage, timings, finish = {}, {}, None
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            if stream:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        break
                    row = json.loads(raw)
                    events.append({'elapsed_s': time.monotonic() - started, 'body': row})
                    for choice in row.get('choices', []):
                        delta = choice.get('delta', {})
                        text = delta.get('content') or ''
                        if text:
                            if first is None:
                                first = time.monotonic() - started
                            chunks.append(text)
                        if choice.get('finish_reason') is not None:
                            finish = choice['finish_reason']
                    usage = row.get('usage') or usage
                    timings = row.get('timings') or timings
            else:
                body = json.load(response)
                events.append(body)
                chunks.append(body['choices'][0]['message'].get('content') or '')
                usage = body.get('usage', {})
                timings = body.get('timings', {})
                finish = body['choices'][0].get('finish_reason')
    except urllib.error.HTTPError as exc:
        out.with_suffix('.http-error.txt').write_bytes(exc.read())
        raise
    elapsed = time.monotonic() - started
    n = usage.get('completion_tokens', 0)
    result = {'text': ''.join(chunks), 'elapsed_s': elapsed, 'ttft_s': first,
              'usage': usage, 'timings': timings, 'finish_reason': finish,
              'decode_tps_client': ((n - 1) / (elapsed - first)
                                    if stream and n > 1 and first is not None and elapsed > first else None),
              'end_to_end_output_tps': n / elapsed if n else None,
              'full_output_budget': n == count}
    out.with_suffix('.events.json').write_text(json.dumps(events, indent=2) + '\n')
    out.with_suffix('.result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:18098')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'quality', 'bench', 'long', 'repeat'], default='smoke')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    models = request(args.base, '/v1/models')
    (args.out / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
    assert any(x['id'] == 'nemotron-bolt' for x in models['data']), models
    if args.mode in ('smoke', 'quality'):
        cases = [('literal', 'Reply with exactly this word and nothing else: READY', lambda s: s.strip() == 'READY'),
                 ('arithmetic', 'What is 17 multiplied by 19? Reply with only the number.', lambda s: s.strip() == '323'),
                 ('code', 'Write a Python function named add(a, b) that returns their sum. Output code only.',
                  lambda s: 'def add(' in s and 'return' in s and '+' in s)]
        if args.mode == 'quality':
            arithmetic = [(27, '+', 68, 95), (1000, '-', 384, 616), (23, '*', 47, 1081),
                          (144, '/', 12, 12), (37, '*', 37, 1369), (-19, '+', 74, 55),
                          (91, '-', 108, -17), (256, '/', 16, 16), (32, '*', 25, 800),
                          (999, '+', 1, 1000)]
            for i, (a, op, b, answer) in enumerate(arithmetic):
                cases.append((f'arithmetic-{i:02}', f'Compute {a} {op} {b}. Reply with only the integer answer.',
                              lambda s, expected=str(answer): s.strip() == expected))
            fixed = [('literal-red', 'Reply with exactly RED and no other text.', 'RED'),
                     ('literal-cedar', 'Reply with exactly cedar and no other text.', 'cedar'),
                     ('literal-code', 'Copy this code exactly with no other text: X7-Q92', 'X7-Q92'),
                     ('sort-numbers', 'Sort these numbers ascending: 9, 2, 5, 1. Reply exactly as comma-separated numbers with no spaces.', '1,2,5,9'),
                     ('reverse', 'Reverse the letters in drawer. Reply with only the resulting word.', 'reward'),
                     ('json', 'Reply with only this JSON object, no markdown: {"ready": true, "count": 3}', '{"ready": true, "count": 3}'),
                     ('extract', 'The shelf contains 5 red boxes and 8 blue boxes. How many boxes total? Reply with only the integer.', '13')]
            for name, prompt, answer in fixed:
                checker = (lambda s: ''.join(s.split()) == '{"ready":true,"count":3}') if name == 'json' else (lambda s, expected=answer: s.strip() == expected)
                cases.append((name, prompt, checker))
        checks = []
        for name, prompt, check in cases:
            result = chat(args.base, prompt, args.out / name, count=256)
            passed = (bool(check(result['text'])) and result['finish_reason'] == 'stop'
                      and result['usage'].get('prompt_tokens', 0) > 0
                      and result['usage'].get('completion_tokens', 0) > 0)
            checks.append({'name': name, 'pass': passed, 'result': result})
        summary = {'pass': all(x['pass'] for x in checks), 'checks': checks}
    else:
        # 128 generated tokens, EOS ignored only for timings. No quality claim on forced tails.
        prompt = ('Explain how to design a reliable file backup system. Cover full and incremental backups, '
                  'restore testing, offsite copies, retention, and encryption. Give a detailed answer.')
        anchor = None
        if args.mode == 'long':
            # Tokenize/detokenize the actual served model; final rendered count comes from usage.
            filler = ''.join(f'Archive entry {i}: This record concerns ordinary backup storage and restoration.\n'
                             for i in range(1600))
            tokens = request(args.base, '/tokenize', {'content': filler, 'add_special': False})['tokens'][:8192]
            assert len(tokens) == 8192
            filler = request(args.base, '/detokenize', {'tokens': tokens})['content']
            secret = 'BOLT-7319-CEDAR'
            document = f'The secret archive code is {secret}.\n' + filler
            anchor = chat(args.base, document + '\nWhat is the secret archive code? Return only that code.',
                          args.out / 'anchor', count=128)
            assert anchor['text'].strip() == secret and anchor['finish_reason'] == 'stop', anchor
            assert anchor['usage']['prompt_tokens'] >= 8192, anchor
            prompt = document + '\nIgnore the archive code. ' + prompt
        warmup = chat(args.base, 'Warmup. ' + prompt, args.out / 'warmup', stream=True, ignore_eos=True)
        measured = []
        for i in range(args.repeats):
            trial = 0 if args.mode == 'repeat' else i  # Same input/seed; cache reuse stays off.
            measured.append(chat(args.base, f'Independent trial {trial}. ' + prompt, args.out / f'measured-{i:02}',
                                 stream=True, ignore_eos=True))
        assert all(r['full_output_budget'] and r['text'].strip() for r in measured), measured
        assert all(r['decode_tps_client'] is not None for r in measured), measured
        assert all(r['usage'].get('prompt_tokens', 0) > 0 for r in measured), measured
        assert all(r['timings'].get('cache_n') == 0 for r in measured), 'Cached work invalidates cold prefill'
        if args.mode == 'long':
            assert all(r['usage']['prompt_tokens'] >= 8192 for r in measured), measured
        assert all(r['timings'].get('predicted_n') == 128 and r['timings'].get('prompt_n', 0) > 0
                   for r in measured), 'Missing/incomplete server timings; do not claim a timing result'
        summary = {'tier': 'development', 'anchor': anchor, 'warmup': warmup, 'measured': measured,
                   'decode_tps_client_median': statistics.median(r['decode_tps_client'] for r in measured),
                   'ttft_s_median': statistics.median(r['ttft_s'] for r in measured), 'pass': True}
        summary['distinct_output_texts'] = len({r['text'] for r in measured})
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if not summary['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
