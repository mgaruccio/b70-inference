"""End-to-end checks of the live Flint server on :8080 (API key on stdin, never printed).

Usage (guest): sudo cat /etc/b70-flint/api-key | python3 -I validate.py OUT_DIR [checks...]
checks: auth parity tools long multiturn (default: all). Writes OUT_DIR/<check>.json."""
import http.client, json, os, pathlib, random, sys, time

HOST, PORT, MODEL = '127.0.0.1', 8080, 'flint'
KEY = sys.stdin.readline().strip()
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
checks = sys.argv[2:] or ['auth', 'parity', 'tools', 'long', 'multiturn']
SPIKE = pathlib.Path.home() / 'flint-vllm-spike'


def req(method, path, body=None, key=True, timeout=600):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    h = {'Content-Type': 'application/json'}
    if key:
        h['Authorization'] = 'Bearer ' + KEY
    c.request(method, path, json.dumps(body) if body is not None else None, h)
    r = c.getresponse(); raw = r.read(); c.close()
    try:
        return r.status, json.loads(raw)
    except Exception:
        return r.status, raw[:200].decode(errors='replace')


def metrics():
    st, txt = 0, ''
    c = http.client.HTTPConnection(HOST, PORT, timeout=10)
    c.request('GET', '/metrics'); r = c.getresponse(); txt = r.read().decode(); c.close()
    vals = {}
    for line in txt.splitlines():
        for name in ('spec_decode_num_drafts_total', 'spec_decode_num_accepted_tokens_total',
                     'prefix_cache_hits_total', 'prefix_cache_queries_total'):
            if line.startswith('vllm:' + name + '{'):
                vals[name] = float(line.rsplit(' ', 1)[1])
    return vals


def stream_chat(messages, max_tokens, tools=None, temperature=0, **extra):
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature,
                stream=True, stream_options={'include_usage': True}, **extra)
    if tools:
        body['tools'] = tools
    c = http.client.HTTPConnection(HOST, PORT, timeout=900)
    m0 = metrics()
    t0 = time.monotonic()
    c.request('POST', '/v1/chat/completions', json.dumps(body),
              {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read(500))
    first = last = None; content = []; reasoning = []; usage = None; finish = None; tcalls = {}
    while line := r.readline():
        line = line.strip()
        if line == b'data: [DONE]':
            break
        if not line.startswith(b'data: '):
            continue
        f = json.loads(line[6:])
        usage = f.get('usage') or usage
        for ch in f.get('choices', ()):
            d = ch.get('delta') or {}
            got = False
            if d.get('content'):
                content.append(d['content']); got = True
            rc = d.get('reasoning_content') or d.get('reasoning')
            if rc:
                reasoning.append(rc); got = True
            for tc in d.get('tool_calls') or ():
                e = tcalls.setdefault(tc.get('index', 0), {'name': '', 'arguments': ''})
                fn = tc.get('function') or {}
                e['name'] += fn.get('name') or ''
                e['arguments'] += fn.get('arguments') or ''
                got = True
            if got:
                now = time.monotonic(); first = first or now; last = now
            finish = ch.get('finish_reason') or finish
    c.close()
    m1 = metrics()
    nd = m1.get('spec_decode_num_drafts_total', 0) - m0.get('spec_decode_num_drafts_total', 0)
    na = (m1.get('spec_decode_num_accepted_tokens_total', 0)
          - m0.get('spec_decode_num_accepted_tokens_total', 0))
    hits = m1.get('prefix_cache_hits_total', 0) - m0.get('prefix_cache_hits_total', 0)
    n = usage['completion_tokens'] if usage else 0
    return dict(content=''.join(content), reasoning=''.join(reasoning), finish=finish, usage=usage,
                tool_calls=[tcalls[k] for k in sorted(tcalls)], ttft_ms=((first or t0) - t0) * 1e3,
                decode_tps=(n - 1) / (last - first) if first and last and last > first else None,
                acceptance_length=1 + na / nd if nd else None, prefix_cache_hit_tokens=hits)


def save(name, obj):
    (out / f'{name}.json').write_text(json.dumps(obj, indent=1))
    print(name, json.dumps(obj)[:1500], flush=True)


if 'auth' in checks:
    r = {}
    r['health_nokey'] = req('GET', '/health', key=False)[0]
    r['models_nokey'] = req('GET', '/v1/models', key=False)[0]
    r['invocations_nokey'] = req('POST', '/invocations', {'model': MODEL, 'prompt': 'hi',
                                                         'max_tokens': 1}, key=False)[0]
    r['tokenize_nokey'] = req('POST', '/tokenize', {'model': MODEL, 'prompt': 'hi'}, key=False)[0]
    r['docs_nokey'] = req('GET', '/docs', key=False)[0]
    c = http.client.HTTPConnection(HOST, PORT, timeout=10)
    c.request('GET', '/v1/models', headers={'Authorization': 'Bearer wrong'})
    r['models_wrongkey'] = c.getresponse().status; c.close()
    st, body = req('GET', '/v1/models')
    r['models_key'] = st
    r['model_ids'] = [m['id'] for m in body.get('data', [])] if isinstance(body, dict) else body
    r['metrics_loopback'] = req('GET', '/metrics', key=False)[0]
    r['PASS'] = (r['health_nokey'] == 200 and r['models_nokey'] == 401 and r['models_wrongkey'] == 401
                 and r['invocations_nokey'] == 401 and r['tokenize_nokey'] == 401
                 and r['docs_nokey'] == 401 and r['models_key'] == 200 and r['model_ids'] == [MODEL]
                 and r['metrics_loopback'] == 200)
    save('auth', r)

if 'parity' in checks:
    prompts = json.loads((SPIKE / 'benchmark-prompts.json').read_text())
    long_prompt = ('Review the following requests and summarize the shared engineering themes.\n\n'
                   + '\n\n'.join(prompts * 2))
    rows = []
    for i, p in enumerate(prompts + [long_prompt]):
        st, r = req('POST', '/v1/completions', {'model': MODEL, 'prompt': p, 'temperature': 0,
                                                'seed': 42, 'max_tokens': 128, 'ignore_eos': True,
                                                'return_token_ids': True, 'logprobs': 5})
        assert st == 200, r
        ch, = r['choices']
        rows.append({'case': i, 'prompt_len': len(ch['prompt_token_ids']), 'token_ids': ch['token_ids'],
                     'top': ch['logprobs']['top_logprobs'], 'lp': ch['logprobs']['token_logprobs'],
                     'tokens': ch['logprobs']['tokens']})
    (out / 'parity.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    print('parity captured', len(rows), flush=True)

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'Current weather for a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}},
    {'type': 'function', 'function': {
        'name': 'get_time', 'description': 'Current local time for a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}},
]

if 'tools' in checks:
    r = {}
    m = [{'role': 'user', 'content': 'What is the weather in Paris right now? Use the tool.'}]
    s = stream_chat(m, 1024, tools=TOOLS)
    r['single_stream'] = dict(finish=s['finish'], tool_calls=s['tool_calls'], content=s['content'][:200])
    st, ns = req('POST', '/v1/chat/completions', dict(model=MODEL, messages=m, max_tokens=1024,
                                                      temperature=0, tools=TOOLS))
    msg = ns['choices'][0]['message']
    r['single_nonstream'] = dict(status=st, finish=ns['choices'][0]['finish_reason'],
                                 tool_calls=[(t['function']['name'], t['function']['arguments'])
                                             for t in msg.get('tool_calls') or []],
                                 has_reasoning=bool(msg.get('reasoning_content') or msg.get('reasoning')))
    m2 = [{'role': 'user', 'content': 'Get both the weather and the local time in Tokyo. '
                                      'Call both tools at once.'}]
    s2 = stream_chat(m2, 1024, tools=TOOLS)
    r['parallel_stream'] = dict(finish=s2['finish'], tool_calls=s2['tool_calls'])
    # multi-turn with the tool result
    call = msg['tool_calls'][0] if msg.get('tool_calls') else None
    if call:
        m3 = m + [{'role': 'assistant', 'content': msg.get('content') or '', 'tool_calls': [call]},
                  {'role': 'tool', 'tool_call_id': call['id'],
                   'content': json.dumps({'city': 'Paris', 'temp_c': 17, 'sky': 'light rain'})}]
        s3 = stream_chat(m3, 1024, tools=TOOLS)
        r['tool_result_turn'] = dict(finish=s3['finish'], content=s3['content'][:300],
                                     mentions_17=('17' in s3['content']))
    ok_single = (r['single_stream']['tool_calls'] and r['single_stream']['tool_calls'][0]['name'] == 'get_weather'
                 and 'paris' in r['single_stream']['tool_calls'][0]['arguments'].lower())
    ok_ns = r['single_nonstream']['tool_calls'] and r['single_nonstream']['tool_calls'][0][0] == 'get_weather'
    names = sorted(t['name'] for t in r['parallel_stream']['tool_calls'])
    r['PASS'] = bool(ok_single and ok_ns and names == ['get_time', 'get_weather']
                     and r.get('tool_result_turn', {}).get('mentions_17'))
    save('tools', r)

if 'long' in checks:
    src = sorted((SPIKE / '.venv/lib/python3.12/site-packages/vllm/v1').rglob('*.py'))
    filler = '\n\n'.join(p.read_text() for p in src)
    rng = random.Random(7)
    res = []
    for target in [int(t) for t in os.environ.get("FLINT_LONG_TARGETS", "2000,8000,15000,21500").split(",")]:
        word = ''.join(rng.choice('abcdefghjkmnpqrstuvwxyz') for _ in range(8))
        # ~3.6 chars/token for Python source with this tokenizer (checked by usage below)
        text = filler[: int(target * 3.4)]
        cut = len(text) // 2
        doc = text[:cut] + f'\n# NOTE: the deployment passphrase is "{word}".\n' + text[cut:]
        m = [{'role': 'user', 'content': 'Here is a source dump:\n\n' + doc +
              '\n\nQuestion: what is the deployment passphrase mentioned in a NOTE comment? '
              'Answer with just the passphrase.'}]
        s = stream_chat(m, 2048)
        res.append(dict(target=target, prompt_tokens=s['usage']['prompt_tokens'],
                        completion_tokens=s['usage']['completion_tokens'], ttft_ms=round(s['ttft_ms']),
                        decode_tps=s['decode_tps'] and round(s['decode_tps'], 1),
                        acceptance_length=s['acceptance_length'] and round(s['acceptance_length'], 2),
                        found=word in s['content'], answer=s['content'][-80:], finish=s['finish']))
        print(res[-1], flush=True)
    save('long', dict(results=res, PASS=all(x['found'] for x in res)))

if 'multiturn' in checks:
    # agent-like: long fixed system prompt + growing conversation; reports TTFT per turn
    sysmsg = ('You are a coding agent. Tools and policies follow.\n' +
              (SPIKE / '.venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py').read_text()[:24000])
    msgs = [{'role': 'system', 'content': sysmsg}]
    turns = []
    for q in ['Summarize what the scheduler does in two sentences.',
              'Which method handles preemption? One line.',
              'Name one data structure it uses for waiting requests. One line.']:
        msgs.append({'role': 'user', 'content': q})
        s = stream_chat(msgs, 512)
        msgs.append({'role': 'assistant', 'content': s['content']})
        turns.append(dict(prompt_tokens=s['usage']['prompt_tokens'], ttft_ms=round(s['ttft_ms']),
                          decode_tps=s['decode_tps'] and round(s['decode_tps'], 1),
                          acceptance_length=s['acceptance_length'] and round(s['acceptance_length'], 2),
                          prefix_cache_hit_tokens=s['prefix_cache_hit_tokens'],
                          answer=s['content'][:160]))
        print(turns[-1], flush=True)
    save('multiturn', dict(turns=turns))
