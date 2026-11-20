"""Quality eval of the live Flint server on :8080 (whichever backend is active).

Standard subset (BENCHMARKING_STANDARDS.md §9.2): GSM8K 100 + IFEval 100 (seeded samples)
+ HumanEval+ 164, chat API, thinking on, greedy, max_tokens 4096, concurrency 1.
API key on stdin (never printed).

  sudo cat /etc/b70-flint/api-key | .venv/bin/python -I eval_flint.py run LABEL OUT_DIR
  .venv/bin/python -I eval_flint.py score OUT_DIR/LABEL      # IFEval + GSM8K + evalplus
Resumable: completed items in OUT_DIR/LABEL/<task>.jsonl are skipped."""
import http.client, json, pathlib, random, re, subprocess, sys, time

DATA = pathlib.Path.home() / 'flint-eval/data'
MAX_TOKENS, N_GSM, N_IFE, SEED = 4096, 100, 100, 0
THINK = re.compile(r'<think>.*?</think>', re.S)


def chat(key, content):
    body = dict(model='flint', messages=[{'role': 'user', 'content': content}],
                max_tokens=MAX_TOKENS, temperature=0, seed=SEED)
    for attempt in range(3):
        try:
            c = http.client.HTTPConnection('127.0.0.1', 8080, timeout=1800)
            t0 = time.monotonic()
            c.request('POST', '/v1/chat/completions', json.dumps(body),
                      {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
            r = c.getresponse(); raw = r.read(); c.close()
            if r.status != 200:
                raise RuntimeError(f'HTTP {r.status}: {raw[:300]!r}')
            d = json.loads(raw)
            msg = d['choices'][0]['message']
            text = msg.get('content') or ''
            reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
            if '<think>' in text or '</think>' in text:  # unsplit reasoning (llama.cpp fallback)
                reasoning += ''.join(THINK.findall(text))
                text = THINK.sub('', text).split('</think>')[-1]
            return dict(response=text.strip(), reasoning_chars=len(reasoning),
                        finish=d['choices'][0]['finish_reason'], usage=d.get('usage'),
                        wall_s=time.monotonic() - t0)
        except Exception as ex:  # transient connection errors
            if attempt == 2:
                raise
            print('retry', repr(ex)[:200], flush=True); time.sleep(10)


def items():
    rng = random.Random(SEED)
    gsm = [json.loads(l) for l in open(DATA / 'gsm8k_test.jsonl')]
    for i in sorted(rng.sample(range(len(gsm)), N_GSM)):
        q = gsm[i]['question']
        yield 'gsm8k', f'gsm8k/{i}', (q + "\n\nSolve the problem step by step. On the last line, give "
                                      "the final answer as 'Answer: <number>'."), gsm[i]['answer']
    rng = random.Random(SEED)
    ife = [json.loads(l) for l in open(DATA / 'ifeval_input_data.jsonl')]
    for i in sorted(rng.sample(range(len(ife)), N_IFE)):
        yield 'ifeval', f'ifeval/{ife[i]["key"]}', ife[i]['prompt'], None
    from evalplus.data import get_human_eval_plus
    for tid, p in get_human_eval_plus().items():
        yield 'humaneval', tid, ('Complete the following Python function. Reply with the complete '
                                 'function (including imports it needs) in a single ```python code '
                                 'block.\n\n```python\n' + p['prompt'] + '```'), None


def run(label, out):
    key = sys.stdin.readline().strip()
    d = pathlib.Path(out) / label; d.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in d.glob('*.jsonl'):
        done |= {json.loads(l)['id'] for l in open(f)}
    n = 0
    for task, iid, prompt, gold in items():
        if iid in done:
            continue
        r = chat(key, prompt)
        r.update(id=iid, task=task, prompt=prompt, gold=gold)
        with open(d / f'{task}.jsonl', 'a') as fh:
            fh.write(json.dumps(r) + '\n')
        n += 1
        print(task, iid, r['finish'], r['usage'] and r['usage']['completion_tokens'],
              round(r['wall_s'], 1), flush=True)
    (d / 'run.done').write_text(f'{n} new items\n')


def num(s):
    s = s.replace(',', '').replace('$', '').strip().rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None


def score(d):
    d = pathlib.Path(d); res = {}
    rows = [json.loads(l) for l in open(d / 'gsm8k.jsonl')]
    ok = 0
    for r in rows:
        gold = num(r['gold'].split('####')[-1])
        m = re.findall(r'Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)', r['response'])
        pred = num(m[-1]) if m else None
        if pred is None:
            nums = re.findall(r'-?[\d,]*\.?\d+', r['response'])
            pred = num(nums[-1]) if nums else None
        r['correct'] = pred is not None and gold is not None and abs(pred - gold) < 1e-6
        ok += r['correct']
    res['gsm8k'] = dict(n=len(rows), accuracy=ok / len(rows),
                        truncated=sum(r['finish'] == 'length' for r in rows))
    sys.path.insert(0, str(DATA))
    from instruction_following_eval import evaluation_lib
    rows = [json.loads(l) for l in open(d / 'ifeval.jsonl')]
    inputs = {x.prompt: x for x in evaluation_lib.read_prompt_list(str(DATA / 'ifeval_input_data.jsonl'))}
    p2r = {r['prompt']: r['response'] for r in rows}
    for mode, fn in (('strict', evaluation_lib.test_instruction_following_strict),
                     ('loose', evaluation_lib.test_instruction_following_loose)):
        outs = [fn(inputs[r['prompt']], p2r) for r in rows]
        res[f'ifeval_{mode}'] = dict(
            n=len(outs), prompt_level=sum(o.follow_all_instructions for o in outs) / len(outs),
            instruction_level=(sum(sum(o.follow_instruction_list) for o in outs)
                               / sum(len(o.follow_instruction_list) for o in outs)))
    res['ifeval_truncated'] = sum(r['finish'] == 'length' for r in rows)
    rows = [json.loads(l) for l in open(d / 'humaneval.jsonl')]
    from evalplus.data import get_human_eval_plus
    probs = get_human_eval_plus()
    with open(d / 'humaneval_samples.jsonl', 'w') as fh:
        for r in rows:
            blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', r['response'], re.S)
            code = blocks[-1] if blocks else r['response']
            if f"def {probs[r['id']]['entry_point']}" not in code:
                code = probs[r['id']]['prompt'] + code
            fh.write(json.dumps({'task_id': r['id'], 'solution': code}) + '\n')
    ev = subprocess.run([sys.executable, '-m', 'evalplus.evaluate', 'humaneval',
                         '--samples', str(d / 'humaneval_samples.jsonl'), '--i_just_wanna_run'],
                        capture_output=True, text=True)
    (d / 'evalplus.log').write_text(ev.stdout + ev.stderr)
    for line in ev.stdout.splitlines():
        m = re.search(r'pass@1:\s*([0-9.]+)', line)
        if m:
            res.setdefault('humaneval_pass1', []).append(float(m.group(1)))
    res['humaneval'] = dict(n=len(rows), base_pass1=(res.get('humaneval_pass1') or [None])[0],
                            plus_pass1=(res.get('humaneval_pass1') or [None, None])[-1],
                            truncated=sum(r['finish'] == 'length' for r in rows))
    res.pop('humaneval_pass1', None)
    tok = [json.loads(l)['usage']['completion_tokens'] for f in ('gsm8k', 'ifeval', 'humaneval')
           for l in open(d / f'{f}.jsonl')]
    res['completion_tokens_mean'] = sum(tok) / len(tok)
    (d / 'score.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    {'run': lambda: run(sys.argv[2], sys.argv[3]), 'score': lambda: score(sys.argv[2])}[sys.argv[1]]()
