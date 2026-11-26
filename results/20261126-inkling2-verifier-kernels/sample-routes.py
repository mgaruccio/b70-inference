"""Development-only routing sample on an eager loopback engine. Do not time this run."""
import json
import statistics
import sys
from pathlib import Path

from checks import MODEL, PROMPTS, fixture, post

root = Path(sys.argv[1])
for n in (1, 8, 32):
    for kind in ('text', 'random'):
        label = f'{kind}-c{n}'
        prompts = ([f'Example {i}: ' + PROMPTS[i % len(PROMPTS)] for i in range(n)] if kind == 'text'
                   else [fixture(512, 42+i) for i in range(n)])
        marker = root/'case'
        marker.write_text(label)
        try:
            data, _ = post('/v1/completions', dict(model=MODEL, prompt=prompts, temperature=0,
                                                   ignore_eos=True, max_tokens=96, seed=42))
            assert data['usage']['completion_tokens'] == 96*n
        finally:
            marker.unlink()
        width = 8 if n == 1 else 4*n
        rows = [json.loads(row) for row in (root/'visits.jsonl').read_text().splitlines()]
        selected = [row[-1] for row in rows if row[0] == label and row[3] == row[4] == width]
        if not selected:
            raise RuntimeError(f'{label}: no full batch of width {width}')
        print(label, len(selected), 'samples, median expert union', statistics.median(selected), flush=True)
