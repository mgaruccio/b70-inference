#!/usr/bin/env python3
"""Compare this completed control with the saved parent captures; no inference."""
import json
import runpy
import socket
from pathlib import Path

assert socket.gethostname() == 'inference-host'
ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-legacy-scan')
PARENT = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-verify-trace'
analyzer = runpy.run_path(str(ROOT / 'analyze-trace.py'))
load = analyzer['load']
assert load(ROOT / 'run-r1/analysis.json')['restoration_verified']
result = {}
for label, prior in (
    ('S0', PARENT / 'run-r1/S0/alignment-after-prefill-fix.json'),
    ('C0', PARENT / 'continuation-c0/run-r1/C0/alignment.json'),
):
    current = ROOT / 'run-r1' / label / 'alignment.json'
    old, new = load(prior), load(current)
    assert old['emitted_ids'] == new['emitted_ids'] and old['prompt_ids'] == new['prompt_ids']
    assert len(old['samples']) == len(new['samples']) == 128
    rows = []
    for i, (x, y) in enumerate(zip(old['samples'], new['samples'])):
        assert x['position'] == y['position'] and x['vocab'] == y['vocab']
        xb = analyzer['raw_bytes'](prior.parent / 'capture', x)
        yb = analyzer['raw_bytes'](current.parent / 'capture', y)
        ob, nb = old['batches'][x['batch_index']], new['batches'][y['batch_index']]
        rows.append(dict(ordinal=i + 1, raw_bytes_equal=xb == yb,
                         batch_row_equal=x['batch_row'] == y['batch_row'],
                         batch_inputs_equal=all(ob[k] == nb[k] for k in
                                                ('off', 'n', 'tokens', 'pos', 'outputs', 'seq'))))
    result[label] = dict(prior_alignment=str(prior), new_alignment=str(current),
                         prior_capture=str(prior.parent / 'capture'),
                         new_capture=str(current.parent / 'capture'),
                         all_emitted_ids_equal=True, accepted=new['accepted'],
                         proposed=new['proposed'], rejections=new['rejection_events'], rows=rows)
    print(label, 'equal raw rows', sum(r['raw_bytes_equal'] for r in rows), '/', len(rows),
          'equal batch inputs', sum(r['batch_inputs_equal'] for r in rows))
(ROOT / 'legacy-vs-parent.json').write_text(json.dumps(result, indent=2) + '\n')
