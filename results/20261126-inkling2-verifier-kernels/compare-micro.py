"""Print kernel timings in run order; mismatching fixture outputs invalidate a run."""
import json
import sys
from pathlib import Path

paths = list(map(Path, sys.argv[1:]))
runs = [[(f['fixture'], f['unique_experts'], f['output_sha256'], f['median_us'])
         for f in json.loads(path.read_text())['fixtures']] for path in paths]
for path, run in zip(paths, runs):
    if [r[:3] for r in run] != [r[:3] for r in runs[0]]:
        raise ValueError(f'Fixture or exact output mismatch in {path.name}')
print('fixture,experts,' + ','.join(p.stem for p in paths))
for group in zip(*runs):
    print(f'{group[0][0]},{group[0][1]},' + ','.join(f'{r[3]:.2f}' for r in group))
