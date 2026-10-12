#!/usr/bin/env python3
"""Run only the authorized cells, stopping on any diagnostic/restoration failure."""
import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    controller = Path(__file__).resolve().with_name('run-pilot.py')
    (root / 'run-campaign.py').write_bytes(Path(__file__).read_bytes())
    (root / 'run-pilot.py').write_bytes(controller.read_bytes())
    cells = list(itertools.product((8192, 8320, 6656), ('on', 'off'), (16384, 32768)))
    random.Random(20261012).shuffle(cells)
    # Separate baseline-after anchors for the previously measured cache-off path.
    cells.extend([(8192, 'off', 16384), (8192, 'off', 32768)])
    state = {'tier': 'development', 'seed': 20261012, 'order': cells, 'completed': [], 'status': 'running'}
    def save():
        (root / 'campaign.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    save()
    try:
        for index, (budget, cache, length) in enumerate(cells):
            name = f'{index:02d}-{budget}-{cache}-{length}'
            with (root / (name + '.controller.log')).open('w') as log:
                result = subprocess.run([sys.executable, '-u', str(root / 'run-pilot.py'),
                    '--output', str(root / name), '--budget', str(budget),
                    '--prefix-cache', cache, '--length', str(length)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Cell {name} failed (exit {result.returncode}); see retained log')
            summary = json.loads((root / name / 'summary.json').read_text())
            assert summary['status'] == 'captured-needs-review' and summary['cleanup']['production_running']
            state['completed'].append(name)
            save()
            print('Captured and restored production:', name, flush=True)
        state['status'] = 'captured-needs-review'
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
