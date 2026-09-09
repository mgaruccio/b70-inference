#!/usr/bin/env python3
"""One-off public-API regression replay; keep with this run's artifacts."""
import argparse
import datetime
import json
from pathlib import Path
import signal
import time

import qwen38_dflash2_probe as dflash
import qwen38_long_context_bench as cold
from qwen38_standard_bench import run_logged

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--draft-int4', type=Path)
p.add_argument('--guard', type=Path, required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
args.mode, args.context, args.graph, args.audit, args.suite = 'dflash2', 32768, True, False, 'boundary-fix'
args.patch = root / 'patch-vllm-qwen38-dflash2-bf16.py'
args.prefill_patch = root / 'patch-vllm-qwen38-xpu-prefill.py'
def interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')
for signum in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, interrupted)
cell = dflash.DFlashCell(args)
cell.summary['started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(cell.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
(cell.out / Path(cold.__file__).name).write_bytes(Path(cold.__file__).read_bytes())
try:
    cell.start()
    run_logged(['docker', 'exec', 'qwen38', 'python', '-m', 'vllm.collect_env'], cell.out / 'collect_env.txt')
    run_logged(['docker', 'exec', 'qwen38', 'cat', '/model/config.json'], cell.out / 'target-config.json')
    cell.gates()
    cell.quality()
    run_logged(['python3', cold.__file__, '--out', str(cell.out / 'long-context'),
                '--confirm-prefix-cache-disabled'], cell.out / 'long-context-console.txt', timeout=3600)
    summary = json.loads((cell.out / 'long-context/summary.json').read_text())
    assert summary['status'] == 'completed', summary['status']
    api = cold.PublicAPI('http://127.0.0.1:8000')
    original = json.loads((cell.out / 'long-context/points/length-32640/measured-01/request.json').read_text())
    nearby = []
    for length in range(32634, 32640):
        out = cell.out / 'nearby' / str(length)
        out.mkdir(parents=True)
        request = dict(original)
        # Remove only middle-body tokens: retain the original template/header/footer.
        request['prompt'] = original['prompt'][:128] + original['prompt'][128 + 32640 - length:]
        assert len(request['prompt']) == length
        (out / 'request.json').write_text(json.dumps(request) + '\n')
        (out / 'metrics-before.prom').write_bytes(api.get('/metrics').body)
        events = []
        started = time.monotonic()
        try:
            for event in api.stream('/v1/completions', request):
                events.append(event)
        finally:
            (out / 'sse.jsonl').write_text(''.join(json.dumps({'monotonic_s': t, 'raw': raw.decode(errors='replace') if isinstance(raw, bytes) else raw}) + '\n' for t, raw in events))
        parsed = cold.parse_sse_events(events)
        result = {'requested_length': length, 'elapsed_s': time.monotonic() - started, 'stream': parsed}
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        assert not parsed['parse_errors'], parsed['parse_errors']
        assert parsed['done_monotonic'] is not None and parsed['finish_reason'] == 'length'
        assert parsed['usage']['prompt_tokens'] == length and parsed['usage']['completion_tokens'] == 128
        assert api.get('/health').status == 200
        (out / 'metrics-after.prom').write_bytes(api.get('/metrics').body)
        nearby.append({'prompt_tokens': length, 'completion_tokens': 128, 'health': 200})
        print('NEARBY_PASS=' + json.dumps(nearby[-1]), flush=True)
    cell.summary.update(status='boundary_api_passed', nearby=nearby)
except (Exception, KeyboardInterrupt) as error:
    cell.summary.update(status='failed', error=repr(error))
    raise
finally:
    cell.summary['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cell.close()
