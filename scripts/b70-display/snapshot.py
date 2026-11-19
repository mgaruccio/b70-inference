#!/usr/bin/env python3
"""Read only llama.cpp service health and completed-request timings inside b70-dev."""
import json
import os
import re
import subprocess
import urllib.error
import urllib.request

UNIT = 'flint-inference.service'
TIMING = re.compile(
    r'^\S+ I slot print_timing: id\s+(\d+) \| task\s+(\d+) \|\s+'
    r'(prompt eval|eval) time\s*=\s*\d+\.\d+ ms /\s*(\d+) tokens '
    r'\(\s*\d+\.\d+ ms per token,\s*(\d+\.\d+) tokens per second\)$'
)


def latest_request(lines):
    """Expect newest first; pair prompt with the same slot/task as the latest decode."""
    latest = None
    for line in lines:
        entry = json.loads(line)
        match = TIMING.fullmatch(entry.get('MESSAGE', ''))
        if not match:
            continue
        slot, task, phase, tokens, rate = match.groups()
        if latest is None and phase == 'eval':
            latest = dict(decode=float(rate), tokens=int(tokens),
                          completed=int(entry['__REALTIME_TIMESTAMP']) // 1_000_000)
            request = slot, task
        elif latest is not None and (slot, task) == request:
            if phase == 'prompt eval':
                latest['prompt'] = float(rate)
            break
    return latest


def health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open('http://127.0.0.1:8080/health', timeout=1) as response:
            return 'READY' if json.loads(response.read(4096)).get('status') == 'ok' else 'UNAVAILABLE'
    except urllib.error.HTTPError as error:
        return 'LOADING' if error.code == 503 else 'UNAVAILABLE'
    except (OSError, ValueError):
        return 'UNAVAILABLE'


def main():
    if os.environ.get('SSH_ORIGINAL_COMMAND') != 'snapshot':
        raise SystemExit('Only the snapshot command is permitted')
    state = health()
    invocation = subprocess.check_output(
        ['/usr/bin/systemctl', 'show', UNIT, '-p', 'InvocationID', '--value'],
        text=True, timeout=2,
    ).strip()
    stats = None
    if re.fullmatch(r'[0-9a-f]{32}', invocation) and invocation != '0' * 32:
        result = subprocess.run([
            '/usr/bin/journalctl', '--boot', '--unit', UNIT, '--reverse',
            '--since', '24 hours ago', '--lines', '64', '--no-pager',
            '--grep', 'I slot print_timing:.*eval time', '--output=json',
            '--output-fields=MESSAGE,__REALTIME_TIMESTAMP', f'_SYSTEMD_INVOCATION_ID={invocation}',
        ], capture_output=True, text=True, timeout=3)
        # journalctl --grep exits 1 when there are no matching records.
        if result.returncode and not (result.returncode == 1 and not result.stdout and not result.stderr):
            result.check_returncode()
        stats = latest_request(result.stdout.splitlines())
    print(json.dumps({'state': state, 'latest': stats}, allow_nan=False))


if __name__ == '__main__':
    main()
