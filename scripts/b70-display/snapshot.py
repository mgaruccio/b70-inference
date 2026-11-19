#!/usr/bin/env python3
"""Return only llama.cpp health, token counters, and completed timings from b70-dev."""
from http.client import HTTPConnection, HTTPException
from pathlib import Path
import json
import os
import re
import subprocess
import time

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


def get_json(path, key=None):
    # No proxies or redirects: the existing bearer key never leaves guest loopback.
    with_connection = HTTPConnection('127.0.0.1', 8080, timeout=1.2)
    try:
        headers = {'Authorization': f'Bearer {key}'} if key else {}
        with_connection.request('GET', path, headers=headers)
        response = with_connection.getresponse()
        return response.status, json.loads(response.read(262145)) if response.status == 200 else None
    finally:
        with_connection.close()


def health():
    try:
        status, data = get_json('/health')
        if status == 503:
            return 'LOADING'
        if status == 200 and isinstance(data, dict) and data.get('status') == 'ok':
            return 'READY'
    except (OSError, ValueError, HTTPException):
        pass
    return 'UNAVAILABLE'


def server_key(pid):
    proc = Path('/proc') / str(pid)
    argv = (proc / 'cmdline').read_bytes().decode().split('\0')
    path = Path(argv[argv.index('--api-key-file') + 1])
    if not path.is_absolute():
        path = (proc / 'cwd') / path
    with path.open() as handle:
        return handle.readline(4096).strip()


def active_slots(data):
    if not isinstance(data, list):
        raise ValueError('Invalid slots response')
    slots = []
    for slot in data:
        if slot['is_processing'] is not True:
            continue
        token = slot.get('next_token')
        if isinstance(token, list):
            token = token[0] if token else None
        counts = (slot.get('id'), slot.get('id_task'),
                  slot.get('n_prompt_tokens_processed'),
                  token.get('n_decoded') if isinstance(token, dict) else None)
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError('Missing slot counters')
        slots.append(dict(zip(('slot', 'task', 'prompt', 'decode'), counts)))
    return slots


def read_slots(pid):
    try:
        key = server_key(pid)
        # Fail closed if the VM's expected key file is no longer in use.
        if key:
            status, data = get_json('/slots', key)
            if status == 200:
                return active_slots(data)
    except (OSError, ValueError, IndexError, KeyError, TypeError, HTTPException):
        pass
    return None


def main():
    if os.environ.get('SSH_ORIGINAL_COMMAND') != 'snapshot':
        raise SystemExit('Only the snapshot command is permitted')
    unit = subprocess.check_output([
        '/usr/bin/systemctl', 'show', UNIT, '-p', 'InvocationID', '-p', 'MainPID',
    ], text=True, timeout=2)
    props = dict(line.split('=', 1) for line in unit.splitlines())
    invocation = props['InvocationID']
    pid = int(props['MainPID'])
    slots = read_slots(pid) if pid > 0 else None
    sampled = time.monotonic()
    state = 'READY' if slots is not None else health()
    stats = None
    # No journal read during active requests; fetch last timings as soon as idle.
    if not slots and re.fullmatch(r'[0-9a-f]{32}', invocation) and invocation != '0' * 32:
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
    print(json.dumps({'state': state, 'latest': stats, 'slots': slots,
                      'sampled': sampled, 'invocation': invocation}, allow_nan=False))


if __name__ == '__main__':
    main()
