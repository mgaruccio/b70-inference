#!/usr/bin/env python3
"""Update the ThermalWriter portrait layout from b70-dev's restricted snapshot."""
from datetime import datetime
from pathlib import Path
import json
import math
import subprocess
import time

CONFIG = Path.home() / '.config/thermalwriter/b70-display'
LAYOUT = 'svg/b70-inference.svg'
STATES = {'READY', 'LOADING', 'UNAVAILABLE'}
POLL_SECONDS = 2
MAX_RATE_WINDOW = POLL_SECONDS * 2.5


def snapshot():
    raw = subprocess.check_output([
        '/usr/bin/ssh', '-F', '/dev/null', '-nT', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=3', '-o', 'StrictHostKeyChecking=yes',
        '-o', f'UserKnownHostsFile={CONFIG / "known_hosts"}',
        '-o', 'HostKeyAlgorithms=ssh-ed25519', '-o', 'IdentitiesOnly=yes',
        '-o', 'ControlMaster=auto', '-o', 'ControlPersist=5s',
        '-o', f'ControlPath={CONFIG / "snapshot.sock"}', '-o', 'ServerAliveInterval=3',
        '-o', 'IdentityAgent=none', '-i', str(CONFIG / 'snapshot_ed25519'),
        'mike@192.168.122.70', 'snapshot',
    ], text=True, stderr=subprocess.DEVNULL, timeout=8)
    return json.loads(raw)


def rate(value):
    if type(value) in (int, float) and math.isfinite(value) and 0 <= value < 1_000_000:
        return f'{value:,.1f}'
    return '--'


def live_rate(current, prior, field):
    if not prior or current['invocation'] != prior.get('invocation') or not prior.get('slots'):
        return None
    seconds = current['sampled'] - prior['sampled']
    if not 0.25 <= seconds <= MAX_RATE_WINDOW:
        return None
    prior_counts = {(s['slot'], s['task']): s[field] for s in prior['slots']}
    delta = 0
    for slot in current['slots']:
        before = prior_counts.get((slot['slot'], slot['task']))
        if before is not None and slot[field] > before:
            delta += slot[field] - before
    return delta / seconds if delta else None


def display_values(data, prior=None):
    state = data['state']
    if state not in STATES | {'OFFLINE'}:
        raise ValueError('Invalid inference state')
    values = dict(llm_state=state, llm_title='LAST COMPLETED REQUEST', llm_prompt='--',
                  llm_decode='--', llm_last='No recent completion')
    slots = data.get('slots')
    if slots:
        tokens = sum(s['decode'] for s in slots)
        phase = 'Generating' if tokens else 'Processing prompt'
        values.update(llm_title='LIVE / ACTIVE REQUESTS',
                      llm_prompt=rate(live_rate(data, prior, 'prompt')),
                      llm_decode=rate(live_rate(data, prior, 'decode')),
                      llm_last=f'{phase} / {tokens:,d} output tokens')
    elif latest := data.get('latest'):
        completed = datetime.fromtimestamp(latest['completed']).astimezone()
        today = datetime.now(completed.tzinfo).date()
        stamp = completed.strftime('%H:%M' if completed.date() == today else '%b %d %H:%M')
        values.update(llm_prompt=rate(latest.get('prompt')), llm_decode=rate(latest.get('decode')),
                      llm_last=f'{stamp} / {latest["tokens"]:,d} output tokens')
    return values


def layout_is_active():
    result = subprocess.check_output([
        '/usr/bin/busctl', '--user', '--timeout=4s', '--json=short', 'get-property',
        'com.thermalwriter.Service', '/com/thermalwriter/display',
        'com.thermalwriter.Display', 'ActiveLayout',
    ], text=True, timeout=5)
    return json.loads(result)['data'] == LAYOUT

def push(values):
    args = [x for pair in values.items() for x in pair]
    subprocess.run([
        '/usr/bin/busctl', '--user', '--timeout=4s', '--', 'call',
        'com.thermalwriter.Service', '/com/thermalwriter/display',
        'com.thermalwriter.Display', 'SetLayoutVars', 'sa{ss}',
        LAYOUT, str(len(values)), *args,
    ], check=True, timeout=5)


def main():
    previous = prior = None
    while True:
        started = time.monotonic()
        try:
            if not layout_is_active():
                previous = prior = None
            else:
                try:
                    data = snapshot()
                    values = display_values(data, prior)
                    prior = data
                except (OSError, ValueError, KeyError, TypeError, OverflowError, subprocess.SubprocessError):
                    prior = None
                    values = display_values({'state': 'OFFLINE'})
                if values != previous:
                    push(values)
                    previous = values
                    print('Display updated: ' + str(values), flush=True)
        except (OSError, subprocess.SubprocessError):
            prior = None
            print('ThermalWriter unavailable; will retry', flush=True)
        time.sleep(max(0.5, POLL_SECONDS - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
