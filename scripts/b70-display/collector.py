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


def snapshot():
    raw = subprocess.check_output([
        '/usr/bin/ssh', '-F', '/dev/null', '-nT', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=3', '-o', 'StrictHostKeyChecking=yes',
        '-o', f'UserKnownHostsFile={CONFIG / "known_hosts"}',
        '-o', 'HostKeyAlgorithms=ssh-ed25519', '-o', 'IdentitiesOnly=yes',
        '-o', 'IdentityAgent=none', '-i', str(CONFIG / 'snapshot_ed25519'),
        'mike@192.168.122.70', 'snapshot',
    ], text=True, stderr=subprocess.DEVNULL, timeout=8)
    return json.loads(raw)


def rate(value):
    if type(value) in (int, float) and math.isfinite(value) and 0 <= value < 1_000_000:
        return f'{value:,.1f}'
    return '--'


def display_values(data):
    state = data['state']
    if state not in STATES | {'OFFLINE'}:
        raise ValueError('Invalid inference state')
    values = dict(llm_state=state, llm_prompt='--', llm_decode='--', llm_last='No recent completion')
    latest = data.get('latest')
    if latest:
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
        '/usr/bin/busctl', '--user', '--timeout=4s', 'call',
        'com.thermalwriter.Service', '/com/thermalwriter/display',
        'com.thermalwriter.Display', 'SetLayoutVars', 'sa{ss}',
        LAYOUT, str(len(values)), *args,
    ], check=True, timeout=5)


def main():
    previous = None
    while True:
        try:
            values = display_values(snapshot())
        except (OSError, ValueError, KeyError, TypeError, OverflowError, subprocess.SubprocessError):
            values = display_values({'state': 'OFFLINE'})
        try:
            if not layout_is_active():
                previous = None
            elif values != previous:
                push(values)
                previous = values
                print('Display updated: ' + str(values), flush=True)
        except (OSError, subprocess.SubprocessError):
            print('ThermalWriter unavailable; will retry', flush=True)
        time.sleep(10)


if __name__ == '__main__':
    main()
