#!/usr/bin/env python3
"""Temporary read-only MTP metrics bridge; uses stdlib, existing logs and SSH.

Run --snapshot on the training host, or serve from the desktop with --ssh.
No weights are read, no trainer process is changed, and no history is stored.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def tail(path, limit=524288):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit))
            data = f.read().decode('utf-8', errors='replace')
        return data.split('\n', 1)[1] if size > limit else data
    except FileNotFoundError:
        return ''


def read_json(path):
    return json.loads(path.read_text())


def active_commands(root):
    commands = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if (args and 'python' in Path(args[0]).name
                    and any(a.endswith('/glimmer_recursive_mtp.py') for a in args)
                    and any(a.startswith(str(root) + '/') for a in args)):
                commands.append(args)
        except (OSError, UnicodeError):
            continue
    return commands


def snapshot(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'run directory missing: {root}')
    commands = active_commands(root)
    result = {'observed_at': time.time(), 'active': bool(commands), 'commands': commands,
              'training': [], 'captures': [], 'complete': (root / 'stage0/stage0-complete.json').exists()}
    # Bounded log reads; never load checkpoint tensors into the monitoring process.
    for path in sorted(root.glob('**/*-training.jsonl')):
        rows = []
        for line in tail(path).splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A trainer may currently be appending its last line.
            if isinstance(row, dict) and 'update' in row:
                rows.append(row)
        if rows:
            latest = rows[-1].copy()
            latest['validation'] = next((r['validation'] for r in reversed(rows) if 'validation' in r), None)
            latest['probe'] = next((r['probe'] for r in reversed(rows) if 'probe' in r), None)
            latest['diagnostic'] = next((r['training_set_diagnostic_NOT_validation'] for r in reversed(rows)
                                       if 'training_set_diagnostic_NOT_validation' in r), None)
            result['training'].append({'stage': str(path.parent.relative_to(root)), 'mtime': path.stat().st_mtime,
                                       'latest': latest})
    for path in sorted(root.glob('**/capture/index.json')):
        data = read_json(path)
        result['captures'].append({'stage': str(path.parent.relative_to(root)), 'mtime': path.stat().st_mtime,
                                   'status': data['status'], 'token_count': data.get('token_count'),
                                   'root_count': data.get('root_count'), 'splits': data.get('split_summaries', {})})
    result['validation_reports'] = []
    for path in sorted(root.glob('**/validation.json')):
        data = read_json(path)
        if data.get('split') != 'validation':
            raise ValueError(f'not a heldout validation report: {path}')
        result['validation_reports'].append({'stage': str(path.parent.relative_to(root)) + '/selected-head',
                                             'offline': data['offline'], 'probe': data['probe']})
    logs = list(root.glob('**/*.log'))
    if logs:
        latest_log = max(logs, key=lambda p: p.stat().st_mtime)
        text = tail(latest_log)
        result['latest_log_at'] = latest_log.stat().st_mtime
        result['latest_log'] = str(latest_log.relative_to(root))
        result['latest_log_error'] = any(t in text for t in ('Traceback (most recent call last)',
                                                           'Fatal Python error:', 'AssertionError:',
                                                           'CUDA out of memory'))
    try:
        completed = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                   check=True, timeout=5)
        result['gpu'] = [list(map(float, line.split(','))) for line in completed.stdout.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError):
        result['gpu'] = None
    return result


def labels(values):
    def escape(value):
        return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')
    return '{' + ','.join(f'{k}="{escape(v)}"' for k, v in values.items()) + '}' if values else ''


def render(data, reachable=True):
    lines = []
    def metric(name, value, **tags):
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)) and math.isfinite(value):
            lines.append(f'mtp_{name}{labels(tags)} {value}')
    metric('source_reachable', reachable)
    if not reachable:
        return '\n'.join(lines) + '\n'
    metric('observed_timestamp_seconds', data['observed_at'])
    metric('process_active', data['active'])
    # 0=stopped/idle (not proof of failure), 1=capture, 2=train, 3=validation/eval,
    # 4=Stage-0 complete ONLY, 5=observed error with no active process.
    phase = 0
    for args in data['commands']:
        phase = 1 if 'capture-generated' in args else 2 if 'train' in args else 3
    if not data['active']:
        if data.get('latest_log_error'):
            phase = 5
        elif data['complete']:
            phase = 4
    metric('phase', phase)
    metric('latest_log_timestamp_seconds', data.get('latest_log_at'))
    metric('latest_log_error', data.get('latest_log_error', False))
    for capture in data['captures']:
        tags = {'stage': capture['stage']}
        metric('capture_complete', capture['status'] == 'complete', **tags)
        metric('capture_sequence_tokens', capture['token_count'], **tags)
        metric('capture_unique_roots', capture['root_count'], **tags)
        for split, values in capture['splits'].items():
            metric('capture_split_tokens', values['sequence_token_count'], split=split, **tags)
            metric('capture_split_budget', values['budget'], split=split, **tags)
            metric('capture_split_roots', values['root_count'], split=split, **tags)
    for training in data['training']:
        row = training['latest']
        tags = {'stage': training['stage'], 'variant': row['variant']}
        metric('training_log_timestamp_seconds', training['mtime'], **tags)
        for key in ('update', 'planned_updates', 'depth', 'lr', 'loss', 'grad_norm',
                    'root_exposures', 'loss_position_exposures', 'eta_s', 'elapsed_s'):
            metric('train_' + key, row.get(key), **tags)
        for depth in row.get('per_depth', []):
            for key in ('ce', 'teacher_kl', 'normalized_mse', 'cosine_distance'):
                metric('train_' + key, depth.get(key), depth=str(depth['depth']), **tags)
        for kind in ('validation', 'diagnostic'):
            values = row.get(kind)
            if not values:
                continue
            for depth in values.get('per_depth', []):
                for key in ('teacher_kl', 'teacher_argmax_agreement', 'normalized_mse', 'cosine_distance'):
                    metric(kind + '_' + key, depth.get(key), depth=str(depth['depth']), **tags)
        probe = row.get('probe')
        if probe:
            metric('validation_draft_acceptance', probe.get('draft_acceptance_rate'), **tags)
    for report in data.get('validation_reports', []):
        tags = {'stage': report['stage'], 'variant': 'selected-head'}
        metric('validation_draft_acceptance', report['probe']['draft_acceptance_rate'], **tags)
        for depth in report['offline']['per_depth']:
            for key in ('teacher_kl', 'teacher_argmax_agreement', 'normalized_mse', 'cosine_distance'):
                metric('validation_' + key, depth.get(key), depth=str(depth['depth']), **tags)
    for i, gpu in enumerate(data.get('gpu') or []):
        metric('gpu_utilization_percent', gpu[0], gpu=str(i))
        metric('gpu_memory_used_bytes', gpu[1] * 1024**2, gpu=str(i))
        metric('gpu_memory_total_bytes', gpu[2] * 1024**2, gpu=str(i))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--ssh')
    parser.add_argument('--remote-script', default='/home/shadeform/mtp-training-code/scripts/experiments/glimmer_mtp_metrics.py')
    parser.add_argument('--listen', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9110)
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(snapshot(args.run_dir), allow_nan=False))
        return
    if not args.ssh:
        parser.error('serving requires --ssh; --snapshot is the remote read-only command')
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', args.ssh,
               shlex.join(['python3', args.remote_script, '--snapshot', '--run-dir', args.run_dir])]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/metrics':
                self.send_error(404)
                return
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=12, check=True)
                payload = render(json.loads(process.stdout)).encode()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                print(f'source observation failed: {exc}', flush=True)
                # Reachability failure must remove current training gauges, not replay stale values.
                payload = render({}, reachable=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    HTTPServer((args.listen, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
