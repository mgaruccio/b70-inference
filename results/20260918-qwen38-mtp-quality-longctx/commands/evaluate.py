"""Trusted wrapper: invoked ONLY inside the isolated, offline evaluator container."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.umask(0o077)
work = Path('/tmp/evaluation')
work.mkdir()
shutil.copyfile('/samples.jsonl', work / 'samples.jsonl')
command = [sys.executable, '-m', 'evalplus.evaluate', '--dataset', 'humaneval',
           '--samples', str(work / 'samples.jsonl'), '--parallel', '2',
           '--min-time-limit', '1', '--gt-time-limit-factor', '4']
process = subprocess.run(command, cwd=work, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=1700)
result_path = work / 'samples.eval_results.json'
result = json.loads(result_path.read_text()) if result_path.is_file() else None
print(json.dumps({'command': command, 'returncode': process.returncode,
                  'log': process.stdout, 'results': result}), flush=True)
if process.returncode or result is None:
    raise SystemExit(process.returncode or 1)
