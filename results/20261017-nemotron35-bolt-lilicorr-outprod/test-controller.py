#!/usr/bin/env python3
"""Remote CPU-only static checks; actual GPU/API checks are separate gates."""
import ast
from pathlib import Path
import runpy
import socket
import subprocess

assert socket.gethostname() == 'inference-host'
path = Path('run-outprod-activation.py')
ast.parse(path.read_text(), filename=str(path))
ast.parse(Path('analyze-and-verify.py').read_text(), filename='analyze-and-verify.py')
module = runpy.run_path(str(path))
assert module['ROOT'] != module['DRAFT_ROOT']
assert module['CANDIDATE'].name.endswith('lilicorr-outprod-r1')
assert module['ACCEPTANCE_RE'].findall('accepted  0/ 4 draft tokens\naccepted  2/ 4 draft tokens\naccepted  4/ 4 draft tokens') == [('0', '4'), ('2', '4'), ('4', '4')]
spec = module['SPEC']
assert spec[spec.index('--spec-draft-n-max') + 1] == '4'
assert '--no-spec-draft-backend-sampling' in spec
assert not any('synth' in option for option in spec)
for filename in ('build-candidate.sh', 'native-outprod.sh'):
    subprocess.run(['bash', '-n', filename], check=True)
    print('BASH_SYNTAX_PASS', filename)
print('CONTROLLER_SYNTAX_AND_FLAGS_PASS')
print('PADDED_ACCEPTANCE_PARSER_PASS')
