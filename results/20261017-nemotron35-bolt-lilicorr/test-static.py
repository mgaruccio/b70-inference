#!/usr/bin/env python3
"""Run on inference-host; CPU-only syntax, GGUF reader fixture, and flag checks."""
import ast
import json
from pathlib import Path
import runpy
import socket
import struct
import tempfile

assert socket.gethostname() == 'inference-host'
for filename in ['prepare-draft.py', 'run-activation.py', 'verify-results.py']:
    ast.parse(Path(filename).read_text(), filename=filename)
    print('SYNTAX_PASS', filename)
module = runpy.run_path('prepare-draft.py')


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'sample.gguf'
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 1, 2)
                     + string('general.architecture') + struct.pack('<I', 8) + string('dflash')
                     + string('dflash.target_layer_ids') + struct.pack('<IIQII', 9, 4, 2, 10, 20)
                     + string('test.weight') + struct.pack('<IQQIQ', 2, 4, 8, 1, 0))
    metadata, tensors = module['header'](path)
    assert metadata == {'general.architecture': 'dflash', 'dflash.target_layer_ids': [10, 20]}
    assert tensors == [{'name': 'test.weight', 'dimensions': [4, 8], 'type': 1, 'offset': 0}]
    print('HEADER_SCALAR_ARRAY_TENSOR_PASS')
controller = runpy.run_path('run-activation.py')
assert controller['SPEC'][controller['SPEC'].index('--spec-draft-n-max') + 1] == '4'
assert '--no-spec-draft-backend-sampling' in controller['SPEC']
assert '--spec-synth-len' not in controller['SPEC']
print('SPEC_FLAGS_PASS')
padded = 'accepted  0/ 4 draft tokens\naccepted  2/ 4 draft tokens (restore checkpoint)\naccepted  4/ 4 draft tokens'
assert controller['ACCEPTANCE_RE'].findall(padded) == [('0', '4'), ('2', '4'), ('4', '4')]
print('PADDED_ACCEPTANCE_LOGS_PASS')
target = json.loads(Path('target-header.json').read_text())
draft = json.loads(Path('draft-header.json').read_text())
tm, dm = target['metadata'], draft['metadata']
arch = tm['general.architecture']
layers = dm['dflash.target_layers']
fc = next(t for t in draft['tensors'] if t['name'] == 'fc.weight')
assert tm[arch + '.embedding_length'] == dm['dflash.embedding_length']
assert all(0 <= i < tm[arch + '.block_count'] for i in layers)
assert fc['dimensions'] == [len(layers) * tm[arch + '.embedding_length'], dm['dflash.embedding_length']]
assert 4 <= dm['dflash.block_size'] - 1
print('REAL_DRAFT_TARGET_SHAPES_PASS')
print('BOLT_LILICORR_STATIC_CHECKS_PASSED')
