#!/usr/bin/env python3
"""Targeted analyzer checks; real trace/API execution remains required."""
import gzip
import json
from pathlib import Path
import runpy
import tempfile

analyze = runpy.run_path(str(Path(__file__).with_name('analyze-shapes.py')))['analyze']
with tempfile.TemporaryDirectory() as temp:
    cell = Path(temp)
    (cell / 'torch').mkdir()
    (cell / 'iterations.json').write_text(json.dumps([{'context_tokens': 6656}, {'context_tokens': 0}]))
    (cell / 'server.log').write_text('onednn_verbose,primitive,create:cache_miss,gpu,matmul,fixture\n')
    native = {'cat': 'cpu_op', 'name': '_xpu_C::int4_gemm_w4a16', 'ts': 12, 'dur': 100000,
              'args': {'External id': 41, 'Input Dims': [[6656, 128], [16, 256], [], [1, 256], [1], [], []]}}
    events = [
        {'cat': 'user_annotation', 'name': 'execute_context_0(6656)', 'ts': 10, 'dur': 10},
        native,
        {'cat': 'xpu_runtime', 'ts': 13, 'name': 'launch', 'args': {'correlation': 51, 'External id': 41}},
        {'cat': 'kernel', 'ph': 'X', 'name': 'gemm_kernel', 'ts': 1000, 'dur': 2000,
         'args': {'correlation': 51}},
        {'cat': 'kernel', 'ph': 'X', 'name': 'gemm_kernel', 'ts': 4000, 'dur': 500,
         'args': {'External id': 999}},
    ]
    path = cell / 'torch/test.pt.trace.json.gz'
    def write():
        with gzip.open(path, 'wt') as stream:
            json.dump({'traceEvents': events}, stream)
    write()
    report = analyze(cell)
    row = report['shapes'][0]
    assert (row['M'], row['K'], row['N']) == (6656, 128, 256)
    assert row['device_ms'] == 2.0, 'Do not count asynchronous CPU scope as GPU time'
    assert row['kernel_calls'] == 1 and row['phase_operator_calls'] == {'6656': 1}
    assert report['unmapped_generic_gemm_kernels'] == 1
    assert report['onednn_matmul_create_lines'] == 1
    native['args']['Input Dims'][1][0] = 17
    write()
    try:
        analyze(cell)
    except AssertionError:
        pass
    else:
        raise AssertionError('Must reject incompatible packed dimensions')
parse_u4 = runpy.run_path(str(Path(__file__).with_name('analyze-shapes.py')))['parse_u4_creation']
records = parse_u4([
    '[EngineCore] onednn_verbose,v1,123,primitive,create:cache_miss,gpu,matmul,jit:gemm:any,undef,src:f16 wei:u4,attrs,,6656x5120:5120x34816,0.1',
    'onednn_verbose,v1,124,primitive,create:dispatch,gemm,gpu,gemm,jit:xe_hp:gemm:any,undef,src_a:f16 src_b:u4,attrs,,6656x5120:5120x34816,unsupported datatype combination,source.cpp:106',
    'onednn_verbose,v1,125,primitive,create:cache_hit,gpu,matmul,jit:gemm:any,undef,src:f16 wei:f16,attrs,,8192x5120:5120x96,0.1',
])
assert len(records) == 2
assert records[0]['implementation'] == 'jit:gemm:any'
assert records[1]['implementation'] == 'jit:xe_hp:gemm:any'
assert records[1]['reason'] == 'unsupported datatype combination'
assert records[0]['problem'] == records[1]['problem'] == '6656x5120:5120x34816'
print('3 targeted shape/correlation/dispatch checks passed')
