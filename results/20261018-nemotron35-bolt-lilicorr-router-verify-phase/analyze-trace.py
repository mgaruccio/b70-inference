#!/usr/bin/env python3
"""Pure native-log gate; neither public fidelity nor the separate N9 byte gate."""
import argparse
import json
import re
from pathlib import Path

if not __debug__:
    raise RuntimeError('do not disable assertions in this experimental gate')

ROUTER_WIDTHS = {
    'blk.1.ffn_gate_inp.weight': 240, 'blk.8.ffn_gate_inp.weight': 304,
    'blk.37.ffn_gate_inp.weight': 368, 'blk.21.ffn_gate_inp.weight': 432,
    'blk.48.ffn_gate_inp.weight': 496,
}
UNRELATED_WIDTHS = {('blk.1.unrelated.weight' if m == 240 else f'blk.1.width{m}.unrelated.weight'): m
                    for m in ROUTER_WIDTHS.values()}
TAGS = ('router_single2', 'router_columns8_r3', 'router_prefill')
ROUTE_STRINGS = {
    'route', 'weight', 'aux_weight', 'op', 'dst_name',
    'src0_type', 'aux_type', 'src1_type', 'dst_type', 'backend_ctx',
}
ROUTE_INTS = {'op_id', 'reorder', 'src1_prec', 'operand_prec', 'accum_prec',
              'dynamic_prec', 'device', 'expert', 'result'} | {
    f'{prefix}{kind}{i}' for prefix in ('', 'src1_', 'dst_', 'ids_')
    for kind in ('ne', 'nb') for i in range(4)
}
ROUTE_FIELDS = ROUTE_STRINGS | ROUTE_INTS
# Recognize exact terminal-colored OK in place. Never normalize raw log bytes.
OK = r'(?:\x1b\[[0-9;]*m)*OK(?:\x1b\[[0-9;]*m)*'


def native_cases(tag):
    assert tag in TAGS
    if tag == 'router_prefill':
        return {('blk.48.ffn_gate_inp.weight', 496, 0, n) for n in (4, 5)}
    ns, control_n = (range(1, 10), 8) if tag == 'router_columns8_r3' else ((1, 2, 3), 2)
    cases = {(w, m, 0, n) for w, m in ROUTER_WIDTHS.items() for n in ns}
    cases |= {(w, m, layout, control_n) for w, m in ROUTER_WIDTHS.items() for layout in (1, 2)}
    cases |= {(w, m, 0, control_n) for w, m in UNRELATED_WIDTHS.items()}
    if tag == 'router_columns8_r3':
        cases |= {('blk.1.width256.ffn_gate_inp.weight', 256, 0, n) for n in (4, 8)}
        cases.remove(('blk.48.ffn_gate_inp.weight', 496, 0, 9))
    assert len(cases) == (61 if tag == 'router_columns8_r3' else 30)
    return cases


def expected_accum_precision(key, tag):
    # Negative name/view/width fixtures are marked too; old30 and prefill are not.
    return 10 if tag == 'router_columns8_r3' and 3 <= key[-1] <= 8 else 0


def split_case(key, tag='router_columns8_r3'):
    weight, m, layout, n = key
    return (weight.endswith('.ffn_gate_inp.weight') and m in ROUTER_WIDTHS.values() and layout == 0
            and (n == 2 or (3 <= n <= 8 and expected_accum_precision(key, tag) == 10)))


def route_records(log):
    result = []
    for line in log.splitlines():
        if 'BOLT_ROUTE' not in line:
            continue
        assert 'BOLT_VERIFY' not in line and line.count('BOLT_ROUTE ') == 1, 'bad native route framing'
        tail = line.split('BOLT_ROUTE ', 1)[1]
        pairs = tail.split()
        assert pairs and all(re.fullmatch(r'[a-z_][a-z_0-9]*=[^\s=]+', p) for p in pairs), tail
        fields = [p.split('=', 1) for p in pairs]
        assert len({k for k, _ in fields}) == len(fields), 'duplicate route field'
        assert {k for k, _ in fields} == ROUTE_FIELDS, tail
        row = {}
        for key, value in fields:
            if key in ROUTE_INTS:
                assert re.fullmatch(r'-?\d+', value), tail
                row[key] = int(value)
            else:
                row[key] = value
        assert re.fullmatch(r'0x[0-9a-fA-F]+', row['backend_ctx']), row
        assert row['route'] in ('MUL_MAT_SYCL', 'LIBRARY_MKL_F32', 'LIBRARY_DNNL_F32'), row
        result.append(row)
    assert result, 'missing actual native route records (BOLT_VERIFY_CAPTURE_DIR must be set)'
    return result


def assert_tensor(route, prefix, shape, strides):
    assert tuple(route[f'{prefix}ne{i}'] for i in range(4)) == shape, route
    assert tuple(route[f'{prefix}nb{i}'] for i in range(4)) == strides, route


def assert_router_operations(routes, key, tag):
    weight, m, layout, n = key
    split = split_case(key, tag)
    calls, view_n = (n, 1) if split else (1, n)
    assert len(routes) == 2 * calls, ('outer/inner pair count', key, len(routes))
    stride = (4096 + (16 if layout == 2 else 0)) * 4
    # Existing F32 library dispatch threshold, with DNN enabled on the B70.
    library = 'LIBRARY_MKL_F32' if m * view_n * 4096 < 256**3 else 'LIBRARY_DNNL_F32'
    for outer, inner in zip(routes[::2], routes[1::2]):
        assert (outer['route'], inner['route']) == ('MUL_MAT_SYCL', library), (outer, inner)
        assert outer['operand_prec'] == -1 and inner['operand_prec'] == 10
        assert {k: v for k, v in outer.items() if k not in ('route', 'operand_prec')} == {
            k: v for k, v in inner.items() if k not in ('route', 'operand_prec')}
        for route in (outer, inner):
            assert route['weight'] == weight and route['op'] == 'MUL_MAT' and route['op_id'] == 29, route
            assert all(route[k] == 'f32' for k in ('src0_type', 'src1_type', 'dst_type')), route
            assert route['aux_weight'] == route['aux_type'] == '-', route
            assert route['reorder'] == route['device'] == 0, route
            assert route['src1_prec'] == route['dynamic_prec'] == 0, route
            assert route['accum_prec'] == expected_accum_precision(key, tag), route
            assert route['expert'] == route['result'] == -1, route
            assert_tensor(route, '', (4096, m, 1, 1), (4, 16384, 16384*m, 16384*m))
            assert_tensor(route, 'src1_', (4096, view_n, 1, 1), (4, stride, stride*n, stride*n))
            assert_tensor(route, 'dst_', (m, view_n, 1, 1), (4, 4*m, 4*m*n, 4*m*n))
            assert_tensor(route, 'ids_', (0, 0, 0, 0), (0, 0, 0, 0))
    assert len({r['backend_ctx'] for r in routes}) == len({r['dst_name'] for r in routes}) == 1


def check_router_native_output(log, tag='router_columns8_r3'):
    """Require numerical OK and exact precision/route pairs for every selected case."""
    expected = native_cases(tag)
    count = str(len(expected))
    assert re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log) == [(count, count)]
    assert re.findall(r'(?m)^Backend \d+/\d+: (\S+)\s*$', log) == ['SYCL0', 'CPU']
    backend_status = re.findall(r'(?m)^[ \t]*Backend SYCL0: ([^\r\n]*)$', log)
    assert len(backend_status) == 1 and re.fullmatch(OK + r'[ \t]*', backend_status[0])
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', log) == [('2', '2')]
    # Only the backend not selected by -b may skip. Keep all raw ANSI bytes.
    assert len(re.findall(r'(?m)^Backend 2/2: CPU\n[ \t]+Skipping\n', log)) == 1
    assert len(re.findall(r'(?i)skipp', log)) == 1
    assert not re.search(r'(?i)not supported|unsupported|unexpected|err\s*=|failed|BOLT_VERIFY error', log)
    prefix = {'router_single2': 'router_single2=1,',
              'router_columns8_r3': 'router_columns8=1,router_single2=1,',
              'router_prefill': 'router_prefill=1,router_single2=1,'}[tag]
    matches = re.findall(
        r'(?m)^\s*MUL_MAT\(' + re.escape(prefix) + r'weight=([^,\n]+),layout=([012]),'
        r'type_a=f32,type_b=f32,m=(\d+),n=(\d+),k=4096,bs=\[1,1\],nr=\[1,1\],'
        r'per=\[0,1,2,3\],k_v=0,o=1,src_overlap=0,m_v=0,pad=0\): ' + OK + r'[ \t]*$', log)
    keys = [(w, int(m), int(layout), int(n)) for w, layout, m, n in matches]
    assert len(keys) == len(expected) and set(keys) == expected, ('native cases', keys)
    assert len(re.findall(r'(?m)^\s*MUL_MAT\(', log)) == len(expected), 'extra/unparsed native cases'
    routes = route_records(log)
    grouped = {}
    patterns = {'router_single2': r'router_single2-l([012])-n([123])',
                'router_columns8_r3': r'router_columns8-m(\d+)-l([012])-n([1-9])',
                'router_prefill': r'router_prefill-m(496)-n([45])'}
    for route in routes:
        identity = re.fullmatch(patterns[tag], route['dst_name'])
        assert identity, route
        values = list(map(int, identity.groups()))
        if tag == 'router_single2':
            m, layout, n = [route['ne1']] + values
        elif tag == 'router_prefill':
            m, n = values
            layout = 0
        else:
            m, layout, n = values
        key = (route['weight'], m, layout, n)
        assert key in expected, ('unexpected native route', key)
        grouped.setdefault(key, []).append(route)
    assert set(grouped) == expected, ('missing actual native cases', expected - set(grouped))
    assert len({r['backend_ctx'] for r in routes}) == 1
    for key, group in grouped.items():
        assert_router_operations(group, key, tag)
    return dict(tag=tag, cases=len(expected), route_records=len(routes),
                split_cases=sum(split_case(key, tag) for key in expected), public_api_fidelity='NOT_TESTED',
                device_batched_vs_singleton_byte_equality='NOT_TESTED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', choices=TAGS, required=True)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    # Decode bytes directly: not even universal-newline rewriting of input logs.
    result = check_router_native_output(args.log.read_bytes().decode('utf-8'), args.tag)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
