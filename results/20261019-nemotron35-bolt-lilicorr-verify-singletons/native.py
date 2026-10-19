#!/usr/bin/env python3
"""CPU-only command printer / strict native-log checker. Does not launch a runtime."""
import argparse
import json
import re
import runpy
from pathlib import Path

if not __debug__:
    raise RuntimeError('do not disable assertions in this experimental gate')
ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-verify-phase'
if not PHASE.is_dir():
    PHASE = Path('/router-verify-phase')
OLD = runpy.run_path(str(PHASE / 'analyze-trace.py'))
OK = OLD['OK']
Q8_WEIGHTS = ['blk.1.ffn_latent_down.weight', 'blk.1.ffn_latent_up.weight',
              'blk.7.attn_q.weight', 'blk.7.attn_k.weight', 'blk.7.attn_v.weight', 'blk.7.attn_output.weight']
ID = r'(q8|iq2)-p([0-5])-n([1-9])-m([01])-l([0-3])(?:-a([02]))?'
SINGLE = 'MOE_SINGLE_MMVQ_PLAIN_Q8_1'
LIBRARIES = {'LIBRARY_MKL_F16', 'LIBRARY_DNNL_F16', 'LIBRARY_MKL_F32', 'LIBRARY_DNNL_F32'}


def cases(kind):
    assert kind in ('q8', 'iq2')
    result = set()
    for p in range(6 if kind == 'q8' else 2):
        result |= {(kind, p, n, int(n >= 3), 0) for n in range(1, 10)}
        result |= {(kind, p, n, 0, 0) for n in ((4,) if kind == 'q8' else (4, 8))}
        result |= {(kind, p, 4, 1, layout) for layout in (1, 2)}
    if kind == 'q8':
        result |= {(kind, 5, n, 1, 3) for n in range(3, 9)}
    result = {(*key, 1) for key in result}
    result |= {(kind, p, n, 1, 0, annotation)
               for p in ((5,) if kind == 'q8' else (0, 1))
               for n in (4, 8) for annotation in (0, 2)}
    assert len(result) == (82 if kind == 'q8' else 34)
    return result


def identity(key):
    kind, p, n, mark, layout, annotation = key
    return f'{kind}-p{p}-n{n}-m{mark}-l{layout}' + ('' if annotation == 1 else f'-a{annotation}')


def parse_id(text):
    m = re.fullmatch(ID, text)
    assert m, text
    return (m[1], *map(int, m.groups()[1:5]), int(m[6]) if m[6] is not None else 1)


def dimensions(key):
    kind, p, _, _, _, _ = key
    if kind == 'q8':
        return (1024 if p == 1 else 4096, 1024 if p == 0 else 256 if p in (3, 4) else 4096, 1)
    return (1024, 2048, 1) if p == 0 else (2048, 1024, 9)


def exact(key):
    kind, _, n, mark, layout, annotation = key
    return layout in ((0, 3) if kind == 'q8' else (0,)) and (n <= 2 or (mark == annotation == 1 and n <= 8))


def route_records(log):
    result = []
    for line in log.splitlines():
        if 'BOLT_ROUTE' not in line:
            continue
        assert line.count('BOLT_ROUTE ') == 1 and 'BOLT_VERIFY' not in line
        pairs = line.split('BOLT_ROUTE ', 1)[1].split()
        assert all(re.fullmatch(r'[a-z_][a-z_0-9]*=[^\s=]+', p) for p in pairs)
        fields = [p.split('=', 1) for p in pairs]
        assert len({k for k, _ in fields}) == len(fields)
        assert {k for k, _ in fields} == OLD['ROUTE_FIELDS']
        row = {}
        for k, v in fields:
            if k in OLD['ROUTE_INTS']:
                assert re.fullmatch(r'-?\d+', v)
                v = int(v)
            row[k] = v
        assert re.fullmatch(r'0x[0-9a-fA-F]+', row['backend_ctx'])
        result.append(row)
    assert result, 'missing actual native route records'
    return result


def check_routes(rows, key, col):
    kind, p, n, mark, layout, _ = key
    k, m, used = dimensions(key)
    wide = col == 'wide'
    expected_mark = 10 if wide and mark else 0
    for r in rows:
        assert r['src0_type'] == ('q8_0' if kind == 'q8' else 'iq2_xxs'), r
        assert r['src1_type'] == r['dst_type'] == 'f32'
        assert r['aux_weight'] == r['aux_type'] == '-'
        assert r['accum_prec'] == expected_mark and r['src1_prec'] == 0
        assert r['dynamic_prec'] == r['device'] == 0
        assert r['weight'] == (Q8_WEIGHTS[p] if kind == 'q8' else f'blk.1.ffn_{"up" if p == 0 else "down"}_exps.weight')
        assert (r['ne0'], r['ne1']) == (k, m)
        assert r['src1_ne0'] == k and r['dst_ne0'] == m
        assert r['op'] == ('MUL_MAT' if kind == 'q8' else 'MUL_MAT_ID')
        assert r['op_id'] == (29 if kind == 'q8' else 30)
        assert r['ne3'] == r['src1_ne3'] == r['dst_ne3'] == 1
        assert r['nb0'] == (34 if kind == 'q8' else 66)
        assert r['src1_nb0'] == r['dst_nb0'] == 4
    routes = [r['route'] for r in rows]
    if kind == 'q8':
        if not wide or exact(key):
            assert routes == ['DMMV_ESIMD'], (key, col, routes)
        elif n <= 8:
            assert routes == ['MMVQ_REORDERED'], (key, routes)
        else:
            assert len(rows) == 2 and routes[0] == 'MUL_MAT_SYCL' and routes[1] in LIBRARIES, routes
        width = n if wide else 1
        for r in rows:
            assert (r['src1_ne1'], r['dst_ne1'], r['src1_ne2'], r['dst_ne2']) == (width, width, 1, 1)
            assert (r['src1_nb0'], r['src1_nb1']) == (4, 4 * (k + (16 if wide and layout == 2 else 0)))
            assert (r['dst_nb0'], r['dst_nb1']) == (4, 4*m)
            assert r['expert'] == r['result'] == -1
            assert r['ne2'] == 1
            assert all(r[f'ids_{v}{i}'] == 0 for v in ('ne', 'nb') for i in range(4))
            if r['route'] in ('DMMV_ESIMD', 'MMVQ_REORDERED'):
                assert r['reorder'] == 1
        return
    if not wide or exact(key):
        assert routes == [SINGLE] * (n if wide else 1), (key, col, routes)
        for r in rows:
            assert r['reorder'] == 0 and r['expert'] == r['result'] == -1
            assert (r['src1_ne1'], r['src1_ne2'], r['dst_ne1'], r['dst_ne2'], r['ids_ne0'], r['ids_ne1']) == (used, 1, 9, 1, 9, 1)
            assert (r['src1_nb1'], r['src1_nb2'], r['src1_nb3']) == (4*k, 4*k*used, 4*k*used*(n if wide else 1))
            assert (r['dst_nb1'], r['dst_nb2'], r['dst_nb3']) == (4*m, 4*m*9, 4*m*9*(n if wide else 1))
            assert (r['ids_nb1'], r['ids_nb2'], r['ids_nb3']) == ((64, 64*n, 64*n) if wide else (36, 36, 36))
        return
    # Negative layouts/marks/widths must stay on the original grouped/per-expert path.
    assert SINGLE not in routes
    fallback = rows
    if layout != 1:
        assert routes[:2] == ['MOE_GROUPED_GEMM_ATTEMPT', 'MOE_GROUPED_GEMM_RESULT']
        assert rows[0]['result'] == -1 and rows[1]['result'] in (0, 1)
        for r in rows[:2]:
            assert (r['src1_ne1'], r['src1_ne2'], r['dst_ne1'], r['dst_ne2'], r['ids_ne1']) == (used, n, 9, n, n)
        fallback = rows[2:]
        if rows[1]['result'] == 1:
            assert not fallback
            return
    # Fixed cyclic native IDs select every expert for these n>=4 controls.
    assert [r['route'] for r in fallback] == ['MOE_PER_EXPERT_FALLBACK', 'MMVQ_PLAIN'] * 16
    for expert, (outer, inner) in enumerate(zip(fallback[::2], fallback[1::2])):
        assert outer['expert'] == expert and inner['expert'] == -1
        count = sum((token*9 + slot) % 16 == expert for token in range(n) for slot in range(9))
        for r in (outer, inner):
            assert r['src1_ne1'] == r['dst_ne1'] == count and r['src1_ne2'] == r['dst_ne2'] == 1


def check(log, kind):
    expected = cases(kind)
    count = str(len(expected))
    assert re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', log) == [(count, count)]
    assert re.findall(r'(?m)^Backend \d+/\d+: (\S+)\s*$', log) == ['SYCL0', 'CPU']
    assert len(re.findall(r'(?m)^[ \t]*Backend SYCL0: ' + OK + r'[ \t]*$', log)) == 1
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', log) == [('2', '2')]
    assert len(re.findall(r'(?m)^Backend 2/2: CPU\n[ \t]+Skipping\n', log)) == 1
    assert len(re.findall(r'(?i)skipp', log)) == 1
    assert not re.search(r'(?i)not supported|unsupported|unexpected|err\s*=|failed|BOLT_VERIFY error', log)
    op = 'MUL_MAT' if kind == 'q8' else 'MUL_MAT_ID'
    matches = re.findall(r'(?m)^\s*' + op + r'\(verify_singletons=1,case=([^,\n)]+)\): ' + OK + r'[ \t]*$', log)
    assert len(matches) == len(expected) and {parse_id(x) for x in matches} == expected
    assert len(re.findall(r'(?m)^\s*MUL_MAT(?:_ID)?\(', log)) == len(expected)
    byte_rows = re.findall(r'(?m)^BOLT_SINGLETON case=(\S+) bytes=(\d+) required=([01]) mismatches=(\d+)$', log)
    assert len(byte_rows) == len(expected) and {parse_id(x[0]) for x in byte_rows} == expected
    assert log.count('BOLT_SINGLETON') == len(expected)
    for ident, size, required, mismatches in byte_rows:
        key = parse_id(ident)
        _, m, _ = dimensions(key)
        assert int(size) == 4*m*key[2]*(9 if kind == 'iq2' else 1)
        assert int(required) == int(exact(key))
        assert 0 <= int(mismatches) <= int(size)
        if exact(key):
            assert int(mismatches) == 0, (ident, 'device byte mismatch')
    routes = route_records(log)
    grouped = {}
    for r in routes:
        match = re.fullmatch('(' + ID + r')-(wide|s[0-8])', r['dst_name'])
        assert match, r
        key, col = parse_id(match[1]), match.groups()[-1]
        assert key in expected and (col == 'wide' or int(col[1:]) < key[2])
        grouped.setdefault((key, col), []).append(r)
    assert set(grouped) == {(key, col) for key in expected for col in ['wide'] + [f's{i}' for i in range(key[2])]}
    assert len({r['backend_ctx'] for r in routes}) == 1
    for (key, col), rows in grouped.items():
        check_routes(rows, key, col)
    return dict(kind=kind, cases=len(expected), byte_exact_cases=sum(map(exact, expected)),
                route_records=len(routes), public_fidelity='NOT_TESTED')


def native_command():
    # Exact inherited F32 router controls are printed unchanged, N9 stays separate.
    old = runpy.run_path(str(PHASE / 'run-trace.py'))['NATIVE_COMMAND']
    return old + r'''
for kind in q8 iq2; do
    test ! -e "/capture/verify-singletons-$kind.log"
    GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 \
        -p "^verify_singletons=1,case=$kind-" 2>&1 | tee "/capture/verify-singletons-$kind.log"
    python3 -B /verify-singletons/native.py --kind "$kind" "/capture/verify-singletons-$kind.log"
done
echo BOLT_VERIFY_SINGLETONS_NATIVE_PASSED'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-native-command', action='store_true')
    parser.add_argument('--kind', choices=['q8', 'iq2'])
    parser.add_argument('log', nargs='?', type=Path)
    args = parser.parse_args()
    if args.print_native_command:
        assert not args.kind and not args.log
        print(native_command())
    else:
        assert args.kind and args.log
        print(json.dumps(check(args.log.read_bytes().decode('utf-8'), args.kind), indent=2))
