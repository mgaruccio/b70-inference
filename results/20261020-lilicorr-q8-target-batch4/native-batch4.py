#!/usr/bin/env python3
"""Keep every existing numeric/byte/route gate; require truthful Q8 n4 launch metadata."""
import argparse
import hashlib
import json
import re
import runpy
from pathlib import Path

assert __debug__, 'assertions required'
HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / 'iq2-native-base.py'
assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() == '5bf16cd5ded9ee20866bf9dd259758f1d5a7ef24cd091b05b334926d5691963f'
BASE = runpy.run_path(str(BASE_PATH))
OLD_FROZEN = BASE['FROZEN']
PREFIX = 'BOLT_Q8_VERIFY_BATCH4'
FIELDS = {'weight', 'dst_name', 'n', 'row_low', 'row_high', 'rowpairs', 'wg', 'physical_launches'}


def expected_q8_cases():
    return {OLD_FROZEN['identity'](key): key for key in OLD_FROZEN['cases']('q8')
            if key[2] == 4 and key[4] in (0, 3) and OLD_FROZEN['exact'](key)}


def check_q8_launches(log):
    expected = expected_q8_cases()
    lines = re.findall(r'(?m)^' + PREFIX + r' (.+)$', log)
    assert len(lines) == log.count(PREFIX) == len(expected) == 7, ('n4 physical launch record count', len(lines))
    routes = OLD_FROZEN['route_records'](log)
    seen = set()
    for line in lines:
        tokens = line.split()
        assert len(tokens) == len(FIELDS) and all(token.count('=') == 1 for token in tokens), line
        row = dict(token.split('=', 1) for token in tokens)
        assert set(row) == FIELDS, row
        assert row['dst_name'].endswith('-wide'), row
        ident = row['dst_name'][:-5]
        assert ident in expected and ident not in seen, row
        seen.add(ident)
        k, m, _used = OLD_FROZEN['dimensions'](expected[ident])
        blocks = k // 32
        wg = 8 if blocks >= 64 else 4 if blocks >= 32 else 2 if blocks >= 16 else 1
        for field, value in dict(n=4, row_low=0, row_high=m, rowpairs=(m + 1) // 2, wg=wg, physical_launches=1).items():
            assert row[field] == str(value), (field, row)
        parent = [record for record in routes if record['dst_name'] == row['dst_name']]
        assert len(parent) == 1 and parent[0]['route'] == 'DMMV_ESIMD', parent
        assert parent[0]['weight'] == row['weight'] and parent[0]['reorder'] == 1, row
        assert parent[0]['accum_prec'] == 10 and parent[0]['src1_ne1'] == 4, parent
    assert seen == set(expected)
    return len(seen)


def check(log, kind):
    result = BASE['check'](log, kind)  # Frozen enumeration/negative/canary/numeric/byte checks first.
    if kind == 'q8':
        result = dict(result, physical_n4_launches=check_q8_launches(log))
    else:
        assert PREFIX not in log, 'Q8 launch outside normal Q8 MUL_MAT gate'
    return result


# A new private mapping, not mutation of BASE's live globals or the public gate.
FROZEN = dict(OLD_FROZEN, check=check)


def native_command():
    return BASE['native_command']()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-native-command', action='store_true')
    parser.add_argument('--kind', choices=['q8', 'iq2'])
    parser.add_argument('log', nargs='?', type=Path)
    args = parser.parse_args()
    if args.print_native_command:
        assert not args.kind and not args.log
        print(native_command())
        return
    assert args.kind and args.log
    print(json.dumps(check(args.log.read_bytes().decode('utf-8'), args.kind), indent=2))


if __name__ == '__main__':
    main()
