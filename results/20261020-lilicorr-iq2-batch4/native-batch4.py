#!/usr/bin/env python3
"""Strict native-route qualifier for the truthful IQ2 batch-4 parent route.

The frozen native checker remains the authority for case enumeration, byte
comparisons, and all non-batch routes.  This module only replaces its private
route dispatcher for the six eligible wide IQ2 parent cases.
"""
import argparse
import json
import os
import re
import runpy
from pathlib import Path


if not __debug__:
    raise RuntimeError('do not disable assertions in this qualification gate')

HERE = Path(__file__).resolve().parent
BATCH_ROUTE = 'MOE_IQ2_BATCH4_MMVQ_PLAIN_Q8_1'


def _frozen_path():
    candidates = []
    override = os.environ.get('IQ2_FROZEN_NATIVE')
    if override:
        candidates.append(Path(override))
    candidates.extend((
        HERE / 'native.py',
        HERE.parent / '20261019-nemotron35-bolt-lilicorr-verify-singletons' / 'native.py',
        Path('/verify-singletons/native.py'),
    ))
    for path in candidates:
        if path.is_file() and path.resolve() != Path(__file__).resolve():
            return path
    raise FileNotFoundError('cannot locate the frozen verify-singletons/native.py')


FROZEN_PATH = _frozen_path()
FROZEN = runpy.run_path(str(FROZEN_PATH))
_ORIGINAL_CHECK_ROUTES = FROZEN['check_routes']


def _eligible(key, col):
    kind, _, n, _, layout, _ = key
    return (kind == 'iq2' and col == 'wide' and 2 <= n <= 4 and layout == 0
            and FROZEN['exact'](key))


def _check_batch_parent(row, key, col):
    """Validate one physical parent record; never synthesize singleton rows."""
    kind, p, n, mark, layout, _ = key
    k, m, used = FROZEN['dimensions'](key)
    assert col == 'wide' and kind == 'iq2'
    assert row['route'] == BATCH_ROUTE

    # Keep the frozen parser's common metadata assertions in force for this
    # new route too, then validate the truthful parent geometry and strides.
    assert row['src0_type'] == 'iq2_xxs', row
    assert row['src1_type'] == row['dst_type'] == 'f32', row
    assert row['aux_weight'] == row['aux_type'] == '-', row
    assert row['accum_prec'] == (10 if mark else 0) and row['src1_prec'] == 0, row
    assert row['operand_prec'] == -1, row
    assert row['dynamic_prec'] == row['device'] == 0, row
    assert row['weight'] == f'blk.1.ffn_{"up" if p == 0 else "down"}_exps.weight', row
    assert row['op'] == 'MUL_MAT_ID' and row['op_id'] == 30, row
    assert row['backend_ctx'] and re.fullmatch(r'0x[0-9a-fA-F]+', row['backend_ctx']), row
    assert row['dst_name'] == FROZEN['identity'](key) + '-wide', row
    assert row['reorder'] == 0 and row['expert'] == row['result'] == -1, row

    assert (row['ne0'], row['ne1'], row['ne2'], row['ne3']) == (k, m, 16, 1), row
    weight_row = (k // 256) * 66
    assert (row['nb0'], row['nb1'], row['nb2'], row['nb3']) == (
        66, weight_row, weight_row * m, weight_row * m * 16), row

    assert (row['src1_ne0'], row['src1_ne1'], row['src1_ne2'], row['src1_ne3']) == (
        k, used, n, 1), row
    assert (row['src1_nb0'], row['src1_nb1'], row['src1_nb2'], row['src1_nb3']) == (
        4, 4 * k, 4 * k * used, 4 * k * used * n), row

    assert (row['dst_ne0'], row['dst_ne1'], row['dst_ne2'], row['dst_ne3']) == (
        m, 9, n, 1), row
    assert (row['dst_nb0'], row['dst_nb1'], row['dst_nb2'], row['dst_nb3']) == (
        4, 4 * m, 4 * m * 9, 4 * m * 9 * n), row

    # IDs intentionally retain the parent fixture's padded 64-byte row pitch.
    assert (row['ids_ne0'], row['ids_ne1'], row['ids_ne2'], row['ids_ne3']) == (
        9, n, 1, 1), row
    assert (row['ids_nb0'], row['ids_nb1'], row['ids_nb2'], row['ids_nb3']) == (
        4, 64, 64 * n, 64 * n), row


def check_routes(rows, key, col):
    batch_rows = [row for row in rows if row['route'] == BATCH_ROUTE]
    if not _eligible(key, col):
        # Every old case, scalar reference, negative layout/annotation, wider
        # case, and Q8 case stays under the frozen route checker unchanged.
        assert not batch_rows, (key, col, [row['route'] for row in rows])
        return _ORIGINAL_CHECK_ROUTES(rows, key, col)

    # Eligible wide IQ2 cases must report exactly one physical parent record.
    assert len(rows) == 1 and len(batch_rows) == 1, (key, col, [row['route'] for row in rows])
    _check_batch_parent(batch_rows[0], key, col)


# The frozen check() function resolves check_routes through its own globals.
# Install only this narrow dispatcher; check(), byte assertions, and case
# enumeration remain the frozen implementation.
FROZEN['check_routes'] = check_routes
FROZEN['check'].__globals__['check_routes'] = check_routes


def check(log, kind):
    return FROZEN['check'](log, kind)


def native_command():
    """Retain the original command surface for lead-side adapter compatibility."""
    return FROZEN['native_command']()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-native-command', action='store_true')
    parser.add_argument('--kind', choices=['q8', 'iq2'])
    parser.add_argument('log', nargs='?', type=Path)
    args = parser.parse_args(argv)
    if args.print_native_command:
        assert not args.kind and not args.log
        print(native_command())
        return 0
    assert args.kind and args.log
    print(json.dumps(check(args.log.read_bytes().decode('utf-8'), args.kind), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
