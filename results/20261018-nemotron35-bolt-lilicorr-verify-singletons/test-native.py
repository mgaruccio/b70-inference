#!/usr/bin/env python3
"""Parser unit tests only. Synthetic strings are never saved as native evidence."""
import runpy
import subprocess
import unittest
from pathlib import Path

M = runpy.run_path(str(Path(__file__).with_name('native.py')))


def row(key, col, route, **changes):
    kind, p, n, mark, layout = key
    k, m, used = M['dimensions'](key)
    r = {k: 0 for k in M['OLD']['ROUTE_INTS']}
    r.update({k: '-' for k in M['OLD']['ROUTE_STRINGS']})
    r.update(route=route, weight=M['Q8_WEIGHTS'][p] if kind == 'q8' else f'blk.1.ffn_{"up" if p == 0 else "down"}_exps.weight',
             op='MUL_MAT' if kind == 'q8' else 'MUL_MAT_ID', op_id=29 if kind == 'q8' else 30,
             dst_name=M['identity'](key) + '-' + col, backend_ctx='0xabc',
             src0_type='q8_0' if kind == 'q8' else 'iq2_xxs', src1_type='f32', dst_type='f32',
             accum_prec=10 if col == 'wide' and mark else 0, expert=-1, result=-1,
             ne0=k, ne1=m, ne2=1 if kind == 'q8' else 16, ne3=1,
             nb0=34 if kind == 'q8' else 66, src1_ne0=k, dst_ne0=m, src1_nb0=4, dst_nb0=4,
             src1_ne3=1, dst_ne3=1)
    if kind == 'q8':
        width = n if col == 'wide' else 1
        r.update(src1_ne1=width, dst_ne1=width, src1_ne2=1, dst_ne2=1,
                 src1_nb1=4*(k + (16 if col == 'wide' and layout == 2 else 0)), dst_nb1=4*m, reorder=1)
    else:
        wide = col == 'wide'
        r.update(src1_ne1=used, src1_ne2=1, dst_ne1=9, dst_ne2=1, ids_ne0=9, ids_ne1=1,
                 src1_nb1=4*k, src1_nb2=4*k*used, src1_nb3=4*k*used*(n if wide else 1),
                 dst_nb1=4*m, dst_nb2=4*m*9, dst_nb3=4*m*9*(n if wide else 1),
                 ids_nb1=64 if wide else 36, ids_nb2=64*n if wide else 36, ids_nb3=64*n if wide else 36)
    r.update(changes)
    return r


def serialize(r):
    return 'BOLT_ROUTE ' + ' '.join(f'{k}={v}' for k, v in sorted(r.items()))


def synthetic(kind, grouped=True):
    # Independent fixture construction of the documented command's finite cases.
    keys = []
    for p in range(6 if kind == 'q8' else 2):
        keys += [(kind, p, n, int(n > 2), 0) for n in range(1, 10)]
        keys += [(kind, p, 4, 0, 0), (kind, p, 4, 1, 1), (kind, p, 4, 1, 2)]
        if kind == 'iq2':
            keys += [(kind, p, 8, 0, 0)]
    if kind == 'q8':
        keys += [(kind, 5, n, 1, 3) for n in range(3, 9)]
    lines = ['Backend 1/2: SYCL0']
    for key in keys:
        _, p, n, mark, layout = key
        ident = M['identity'](key)
        _, m, used = M['dimensions'](key)
        records = []
        if kind == 'q8':
            if M['exact'](key):
                records += [row(key, 'wide', 'DMMV_ESIMD')]
            elif n < 9:
                records += [row(key, 'wide', 'MMVQ_REORDERED')]
            else:
                records += [row(key, 'wide', 'MUL_MAT_SYCL'), row(key, 'wide', 'LIBRARY_MKL_F32')]
            records += [row(key, f's{i}', 'DMMV_ESIMD') for i in range(n)]
        else:
            if M['exact'](key):
                records += [row(key, 'wide', M['SINGLE']) for i in range(n)]
            else:
                if layout != 1:
                    shape = dict(src1_ne1=used, src1_ne2=n, dst_ne1=9, dst_ne2=n, ids_ne1=n)
                    records += [row(key, 'wide', 'MOE_GROUPED_GEMM_ATTEMPT', **shape),
                                row(key, 'wide', 'MOE_GROUPED_GEMM_RESULT', result=int(grouped), **shape)]
                if layout == 1 or not grouped:
                    for expert in range(16):
                        count = sum((t*9+s) % 16 == expert for t in range(n) for s in range(9))
                        shape = dict(src1_ne1=count, dst_ne1=count)
                        records += [row(key, 'wide', 'MOE_PER_EXPERT_FALLBACK', expert=expert, **shape),
                                    row(key, 'wide', 'MMVQ_PLAIN', **shape)]
            records += [row(key, f's{i}', M['SINGLE']) for i in range(n)]
        lines += [serialize(r) for r in records]
        lines += [f'BOLT_SINGLETON case={ident} bytes={4*m*n*(9 if kind == "iq2" else 1)} required={int(M["exact"](key))} mismatches=0']
        op = 'MUL_MAT' if kind == 'q8' else 'MUL_MAT_ID'
        lines += [f'  {op}(verify_singletons=1,case={ident}): OK']
    lines += [f'{len(keys)}/{len(keys)} tests passed', 'Backend SYCL0: OK', 'Backend 2/2: CPU', '  Skipping', '2/2 backends passed']
    return '\n'.join(lines) + '\n'


class NativeTests(unittest.TestCase):
    def test_complete_and_colored(self):
        for kind in ('q8', 'iq2'):
            for grouped in (False, True):
                log = synthetic(kind, grouped)
                result = M['check'](log, kind)
                self.assertEqual(result['cases'], 78 if kind == 'q8' else 26)
                self.assertEqual(result['byte_exact_cases'], 54 if kind == 'q8' else 16)
                M['check'](log.replace(': OK', ': \x1b[32mOK\x1b[0m'), kind)

    def test_rejects_bad_evidence(self):
        for kind in ('q8', 'iq2'):
            log = synthetic(kind)
            route = next(line for line in log.splitlines() if line.startswith('BOLT_ROUTE'))
            mutations = [
                log.replace('Backend SYCL0: OK', 'Backend SYCL0: FAIL'),
                log.replace('2/2 backends passed', '1/2 backends passed'),
                log.replace('mismatches=0', 'mismatches=1', 1),
                log.replace('required=1', 'required=0', 1),
                log.replace('src1_ne3=1', 'src1_ne3=2', 1),
                log.replace('accum_prec=10', 'accum_prec=0', 1),
                log.replace('src1_prec=0', 'src1_prec=10', 1),
                log.replace('device=0', 'device=1', 1),
                log.replace(route + '\n', '', 1), log + route + '\n',
                log.replace(route, route + ' device=0', 1),
                log + 'BOLT_SINGLETON malformed\n', log + '  MUL_MAT(extra): OK\n',
                log.replace('bytes=', 'bytes=1', 1), log.replace('mismatches=0', 'mismatches=-1', 1),
                log + 'unsupported\n', log + 'ERR = 0.1\n', log + 'Skipping\n',
                log.replace('backend_ctx=0xabc', 'backend_ctx=0xdef', 1),
            ]
            if kind == 'iq2':
                mutations += [log.replace('ids_nb2=192', 'ids_nb2=128', 1),
                              log.replace('route=' + M['SINGLE'], 'route=MOE_GROUPED_GEMM_RESULT', 1)]
            else:
                mutations += [log.replace('route=DMMV_ESIMD', 'route=MMVQ_REORDERED', 1),
                              log.replace('reorder=1', 'reorder=0', 1)]
            for i, bad in enumerate(mutations):
                with self.subTest(kind=kind, mutation=i), self.assertRaises(AssertionError):
                    M['check'](bad, kind)

    def test_command_is_inherited_plus_new_selectors(self):
        command = M['native_command']()
        inherited = runpy.run_path(str(M['PHASE'] / 'run-trace.py'))['NATIVE_COMMAND']
        self.assertTrue(command.startswith(inherited + '\n'))
        self.assertIn('^verify_singletons=1,case=$kind-', command)
        self.assertIn('test ! -e', command)
        self.assertNotIn('docker ', command)
        subprocess.run(['bash', '-n'], input=command, text=True, check=True)


if __name__ == '__main__':
    unittest.main()
