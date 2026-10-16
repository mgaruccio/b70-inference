#!/usr/bin/env python3
"""Stdlib-only NVFP4 layout/arithmetic model checks, NOT SYCL execution tests.

The actual backend regression cases are in nvfp4-reorder.patch. These checks
exercise independent scalar decode vs the proposed addressing/SWAR dot algebra;
they cannot validate C++ compilation, queue ordering, GPU loads or reduction.
"""
import itertools
import math
import random
import struct
import unittest

VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
QK, SUB, QS, DS, BLOCK = 64, 16, 32, 4, 36


def scale(code):
    # Independent CPU convention: integer E2M1 lookup values are doubled.
    if code in (0, 127):
        return 0.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    raw = (math.ldexp(mantissa, -9) if exponent == 0 else
           math.ldexp(1.0 + mantissa / 8.0, exponent - 7))
    return raw / 2.0


def signed_bytes(word):
    return tuple(struct.unpack('4b', struct.pack('<I', word & 0xffffffff)))


def swar(word):
    # Existing get_int_from_mxfp4's byte-lane arithmetic, reused by the patch.
    m = word & 0x07070707
    ge5 = ((m + 0x03030303) >> 3) & 0x01010101
    ge6 = ((m + 0x02020202) >> 3) & 0x01010101
    ge7 = ((m + 0x01010101) >> 3) & 0x01010101
    mag = m + ge5 + ge6 + 3 * ge7
    nz = ((mag + 0x7f7f7f7f) >> 7) & 0x01010101
    neg = (word >> 3) & nz & 0x01010101
    return signed_bytes((mag ^ (neg * 0xff)) + neg)


def unpack_word(word):
    return swar(word & 0x0f0f0f0f), swar((word >> 4) & 0x0f0f0f0f)


def reorder(aos):
    assert len(aos) % BLOCK == 0
    blocks = [aos[i:i + BLOCK] for i in range(0, len(aos), BLOCK)]
    return b''.join(b[DS:] for b in blocks) + b''.join(b[:DS] for b in blocks)


def scalar_decode(aos):
    out = []
    for off in range(0, len(aos), BLOCK):
        for sub in range(DS):
            d = scale(aos[off + sub])
            packed = aos[off + DS + sub * 8:off + DS + (sub + 1) * 8]
            out.extend(d * VALUES[q & 15] for q in packed)
            out.extend(d * VALUES[q >> 4] for q in packed)
    return out


def reordered_decode(soa, ne):
    out = [None] * ne
    for ib in range(ne // QK):
        for tid in range(QS):
            sub, j = divmod(tid, 8)
            d = scale(soa[ne // 2 + ib * DS + sub])
            q = soa[ib * QS + sub * 8 + j]
            y0 = ib * QK + sub * SUB + j
            out[y0] = d * VALUES[q & 15]
            out[y0 + 8] = d * VALUES[q >> 4]
    return out


def reordered_dot(soa, ncols, nrows, row, q8, q8scales):
    nblocks = ncols // QK
    total = 0.0
    # The two lanes have iqs 0 and 4; each covers two independent subblocks.
    for ib in range(nblocks):
        bx = row * nblocks + ib
        doff = ncols // 2 * nrows + bx * DS
        for iqs in (0, 4):
            q4 = struct.unpack_from('<4I', soa, bx * QS + 4 * iqs)
            for i in range(2):
                sub = iqs // 2 + i
                v0, v1 = unpack_word(q4[2 * i]), unpack_word(q4[2 * i + 1])
                u = q8[ib * QK + sub * SUB:ib * QK + (sub + 1) * SUB]
                dot = lambda a, b: sum(x * y for x, y in zip(a, b))
                sumi = (dot(v0[0], u[0:4]) + dot(v0[1], u[8:12]) +
                        dot(v1[0], u[4:8]) + dot(v1[1], u[12:16]))
                total += scale(soa[doff + sub]) * q8scales[ib * 2 + sub // 2] * sumi
    return total


class LayoutTests(unittest.TestCase):
    def test_exhaustive_swar_codes(self):
        for codes in itertools.product(range(16), repeat=4):
            packed = sum(code << (8 * i) for i, code in enumerate(codes))
            self.assertEqual(swar(packed), tuple(VALUES[code] for code in codes))
        for byte in range(256):
            lo, hi = unpack_word(byte * 0x01010101)
            self.assertEqual(lo, (VALUES[byte & 15],) * 4)
            self.assertEqual(hi, (VALUES[byte >> 4],) * 4)

    def test_all_scale_codes_subblocks_and_packed_bytes(self):
        # All scale-byte codes, including bit-7 aliases as decoded by upstream.
        for code in range(256):
            aos = bytes([code] * DS + list(range(QS)))
            for start in range(0, 256, QS):
                aos = aos[:DS] + bytes(range(start, start + QS))
                self.assertEqual(reordered_decode(reorder(aos), QK), scalar_decode(aos))
        self.assertEqual(scale(0), 0)
        self.assertEqual(scale(1), 2 ** -10)
        self.assertEqual(scale(8), 2 ** -7)
        self.assertEqual(scale(126), 224)
        self.assertEqual(scale(127), 0)

    def test_matrices_and_multicolumn_dot(self):
        rng = random.Random(7319)
        for ncols, nrows in ((64, 37), (192, 5), (4096, 3), (8192, 3)):
            aos = bytes(rng.randrange(256) for _ in range(ncols * nrows // QK * BLOCK))
            soa = reorder(aos)
            ref = scalar_decode(aos)
            self.assertEqual(len(aos), len(soa))
            self.assertEqual(reordered_decode(soa, ncols * nrows), ref)
            blocks = ncols * nrows // QK
            self.assertEqual(aos, b''.join(soa[blocks * QS + ib * DS:blocks * QS + (ib + 1) * DS] +
                                          soa[ib * QS:(ib + 1) * QS] for ib in range(blocks)))
            for nout in ((1, 2, 8, 9, 128) if ncols <= 192 else (1, 8)):
                for _ in range(nout):
                    q8 = [rng.randrange(-127, 128) for _ in range(ncols)]
                    ds = [2.0 ** rng.randrange(-10, -1) for _ in range(ncols // 32)]
                    for row in range(nrows):
                        expected = sum(ref[row * ncols + j] * q8[j] * ds[j // 32] for j in range(ncols))
                        self.assertEqual(reordered_dot(soa, ncols, nrows, row, q8, ds), expected)

    def test_real_shape_addresses_and_alignment(self):
        for ncols, nrows in ((4096, 18560), (8192, 4096)):
            total_blocks = ncols * nrows // QK
            nbytes = total_blocks * BLOCK
            self.assertLessEqual(ncols * nrows, 2 ** 31 - 1)
            self.assertEqual(total_blocks * QS + total_blocks * DS, nbytes)
            for row in (0, nrows // 2, nrows - 1):
                for ib in (0, ncols // QK - 1):
                    bx = row * (ncols // QK) + ib
                    self.assertLessEqual(bx * QS + QS, total_blocks * QS)
                    self.assertLessEqual(total_blocks * QS + bx * DS + DS, nbytes)
                    for iqs in (0, 4):
                        self.assertEqual((bx * QS + 4 * iqs) % 16, 0)
                        for i in range(2):
                            self.assertEqual((ib * QK + (iqs // 2 + i) * SUB) % 16, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
