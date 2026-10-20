#!/usr/bin/env python3
"""CPU/source/offset checks ONLY. No C++ build, SYCL runtime, or numerical proof."""
import hashlib
import itertools
import os
from pathlib import Path
import re
import runpy
import shutil
import struct
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = Path(os.environ.get('IQ2_SOURCE_ROOT', '/tmp/lilicorr-verify-r2-source-inspect-20261020'))
PREFIX = Path('ggml/src/ggml-sycl')
FILES = ('ggml-sycl.cpp', 'mmvq.cpp', 'mmvq.hpp')
HOST_START = '// Called only after single2 eligibility and ALL singleton-view preflights.\n'
HOST_END = 'static bool ggml_sycl_mul_mat_id_iq2_single2('
KERNEL_START = '// Plain IQ2 only. Each token calls the unchanged singleton arithmetic with its own pointers.\n'
KERNEL_END = 'bool ggml_sycl_mul_mat_vec_q_id('
HEADER_START = '// Preflighted plain IQ2_XXS, 2..4 tokens; vy is plain Q8_1. All strides are bytes.\n'
HEADER_END = '// Reorder (SoA) variant of the fused MoE expert GEMV.'
CALL = '    if (ggml_sycl_mul_mat_id_iq2_batch4(ctx, src0, src1, ids, dst)) return true;\n'


def section(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only scratch beneath this checkout is written; the supplied source is read-only.
        scratch = ROOT / '.scratch'
        scratch.mkdir(exist_ok=True)
        ignore = scratch / '.gitignore'
        if not ignore.exists():
            ignore.write_text('*\n')
        cls.temp = tempfile.TemporaryDirectory(prefix='iq2-batch4-test-', dir=scratch)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.tree = Path(cls.temp.name)
        (cls.tree / PREFIX).mkdir(parents=True)
        for name in FILES:
            shutil.copyfile(SOURCE / PREFIX / name, cls.tree / PREFIX / name)
        command = ['git', 'apply', '--whitespace=error', '--directory', str(cls.tree.relative_to(ROOT))]
        for flags in (['--check'], []):
            subprocess.run(command + flags + [str(HERE / 'iq2-batch4.patch')], cwd=ROOT, check=True)
        cls.base = {name: (SOURCE / PREFIX / name).read_text() for name in FILES}
        cls.patched = {name: (cls.tree / PREFIX / name).read_text() for name in FILES}
        cls.host = section(cls.patched['ggml-sycl.cpp'], HOST_START, HOST_END)
        cls.kernel = section(cls.patched['mmvq.cpp'], KERNEL_START, KERNEL_END)

    def test_source_pins(self):
        self.assertEqual((SOURCE / 'UPSTREAM_COMMIT').read_text().strip(),
                         'b6334e37e6ef286fc5091a41b6561b2113d915fb')
        for line in (HERE / 'source-pins.sha256').read_text().splitlines():
            digest, path = line.split()
            self.assertEqual(hashlib.sha256((SOURCE / path).read_bytes()).hexdigest(), digest, path)

    def test_all_original_code_and_fallbacks_unchanged(self):
        # Exact full-file comparison, not just selected guards: includes all dtype/mark/stride/
        # range checks, n=1 and n=5..8 paths, quantized vec-dot, XOR masks, K loop, NVFP4, etc.
        self.assertEqual(self.patched['ggml-sycl.cpp'].count(CALL), 1)
        self.assertEqual(self.patched['ggml-sycl.cpp'].replace(self.host, '', 1).replace(CALL, '', 1),
                         self.base['ggml-sycl.cpp'])
        self.assertEqual(self.patched['mmvq.cpp'].replace(self.kernel, '', 1), self.base['mmvq.cpp'])
        header = section(self.patched['mmvq.hpp'], HEADER_START, HEADER_END)
        self.assertEqual(self.patched['mmvq.hpp'].replace(header, '', 1), self.base['mmvq.hpp'])
        paths = re.findall(r'^\+\+\+ b/(.+)$', (HERE / 'iq2-batch4.patch').read_text(), re.M)
        self.assertEqual(paths, [str(PREFIX / name) for name in FILES])

    def test_preflight_and_enqueue_order(self):
        text = self.patched['ggml-sycl.cpp']
        self.assertIn('!ggml_sycl_iq2_single2_range(ctx, &dst_token[t])) return false;\n    }\n' + CALL, text)
        self.assertIn('if (n != 4 || !ggml_is_contiguous(src1)) return false;', self.host)
        self.assertIn('for (const ggml_tensor * t : {src0, src1, ids}) {\n'
                      '        if (!ggml_sycl_verify_singletons_disjoint(ctx, t, dst)) return false;', self.host)
        enqueue = self.host.index('    quantize_row_q8_1_sycl<quantize_q8_1>(')
        self.assertNotIn('return false;', self.host[enqueue:])
        self.assertLess(self.host.index('if (!src1_ddq) return false;'), enqueue)
        self.assertLess(enqueue, self.host.index('    ggml_sycl_mul_mat_vec_q_id_iq2_batch4('))
        self.assertEqual(self.host.count('ctx.stream()'), 1)
        self.assertEqual(self.host.count('ctx.pool()'), 1)
        self.assertEqual(self.host.count('quantize_row_q8_1_sycl<'), 1)
        self.assertEqual(self.kernel.count('stream->submit('), 1)
        self.assertEqual(self.kernel.count('cgh.parallel_for('), 1)
        for forbidden in ('wait(', 'wait_and_throw', 'barrier(', 'queue(', 'reorder(', 'memcpy('):
            self.assertNotIn(forbidden, self.host + self.kernel)

    def test_geometry_and_iq2_specialization(self):
        original = section(self.base['mmvq.cpp'], 'static void launch_mul_mat_vec_q_moe(', KERNEL_END)
        for line in ('const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);',
                     'sycl::nd_range<3>(block_nums * block_dims, block_dims),',
                     '[=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {'):
            self.assertIn(line, original)
            self.assertIn(line, self.kernel)
        self.assertIn('block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num_y)', self.kernel)
        self.assertIn('const size_t token = item.get_group(0);', self.kernel)
        spec = 'QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe'
        self.assertIn('launch_mul_mat_vec_q_moe<' + spec + '>', self.base['mmvq.cpp'])
        self.assertIn('mul_mat_vec_q_moe<' + spec + '>', self.kernel)
        self.assertNotRegex(self.kernel, r'reqd_sub_group_size\(\d+\)')
        self.assertIn('(size_t) src1->ne[1] * bytes_per_qrow, ids->nb[1], dst->nb[2], stream);', self.host)
        self.assertIn('src0->nb[2], dst->nb[1], src1->ne[1] == 1 ? 0 : bytes_per_qrow,', self.host)

    def test_token_offsets_ids_duplicates_and_canaries(self):
        # Evaluate the actual added token-offset expressions; no floating-point arithmetic.
        expressions = re.findall(r'token \* (?:src1|ids|dst)_token_stride', self.kernel)
        self.assertEqual(expressions, ['token * src1_token_stride', 'token * dst_token_stride', 'token * ids_token_stride'])
        used, m, guard = 9, 7, 64
        for n, ne11, k, id_padding, dst_padding in itertools.product((2, 3, 4), (1, 9), (256, 768, 1024, 2048), (0, 7, 15), (0, 20)):
            # Padded dst tests the wrapper's byte-stride handling only: the unchanged host
            # eligibility rejects padded dst. Padded IDs are host-eligible.
            qrow = ((k + 511) // 512) * 512 // 32 * 36
            strides = dict(src1_token_stride=ne11*qrow, ids_token_stride=4*(used+id_padding),
                           dst_token_stride=4*m*used+dst_padding)
            ids = bytearray(b'\xa5' * (guard*2 + n*strides['ids_token_stride']))
            orders = [[t, (t+3) % 16, t, 15, 0, (t*7+4) % 16, 1, 1, (t*5+6) % 16] for t in range(n)]
            for t, order in enumerate(orders):
                for slot, expert in enumerate(order):
                    struct.pack_into('<i', ids, guard+t*strides['ids_token_stride']+4*slot, expert)
            before_ids = bytes(ids)
            dst = bytearray(b'\xa5' * (guard*2 + n*strides['dst_token_stride']))
            reference = bytearray(dst)
            written = set()
            for token, slot, row in itertools.product(range(n), range(used), range(m)):
                qoff, doff, ioff = [eval(expr, {'__builtins__': {}}, dict(token=token, **strides)) for expr in expressions]
                expert = struct.unpack_from('<i', ids, guard+ioff+4*slot)[0]
                self.assertEqual(expert, orders[token][slot])
                qbase = qoff + (0 if ne11 == 1 else slot*qrow)
                expected_qbase = (token*ne11 + (0 if ne11 == 1 else slot))*qrow
                self.assertEqual(qbase, expected_qbase)
                self.assertLessEqual(qbase + k//32*36, n*ne11*qrow)
                address = guard + doff + slot*m*4 + row*4
                expected = guard + token*(used*m*4+dst_padding) + (slot*m+row)*4
                self.assertEqual(address, expected)
                self.assertNotIn(address, written)
                written.add(address)
                payload = (expert << 16) | (token << 12) | (slot << 8) | row
                struct.pack_into('<I', dst, address, payload)
                struct.pack_into('<I', reference, expected, payload)
            self.assertEqual(dst, reference)  # all padding and both canaries included
            self.assertEqual(ids, before_ids)
            self.assertEqual(len(written), n*used*m)

    def test_quantizer_flattened_row_offsets_and_padding(self):
        # WARP_SIZE=16 in the frozen build: two F32 elements per work item, QK8_1=32.
        # This models addresses/written bytes only, NOT quantization/reduction numerics.
        for n, ne11, k in itertools.product((2, 3, 4), (1, 9), (256, 768, 1024, 2048)):
            pitch, real_blocks = ((k+511)//512)*16, k//32
            qrow, guard = pitch*36, 64
            old = bytearray(b'\xa5' * (guard*2 + n*ne11*qrow))
            batch = bytearray(old)
            for token, row, block in itertools.product(range(n), range(ne11), range(real_blocks)):
                singleton_group = row*real_blocks+block
                batch_group = (token*ne11+row)*real_blocks+block
                for lane, element in itertools.product(range(16), range(2)):
                    self.assertEqual(token*ne11*k + singleton_group*32 + 2*lane + element,
                                     batch_group*32 + 2*lane + element)
                singleton_out = token*ne11*qrow + (singleton_group % real_blocks + row*pitch)*36
                batch_out = (batch_group % real_blocks + (batch_group//real_blocks)*pitch)*36
                self.assertEqual(singleton_out, batch_out)
                old[guard+singleton_out:guard+singleton_out+36] = b'\x55'*36
                batch[guard+batch_out:guard+batch_out+36] = b'\x55'*36
            self.assertEqual(batch, old)
            self.assertEqual(batch.count(0x55), n*ne11*real_blocks*36)
            self.assertEqual(batch[:guard] + batch[-guard:], b'\xa5'*(guard*2))

    def test_scratch_cap_and_integer_boundaries(self):
        self.assertIn('std::min<size_t>(1u << 20,', self.host)
        self.assertIn('ggml_backend_sycl_buffer_type_get_max_size(src1->buffer->buft) / 2', self.host)
        self.assertIn('if (bytes_per_qrow > max_q8_bytes / qrows) return false;', self.host)
        self.assertLess(self.host.index('max_q8_bytes / qrows'), self.host.index('ctx.stream()'))
        for max_alloc, rows in itertools.product((0, 2048, 1 << 20, 1 << 32, 1 << 35), (2, 4, 36, (1 << 31)-1)):
            limit = min(1 << 20, max_alloc//2)
            for bpr in (576, 2304, limit//rows, limit//rows+1, (1 << 31)-1):
                rejected = bpr > limit//rows
                self.assertEqual(rejected, bpr*rows > limit)
                if not rejected:
                    self.assertLessEqual(bpr*rows, (1 << 32)-1)
                    self.assertLessEqual(bpr*rows*1.05, max_alloc)
        self.assertEqual(4*1*(1024//32)*36, 4608)   # native up, n=4
        self.assertEqual(4*9*(2048//32)*36, 82944)  # native down, n=4

    def test_truthful_parent_route_and_frozen_parser_rejection(self):
        route = 'MOE_IQ2_BATCH4_MMVQ_PLAIN_Q8_1'
        self.assertEqual(self.host.count('bolt_route_record('), 1)
        self.assertIn('bolt_route_record("' + route + '",\n'
                      '                      ctx, src0, src1, dst, ctx.device, bolt_route_src1_precision(dst), -1, nullptr, ids, -1, -1);', self.host)
        self.assertNotIn('MOE_SINGLE_MMVQ_PLAIN_Q8_1', self.host)
        native = runpy.run_path(str(ROOT / 'results/20261019-nemotron35-bolt-lilicorr-verify-singletons/native.py'))
        # Synthetic metadata only: deliberately prove the OLD parser does NOT qualify this route.
        r = dict(route=route, src0_type='iq2_xxs', src1_type='f32', dst_type='f32', aux_weight='-', aux_type='-',
                 accum_prec=10, src1_prec=0, dynamic_prec=0, device=0, weight='blk.1.ffn_up_exps.weight',
                 ne0=1024, ne1=2048, ne2=16, ne3=1, nb0=66, src1_ne0=1024, dst_ne0=2048,
                 src1_ne1=1, dst_ne1=9, src1_ne2=4, dst_ne2=4, src1_ne3=1, dst_ne3=1,
                 src1_nb0=4, dst_nb0=4, ids_ne0=9, ids_ne1=4, reorder=0, expert=-1, result=-1,
                 op='MUL_MAT_ID', op_id=30)
        with self.assertRaises(AssertionError) as failure:
            native['check_routes']([r], ('iq2', 0, 4, 1, 0, 1), 'wide')
        self.assertEqual(failure.exception.args[0], (('iq2', 0, 4, 1, 0, 1), 'wide', [route]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
