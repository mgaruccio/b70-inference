#!/usr/bin/env python3
"""Supplementary CPU/source tests, NOT a SYCL build or native/fidelity/speed gate."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PINS = json.loads((HERE / 'source-pins.json').read_text())
SOURCE = Path(os.environ.get('Q8_TARGET_SOURCE_ROOT', PINS['original_snapshot'])).resolve()
PREFIX = Path('ggml/src/ggml-sycl')
PATCH = HERE / 'q8-target-batch4.patch'
SEED_PATCH = ROOT / PINS['seed_patch']
ENTRY = 'ggml_sycl_op_dequantize_mul_mat_vec_q8_0_batch4'
LAUNCH_START = '// Private n=4 submission: adjacent groups share a weight row pair, not SLM.\n'
LAUNCH_END = 'static void dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd('
ENTRY_START = '// Caller retains the private tensor/layout/scope eligibility and allocation bounds.\n'
ENTRY_END = 'static void dequantize_mul_mat_vec_q4_K_sycl_reorder('
HEADER_START = '// Private n=4 target Q8_0/F32 path, only after ggml_sycl_latent_dmmv2_eligible.\n'
HOST_START = '                    bool batch4_launched = false;\n'
HOST_END = '                    for (int64_t col = 0; col < src1_ncols; ++col) {\n'
OLD_COMMENT = ('                    // The DMMV callback ignores ncols. Keep its F32 ESIMD arithmetic,\n'
               '                    // row-low offset and queue ordering, but launch once per column.\n')
NEW_COMMENT = ('                    // Preserve private eligibility and native F32 arithmetic. Only n=4\n'
               '                    // may share a submission; the old callback loop remains the fallback.\n')


def section(text, start, end):
    first = text.index(start)
    return text[first:text.index(end, first)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(tree):
    return {p.relative_to(tree).as_posix(): sha(p) for p in sorted(tree.rglob('*')) if p.is_file()}


def run(command, cwd=ROOT):
    print(f'$ (cd {shlex.quote(str(cwd))} && {shlex.join(map(str, command))})', flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='' if result.stdout.endswith('\n') else '\n', flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result.stdout


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # All mutations stay below this new package; source input is read-only.
        cls.temp = tempfile.TemporaryDirectory(prefix='.cpu-', dir=HERE)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cls.seed = cls.work / 'seed'
        shutil.copytree(SOURCE, cls.seed)
        output = run(['patch', '--batch', '--forward', '--fuzz=0', '--verbose', '-p1',
                      '-d', str(cls.seed), '-i', str(SEED_PATCH)])
        assert not re.search(r'offset|with fuzz|FAILED', output)
        cls.seed_hashes = tree_hashes(cls.seed)
        cls.gnu = cls.work / 'gnu'
        shutil.copytree(cls.seed, cls.gnu)
        output = run(['patch', '--batch', '--forward', '--fuzz=0', '--verbose', '-p1',
                      '-d', str(cls.gnu), '-i', str(PATCH)])
        assert not re.search(r'offset|with fuzz|FAILED', output)
        cls.candidate_hashes = tree_hashes(cls.gnu)
        cls.base = {n: (cls.seed / PREFIX / n).read_text() for n in ('ggml-sycl.cpp', 'dmmv.cpp', 'dmmv.hpp')}
        cls.candidate = {n: (cls.gnu / PREFIX / n).read_text() for n in cls.base}
        cls.launch = section(cls.candidate['dmmv.cpp'], LAUNCH_START, LAUNCH_END)
        cls.entry = section(cls.candidate['dmmv.cpp'], ENTRY_START, ENTRY_END)
        cls.host = section(cls.candidate['ggml-sycl.cpp'], HOST_START, HOST_END)

    def test_01_source_pins_and_only_allowed_files(self):
        self.assertEqual((SOURCE / 'UPSTREAM_COMMIT').read_text().strip(), PINS['upstream_commit'])
        self.assertEqual(sha(SEED_PATCH), PINS['seed_patch_sha256'])
        self.assertEqual(sha(PATCH), PINS['candidate_patch_sha256'])
        for stage, tree in [('original', SOURCE), ('seed', self.seed), ('candidate', self.gnu)]:
            for path, digest in PINS[stage].items():
                self.assertEqual(sha(tree / path), digest, (stage, path))
        self.assertEqual(len(self.seed_hashes), 178)  # 177 SYCL files + UPSTREAM_COMMIT
        self.assertEqual(self.seed_hashes.keys(), self.candidate_hashes.keys())
        self.assertEqual(sorted(p for p in self.seed_hashes if self.seed_hashes[p] != self.candidate_hashes[p]),
                         sorted(PINS['allowed_changes']))
        self.assertEqual(re.findall(r'^\+\+\+ b/(.+)$', PATCH.read_text(), re.M), PINS['allowed_changes'])
        # No newly introduced trailing whitespace (raw context preserves its source).
        for line in PATCH.read_text().splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                self.assertEqual(line, line.rstrip(), line)

    def test_02_git_forward_reverse_reforward_matches_gnu_exactly(self):
        tree = self.work / 'git-roundtrip'
        shutil.copytree(self.seed, tree)
        directory = str(tree.relative_to(ROOT))
        command = ['git', 'apply', '--whitespace=error', '--directory', directory]
        for flags, expected in [([], self.candidate_hashes), (['--reverse'], self.seed_hashes),
                                ([], self.candidate_hashes)]:
            run(command + flags + ['--check', str(PATCH)])
            run(command + flags + [str(PATCH)])
            self.assertEqual(tree_hashes(tree), expected)
        print('GNU exact forward == Git forward == Git re-forward; reverse == passed seed (all 178 files).')

    def test_03_all_original_code_restored_except_two_row_index_edits(self):
        text = self.candidate['dmmv.cpp']
        text = text.replace('#include <climits>\n#include <cstdint>\n', '', 1)
        text = text.replace(self.launch, '', 1).replace(self.entry, '', 1)
        text = text.replace('const sycl::nd_item<1> & it, const int row_pair_override = -1)',
                            'const sycl::nd_item<1> & it)', 1)
        text = text.replace('row_pair_override < 0 ? it.get_group(0) : row_pair_override', 'it.get_group(0)', 1)
        self.assertEqual(text, self.base['dmmv.cpp'])
        header = self.candidate['dmmv.hpp']
        self.assertEqual(header.replace(section(header, HEADER_START, '#endif // GGML_SYCL_DMMV_HPP'), '', 1),
                         self.base['dmmv.hpp'])
        host = self.candidate['ggml-sycl.cpp']
        self.assertEqual(host.replace(self.host, '', 1).replace(NEW_COMMENT, OLD_COMMENT, 1),
                         self.base['ggml-sycl.cpp'])
        # Thus native MAC/STRIPE8, SIMD32 accs, SLM, barrier and increasing-p reduction,
        # all original launches/thresholds, precision, route family, IQ2 and non4 paths match.

    def test_04_native_local_geometry_and_new_launch(self):
        base = self.base['dmmv.cpp']
        native = section(base, 'template <int WG>\nstatic void q8_0_esimd_launch(', LAUNCH_END)
        for line in ('sycl::local_accessor<float, 1> lmem(sycl::range<1>(WG * 2), h);',
                     '[=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {'):
            self.assertIn(line, native)
            self.assertIn(line, self.launch)
        self.assertIn('sycl::nd_range<1>(sycl::range<1>(workgroups * WG), sycl::range<1>(WG))', self.launch)
        self.assertIn('const size_t workgroups = ((size_t) nrows + 1) / 2 * 4;', self.launch)
        self.assertIn('vx, y + token * src1_stride, dst + token * dst_stride,\n'
                      '                    ncols, nrows, lmem, it, row_pair);', self.launch)
        self.assertEqual(self.launch.count('stream->submit('), 1)
        self.assertEqual(self.launch.count('h.parallel_for('), 1)
        self.assertEqual(self.launch.count('dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>('), 1)
        self.assertNotRegex(self.launch + self.entry + self.host, r'\bsimd\s*<')
        for forbidden in ('barrier(', 'wait(', 'queue(', 'alloc(', 'quantize_row_', 'to_fp16', 'memcpy('):
            self.assertNotIn(forbidden, self.launch + self.entry + self.host)
        native_select = section(base, LAUNCH_END, '#endif // GGML_SYCL_DMMV_HAS_ESIMD')
        self.assertEqual(re.findall(r'nblk_row >= (\d+)', native_select), ['64', '32', '16'])
        self.assertEqual(re.findall(r'nblk_row >= (\d+)', self.entry), ['64', '32', '16'])
        self.assertEqual(re.findall(r'q8_0_esimd_launch_batch4<(\d+)>', self.entry), ['8', '4', '2', '1'])
        # The launcher lies inside the pre-existing ESIMD guard; entry has a false-only stub.
        self.assertLess(self.candidate['dmmv.cpp'].index(LAUNCH_START),
                        self.candidate['dmmv.cpp'].index('#endif // GGML_SYCL_DMMV_HAS_ESIMD'))
        enabled, disabled = section(self.entry, '#ifdef GGML_SYCL_DMMV_HAS_ESIMD', '\n#endif').split('#else')
        first_launch = enabled.index('q8_0_esimd_launch_batch4<')
        self.assertNotIn('return false;', enabled[first_launch:])
        self.assertNotIn('q8_0_esimd_launch_batch4<', disabled)
        self.assertEqual(disabled.count('return false;'), 1)

    def test_05_non4_fallback_and_truthful_diagnostic(self):
        host = self.candidate['ggml-sycl.cpp']
        base = self.base['ggml-sycl.cpp']
        eligibility = 'static bool ggml_sycl_latent_dmmv2_eligible('
        self.assertEqual(section(host, eligibility, 'template <template <int> typename quantize_f>'),
                         section(base, eligibility, 'template <template <int> typename quantize_f>'))
        original_loop = section(base, HOST_END, '\n                } else {')
        self.assertIn(original_loop, host)  # byte-for-byte, including whitespace
        self.assertIn('ggml_sycl_latent_dmmv2_eligible(ctx, src0, src1, dst)) {\n' + NEW_COMMENT +
                      '                    GGML_ASSERT(src0_extra && src0_extra->optimized_feature.reorder);\n' +
                      self.host + original_loop, host)
        self.assertIn('if (src1_ncols == 4) {\n'
                      '                        SYCL_CHECK(CHECK_TRY_ERROR(batch4_launched = ' + ENTRY + '(', self.host)
        self.assertIn('ne10, dst_on_device ? ne0 : row_diff, stream, batch4_wg)', self.host)
        self.assertTrue(self.host.endswith('                    } else\n'))
        self.assertIn('if (batch4_launched) {\n                        if (bolt_route_enabled()) {', self.host)
        self.assertLess(self.host.index(ENTRY), self.host.index('BOLT_Q8_VERIFY_BATCH4'))
        for literal in ('weight=%s', 'dst_name=%s', 'n=4', 'row_low=%', 'row_high=%', 'rowpairs=%',
                        'wg=%d physical_launches=1\\n', '(row_diff + 1) / 2, batch4_wg'):
            self.assertIn(literal, self.host)
        self.assertEqual(host.count('BOLT_Q8_VERIFY_BATCH4'), 1)
        self.assertNotIn('bolt_route_record(', self.host)
        self.assertNotIn('getenv', self.host + self.entry)  # no knob, existing default-off logger only

    def test_06_row_pair_token_bijection_every_wg(self):
        self.assertIn('const size_t token = group % 4;', self.launch)
        self.assertIn('const int row_pair = (int) (group / 4);', self.launch)
        cases = 0
        for nrows, wg in itertools.product(list(range(1, 130)) + [255, 256, 1023, 1024], [1, 2, 4, 8]):
            groups = ((nrows + 1) // 2) * 4
            observed = [(group // 4, group % 4) for group in range(groups)]
            self.assertEqual(set(observed), set(itertools.product(range((nrows + 1) // 2), range(4))))
            self.assertEqual(len(set(observed)), groups)
            self.assertEqual([group % 4 for group in range(4)], list(range(4)))
            writes = []
            for row_pair, token in observed:
                tids = [global_id % wg for global_id in range((row_pair * 4 + token) * wg,
                                                             (row_pair * 4 + token + 1) * wg)]
                self.assertEqual(tids, list(range(wg)))
                self.assertEqual(sorted(tid * 2 + lane for tid in tids for lane in (0, 1)), list(range(wg * 2)))
                row0 = row_pair * 2
                writes.append((token, row0))
                if row0 + 1 < nrows:
                    writes.append((token, row0 + 1))
            self.assertEqual(sorted(writes), list(itertools.product(range(4), range(nrows))))
            cases += 1
        print(f'Row-pair/token bijection, local IDs/SLM slots and odd-row stores: {cases} geometries.')

    def test_07_column_offsets_and_canaries_vs_original_loop(self):
        cases = 0
        for nrows, row_low, dst_on_device, pad, nblk in itertools.product(
                [1, 2, 3, 7, 8], [0, 5], [False, True], [0, 9], [1, 15, 16, 17, 31, 32, 33, 63, 64, 65]):
            k = nblk * 32
            wg = 8 if nblk >= 64 else 4 if nblk >= 32 else 2 if nblk >= 16 else 1
            ne0 = row_low + nrows + pad
            stride = ne0 if dst_on_device else nrows
            base = row_low if dst_on_device else 0
            guard = 32
            actual = bytearray(b'\xa5' * (guard * 2 + 4 * ne0 * 4))
            expected = bytearray(actual)
            for token, row in itertools.product(range(4), range(nrows)):
                struct.pack_into('<I', expected, guard + 4 * (base + token * stride + row), token * 1000 + row)
            touched = set()
            for group in range(((nrows + 1) // 2) * 4):
                token, row_pair = group % 4, group // 4
                row0 = row_pair * 2
                # Actual native per-thread stripe/tail ordering; no FP simulation.
                blocks = []
                for tid in range(wg):
                    ib = 0
                    while ib + wg * 8 <= nblk:
                        blocks.extend(range(ib + tid * 8, ib + tid * 8 + 8))
                        ib += wg * 8
                    blocks.extend(range(ib + tid, nblk, wg))
                self.assertEqual(sorted(blocks), list(range(nblk)))
                for block in blocks:
                    y_begin = token * k + block * 32
                    self.assertGreaterEqual(y_begin, token * k)
                    self.assertLessEqual(y_begin + 32, (token + 1) * k)
                for row in range(row0, min(row0 + 2, nrows)):
                    # Reordered qs/scales are based on row_diff, not absolute row_low.
                    q_end = (row * nblk + nblk) * 32
                    d_end = nrows * nblk * 32 + (row * nblk + nblk) * 2
                    self.assertLessEqual(q_end, nrows * nblk * 32)
                    self.assertLessEqual(d_end, nrows * nblk * 34)
                    offset = guard + 4 * (base + token * stride + row)
                    self.assertNotIn(offset, touched)
                    touched.add(offset)
                    struct.pack_into('<I', actual, offset, token * 1000 + row)
            self.assertEqual(actual, expected)  # gaps, both canaries and untouched row-low prefix
            self.assertEqual(len(touched), 4 * nrows)
            cases += 1
        print(f'Column/row-low/reordered-weight offsets, native stripe coverage and canaries: {cases} cases.')

    def test_08_actual_host_preflight_with_cpu_launch_spy(self):
        # Compile ONLY the actual scalar preflight/selector, with an inert submission spy.
        # This is not compilation of ESIMD/SYCL code or a replacement for the native gate.
        declaration = section(self.candidate['dmmv.hpp'], HEADER_START, '#endif // GGML_SYCL_DMMV_HPP')
        prelude = r'''
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
namespace dpct { using queue_ptr = void *; }
#define GGML_UNUSED(x) (void) (x)
constexpr int QK8_0 = 32;
struct block_q8_0 { unsigned char bytes[34]; };
static int calls, selected;
static bool throw_on_launch;
template <int WG>
static void q8_0_esimd_launch_batch4(const void *, const float *, float *, int k,
                                    int rows, size_t ys, size_t ds, dpct::queue_ptr stream) {
    assert(k > 0 && rows > 0 && ys == (size_t) k && ds >= (size_t) rows && stream);
    ++calls; selected = WG;
    if (throw_on_launch) throw std::runtime_error("synthetic submission failure");
}
struct Input {
    uintptr_t vx = uintptr_t(1) << 20, y = uintptr_t(1) << 44, dst = uintptr_t(1) << 56;
    int64_t k = 2048, low = 0, high = 7, ys = 2048, ds = 7;
    dpct::queue_ptr queue = reinterpret_cast<void *>(uintptr_t(16));
};
'''
        tests = r'''
static int checks;
static void check(Input a, bool expected, int wg = 8) {
    calls = selected = 0;
    int launched = -123;
    bool result = ggml_sycl_op_dequantize_mul_mat_vec_q8_0_batch4(
        reinterpret_cast<const void *>(a.vx), reinterpret_cast<const float *>(a.y),
        reinterpret_cast<float *>(a.dst), a.k, a.low, a.high, a.ys, a.ds, a.queue, launched);
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    assert(result == expected);
    assert(calls == (expected ? 1 : 0));
    assert(launched == (expected ? wg : -123));
    assert(selected == (expected ? wg : 0));
#else
    (void) expected; (void) wg;
    assert(!result && !calls && launched == -123);
#endif
    ++checks;
}
int main() {
    static_assert(sizeof(uintptr_t) == 8, "this fixture tests 64-bit host bounds");
    for (int blocks : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128}) {
        for (int rows : {1, 2, 3, 8, 9}) {
            Input a; a.k = a.ys = blocks * 32; a.high = rows; a.ds = rows + 5;
            check(a, true, blocks >= 64 ? 8 : blocks >= 32 ? 4 : blocks >= 16 ? 2 : 1);
            a.low = 17; a.high += a.low; check(a, true, blocks >= 64 ? 8 : blocks >= 32 ? 4 : blocks >= 16 ? 2 : 1);
        }
    }
    Input a;
    a.vx = 0; check(a, false); a = {};
    a.y = 0; check(a, false); a = {};
    a.dst = 0; check(a, false); a = {};
    a.queue = nullptr; check(a, false); a = {};
    ++a.vx; check(a, false); a = {};
    ++a.y; check(a, false); a = {};
    ++a.dst; check(a, false); a = {};
    for (int64_t k : {INT64_MIN, int64_t(-1), int64_t(0), int64_t(33), int64_t(INT_MAX) + 1}) {
        a.k = a.ys = k; check(a, false); a = {};
    }
    a.low = -1; check(a, false); a = {};
    a.high = 0; check(a, false); a = {};
    a.high = INT64_MIN; check(a, false); a = {};
    a.low = INT64_MAX; check(a, false); a = {};
    a.high = INT_MAX; a.ds = INT_MAX; check(a, false); a = {};
    a.high = INT64_MAX; a.ds = INT64_MAX; check(a, false); a = {};
    a.low = INT64_MAX - 7; a.high = INT64_MAX; check(a, true); a = {};
    a.ds = 6; check(a, false); a = {};
    a.ds = -1; check(a, false); a = {};
    a.ys = 2047; check(a, false); a = {};
    a.ys = INT64_MAX; check(a, false); a = {};
    a.ds = PTRDIFF_MAX / sizeof(float) / 4; check(a, true);
    ++a.ds; check(a, false); a = {};
    a.k = a.ys = INT_MAX - 31; a.high = a.ds = 1; check(a, true); a = {};
    a.k = a.ys = 32; a.high = a.ds = INT_MAX - 1; check(a, true, 1); a = {};
    a.vx = UINTPTR_MAX & ~uintptr_t(15); check(a, false); a = {};
    a.y = UINTPTR_MAX & ~uintptr_t(3); check(a, false); a = {};
    a.dst = UINTPTR_MAX & ~uintptr_t(3); check(a, false); a = {};
    a.dst = a.vx; check(a, false); a = {};
    a.dst = a.y; check(a, false); a = {};
    a.dst = a.y + 3 * a.ys * sizeof(float); check(a, false); a = {};
    a.y = a.dst + (4 * a.ds - 1) * sizeof(float); check(a, false); a = {};
    a.y = a.dst + 4 * a.ds * sizeof(float); check(a, true); a = {};
    a.dst = a.vx + 7 * 64 * 34 - 4; check(a, false); a = {};
    a.dst = a.vx + 7 * 64 * 34; check(a, true); a = {};
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    throw_on_launch = true;
    calls = 0;
    int launched = -123;
    try {
        ggml_sycl_op_dequantize_mul_mat_vec_q8_0_batch4(
            reinterpret_cast<const void *>(a.vx), reinterpret_cast<const float *>(a.y),
            reinterpret_cast<float *>(a.dst), a.k, a.low, a.high, a.ys, a.ds, a.queue, launched);
        assert(false);
    } catch (const std::runtime_error &) {
        assert(calls == 1 && launched == -123); // no false return / partial-launch fallback
    }
    std::printf("Actual host preflight/selector: %d cases, plus propagated submission failure.\n", checks);
#else
    (void) throw_on_launch;
    std::printf("No-ESIMD host stub: %d cases, zero calls and unchanged launched_wg.\n", checks);
#endif
}
'''
        fixture = self.work / 'host-preflight.cpp'
        fixture.write_text(prelude + declaration + self.entry + tests)
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        for esimd in (True, False):
            executable = self.work / ('host-preflight-esimd' if esimd else 'host-preflight-no-esimd')
            command = compiler + ['-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=undefined', '-fno-sanitize-recover=all']
            if esimd:
                command.append('-DGGML_SYCL_DMMV_HAS_ESIMD')
            run(command + [str(fixture), '-o', str(executable)])
            run([str(executable)])


if __name__ == '__main__':
    print('DEVELOPMENT: CPU/source only; no SYCL compilation, GPU, native byte or serving pass.', flush=True)
    print('platform:', platform.platform(), flush=True)
    print('python:', sys.version.replace('\n', ' '), flush=True)
    print('source:', SOURCE, flush=True)
    print('CXX:', os.environ.get('CXX', 'c++'), flush=True)
    for command in (['git', '--version'], ['patch', '--version'], shlex.split(os.environ.get('CXX', 'c++')) + ['--version'],
                    ['git', 'rev-parse', 'HEAD']):
        run(command)
    unittest.main(verbosity=2)
