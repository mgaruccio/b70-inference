#!/usr/bin/env python3
"""Bounded CPU/extracted-source check, not a SYCL build or device/speed proof.

All generated files stay in this package's ignored .scratch and are removed.
Only the existing DPCT byte extraction and subgroup operations are CPU shims;
original/new dots, sign mask, dp4a body, and native K/XOR function are extracted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = Path('ggml/src/ggml-sycl')
OLD = 'vec_dot_iq2_xxs_q8_1_moe'
NEW = OLD + '_n4_acc2'
SCALAR = 'vec_dot_iq2_xxs_q8_1'
TAIL = ('    const float d = (float)bq2->d * (0.5f + aux32) * '
        'bq8_1[ib32].ds[0] * 0.25f;\n    return d * sumi;')


def run(argv, cwd=ROOT, *, input=None):
    print('+', shlex.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=cwd, check=True, timeout=60, input=input)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(text, name, marker='static __dpct_inline__'):
    at = re.search(r'\b' + re.escape(name) + r'\s*\(', text).start()
    start = text.rfind(marker, 0, at)
    assert start >= 0, name
    brace = text.index('{', at)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def int_clone(text, name):
    assert text.count(TAIL) == 1
    return (text.replace('static __dpct_inline__ float', 'static __dpct_inline__ int', 1)
            .replace(name + '(', name + '_sum(', 1).replace(TAIL, '    return sumi;'))


def table(text, name):
    m = re.search(r'GGML_TABLE_BEGIN\((\w+), ' + name + r', (\d+)\)(.*?)GGML_TABLE_END\(\)', text, re.S)
    assert m, name
    return f'static const {m[1]} {name}[{m[2]}] = {{{m[3]}}};\n'


def block(text, name):
    end = text.index('} ' + name + ';') + len('} ' + name + ';')
    start = text.rfind('typedef struct {', 0, end)
    return text[start:end]


PREAMBLE = r'''
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#define __dpct_inline__ inline
#define QK_K 256
#define QK8_1 32
#define QI2_XXS (QK_K / (4*4))
#define VDR_IQ2_XXS_Q8_1_MMVQ 1
#define WARP_SIZE 16
#define GGML_EXTENSION
#define GGML_COMMON_AGGR_S
#define GGML_COMMON_AGGR_U
using ggml_half = _Float16;
struct alignas(4) ggml_half2 {
    ggml_half v[2];
    ggml_half & operator[](int i) { return v[i]; }
    const ggml_half & operator[](int i) const { return v[i]; }
};
namespace sycl {
struct sub_group { int lane; };
template<int> struct nd_item {
    int lane, row, expert;
    int get_group(int dim) const { return dim == 1 ? expert : dim == 2 ? row : 0; }
    int get_local_range(int dim) const { return dim == 2 ? WARP_SIZE : 1; }
    int get_local_id(int dim) const { return dim == 2 ? lane : 0; }
    sub_group get_sub_group() const { return {lane}; }
};
}
namespace dpct {
// CPU implementation of helper.hpp's vec.as<signed/unsigned byte>().convert<T>().
// The accumulator alias and dp4a body below are extracted verbatim from the pin.
template<typename T> std::array<T, 4> extract_and_sign_or_zero_extend4(T val) {
    static_assert(sizeof(T) == 4);
    using byte = std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>;
    std::array<byte, 4> bytes;
    memcpy(bytes.data(), &val, 4);
    return {T(bytes[0]), T(bytes[1]), T(bytes[2]), T(bytes[3])};
}
struct partial { float value; };
static bool capture_partial;
static std::array<std::array<float, WARP_SIZE>, WARP_SIZE> peers;
float permute_sub_group_by_xor(sycl::sub_group g, float value, int mask) {
    if (capture_partial) throw partial{value};
    return peers[mask][g.lane ^ mask];
}
'''

HARNESS = r'''
static uint32_t rng_state = 0x1278a35u;
static uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static ggml_half half(uint16_t bits) {
    ggml_half h;
    memcpy(&h, &bits, 2);
    return h;
}
static uint32_t bits(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}
static uint64_t comparisons = 0;
static void compare(const block_iq2_xxs & x, const block_q8_1 * y, int iqs) {
    const int a = vec_dot_iq2_xxs_q8_1_sum(&x, y, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    const int b = vec_dot_iq2_xxs_q8_1_moe_n4_acc2_sum(&x, y, iqs);
    assert(a == b && std::abs(a) <= 176128);
    assert(bits(vec_dot_iq2_xxs_q8_1_moe(&x, y, iqs)) ==
           bits(vec_dot_iq2_xxs_q8_1_moe_n4_acc2(&x, y, iqs)));
    ++comparisons;
}
static void encode(block_iq2_xxs & x, int iqs, int grid, int sign, int scale) {
    uint8_t * a = reinterpret_cast<uint8_t *>(x.qs + 4*iqs);
    uint32_t encoded = uint32_t(scale) << 28;
    for (int l = 0; l < 4; ++l) {
        a[l] = uint8_t(grid + 37*l);
        encoded |= uint32_t((sign + 23*l) & 127) << (7*l);
    }
    x.qs[4*iqs + 2] = uint16_t(encoded);
    x.qs[4*iqs + 3] = uint16_t(encoded >> 16);
}
static void fill_q8(block_q8_1 & y, int pattern) {
    y.ds[0] = half(0x3800);
    y.ds[1] = half(0);
    for (int j = 0; j < 32; ++j) {
        int v = pattern == 0 ? 0 : pattern == 1 ? -128 : pattern == 2 ? 127 :
                pattern == 3 ? (j % 2 ? -128 : 127) :
                pattern == 4 ? (j == 31 ? -128 : 0) : int(rnd() % 256) - 128;
        y.qs[j] = int8_t(v);
    }
}
static constexpr uint16_t edges[] = {
    0, 0x8000, 1, 0x8001, 0x3ff, 0x83ff, 0x400, 0x8400, 0x401, 0x8401,
    0x3555, 0xb555, 0x3bff, 0xbbff, 0x3c00, 0xbc00, 0x3c01, 0xbc01,
    0x7bfe, 0xfbfe, 0x7bff, 0xfbff
};
static void dot_tests() {
    block_iq2_xxs x{};
    std::array<block_q8_1, 8> y{};
    x.d = half(0x3c00);
    // Each l and iqs sees every actual grid/sign pair; group encodings differ.
    for (int grid = 0; grid < 256; ++grid) for (int sign = 0; sign < 128; ++sign)
        for (int iqs = 0; iqs < 8; ++iqs) {
            encode(x, iqs, grid, sign, (grid + sign + iqs) & 15);
            fill_q8(y[iqs], (grid + sign + iqs) % 6);
            compare(x, y.data(), iqs);
        }
    printf("PASS grid/sign dots: %llu integer+F32-bit comparisons\n", (unsigned long long)comparisons);
    for (int v = -128; v <= 127; ++v) for (int iqs = 0; iqs < 8; ++iqs) {
        encode(x, iqs, rnd() % 256, rnd() % 128, rnd() % 16);
        for (auto & q : y[iqs].qs) q = int8_t(v);
        compare(x, y.data(), iqs);
    }
    // Exhaust every finite binary16 encoding in EACH scale operand (not every pair).
    unsigned finite_scales = 0;
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        ++finite_scales;
        for (int iqs = 0; iqs < 8; ++iqs) {
            encode(x, iqs, rnd() % 256, rnd() % 128, (h + iqs) & 15);
            fill_q8(y[iqs], (h + iqs) % 6);
            x.d = half(h);
            y[iqs].ds[0] = half(edges[(h + iqs) % 22]);
            compare(x, y.data(), iqs);
            x.d = half(edges[(h + iqs) % 22]);
            y[iqs].ds[0] = half(h);
            compare(x, y.data(), iqs);
        }
    }
    assert(finite_scales == 63488);
    // Cartesian signed zero/subnormal/normal/max-finite edges, all high nibbles,
    // all iqs, and zero/extreme/single-impulse/random activation patterns.
    for (uint16_t a : edges) for (uint16_t b : edges) for (int s = 0; s < 16; ++s)
        for (int p = 0; p < 6; ++p) for (int iqs = 0; iqs < 8; ++iqs) {
            encode(x, iqs, rnd() % 256, rnd() % 128, s);
            fill_q8(y[iqs], p);
            x.d = half(a);
            y[iqs].ds[0] = half(b);
            compare(x, y.data(), iqs);
        }
    for (int r = 0; r < 4096; ++r) for (int iqs = 0; iqs < 8; ++iqs) {
        encode(x, iqs, rnd() % 256, rnd() % 128, rnd() % 16);
        fill_q8(y[iqs], 5);
        x.d = half(uint16_t(rnd() % 0x7c00) | (rnd() & 0x8000));
        y[iqs].ds[0] = half(uint16_t(rnd() % 0x7c00) | (rnd() & 0x8000));
        compare(x, y.data(), iqs);
    }
    printf("PASS scales: 63488 finite encodings/operand; 22x22 edges x16 scales x8 iqs x6 Q8 patterns\n");
    printf("PASS dots total: %llu integer+F32-bit comparisons; Q8 includes every -128..127 value\n",
           (unsigned long long)comparisons);
}
template<vec_dot_q_sycl_t DOT>
static std::array<float, 17> native_partials(const block_iq2_xxs * x, const block_q8_1 * y,
                                           int k, int row, int expert, size_t ys) {
    std::array<float, 16> partials{};
    const int32_t ids[2] = {2, 0};
    float dst[6]{};
    dpct::capture_partial = true;
    for (int lane = 0; lane < 16; ++lane) {
        bool caught = false;
        try {
            mul_mat_vec_q_moe<QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, DOT>(
                x, y, dst, ids, k, 3, 3*(k/256)*sizeof(*x), 3*sizeof(float), ys,
                sycl::nd_item<3>{lane, row, expert});
        } catch (dpct::partial p) { partials[lane] = p.value; caught = true; }
        assert(caught);
    }
    // Snapshot simultaneous lanes at each XOR step, then run the actual native
    // function again. No CPU scheduling order is asserted for the real GPU.
    dpct::capture_partial = false;
    auto sums = partials;
    for (int mask = 8; mask > 0; mask >>= 1) {
        dpct::peers[mask] = sums;
        for (int lane = 0; lane < 16; ++lane)
            sums[lane] += dpct::peers[mask][lane ^ mask];
    }
    mul_mat_vec_q_moe<QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, DOT>(
        x, y, dst, ids, k, 3, 3*(k/256)*sizeof(*x), 3*sizeof(float), ys,
        sycl::nd_item<3>{0, row, expert});
    assert(bits(dst[3*expert + row]) == bits(sums[0]));
    std::array<float, 17> result{};
    for (int lane = 0; lane < 16; ++lane) result[lane] = partials[lane];
    result[16] = dst[3*expert + row];
    return result;
}
static void native_tests() {
    uint64_t rows = 0;
    for (int k : {1024, 2048}) for (int fixture = 0; fixture < 16; ++fixture) {
        std::vector<block_iq2_xxs> x(3*3*(k/256));
        std::vector<block_q8_1> y(4*2*(k/32));
        for (auto & b : x) {
            b.d = half(edges[fixture % 22]);
            for (int iqs = 0; iqs < 8; ++iqs) encode(b, iqs, rnd() % 256, rnd() % 128, rnd() % 16);
        }
        for (auto & b : y) fill_q8(b, fixture % 6);
        const size_t ys = fixture % 2 ? (k/32)*sizeof(y[0]) : 0;
        for (int token = 0; token < 4; ++token) for (int e = 0; e < 2; ++e) for (int row = 0; row < 3; ++row) {
            const auto * yt = y.data() + token*2*(k/32);
            const auto a = native_partials<vec_dot_iq2_xxs_q8_1_moe>(x.data(), yt, k, row, e, ys);
            const auto b = native_partials<vec_dot_iq2_xxs_q8_1_moe_n4_acc2>(x.data(), yt, k, row, e, ys);
            for (int lane = 0; lane < 17; ++lane) assert(bits(a[lane]) == bits(b[lane]));
            ++rows;
        }
    }
    printf("PASS extracted native K/XOR: K=1024,2048; %llu rows x16 lanes, 4 token slices, shared/per-expert Q8\n",
           (unsigned long long)rows);
}
int main() {
    static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(ggml_half) == 2);
    static_assert(sizeof(block_iq2_xxs) == 66 && alignof(block_iq2_xxs) == 2);
    static_assert(sizeof(block_q8_1) == 36 && alignof(block_q8_1) == 4 && offsetof(block_q8_1, qs) == 4);
    static_assert(32*43*128 < INT32_MAX);
    static_assert(16*43*128 <= 88064);
    static_assert(std::is_same_v<decltype(dpct::dp4a(int{}, int{}, int{})), int32_t>);
    uint32_t endian = 1;
    assert(*reinterpret_cast<uint8_t *>(&endian) == 1);
    assert(float(half(1)) == std::ldexp(1.0f, -24));
    assert(float(half(0x7bff)) == 65504.0f && bits(float(half(0x8000))) == 0x80000000);
    assert(std::numeric_limits<float>::is_iec559);
    table_tests();
    dot_tests();
    native_tests();
    puts("PASS CPU exactness only; no SDK/device execution, hardware-DP4A, or speed claim");
}
'''


def main():
    assert __debug__, 'do not disable assertions'
    pins = json.loads((HERE / 'source-pins.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=pins['qualified_source_root'])
    parser.add_argument('--codebook', type=Path, default=pins['codebook_path'])
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    print('Development-tier CPU/source check; baseline=immutable batch4; delta=one private n4 dot only', flush=True)
    print('Invocation:', shlex.join([sys.executable, '-B', *sys.argv]), flush=True)
    print('Environment:', platform.platform(), platform.python_version(), 'cwd=' + str(Path.cwd()), flush=True)
    run([args.compiler, '--version'])
    originals = {}
    for name, expected in pins['qualified_sycl_sha256'].items():
        originals[name] = (args.source_root / REL / name).read_bytes()
        assert sha(originals[name]) == expected, name
    codebook = args.codebook.read_bytes()
    assert sha(codebook) == pins['codebook_sha256']
    batch = ROOT / pins['batch4_patch']
    assert sha(batch.read_bytes()) == pins['batch4_patch_sha256']
    patch = HERE / 'iq2-acc2.patch'
    assert re.findall(r'^\+\+\+ b/(.*)$', patch.read_text(), re.M) == [str(REL / 'mmvq.cpp')]
    # Every mutation is confined to this checkout/package, even with external input overrides.
    assert HERE.is_relative_to(ROOT) and not (HERE / '.scratch').is_symlink()
    (HERE / '.scratch').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cpu-', dir=HERE / '.scratch') as tmp:
        work = Path(tmp)
        gnu = work / 'gnu-pristine'
        for name, data in originals.items():
            for target in (work, gnu):
                dst = target / REL / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
        def apply(path, reverse=False):
            opts = ['-R'] if reverse else []
            # Independent pristine trees exercise both delivery paths, not a proxy.
            prefix = str(work.relative_to(ROOT))
            run(['git', 'apply', '--check', '--directory=' + prefix, *opts, path])
            run(['git', 'apply', '--directory=' + prefix, *opts, path])
            direction = '--reverse' if reverse else '--forward'
            command = ['patch', '--directory=' + str(gnu), '--fuzz=0', '--batch', direction, '-p1']
            run([*command, '--dry-run'], input=path.read_bytes())
            run(command, input=path.read_bytes())
            assert all((gnu / REL / name).read_bytes() == (work / REL / name).read_bytes()
                       for name in originals), 'GNU/Git composition differs'
        def files():
            return {name: (work / REL / name).read_bytes() for name in originals}
        apply(batch)
        seed = files()
        assert sha(seed['mmvq.cpp']) == pins['batch4_mmvq_sha256']
        apply(patch)
        candidate = files()
        assert sha(candidate['mmvq.cpp']) == pins['acc2_mmvq_sha256']
        assert [name for name in originals if candidate[name] != seed[name]] == ['mmvq.cpp']
        mmvq = candidate['mmvq.cpp'].decode()
        new = extract(mmvq, NEW)
        assert new.count('int sumi0 = 0;') == new.count('int sumi1 = 0;') == 1
        assert new.count('dpct::dp4a(') == 4
        assert 'if (l < 2)' in new and 'const int sumi = sumi0 + sumi1;' in new
        addition = '// Private n==4 path: packed integer dot with two independent accumulators; keep XXS signs, scales and FP order.\n' + new + '\n\n'
        call = 'VDR_IQ2_XXS_Q8_1_MMVQ, ' + NEW + '>'
        old_call = 'VDR_IQ2_XXS_Q8_1_MMVQ, ' + OLD + '>'
        assert mmvq.count(addition) == mmvq.count(call) == 1
        assert mmvq.count(old_call) == 1  # the non-batch4 selector remains unchanged
        assert mmvq.replace(addition, '', 1).replace(call, call.replace(NEW, OLD), 1) == seed['mmvq.cpp'].decode()
        batch_helper = extract(mmvq, 'ggml_sycl_mul_mat_vec_q_id_iq2_batch4', 'void ')
        assert call in batch_helper
        seed_native = extract(seed['mmvq.cpp'].decode(), 'mul_mat_vec_q_moe', 'template <int qk,')
        candidate_native = extract(mmvq, 'mul_mat_vec_q_moe', 'template <int qk,')
        assert seed_native == candidate_native  # native K/XOR reduction is source-identical
        selector = extract(candidate['ggml-sycl.cpp'].decode(), 'ggml_sycl_mul_mat_id_iq2_batch4', 'static bool')
        assert 'if (n != 4 || !ggml_is_contiguous(src1)) return false;' in selector
        assert '#define GGML_SYCL_MMV_Y 1' in originals['common.hpp'].decode()
        assert pins['cpu_geometry'] == {'warp_size': 16, 'mmv_y': 1, 'native_k': [1024, 2048]}
        apply(patch, reverse=True)
        assert files() == seed
        apply(batch, reverse=True)
        assert files() == originals
        apply(batch)
        apply(patch)
        assert files() == candidate
        print('PASS pins; exact GNU fuzz=0 and Git forward/reverse composition; only new acc2 helper + batch4 callee; all other bytes unchanged', flush=True)
        vec = originals['vecdotq.hpp'].decode()
        old = extract(vec, SCALAR)
        assert old.count(TAIL) == new.count(TAIL) == 1
        helper = originals['dpct/helper.hpp'].decode()
        acc_start = helper.index('template <typename T1, typename T2>\n    using dot_product_acc_t')
        acc_end = helper.index(';', acc_start) + 1
        dp4a = extract(helper, 'dp4a', 'template <typename T1, typename T2, typename T3>')
        native = extract(mmvq, 'mul_mat_vec_q_moe', 'template <int qk,')
        cb = codebook.decode()
        # Actual new packed-weight statements, not a separately hand-coded fast path.
        weight_lines = new[new.index('        const uint64_t grid'):new.index('        // Signed inputs')]
        table_test = r'''
static void table_tests() {
    unsigned count = 0;
    int lo = 255, hi = 0;
    for (int g = 0; g < 256; ++g) for (int s = 0; s < 128; ++s) {
        const uint8_t aux8[] = {uint8_t(g)};
        const uint32_t aux32 = s;
        const int l = 0;
''' + weight_lines + r'''
        int8_t packed[8];
        memcpy(packed, &grid_l, 4);
        memcpy(packed + 4, &grid_h, 4);
        for (int j = 0; j < 8; ++j) {
            int v = int((grid >> (8*j)) & 255);
            assert(v == 8 || v == 25 || v == 43);
            lo = v < lo ? v : lo; hi = v > hi ? v : hi;
            assert(packed[j] == v * ((ksigns_iq2xs[s] & kmask_iq2xs[j]) ? -1 : 1));
            ++count;
        }
    }
    for (unsigned s = 0; s < 256; ++s) for (int j = 0; j < 8; ++j) {
        const auto mask = iq_sign_mask(s >> (4*(j/4)));
        assert(((mask >> (8*(j%4))) & 255) == ((s & (1u << j)) ? 255 : 0));
    }
    assert(lo == 8 && hi == 43);
    printf("PASS tables: 256 grids x128 sign codes x8 = %u signed bytes; all 256 sign masks; values={8,25,43}, nonzero\n", count);
    puts("PASS bound: every 32-product prefix <= 32*43*128 = 176128 < INT32_MAX; signed dp4a inputs; memcpy loads");
}
'''
        cpp = (PREAMBLE + helper[acc_start:acc_end] + '\n' + dp4a + '\n}\n' +
               block(cb, 'block_q8_1') + '\n' + block(cb, 'block_iq2_xxs') + '\n' +
               ''.join(table(cb, n) for n in ['iq2xxs_grid', 'ksigns_iq2xs', 'kmask_iq2xs']) +
               extract(vec, 'iq_sign_mask') + '\n' + old + '\n' + extract(mmvq, OLD) + '\n' + new + '\n' +
               int_clone(old, SCALAR) + '\n' + int_clone(new, NEW) + '\n' +
               'using vec_dot_q_sycl_t = float (*)(const void *, const block_q8_1 *, const int &);\n' +
               native + '\n' + table_test + HARNESS)
        source = work / 'extracted.cpp'
        source.write_text(cpp)
        flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-unknown-pragmas',
                 '-fno-fast-math', '-ffp-contract=off', '-fsanitize=undefined', '-fno-sanitize-recover=all']
        run([args.compiler, *flags, source, '-o', work / 'extracted'])
        run([work / 'extracted'])
    print('PASS cleanup: generated source, binary and seed/candidate copies removed', flush=True)


if __name__ == '__main__':
    main()
