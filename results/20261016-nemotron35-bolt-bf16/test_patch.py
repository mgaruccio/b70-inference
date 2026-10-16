#!/usr/bin/env python3
"""Local source/scalar checks ONLY; does not compile SYCL or execute a backend.

Usage: python3 -B test_patch.py R1_TREE CANDIDATE_TREE
The two trees must contain the pinned test-backend-ops.cpp. Requires host c++.
Counter compilation runs only the actual registration loops, with stub cases.
"""
import pathlib
import re
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest

R1, CANDIDATE = (pathlib.Path(arg).resolve() for arg in sys.argv[1:3])
TEST_FILE = pathlib.Path("tests/test-backend-ops.cpp")
OLD = (R1 / TEST_FILE).read_text()
NEW = (CANDIDATE / TEST_FILE).read_text()
HERE = pathlib.Path(__file__).resolve().parent
LAYOUT = runpy.run_path(str(HERE.parent / "20261016-nemotron35-bolt-nvfp4-reorder/test_layout.py"))


def section(text, begin, end):
    return text.split(begin, 1)[1].split(end, 1)[0]


def bf16_bits(bits):
    # Finite F32 values only; independent round-to-nearest-even integer reference.
    assert bits & 0x7f800000 != 0x7f800000
    return (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000


def as_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


class PatchTests(unittest.TestCase):
    def test_r1_suite_unchanged(self):
        start = "struct test_mul_mat_nvfp4_reorder :"
        self.assertEqual(section(OLD, start, "\n};"), section(NEW, start, "\n};"))
        start = "    // Experimental NVFP4 reorder lifetime/scale regressions."
        self.assertEqual(
            section(OLD, start, "    // Multi-column MMVQ").strip(),
            section(NEW, start, "    // 20 separately selected BF16").strip(),
        )
        for name, selector in (
            ("test_mul_mat_nvfp4_bf16", "nvfp4_bf16_lifetime=1"),
            ("test_mul_mat_nvfp4_bf16_reference", "nvfp4_bf16_reference=1"),
        ):
            body = section(NEW, "struct " + name + " :", "\n};")
            self.assertIn('return "' + selector + '," + test_mul_mat::vars()', body)
            self.assertNotIn('"nvfp4_reorder=1', body)

    def test_actual_registration_counts(self):
        loops = section(NEW, "    // Experimental NVFP4 reorder lifetime/scale regressions.",
                        "    // Multi-column MMVQ")
        # Drop the first comment's remaining text, preserving the C++ loops verbatim.
        loops = loops.split("\n", 1)[1]
        source = """#include <cstdint>
#include <cstdio>
#include <initializer_list>
int counts[3] = {};
"""
        for i, name in enumerate(("test_mul_mat_nvfp4_reorder", "test_mul_mat_nvfp4_bf16",
                                  "test_mul_mat_nvfp4_bf16_reference")):
            source += (f"struct {name} {{ template<class... T> {name}(T...) "
                       f"{{ ++counts[{i}]; }} }};\n")
        source += """struct cases {
    template<class T> void emplace_back(T * p) { delete p; }
};
int main() {
    cases test_cases;
""" + loops + """
    std::printf("R1=%d lifetime=%d reference=%d\\n", counts[0], counts[1], counts[2]);
    return counts[0] != 56 || counts[1] != 20 || counts[2] != 4;
}
"""
        scratch = pathlib.Path.cwd() / ".pi"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="bf16-counts-", dir=scratch) as tmp:
            cpp = pathlib.Path(tmp) / "counts.cpp"
            exe = pathlib.Path(tmp) / "counts"
            cpp.write_text(source)
            subprocess.run(["c++", "-std=c++17", str(cpp), "-o", str(exe)], check=True)
            result = subprocess.run([str(exe)], check=True, text=True, capture_output=True)
        self.assertEqual(result.stdout, "R1=56 lifetime=20 reference=4\n")
        print(result.stdout.strip())

    def test_native_rne_vectors(self):
        body = section(NEW, "struct test_mul_mat_nvfp4_bf16_reference :", "\n};")
        vectors = []
        for name in ("bits", "rounded"):
            array = section(body, "uint32_t " + name + "[] = {", "};")
            vectors.append([int(h, 16) for h in re.findall(r"0x[0-9a-f]+", array)])
        inputs, expected = vectors
        self.assertEqual(len(inputs), 10)
        self.assertEqual([bf16_bits(bits) for bits in inputs], expected)
        self.assertTrue(all(a != b for a, b in zip(inputs, expected)))  # F32 bypass must fail

    def test_nvfp4_values_and_onehot_layout(self):
        for scale in range(128):
            for nibble in LAYOUT["VALUES"]:
                bits = as_bits(LAYOUT["scale"](scale) * nibble)
                self.assertEqual(bf16_bits(bits), bits)
        k, m = 192, 37
        data = bytearray(k*m//64*36)
        for ib in range(len(data)//36):
            data[ib*36:ib*36+4] = bytes([0x38]*4)
        for row in range(m):
            j = row % 64
            block = row*(k//64) + row//64
            data[block*36 + 4 + (j//16)*8 + j%8] = 2 << (4 if j%16 >= 8 else 0)
        a = LAYOUT["scalar_decode"](data)
        self.assertEqual(a, LAYOUT["reordered_decode"](LAYOUT["reorder"](data), k*m))
        for row in range(m):
            self.assertEqual(a[row*k:(row+1)*k], [float(p == row) for p in range(k)])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
