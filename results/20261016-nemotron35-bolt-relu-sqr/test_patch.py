#!/usr/bin/env python3
"""Source-level checks for the F32 RELU->SQR SYCL patch.

Usage: python3 -B test_patch.py R1_TREE CANDIDATE_TREE

This deliberately does not compile SYCL, allocate a device, or run inference.
The native graph cases remain in tests/test-backend-ops.cpp and are run by the
lead on the inference host.
"""
from pathlib import Path
import re
import sys
import unittest


R1, CANDIDATE = (Path(arg).resolve() for arg in sys.argv[1:3])


def source(tree, relative):
    return (tree / relative).read_text()


class PatchTests(unittest.TestCase):
    def test_r1_has_no_relu_sqr_dispatch(self):
        for relative in (
            "ggml/src/ggml-sycl/element_wise.cpp",
            "ggml/src/ggml-sycl/element_wise.hpp",
            "ggml/src/ggml-sycl/ggml-sycl.cpp",
        ):
            self.assertNotIn("ggml_sycl_op_relu_sqr_fused", source(R1, relative))

    def test_fusion_guard_is_f32_contiguous_and_graph_safe(self):
        text = source(CANDIDATE, "ggml/src/ggml-sycl/fusion.cpp")
        header = source(CANDIDATE, "ggml/include/ggml.h")
        match = re.search(r"enum\s+ggml_op\s*\{(?P<body>.*?)\};", header, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("GGML_OP_SQR", match.group("body"))
        self.assertNotIn("GGML_UNARY_OP_SQR", match.group("body"))
        self.assertIn("GGML_UNARY_OP_RELU", text)
        self.assertIn("ops.begin()[1] == GGML_OP_SQR", text)
        self.assertIn("unary_ops.size() == 1", text)
        self.assertIn("unary_ops.begin()[0] == GGML_UNARY_OP_RELU", text)
        self.assertIn("sqr->op != GGML_OP_SQR", text)
        self.assertIn("relu->src[0]->type != GGML_TYPE_F32", text)
        self.assertIn("!ggml_is_contiguous(relu->src[0])", text)
        self.assertIn("!ggml_is_contiguous(sqr)", text)
        self.assertIn("ggml_nelements(sqr) == 0", text)
        self.assertIn("ggml_can_fuse(cgraph, node_idx, ops)", text)
        self.assertNotIn("GGML_UNARY_OP_SQR", text)

    def test_kernel_dispatch_and_debug_marker(self):
        element_wise = source(CANDIDATE, "ggml/src/ggml-sycl/element_wise.cpp")
        self.assertIn("dst[i] = op_sqr(op_relu(x[i]));", element_wise)
        self.assertIn('" : fused RELU_SQR"', element_wise)
        self.assertIn("ggml_sycl_op_relu_sqr_fused", element_wise)
        kernel_pos = element_wise.index("static void relu_sqr_sycl")
        namespace_pos = element_wise.index("namespace ggml_sycl_detail")
        self.assertLess(kernel_pos, namespace_pos)
        self.assertIn("relu_sqr_sycl((const float *)", element_wise)
        self.assertNotIn("ggml_sycl_detail::relu_sqr_sycl", element_wise)

        dispatcher = source(CANDIDATE, "ggml/src/ggml-sycl/ggml-sycl.cpp")
        self.assertIn("cgraph->nodes[i + 1]->op == GGML_OP_SQR", dispatcher)
        self.assertIn("{ GGML_OP_UNARY, GGML_OP_SQR }", dispatcher)
        self.assertIn("{ GGML_UNARY_OP_RELU }))", dispatcher)
        self.assertNotIn("GGML_UNARY_OP_SQR", dispatcher)
        self.assertIn("ggml_sycl_op_relu_sqr_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);", dispatcher)
        self.assertIn("i++;\n            continue;\n        }\n        if (node->op == GGML_OP_UNARY", dispatcher)

    def test_native_case_catalog(self):
        text = source(CANDIDATE, "tests/test-backend-ops.cpp")
        # Four cases from the type loop plus seventeen explicit Bolt/guard cases.
        self.assertEqual(text.count("new test_relu_sqr"), 19)
        self.assertEqual(text.count("new test_relu_sqr(GGML_TYPE_F32"), 17)
        for shape in (
            "{ 2048, 1, 1, 1 }",
            "{ 5376, 1, 1, 1 }",
            "{ 2048, 1024, 1, 1 }",
            "{ 5376, 1024, 1, 1 }",
            "{ 257, 3, 1, 1 }",
        ):
            self.assertIn(shape, text)
        for width in (1536, 2048, 2304, 2560):
            for columns in (1, 31):
                self.assertIn(f"{{ {width}, 9, {columns}, 1 }}", text)
        for tail in ('"strided"', '"reuse"', '"output"', '"consumer"'):
            self.assertIn(tail, text)
        self.assertIn('return "relu_sqr=1," + VARS_TO_STR4(type, ne, layout, tail);', text)
        self.assertIn("checked = { r, side, out };", text)
        self.assertIn("ggml_set_output(r);", text)
        self.assertIn("return type == GGML_TYPE_F32 ? 0.0 : test_case::max_nmse_err();", text)

    def test_scalar_contract_is_finite_pointwise(self):
        # Keep the reference contract visible in this source-only check without
        # pretending that it substitutes for native CPU-vs-SYCL execution.
        for value in (-3.0, -0.0, 0.0, 0.25, 3.0):
            expected = max(value, 0.0) ** 2
            self.assertEqual(expected, max(value, 0.0) * max(value, 0.0))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 -B test_patch.py R1_TREE CANDIDATE_TREE")
    unittest.main(argv=[sys.argv[0]], verbosity=2)
