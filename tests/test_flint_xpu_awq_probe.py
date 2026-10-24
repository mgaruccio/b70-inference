"""Stdlib-only diagnostic tests, NOT an installed-runtime or Flint inference test."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe-flint-xpu-awq.py"
SPEC = importlib.util.spec_from_file_location("flint_awq_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FlintAWQProbeTests(unittest.TestCase):
    def report(self, shapes, zero_points):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = PROBE.report_conversion(shapes, zero_points)
        return code, output.getvalue()

    def test_pinned_failure_metadata(self):
        code, output = self.report(((2, 16, 512), (2, 16, 256)), (False, False))
        self.assertEqual(code, 2)
        self.assertIn("INCOMPATIBLE", output)
        self.assertIn("((2, 128, 64), (2, 128, 32))", output)
        self.assertIn("packing was not converted", output)
        self.assertIn("zero points were omitted", output)
        self.assertIn("w13=5, w2=7", output)

    def test_wrong_shapes_alone_are_a_blocker(self):
        code, output = self.report(((2, 16, 512), (2, 16, 256)), (True, True))
        self.assertEqual(code, 2)
        self.assertNotIn("zero points were omitted", output)

    def test_missing_either_zero_point_is_a_blocker(self):
        for present in ((False, True), (True, False)):
            with self.subTest(present=present):
                code, output = self.report(PROBE.NATIVE_SHAPES, present)
                self.assertEqual(code, 2)
                self.assertIn("zero points were omitted", output)
                self.assertNotIn("packing was not converted", output)

    def test_matching_metadata_is_not_compatibility_success(self):
        code, output = self.report(PROBE.NATIVE_SHAPES, (True, True))
        self.assertEqual(code, 3)
        self.assertIn("UNVERIFIED", output)
        self.assertIn("values/kernel/serving were not tested", output)

    def test_import_does_not_load_torch_or_vllm(self):
        result = subprocess.run(
            [sys.executable, "-S", "-c", (
                "import runpy, sys; runpy.run_path(sys.argv[1]); "
                "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
            ), str(SCRIPT)], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("never run a MoE kernel", " ".join(result.stdout.split()))

    def test_probe_exception_is_unverified_not_incompatible(self):
        output = io.StringIO()
        error = io.StringIO()
        with patch.object(PROBE, "probe_runtime", side_effect=ImportError("no runtime")):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                self.assertEqual(PROBE.main([]), 3)
        self.assertIn("UNVERIFIED", output.getvalue())
        self.assertNotIn("INCOMPATIBLE", output.getvalue())
        self.assertIn("ImportError: no runtime", error.getvalue())
        self.assertIn("MoE kernel dispatched: NO", output.getvalue())


if __name__ == "__main__":
    unittest.main()
