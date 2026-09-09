"""Supplemental checks; the real benchmark clients still require B70/API execution."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import qwen38_mtp_reference as mtp
import qwen38_standard_bench as bench


class StandardBenchmarkTests(unittest.TestCase):
    def test_reference_is_cold_localhost_and_original_unchanged(self):
        original = (ROOT / "scripts" / "start-qwen38.sh").read_bytes()
        text = mtp.launcher_text(original, Path("/tmp/reference"))
        self.assertIn('-p "127.0.0.1:8000:8000"', text)
        self.assertNotIn("0.0.0.0", text)
        self.assertIn("--no-enable-prefix-caching", text)
        self.assertNotIn(" --enable-prefix-caching", text)
        self.assertIn(r'enable_thinking\":false', text)
        self.assertIn("--max-model-len 212992", text)
        self.assertIn("--max-num-seqs 1", text)
        self.assertIn(mtp.IMAGE, text)
        self.assertIn("python /prefill_guard.py; exec vllm serve", text)
        self.assertEqual(original, (ROOT / "scripts" / "start-qwen38.sh").read_bytes())
        subprocess.run(["bash", "-n"], input=text, text=True, check=True)

    def test_reference_rejects_changed_launcher(self):
        original = (ROOT / "scripts" / "start-qwen38.sh").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "launcher changed"):
            mtp.launcher_text(original + b"\n", Path("/tmp/reference"))

    def test_reference_quotes_output_path(self):
        original = (ROOT / "scripts" / "start-qwen38.sh").read_bytes()
        out = Path("/tmp/with space;$(not-a-command)")
        text = mtp.launcher_text(original, out)
        self.assertIn("COOKBOOK='/tmp/with space;$(not-a-command)/reference-source'", text)
        self.assertIn("-v '/tmp/with space;$(not-a-command):/profile'", text)
        subprocess.run(["bash", "-n"], input=text, text=True, check=True)

    def test_metrics_snapshot_preserves_raw_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "spec-metrics").mkdir()
            raw = '# TYPE counter counter\nvllm:spec_decode_num_drafts_total{model="qwen38"} 4\n'
            with patch.object(bench.dflash.probe, "get", return_value=raw):
                bench.snapshot(out, "before")
            self.assertEqual((out / "spec-metrics/before.prom").read_text(), raw)

    def test_command_output_and_argv_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "console.txt"
            bench.run_logged([sys.executable, "-c", "print('observed')"], dest)
            self.assertEqual(dest.read_text(), "observed\n")
            self.assertIn("print", dest.with_suffix(".command.txt").read_text())

    def test_failed_commands_are_not_successful(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "failure.txt"
            with self.assertRaisesRegex(RuntimeError, "command failed"):
                bench.run_logged([sys.executable, "-c", "raise SystemExit(7)"], dest)
            self.assertTrue(dest.exists())
            self.assertTrue(dest.with_suffix(".command.txt").exists())

    def test_mtp_cannot_take_dflash_checkpoint(self):
        argv = ["benchmark", "--out", "/unused", "--betterbench", "/unused",
                "--reference-mtp", "--draft-int4", "/unused", "--guard", "/unused"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                bench.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
