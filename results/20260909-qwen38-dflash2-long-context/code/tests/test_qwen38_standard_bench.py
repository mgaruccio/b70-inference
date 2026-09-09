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

    def test_dflash_context_default_and_explicit_limits(self):
        base = ["benchmark", "--out", "/unused", "--betterbench", "/unused",
                "--guard", "/unused", "--patch", "/unused", "--prefill-patch", "/unused"]
        for extra, expected in (([], 32768), (["--context", "49152"], 49152),
                                (["--context", "65536"], 65536)):
            with self.subTest(context=expected), patch.object(sys, "argv", base + extra), \
                    patch.object(bench.signal, "signal"), \
                    patch.object(bench.dflash, "DFlashCell", side_effect=RuntimeError("stop before host start")) as cell:
                with self.assertRaisesRegex(RuntimeError, "stop before host start"):
                    bench.main()
                self.assertEqual(cell.call_args.args[0].context, expected)
                self.assertEqual(cell.call_args.args[0].mode, "dflash2")

    def test_invalid_context_rejected_before_host_start(self):
        for value in ("511", "212993", "not-an-integer"):
            argv = ["benchmark", "--out", "/unused", "--betterbench", "/unused",
                    "--guard", "/unused", "--context", value]
            with self.subTest(value=value), patch.object(sys, "argv", argv), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main()
                self.assertEqual(error.exception.code, 2)

    def test_mtp_rejects_context_override(self):
        for value in ("49152", "212992"):
            argv = ["benchmark", "--out", "/unused", "--betterbench", "/unused",
                    "--guard", "/unused", "--reference-mtp", "--context", value]
            with self.subTest(value=value), patch.object(sys, "argv", argv), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main()
                self.assertEqual(error.exception.code, 2)

    def test_long_context_keeps_standard_points_and_adds_one_token_over(self):
        for limit in (49152, 65536):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as tmp:
                argv = ["benchmark", "--out", tmp, "--betterbench", "/unused",
                        "--guard", "/unused", "--patch", "/unused", "--prefill-patch", "/unused",
                        "--long-context-only", "--context", str(limit)]
                with patch.object(sys, "argv", argv), patch.object(bench.signal, "signal"), \
                        patch.object(bench.dflash, "DFlashCell") as constructor, \
                        patch.object(bench, "run_logged") as run, contextlib.redirect_stdout(io.StringIO()):
                    cell = constructor.return_value
                    cell.out, cell.summary = Path(tmp), {}
                    bench.main()
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("--lengths") + 1:],
                                     list(map(str, (*bench.cold.DEFAULT_LENGTHS,
                                                    limit - bench.cold.OUTPUT_TOKENS + 1))))
                    self.assertEqual(constructor.call_args.args[0].context, limit)
                    cell.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
