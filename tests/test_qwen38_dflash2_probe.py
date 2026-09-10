"""Focused runner checks; real XPU/API validation is a separate required gate."""
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "experiments"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("dflash_probe", SCRIPTS / "qwen38_dflash2_probe.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def argv(self, mode, graph=False, draft_int4=None, **options):
        args = SimpleNamespace(mode=mode, context=32768, graph=graph, audit=False, draft_int4=draft_int4, **options)
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_gid=109)):
            return runner.server_command(args, Path("/tmp/test with spaces"))

    def test_control_is_unmodified_fp16_target(self):
        argv = self.argv("control")
        self.assertNotIn("B70_DFLASH2_BF16=1", argv)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=0", argv)
        self.assertNotIn("--speculative-config", argv[-1])
        self.assertIn("--dtype float16", argv[-1])
        self.assertIn("--enforce-eager", argv[-1])
        self.assertIn("127.0.0.1:8000:8000", argv)
        self.assertNotIn("gdn_boundary.py", argv[-1])
        self.assertFalse(any("patch_xpu_boundary.py" in part for part in argv))

    def test_draft_is_explicit_and_separate(self):
        argv = self.argv("dflash2")
        self.assertIn("B70_DFLASH2_BF16=1", argv)
        self.assertIn("/tmp/test with spaces/patch_dflash2.py:/overlay.py:ro", argv)
        serve = shlex.split(argv[-1].split("exec ", 1)[1])
        config = json.loads(serve[serve.index("--speculative-config") + 1])
        self.assertEqual(config, {"method": "dflash", "model": "/draft",
                                  "kv_cache_dtype": "auto", "num_speculative_tokens": 7})
        self.assertEqual(serve[serve.index("--dtype") + 1], "float16")
        self.assertNotIn("cascade", argv[-1])
        self.assertNotIn("lmhead", argv[-1])
        self.assertNotIn("B70_DFLASH2_INT4=1", argv)
        self.assertIn("/tmp/test with spaces/patch_xpu_boundary.py:/gdn_boundary.py:ro", argv)
        self.assertIn("python /overlay.py; python /gdn_boundary.py; exec ", argv[-1])

    def test_int4_is_only_an_explicit_draft_override(self):
        argv = self.argv("dflash2", draft_int4=Path("/tmp/disposable draft"))
        self.assertIn("/tmp/disposable draft:/draft:ro", argv)
        self.assertIn(f"{runner.TARGET}:/model:ro", argv)
        self.assertIn("B70_DFLASH2_INT4=1", argv)
        serve = shlex.split(argv[-1].split("exec ", 1)[1])
        spec = json.loads(serve[serve.index("--speculative-config") + 1])
        self.assertEqual(spec["quantization"], "gptq")
        self.assertEqual(spec["num_speculative_tokens"], 7)
        self.assertEqual(spec["kv_cache_dtype"], "auto")
        self.assertEqual(serve[serve.index("--dtype") + 1], "float16")
        self.assertEqual(serve[serve.index("--kv-cache-dtype") + 1], "fp8")
        self.assertIn("python /gdn_boundary.py; exec ", argv[-1])

    def test_int4_rejects_control_and_original_weights_before_start(self):
        for mode, path in (("control", "/tmp/quant"), ("dflash2", str(runner.TARGET)),
                           ("dflash2", str(runner.DRAFT))):
            with patch.object(sys, "argv", ["probe", "--mode", mode, "--out", "/tmp/unused",
                                           "--prefill-patch", "/tmp/unused", "--draft-int4", path]), \
                    patch.object(runner, "DFlashCell") as cell:
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                cell.assert_not_called()

    def test_cache_grouping_is_opt_in_and_does_not_change_dtypes(self):
        baseline = self.argv("dflash2")
        self.assertNotIn("patch_cache_groups.py", baseline[-1])
        self.assertIn("--max-num-batched-tokens 8192", baseline[-1])
        for budget in (2048, 4096, 8192):
            argv = self.argv("dflash2", cache_group_size=8, max_num_batched_tokens=budget)
            self.assertIn("python /profile/patch_cache_groups.py; exec ", argv[-1])
            serve = shlex.split(argv[-1].split("exec ", 1)[1])
            self.assertEqual(serve[serve.index("--max-num-batched-tokens") + 1], str(budget))
            self.assertEqual(serve[serve.index("--kv-cache-dtype") + 1], "fp8")
            self.assertIn("--no-enable-prefix-caching", serve)
            self.assertEqual(json.loads(serve[serve.index("--speculative-config") + 1])["kv_cache_dtype"], "auto")

    def test_unsupported_cache_and_prefill_overrides_are_rejected(self):
        for mode, options in (("control", {"cache_group_size": 8}),
                              ("dflash2", {"cache_group_size": 5}),
                              ("dflash2", {"max_num_batched_tokens": 0})):
            with self.subTest(mode=mode, options=options), self.assertRaises(ValueError):
                self.argv(mode, **options)

    def test_graph_is_opt_in(self):
        argv = self.argv("dflash2", graph=True)
        self.assertIn("--compilation-config", argv[-1])
        self.assertNotIn("--enforce-eager", argv[-1])

    def test_acceptance_uses_measured_requests(self):
        cell = object.__new__(runner.DFlashCell)
        cell.args = SimpleNamespace(mode="dflash2")
        cell.summary = {}
        def row(label, steps, accepted, drafted):
            return {"label": label, "metric_deltas": {
                f'vllm:spec_decode_num_{k}_total{{model_name="qwen38"}}': v
                for k, v in (("drafts", steps), ("accepted_tokens", accepted), ("draft_tokens", drafted))}}
        cell.rows = [row("code-warmup", 100, 100, 700), row("code-0", 2, 5, 14), row("prose-0", 2, 3, 14)]
        cell.check_acceptance()
        self.assertEqual(cell.summary["acceptance"]["mean_emitted_per_step"], 3)
        self.assertEqual(cell.summary["acceptance"]["steps"], 4)
        cell.rows = [row("code-0", 2, 0, 14)]
        with self.assertRaisesRegex(RuntimeError, "nonzero"):
            cell.check_acceptance()


if __name__ == "__main__":
    unittest.main()
