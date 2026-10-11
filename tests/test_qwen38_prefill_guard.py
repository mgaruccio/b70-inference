"""CPU-only tests for the maintained legacy prefill-classification guard."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "scripts/patch-vllm-qwen38-prefill-guard.py"
SPEC = importlib.util.spec_from_file_location("qwen38_prefill_guard", PATCH_PATH)
assert SPEC is not None and SPEC.loader is not None
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)


SOURCE = (
    "from __future__ import annotations\n\n"
    "class GPUModelRunner:\n"
    + patch.OLD_CLASSIFIER
    + "    def invoke(self):\n"
    + "        return self._is_uniform_decode(\n"
    + "            max_num_scheduled_tokens=5,\n"
    + "            uniform_decode_query_len=5,\n"
    + patch.OLD_CALL
    + "        )\n"
)


def _run_fake_vllm(root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(root), environment.get("PYTHONPATH", "")) if item
    )
    return subprocess.run(
        [sys.executable, str(PATCH_PATH)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class PrefillGuardTests(unittest.TestCase):
    def test_anchored_patch_compiles_and_replays_exactly(self) -> None:
        patched = patch.patch_text(SOURCE)
        ast.parse(patched)
        self.assertEqual(patched.count(patch.MARKER), 1)
        self.assertEqual(patched.count(patch.NEW_CLASSIFIER), 1)
        self.assertEqual(patched.count(patch.NEW_CALL), 1)
        self.assertNotIn(patch.OLD_CLASSIFIER, patched)
        self.assertNotIn(patch.OLD_CALL, patched)
        self.assertEqual(patch.patch_text(patched), patched)

    def test_anchor_drift_and_partial_marker_fail_closed(self) -> None:
        broken_classifier = SOURCE.replace(
            "force_uniform_decode: bool | None = None",
            "force_uniform_decode: bool = False",
            1,
        )
        broken_call = SOURCE.replace(
            "num_reqs=num_reqs,", "num_reqs=num_reqs + 1,", 1
        )
        partial_marker = SOURCE + f"\n# {patch.MARKER}\n"
        for broken in (broken_classifier, broken_call, partial_marker):
            with self.subTest(broken=broken[-80:]), self.assertRaises(RuntimeError):
                patch.patch_text(broken)

        patched = patch.patch_text(SOURCE)
        changed_after_patch = patched.replace(
            "not has_prefill  # B70_FIX_UNIFORM_DECODE_PREFILL",
            "has_prefill  # B70_FIX_UNIFORM_DECODE_PREFILL",
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "partial/changed"):
            patch.patch_text(changed_after_patch)

    def test_classifier_cases_preserve_decode_and_force_override_semantics(self) -> None:
        namespace: dict[str, object] = {}
        exec(compile(patch.patch_text(SOURCE), "fixture.py", "exec"), namespace)
        classifier = namespace["GPUModelRunner"]._is_uniform_decode  # type: ignore[attr-defined]
        cases = (
            ("MTP decode", 5, 5, 10, 2, False, None, True),
            ("no-spec decode", 1, 1, 2, 2, False, None, True),
            ("five-token prefill", 5, 5, 5, 1, True, None, False),
            ("one-token prefill", 1, 1, 1, 1, True, None, False),
            ("chunked prefill", 8, 8, 8, 1, True, None, False),
            ("mixed prefill alias", 5, 5, 10, 2, True, None, False),
            ("nonuniform batch", 5, 4, 9, 2, False, None, False),
            ("forced capture true", 5, 4, 9, 2, True, True, True),
            ("forced capture false", 5, 5, 10, 2, False, False, False),
        )
        for name, max_scheduled, query_len, tokens, requests, has_prefill, force, expected in cases:
            kwargs = {
                "max_num_scheduled_tokens": max_scheduled,
                "uniform_decode_query_len": query_len,
                "num_tokens": tokens,
                "num_reqs": requests,
                "has_prefill": has_prefill,
            }
            if force is not None:
                kwargs["force_uniform_decode"] = force
            with self.subTest(name=name):
                self.assertEqual(classifier(**kwargs), expected)

    def test_cli_applies_replays_and_does_not_write_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "vllm"
            target = package / "v1/worker/gpu_model_runner.py"
            target.parent.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            target.write_text(SOURCE, encoding="utf-8")

            first = _run_fake_vllm(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("patched", first.stdout)
            expected = patch.patch_text(SOURCE)
            self.assertEqual(target.read_text(encoding="utf-8"), expected)

            second = _run_fake_vllm(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already patched", second.stdout)
            self.assertEqual(target.read_text(encoding="utf-8"), expected)

            drifted = SOURCE.replace(
                "num_reqs=num_reqs,", "num_reqs=num_reqs + 1,", 1
            )
            target.write_text(drifted, encoding="utf-8")
            before_failure = target.read_bytes()
            failed = _run_fake_vllm(root)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(target.read_bytes(), before_failure)

            invalid = SOURCE + "\nif broken\n"
            target.write_text(invalid, encoding="utf-8")
            before_compile_failure = target.read_bytes()
            failed_compile = _run_fake_vllm(root)
            self.assertNotEqual(failed_compile.returncode, 0)
            self.assertEqual(target.read_bytes(), before_compile_failure)

    def test_launcher_runs_guard_after_existing_five_patches(self) -> None:
        launcher = (ROOT / "scripts/start-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn(
            '-v "$COOKBOOK/patches/patch_uniform_decode_prefill.py:/patch_uniform_decode_prefill.py:ro"',
            launcher,
        )
        self.assertIn(
            "python /patch_mtp.py; python /patch_boundary.py; python /patch_v5.py; "
            "python /patch_s.py; python /patch_m1.py; "
            "python /patch_uniform_decode_prefill.py; exec vllm serve",
            launcher,
        )
        self.assertIn("--max-num-batched-tokens 8192", launcher)
        self.assertIn("--enable-prefix-caching", launcher)
        self.assertIn("VLLM_TARGET_DEVICE=xpu", launcher)
        self.assertIn("libattn_kernels_xe_2-head256-k64.so", launcher)
        self.assertIn("--speculative-config", launcher)
        self.assertIn("num_speculative_tokens", launcher)


if __name__ == "__main__":
    unittest.main()
