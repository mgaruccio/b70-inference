"""Dispatch/replay checks; live short-prefill correctness is tested on B70."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/patch-vllm-qwen38-xpu-prefill.py"
spec = importlib.util.spec_from_file_location("xpu_prefill", PATH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
SOURCE = patch.IMPORT + "class Builder:\n    def build(self, m):\n        result = (\n" + patch.OLD + "        )\n        return result\n"


class PrefillTests(unittest.TestCase):
    def test_replay_and_syntax(self):
        result = patch.patch_text(SOURCE)
        ast.parse(result)
        self.assertEqual(patch.patch_text(result), result)
        self.assertEqual(result.count(patch.MARKER), 1)

    def test_reject_changed_or_partial_patch(self):
        for source in (SOURCE.replace("decode_threshold=1", "decode_threshold=2"),
                       SOURCE + "# " + patch.MARKER,
                       SOURCE + patch.OLD):
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                patch.patch_text(source)

    def test_xpu_real_metadata_uses_existing_prefill_aware_helper(self):
        tree = ast.parse(patch.patch_text(SOURCE))
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "split_decodes_and_prefills")
        flag = next(k.value for k in call.keywords if k.arg == "treat_short_extends_as_decodes")
        expression = compile(ast.Expression(flag), "patched-dispatch", "eval")
        for xpu, state, expected in ((True, [True], False), (True, [False], False),
                                      (True, None, True), (False, [True], True)):
            with self.subTest(xpu=xpu, state=state):
                got = eval(expression, {"current_platform": SimpleNamespace(is_xpu=lambda: xpu),
                                        "m": SimpleNamespace(is_prefilling=state)})
                self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
