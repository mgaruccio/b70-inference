"""Source-hook regressions; real serving verification is separate."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experimental/patch_glimmer_draft_head.py"
spec = importlib.util.spec_from_file_location("draft_head_patch", SCRIPT)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

LOAD = "def load_dflash_model(target_model, vllm_config):\n    dflash_model = target_model\n" + patch.LOAD_OLD
SAMPLE = (
    "import torch\nclass DraftModelSpeculator:\n"
    + patch.SAMPLE_OLD
    + "            return self.model.get_top_tokens(hidden_states)\n"
    "        logits = self.model.compute_logits(hidden_states)\n"
    "        return logits.argmax(dim=-1)\n"
)


class DraftHeadPatchTests(unittest.TestCase):
    def setup_sources(self, root, sample=SAMPLE):
        for relative, source in ((patch.LOAD_PATH, LOAD), (patch.SAMPLE_PATH, sample)):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def test_two_hooks_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.setup_sources(root)
            self.assertEqual(patch.patch_sources(root), [patch.LOAD_PATH, patch.SAMPLE_PATH])
            self.assertEqual(patch.patch_sources(root), [])
            self.assertEqual((root / patch.LOAD_PATH).read_text(), LOAD.replace(patch.LOAD_OLD, patch.LOAD_NEW))
            self.assertEqual((root / patch.SAMPLE_PATH).read_text(), SAMPLE.replace(patch.SAMPLE_OLD, patch.SAMPLE_NEW))

    def test_second_anchor_failure_does_not_write_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.setup_sources(root, SAMPLE.replace("_greedy_sample_draft", "unsupported"))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                patch.patch_sources(root)
            self.assertEqual((root / patch.LOAD_PATH).read_text(), LOAD)

    def test_duplicate_anchor_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.setup_sources(root, SAMPLE + SAMPLE)
            with self.assertRaises(ValueError):
                patch.patch_sources(root)
            self.assertEqual((root / patch.LOAD_PATH).read_text(), LOAD)

    def test_syntax_failure_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.setup_sources(root, SAMPLE + "broken = (\n")
            with self.assertRaises(SyntaxError):
                patch.patch_sources(root)
            self.assertEqual((root / patch.LOAD_PATH).read_text(), LOAD)


if __name__ == "__main__":
    unittest.main()
