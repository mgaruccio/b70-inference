"""The concurrency client must recognize current and legacy reasoning deltas."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


class ReasoningFieldsTest(unittest.TestCase):
    def test_current_and_legacy_fields(self):
        path = Path(__file__).resolve().parents[1] / "scripts/glimmer-phase0-instrument.py"
        spec = importlib.util.spec_from_file_location("glimmer_instrument", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for field in ("reasoning", "reasoning_content"):
            with self.subTest(field=field):
                events = [
                    {"choices": [{"delta": {field: "Thinking."}}]},
                    {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 2}},
                ]
                wire = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                wire += "data: [DONE]\n\n"
                with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(wire.encode())):
                    result = module.stream_once(
                        base="http://test/v1", model="muse", idx=0, prompt="test",
                        max_tokens=4, timeout=1, seed=0, reasoning_strength=None,
                    )
                self.assertEqual(result["chunks"], 2)
                self.assertEqual(result["completion_tokens"], 3)
                self.assertIsNotNone(result["ttft_reasoning_s"])
                self.assertIsNotNone(result["decode_tok_s"])


if __name__ == "__main__":
    unittest.main()
