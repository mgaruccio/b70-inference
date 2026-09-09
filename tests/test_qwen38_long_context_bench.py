"""Focused protocol tests for the cold Qwen long-context benchmark cell."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen38_long_context_bench", ROOT / "scripts/experiments/qwen38_long_context_bench.py"
)
assert SPEC and SPEC.loader
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class LongContextProtocolTests(unittest.TestCase):
    def test_rendered_prompt_is_exact_and_preserves_actual_template_framing(self):
        template = [101, 102, BENCH.IM_END_ID, 103, 104]
        prompt = BENCH.compose_prompt(
            template,
            prefix_tokens=[201, 202],
            body_tokens=[301, 302],
            tail_tokens=[401, 402, 403],
            requested_length=13,
        )
        self.assertEqual(len(prompt), 13)
        self.assertEqual(prompt[:2], template[:2])
        self.assertEqual(prompt[-3:], template[-3:])
        self.assertEqual(prompt[2:-3], [201, 202, 301, 302, 301, 401, 402, 403])

    def test_near_limit_selection_leaves_output_room_and_uses_190k_for_large_config(self):
        lengths, near = BENCH.select_lengths([512, 32768], 32768)
        self.assertEqual(near, 32640)
        self.assertEqual(lengths, [512, 32640, 32768])
        lengths, near = BENCH.select_lengths([512], 212992)
        self.assertEqual(near, 190000)
        self.assertEqual(lengths, [512, 190000])

    def test_quantiles_are_inclusive(self):
        median, q1, q3 = BENCH.inclusive_quartiles([1, 2, 3, 4, 5, 6])
        self.assertEqual((median, q1, q3), (3.5, 2.25, 4.75))
        self.assertEqual(BENCH.inclusive_iqr([1, 2, 3, 4, 5, 6]), 2.5)

    def test_summary_excludes_warmup_and_requires_six_valid_measurements(self):
        def row(kind, value, valid=True):
            return {
                "kind": kind,
                "valid": valid,
                "total_time_s": value,
                "ttft_s": value,
                "decode_elapsed_s": value,
                "decode_tps_post_first": value,
                "e2e_tps": value,
                "prefill_proxy_input_tps": value,
            }

        rows = [row("warmup", 1000)] + [row("measured", value) for value in range(1, 7)]
        summary = BENCH.summarize_measurements(rows)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["warmups_excluded"])
        self.assertEqual(summary["valid_count"], 6)
        self.assertEqual(summary["median"]["e2e_tps"], 3.5)
        self.assertEqual(summary["iqr_inclusive"]["e2e_tps"], 2.5)

        incomplete = BENCH.summarize_measurements(rows[:-1])
        self.assertFalse(incomplete["complete"])
        self.assertEqual(incomplete["median"], {})

    def test_forced_prompt_and_output_count_validation(self):
        self.assertTrue(BENCH.validate_token_counts({"prompt_tokens": 32640, "completion_tokens": 128}, 32640)["valid"])
        invalid = BENCH.validate_token_counts({"prompt_tokens": 32640, "completion_tokens": 127}, 32640)
        self.assertFalse(invalid["valid"])
        self.assertIn("completion_tokens", " ".join(invalid["issues"]))

    def test_capacity_classification_distinguishes_supported_and_unsupported(self):
        self.assertTrue(BENCH.classify_length(32640, 32768)["supported"])
        standard = BENCH.classify_length(32768, 32768)
        self.assertFalse(standard["supported"])
        self.assertEqual(standard["reason"], "prompt_plus_output_exceeds_max_model_len")
        self.assertFalse(BENCH.classify_length(32769, 32768)["supported"])
        self.assertTrue(BENCH.classify_length(190000, 212992)["supported"])

    def test_sse_burst_rate_uses_n_minus_one_after_first_chunk(self):
        events = [
            (10.0, b'data: {"choices":[{"text":"first"}]}\n'),
            # This chunk represents a speculative burst; it is not per-token ITL.
            (12.0, b'data: {"choices":[{"text":"burst"}]}\n'),
            (13.0, b'data: {"choices":[],"usage":{"prompt_tokens":512,"completion_tokens":6}}\n'),
            (14.0, b"data: [DONE]\n"),
        ]
        parsed = BENCH.parse_sse_events(events)
        metrics = BENCH.compute_stream_metrics(parsed, request_started_monotonic=9.0,
                                               stream_end_monotonic=14.0, prompt_tokens=512)
        self.assertEqual(parsed["nonempty_chunk_count"], 2)
        self.assertEqual(metrics["ttft_s"], 1.0)
        self.assertEqual(metrics["decode_elapsed_s"], 4.0)
        self.assertEqual(metrics["decode_tps_post_first"], 1.25)  # (6 - 1) / (14 - 10)
        self.assertEqual(metrics["e2e_tps"], 1.2)
        self.assertIn("not per-token ITL", metrics["rate_formula"])

    def test_nonce_is_deterministic_and_varies_by_point_and_trial(self):
        first = BENCH.deterministic_nonce(8192, 1)
        self.assertEqual(first, BENCH.deterministic_nonce(8192, 1))
        self.assertNotEqual(first, BENCH.deterministic_nonce(8192, 2))
        self.assertNotEqual(first, BENCH.deterministic_nonce(16384, 1))


if __name__ == "__main__":
    unittest.main()
