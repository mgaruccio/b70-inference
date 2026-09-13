#!/usr/bin/env python3
"""CPU-only fixture checks for the bounded MTP4 attribution scripts."""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


annotations = load_script("mtp4_draft_annotations_fixture", "draft-annotations.py")
patch = load_script("mtp4_draft_patch_fixture", "draft-patch.py")
summary = load_script("mtp4_draft_summary_fixture", "summarize-draft.py")


class DraftAttributionFixtureTest(unittest.TestCase):
    def test_conservative_dispatch_and_stage_labels(self):
        draft_stack = [
            {
                "module": "vllm.v1.spec_decode.llm_base_proposer",
                "function": "_determine_batch_execution_and_padding",
                "class": "SpecDecodeBaseProposer",
            }
        ]
        target_stack = [
            {
                "module": "vllm.v1.worker.gpu_model_runner",
                "function": "_get_cudagraph_mode",
                "class": "XPUModelRunner",
            }
        ]
        self.assertEqual(annotations.classify_dispatch_call(draft_stack), "draft")
        self.assertEqual(annotations.classify_dispatch_call(target_stack), "target")
        self.assertEqual(annotations.classify_token_stage(5, 8), "first_five_tokens")
        self.assertEqual(annotations.classify_token_stage(1, 1), "later_one_token")
        self.assertEqual(annotations.classify_token_stage(3, 4), "other")

    def test_draft_span_events_resolve_only_at_stop(self):
        class Event:
            def __init__(self, enable_timing=True):
                self.enable_timing = enable_timing
                self.recorded = False

            def record(self):
                self.recorded = True

            def elapsed_time(self, other):
                self.assert_not_used = other
                return 1.25

        class XPU:

            def __init__(self):
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

        xpu = XPU()
        xpu.Event = Event
        class Torch:
            pass
        torch = Torch()
        torch.xpu = xpu
        session = annotations.AttributionSession(max_scopes=2, max_dispatches=2)
        session.torch = torch
        pending = session._start_draft_span("propose", 0)
        self.assertIsNotNone(pending)
        session._finish_draft_span(pending)
        self.assertEqual(xpu.synchronize_calls, 0)
        session._resolve_draft_spans()
        self.assertEqual(xpu.synchronize_calls, 1)
        self.assertEqual(session.draft_spans[0]["duration_ms"], 1.25)

    def test_campaign_patch_is_compiled_and_idempotent(self):
        source = "import os\n\nimport torch\n\nclass XPUWorker:\n    pass\n"
        patched = patch.patch_text(source)
        compile(patched, "fixture-xpu-worker.py", "exec")
        self.assertEqual(patched.count(patch.MARKER), 1)
        self.assertEqual(patch.patch_text(patched), patched)
        with self.assertRaises(RuntimeError):
            patch.patch_text(patched + patch.DRAFT_IMPORT_BLOCK)

    def test_external_id_mapping_is_exclusive_and_graph_metadata_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            profile = run / "profile"
            profile.mkdir(parents=True)
            annotations_dir = run / "draft-attribution"
            annotations_dir.mkdir()
            events = [
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "b70_draft/phase:propose",
                    "pid": 10,
                    "tid": 20,
                    "ts": 0,
                    "dur": 100,
                },
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "b70_draft/phase:mtp_forward",
                    "pid": 10,
                    "tid": 20,
                    "ts": 10,
                    "dur": 40,
                },
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::mm",
                    "pid": 10,
                    "tid": 20,
                    "ts": 15,
                    "dur": 5,
                    "args": {"External id": 17, "Input Dims": [[1, 2]]},
                },
                {
                    "ph": "kernel",
                    "cat": "kernel",
                    "name": "mm_kernel",
                    "pid": 99,
                    "tid": 100,
                    "ts": 20,
                    "dur": 4,
                    "args": {"External id": 17, "correlation": 1},
                },
                {
                    "ph": "kernel",
                    "cat": "kernel",
                    "name": "unmapped_kernel",
                    "pid": 99,
                    "tid": 100,
                    "ts": 21,
                    "dur": 2,
                    "args": {"External id": 999, "correlation": 2},
                },
            ]
            trace = profile / "rank0.fixture.pt.trace.json.gz"
            with gzip.open(trace, "wt", encoding="utf-8") as output:
                json.dump({"traceEvents": events}, output)
            annotation = {
                "format": "b70-mtp4-draft-attribution-v1",
                "errors": [],
                "dispatches": [
                    {
                        "index": 0,
                        "role": "target",
                        "requested_num_tokens": 5,
                        "returned_runtime_mode": {"name": "FULL", "value": 1},
                        "returned_batch_descriptor": {"num_tokens": 8},
                    }
                ],
                "draft_spans": [
                    {
                        "index": 0,
                        "phase": "propose",
                        "duration_ms": 2.25,
                    }
                ],
                "graph_replays": [
                    {
                        "index": 0,
                        "role": "target",
                        "caller_stack": [
                            {
                                "module": "vllm.v1.worker.gpu_model_runner",
                                "function": "_model_forward",
                                "class": "XPUModelRunner",
                            }
                        ],
                        "duration_ms": 3.5,
                        "campaign_graph_context": {
                            "runtime_mode": {"name": "FULL", "value": 1},
                            "batch_descriptor": {"num_tokens": 8},
                        },
                    }
                ],
            }
            (annotations_dir / "draft-attribution-rank0-session1.json").write_text(
                json.dumps(annotation), encoding="utf-8"
            )
            result = summary.summarize(run)
            eager = result["eager_trace"]
            self.assertEqual(eager["draft_generation_steps"], 1)
            self.assertEqual(eager["draft_kernel_ms"], 0.004)
            self.assertEqual(eager["unmapped_cpu_external_id"]["count"], 1)
            graph = result["graph_replay_attribution"]["events"][0]
            self.assertEqual(graph["stage"], "first_five_tokens")
            self.assertEqual(graph["dispatch_match_index"], 0)
            self.assertEqual(graph["duration_ms"], 3.5)
            self.assertTrue(result["graph_replay_attribution"]["finite_metrics"])
            spans = result["draft_span_attribution"]
            self.assertTrue(spans["observed"])
            self.assertEqual(spans["event_count"], 1)
            self.assertEqual(spans["total_ms"], 2.25)
            self.assertTrue(spans["finite_metrics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
