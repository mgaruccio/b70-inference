#!/usr/bin/env python3
"""CPU-only tests for the target-boundary control shim."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "scripts" / "experiments"
RESULT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CANONICAL_DIR))

import qwen38_step_timing_overlay as canonical  # noqa: E402


shim_spec = importlib.util.spec_from_file_location(
    "target_boundary_control", RESULT_DIR / "target-boundary-control.py"
)
if shim_spec is None or shim_spec.loader is None:
    raise RuntimeError("cannot load target-boundary-control.py")
shim = importlib.util.module_from_spec(shim_spec)
sys.modules["target_boundary_control"] = shim
shim_spec.loader.exec_module(shim)


TARGET_MODULE = "vllm.v1.worker.gpu_model_runner"
DRAFT_MODULE = "vllm.v1.spec_decode.llm_base_proposer"


class FakeEvent:
    clock = 0
    instances: list["FakeEvent"] = []

    def __init__(self, *, enable_timing: bool = False):
        self.enable_timing = enable_timing
        self.recorded: int | None = None
        self.elapsed_calls = 0
        type(self).instances.append(self)

    def record(self) -> None:
        type(self).clock += 1
        self.recorded = type(self).clock

    def elapsed_time(self, other: "FakeEvent") -> float:
        if not self.enable_timing or not other.enable_timing:
            raise AssertionError("false events must never be elapsed")
        if self.recorded is None or other.recorded is None:
            raise AssertionError("elapsed_time called before both records")
        self.elapsed_calls += 1
        return float(other.recorded - self.recorded)


class FakeXpu:
    Event = FakeEvent

    def __init__(self) -> None:
        self.capturing = False
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def is_current_stream_capturing(self) -> bool:
        return self.capturing


class FakeGraph:
    replays = 0
    should_raise: BaseException | None = None

    def replay(self) -> object:
        type(self).replays += 1
        if type(self).should_raise is not None:
            raise type(self).should_raise
        return {"replay": type(self).replays}


ORIGINAL_GRAPH_REPLAY = FakeGraph.replay


class FakeTorch:
    def __init__(self) -> None:
        self.xpu = FakeXpu()
        self.xpu.XPUGraph = FakeGraph

class CudaGraphManager:
    def __init__(self, graph: FakeGraph):
        self.graph = graph

    def run_fullgraph(self) -> object:
        return self.graph.replay()


class ModelCudaGraphManager:
    def __init__(self, graph: FakeGraph):
        self.graph = graph

    def run_fullgraph(self) -> object:
        return self.graph.replay()


class Holder:
    def __init__(self, manager: object):
        self.manager = manager
        self.model = lambda _graph: self.manager.run_fullgraph()  # type: ignore[attr-defined]


def _compiled_functions(module_name: str) -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": module_name}
    exec(
        '''
def _model_forward(self):
    return self.model(None)

def execute_model(self):
    return self._model_forward()

def ambiguous(self):
    return self.model(None)

def propose(self):
    return self.model(None)

def propose_draft_token_ids(self):
    return self.model(None)

        ''',
        namespace,
    )
    return namespace


_TARGET_FUNCTIONS = _compiled_functions(TARGET_MODULE)
_DRAFT_FUNCTIONS = _compiled_functions(DRAFT_MODULE)


def target_holder(manager: object) -> Holder:
    holder = Holder(manager)
    holder._model_forward = types.MethodType(_TARGET_FUNCTIONS["_model_forward"], holder)  # type: ignore[attr-defined]
    holder.execute_model = types.MethodType(_TARGET_FUNCTIONS["execute_model"], holder)  # type: ignore[attr-defined]
    return holder


def ambiguous_holder(manager: object) -> Holder:
    holder = Holder(manager)
    holder.ambiguous = types.MethodType(_TARGET_FUNCTIONS["ambiguous"], holder)  # type: ignore[attr-defined]
    return holder


def draft_holder(manager: object) -> Holder:
    holder = Holder(manager)
    holder.propose = types.MethodType(_DRAFT_FUNCTIONS["propose"], holder)  # type: ignore[attr-defined]
    return holder


def draft_anchor_holder(manager: object) -> Holder:
    holder = Holder(manager)
    holder.propose_draft_token_ids = types.MethodType(  # type: ignore[attr-defined]
        _TARGET_FUNCTIONS["propose_draft_token_ids"], holder
    )
    return holder


class TargetBoundaryControlTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeEvent.clock = 0
        FakeEvent.instances = []
        FakeGraph.replays = 0
        FakeGraph.should_raise = None

    def make_overlay(self, directory: Path, *, max_samples: int = 512):
        torch = FakeTorch()
        overlay = canonical.TimingOverlay(
            torch,
            max_samples=max_samples,
            output_dir=directory,
        )
        overlay.install()
        control = shim.install(overlay)
        # unittest cleanups run LIFO: remove the shim first, then let the
        # canonical overlay restore its own replay hook.
        self.addCleanup(overlay.close)
        self.addCleanup(control.close)
        return overlay, control, torch

    def stop_json(self, overlay: object) -> dict[str, object]:
        path = overlay.stop(reason="fixture")  # type: ignore[attr-defined]
        self.assertIsNotNone(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_install_active_finds_the_canonical_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            overlay, control, _torch = self.make_overlay(Path(tmp))
            module_name = "canonical_fixture_module"
            fake_module = types.SimpleNamespace(active_overlay=lambda: overlay)
            sys.modules[module_name] = fake_module
            self.addCleanup(sys.modules.pop, module_name, None)
            self.assertIs(shim.install_active(module_name), control)
            self.assertIs(control.overlay, overlay)

    def test_target_pair_is_inner_to_canonical_pair_and_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            overlay, control, torch = self.make_overlay(Path(tmp))
            graph = FakeGraph()
            runner = target_holder(CudaGraphManager(graph))
            overlay.start(kind="fixture", profile_prefix="target")

            result = runner.execute_model()
            self.assertEqual(result, {"replay": 1})
            self.assertEqual(FakeGraph.replays, 1)
            self.assertEqual(torch.xpu.synchronize_calls, 0)
            self.assertEqual(control.pending_event_references, 2)
            self.assertEqual(
                [event.enable_timing for event in FakeEvent.instances],
                [False, False, True, True],
            )
            # Keep an extra harness stage value to verify that the shim only
            # adds metadata and does not rewrite canonical event fields.
            overlay._events[0]["provenance"]["stage"] = "future-harness-stage"  # type: ignore[attr-defined]


            data = self.stop_json(overlay)
            self.assertEqual(torch.xpu.synchronize_calls, 1)
            self.assertEqual(control.pending_event_references, 0)
            self.assertEqual(len(data["events"]), 1)  # type: ignore[arg-type]
            event = data["events"][0]  # type: ignore[index]
            control_data = event[shim.CONTROL_KEY]  # type: ignore[index]
            self.assertEqual(control_data["classification"], "target")
            self.assertTrue(control_data["boundary_recorded"])
            self.assertEqual(control_data["replay_index"], event["index"])
            self.assertEqual(event["stage"], "future-harness-stage")
            self.assertEqual(control_data["graph_id"], event["graph_id"])
            self.assertEqual(control_data["pid"], os.getpid())
            self.assertIsInstance(control_data["native_thread_id"], int)
            self.assertIsInstance(
                control_data["start_record"]["before_monotonic_raw_ns"], int
            )
            self.assertIsInstance(
                control_data["end_record"]["after_monotonic_raw_ns"], int
            )
            self.assertIsNone(control_data["boundary_duration_ms"])
            self.assertEqual(
                sum(event.elapsed_calls for event in FakeEvent.instances if not event.enable_timing),
                0,
            )
            self.assertEqual(
                sum(event.elapsed_calls for event in FakeEvent.instances if event.enable_timing),
                1,
            )
            self.assertEqual(event["duration_ms"], 1.0)
            self.assertNotIn("start_event", json.dumps(data))
            self.assertNotIn("FakeEvent", json.dumps(data))

    def test_only_two_target_markers_skip_warmup_inactive_capture_draft_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            overlay, control, torch = self.make_overlay(Path(tmp))
            graph = FakeGraph()
            target = target_holder(CudaGraphManager(graph))
            draft = draft_holder(ModelCudaGraphManager(graph))
            draft_anchor = draft_anchor_holder(CudaGraphManager(graph))
            ambiguous = ambiguous_holder(CudaGraphManager(graph))

            # Warmup is before the canonical active window.
            target.execute_model()
            overlay.start(kind="fixture", profile_prefix="roles")

            # Capture is passed through by both layers and gets no marker.
            torch.xpu.capturing = True
            target.execute_model()
            torch.xpu.capturing = False

            # Both explicit draft anchors and an incomplete target context are
            # observed as unknown, never target.
            draft.propose()
            draft_anchor.propose_draft_token_ids()
            ambiguous.ambiguous()
            target.execute_model()
            target.execute_model()

            data = self.stop_json(overlay)
            self.assertEqual(torch.xpu.synchronize_calls, 1)
            self.assertEqual(control.pending_event_references, 0)
            events = data["events"]  # type: ignore[index]
            self.assertEqual(len(events), 5)  # draft, draft-anchor, ambiguous, 2 target
            controls = [event[shim.CONTROL_KEY] for event in events]
            self.assertEqual(
                sum(item["boundary_recorded"] for item in controls),
                2,
            )
            self.assertEqual(
                [item["classification"] for item in controls],
                ["unknown", "unknown", "unknown", "target", "target"],
            )
            self.assertIn("draft anchor rejected", controls[0]["classification_reason"])
            self.assertIn("draft anchor rejected", controls[1]["classification_reason"])
            self.assertFalse(controls[2]["boundary_recorded"])

            # An inactive replay after stop remains an ordinary graph replay and
            # cannot append a marker to the already-flushed canonical result.
            target.execute_model()
            self.assertEqual(control.pending_boundary_count, 0)

    def test_original_exception_and_result_are_preserved_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            overlay, control, _torch = self.make_overlay(Path(tmp))
            graph = FakeGraph()
            runner = target_holder(CudaGraphManager(graph))
            expected = RuntimeError("native graph failure")
            FakeGraph.should_raise = expected
            overlay.start(kind="fixture", profile_prefix="exception")

            with self.assertRaises(RuntimeError) as raised:
                runner.execute_model()
            self.assertIs(raised.exception, expected)
            self.assertEqual(FakeGraph.replays, 1)
            self.assertEqual(control.pending_event_references, 2)

            data = self.stop_json(overlay)
            self.assertEqual(len(data["events"]), 1)  # type: ignore[arg-type]
            self.assertTrue(data["events"][0][shim.CONTROL_KEY]["boundary_recorded"])  # type: ignore[index]

    def test_no_false_marker_after_canonical_512_sample_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            overlay, control, torch = self.make_overlay(Path(tmp), max_samples=512)
            graph = FakeGraph()
            runner = target_holder(CudaGraphManager(graph))
            overlay.start(kind="fixture", profile_prefix="bound")

            for _ in range(512):
                runner.execute_model()
            self.assertEqual(control.pending_event_references, 1024)
            runner.execute_model()
            self.assertEqual(FakeGraph.replays, 513)
            self.assertEqual(control.pending_boundary_count, 512)
            self.assertEqual(
                sum(event.enable_timing is False for event in FakeEvent.instances),
                1024,
            )
            self.assertEqual(torch.xpu.synchronize_calls, 0)

            data = self.stop_json(overlay)
            self.assertEqual(torch.xpu.synchronize_calls, 1)
            self.assertEqual(len(data["events"]), 512)  # type: ignore[arg-type]
            self.assertEqual(
                sum(event[shim.CONTROL_KEY]["boundary_recorded"] for event in data["events"]),  # type: ignore[index]
                512,
            )

    def test_two_overlay_cycles_restore_shared_graph_hook(self) -> None:
        original = ORIGINAL_GRAPH_REPLAY
        for cycle in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                torch = FakeTorch()
                overlay = canonical.TimingOverlay(torch, output_dir=Path(tmp))
                control: shim.TargetBoundaryControl | None = None
                try:
                    overlay.install()
                    self.assertIsNot(FakeGraph.replay, original)
                    control = shim.install(overlay)
                    overlay.start(kind="cycle", profile_prefix=f"cycle-{cycle}")

                    # The shim owns only its outer wrapper.  Removing it must
                    # leave the canonical wrapper for overlay.close() to remove.
                    control.close()
                    self.assertIs(FakeGraph.replay, control.previous_replay)
                finally:
                    if control is not None:
                        control.close()
                    overlay.close()
                self.assertIs(FakeGraph.replay, original)

    def test_install_before_canonical_is_rejected(self) -> None:
        self.assertIs(FakeGraph.replay, ORIGINAL_GRAPH_REPLAY)
        torch = FakeTorch()
        overlay = canonical.TimingOverlay(torch)
        try:
            with self.assertRaisesRegex(RuntimeError, "after canonical TimingOverlay.install"):
                shim.install(overlay)
        finally:
            overlay.close()
        self.assertIs(FakeGraph.replay, ORIGINAL_GRAPH_REPLAY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
