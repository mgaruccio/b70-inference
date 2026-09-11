#!/usr/bin/env python3
"""CPU-only fixture tests for the disposable step-timing overlay."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import qwen38_step_timing_overlay as overlay
import qwen38_step_timing_patch as patch


class FakeEvent:
    clock = 0
    def __init__(self, *, enable_timing: bool = False):
        assert enable_timing is True
        self.recorded = None
    def record(self):
        FakeEvent.clock += 1
        self.recorded = FakeEvent.clock
    def elapsed_time(self, other):
        assert self.recorded is not None and other.recorded is not None
        return float(other.recorded - self.recorded)

class FakeXpu:
    Event = FakeEvent

    def __init__(self):
        self.capturing = False
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1

    def is_current_stream_capturing(self):
        return self.capturing


class FakeGraph:
    replays = 0

    def replay(self):
        type(self).replays += 1
        return type(self).replays


class FakeTorch:
    def __init__(self):
        self.xpu = FakeXpu()
        self.xpu.XPUGraph = FakeGraph


class DFlashCudaGraphManager:
    def __init__(self, graph):
        self.graph = graph

    def run_fullgraph(self):
        return self.graph.replay()


class ModelCudaGraphManager:
    def __init__(self, graph):
        self.graph = graph

    def run_fullgraph(self):
        return self.graph.replay()


class FakeRunner:
    def __init__(self, manager):
        self.manager = manager

    def execute_model(self):
        return self.manager.run_fullgraph()


class FakeWorker:
    def __init__(self):
        self.rank = 0
        self.local_rank = 0
        self.profile_calls = []

    def profile(self, is_start=True, profile_prefix=None):
        self.profile_calls.append((is_start, profile_prefix))


class StepTimingOverlayTest(unittest.TestCase):
    def setUp(self):
        FakeEvent.clock = 0
        FakeGraph.replays = 0

    def make_overlay(self, directory: Path, *, max_samples=32, auto_start=False):
        fake_torch = FakeTorch()
        instance = overlay.TimingOverlay(
            fake_torch,
            max_samples=max_samples,
            auto_start=auto_start,
            output_dir=directory,
        )
        instance.install()
        self.addCleanup(instance.close)
        return instance, fake_torch

    def test_opt_out_does_not_install_graph_hook(self):
        fake_torch = FakeTorch()
        original = fake_torch.xpu.XPUGraph.replay
        self.assertIsNone(overlay.install(torch_module=fake_torch, enabled=False))
        self.assertIs(fake_torch.xpu.XPUGraph.replay, original)

    def test_capture_replays_are_not_timed(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance, fake_torch = self.make_overlay(Path(tmp))
            instance.start(kind="fixture", profile_prefix="capture")
            fake_torch.xpu.capturing = True
            FakeGraph().replay()
            fake_torch.xpu.capturing = False
            self.assertEqual(instance.recorded_count, 0)
            self.assertEqual(instance._capture_skipped, 1)
            instance.stop()
            self.assertEqual(fake_torch.xpu.synchronize_calls, 0)

    def test_bound_role_labels_and_one_sync_at_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance, fake_torch = self.make_overlay(Path(tmp), max_samples=2)
            instance.start(kind="fixture", profile_prefix="roles", rank=0)
            graph = FakeGraph()
            draft = DFlashCudaGraphManager(graph)
            target = ModelCudaGraphManager(graph)
            draft.run_fullgraph()
            target.run_fullgraph()
            target.run_fullgraph()  # bounded, original replay still executes
            path = instance.stop()
            self.assertEqual(FakeGraph.replays, 3)
            self.assertEqual(instance.recorded_count, 2)
            self.assertEqual(instance._dropped_replays, 1)
            self.assertEqual(fake_torch.xpu.synchronize_calls, 1)
            self.assertIsNotNone(path)
            data = json.loads(Path(path).read_text())
            self.assertEqual([item["role"] for item in data["events"]], ["draft", "target"])
            self.assertTrue(all(item["duration_ms"] > 0 for item in data["events"]))
            self.assertTrue(all(item["descriptor"] is None for item in data["events"]))
            self.assertIn("concurrency_caveat", data["session"])

    def test_host_scope_is_separate_from_nested_graph_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance, _ = self.make_overlay(Path(tmp))
            instance.install_host_scope_hooks([FakeRunner])
            instance.start(kind="fixture", profile_prefix="scope")
            FakeRunner(DFlashCudaGraphManager(FakeGraph())).execute_model()
            path = instance.stop()
            data = json.loads(Path(path).read_text())
            self.assertEqual(len(data["events"]), 1)
            self.assertEqual(len(data["host_scopes"]), 1)
            self.assertEqual(data["host_scopes"][0]["stage"], "execute_model_host")
            self.assertTrue(data["host_scopes"][0]["nested_graph_timing"])
            self.assertEqual(data["host_scopes"][0]["graph_event_indices"], [0])

    def test_cleanup_restores_graph_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_torch = FakeTorch()
            original = fake_torch.xpu.XPUGraph.replay
            instance = overlay.TimingOverlay(fake_torch, output_dir=Path(tmp))
            instance.install()
            self.assertIsNot(fake_torch.xpu.XPUGraph.replay, original)
            instance.start(kind="fixture")
            instance.stop()
            instance.uninstall()
            self.assertIs(fake_torch.xpu.XPUGraph.replay, original)

    def test_auto_window_starts_only_on_non_capture_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance, fake_torch = self.make_overlay(Path(tmp), max_samples=2, auto_start=True)
            fake_torch.xpu.capturing = True
            FakeGraph().replay()
            self.assertFalse(instance.active)
            fake_torch.xpu.capturing = False
            FakeGraph().replay()
            self.assertTrue(instance.active)
            FakeGraph().replay()
            self.assertFalse(instance.active)
            self.assertEqual(instance.recorded_count, 2)
            self.assertEqual(fake_torch.xpu.synchronize_calls, 1)

    def test_profile_wrapper_uses_native_start_stop_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance, _ = self.make_overlay(Path(tmp))
            instance.install_worker_profile(FakeWorker)
            worker = FakeWorker()
            worker.profile(True, "api")
            DFlashCudaGraphManager(FakeGraph()).run_fullgraph()
            worker.profile(False)
            self.assertFalse(instance.active)
            self.assertEqual(worker.profile_calls, [(True, "api"), (False, None)])
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 1)

    def test_patcher_is_compiled_and_idempotent(self):
        source = "import os\n\nimport torch\n\nclass XPUWorker:\n    pass\n"
        patched = patch.patch_text(source)
        compile(patched, "fixture.py", "exec")
        self.assertEqual(patched.count(patch.MARKER), 1)
        self.assertEqual(patch.patch_text(patched), patched)


if __name__ == "__main__":
    unittest.main(verbosity=2)
