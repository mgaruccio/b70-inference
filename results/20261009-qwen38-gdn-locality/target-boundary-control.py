#!/usr/bin/env python3
"""Disposable target-boundary timing shim for the Qwen 3.8 locality cell.

This module is deliberately a thin, importable development diagnostic.  It is
mounted beside the existing ``qwen38_step_timing_overlay`` and installed only
after that overlay has installed its public ``torch.xpu.XPUGraph.replay`` hook:

    import qwen38_step_timing_overlay as timing
    timing.install()
    import target_boundary_control
    control = target_boundary_control.install_active()

``install(overlay)`` is the direct entry point when the caller already has the
canonical ``TimingOverlay`` instance.  ``install_active()`` is a convenience
for an ephemeral launcher injector; it finds the active instance through the
canonical module's public ``active_overlay()`` helper.  Neither entry point
changes vLLM or the canonical timing source.

The shim wraps the replay method that is already installed by the canonical
overlay.  For an active, non-capture replay whose call stack contains the
pinned target ``vllm.v1.worker.gpu_model_runner._model_forward`` and its
``execute_model`` caller, it records a second public XPU event pair with
``enable_timing=False``.  The canonical timing pair remains the inner pair and
its event/duration/result/error path is untouched.  Draft and ambiguous stacks
are recorded as ``unknown`` control metadata but never get a target marker.

False-event completion is deferred until canonical ``stop``.  The shim does
not synchronize per replay and never stores event handles, IDs, or pointers in
the JSON artifact.  Control metadata is added to the canonical overlay's
existing event records; no second service, store, schema, protocol, or receipt
is created.
"""
from __future__ import annotations

import functools
import inspect
import os
import sys
import threading
import time
from typing import Any, Callable


CONTROL_KEY = "target_boundary_control"
TARGET_MODULE = "vllm.v1.worker.gpu_model_runner"
DRAFT_MODULE = "vllm.v1.spec_decode.llm_base_proposer"
GRAPH_OWNER_ATTR = "_b70_target_boundary_control_owner"
CANONICAL_OWNER_ATTR = "_b70_step_timing_owner"


class _PendingBoundary:
    """Internal state for one false event pair.

    The event objects are intentionally kept only in this process-local object
    until the deferred stop resolver has called ``elapsed_time``.  They are
    never copied to the result payload.
    """

    __slots__ = (
        "start_event",
        "end_event",
        "start_record",
        "end_record",
        "metadata",
    )

    def __init__(self, start_event: Any, end_event: Any, metadata: dict[str, Any]) -> None:
        self.start_event = start_event
        self.end_event = end_event
        self.start_record: dict[str, Any] | None = None
        self.end_record: dict[str, Any] | None = None
        self.metadata = metadata


class TargetBoundaryControl:
    """One process-local wrapper around a canonical timing overlay."""

    def __init__(self, overlay: Any) -> None:
        self.overlay = overlay
        self.xpu = getattr(overlay, "xpu", None)
        self.graph_class = self._resolve_graph_class(overlay)
        self.previous_replay: Callable[..., Any] | None = None
        self._installed = False
        self._lock = threading.RLock()
        self._pending: list[_PendingBoundary] = []
        self._control_by_index: dict[int, dict[str, Any]] = {}
        self._stop_sync_attempted = False
        self._saved_instance_attrs: dict[str, tuple[bool, Any]] = {}

    @staticmethod
    def _resolve_graph_class(overlay: Any) -> type[Any]:
        graph_class = getattr(overlay, "_graph_class", None)
        if graph_class is None:
            xpu = getattr(overlay, "xpu", None)
            graph_class = getattr(xpu, "XPUGraph", None)
        if graph_class is None:
            raise RuntimeError("canonical timing overlay has no torch.xpu.XPUGraph")
        return graph_class

    @property
    def pending_event_references(self) -> int:
        """Return the number of retained false-event handles awaiting stop."""
        with self._lock:
            return sum(
                int(boundary.start_event is not None)
                + int(boundary.end_event is not None)
                for boundary in self._pending
            )

    @property
    def pending_boundary_count(self) -> int:
        """Return the number of false pairs awaiting deferred resolution."""
        with self._lock:
            return len(self._pending)

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> "TargetBoundaryControl":
        """Install after, and only after, the canonical replay hook."""
        if self._installed:
            return self

        current = getattr(self.graph_class, "replay", None)
        existing = getattr(current, GRAPH_OWNER_ATTR, None)
        if existing is not None:
            if isinstance(existing, TargetBoundaryControl):
                return existing
            raise RuntimeError("XPUGraph.replay already has another target-boundary shim")

        canonical_owner = getattr(current, CANONICAL_OWNER_ATTR, None)
        if canonical_owner is not self.overlay:
            raise RuntimeError(
                "install target-boundary control only after canonical TimingOverlay.install()"
            )
        if not callable(current):
            raise RuntimeError("torch.xpu.XPUGraph.replay is unavailable")

        self.previous_replay = current
        control = self

        @functools.wraps(current)
        def replay(graph: Any, *args: Any, **kwargs: Any) -> Any:
            return control._replay(current, graph, *args, **kwargs)

        # The canonical overlay's uninstall recognizes its own owner marker and
        # can therefore still restore the original hook when its existing close()
        # lifecycle runs.  The second marker lets this object uninstall itself.
        setattr(replay, GRAPH_OWNER_ATTR, self)
        setattr(replay, CANONICAL_OWNER_ATTR, self.overlay)
        setattr(replay, "_b70_target_boundary_control_original", current)
        self.graph_class.replay = replay  # type: ignore[assignment]

        self._install_instance_hooks()
        self._installed = True
        return self

    def _save_instance_attr(self, name: str) -> Any:
        instance_dict = getattr(self.overlay, "__dict__", {})
        present = name in instance_dict
        previous = instance_dict.get(name) if present else getattr(self.overlay, name, None)
        self._saved_instance_attrs[name] = (present, previous)
        return previous

    def _restore_instance_attr(self, name: str) -> None:
        saved = self._saved_instance_attrs.get(name)
        if saved is None:
            return
        present, previous = saved
        if present:
            setattr(self.overlay, name, previous)
        else:
            try:
                delattr(self.overlay, name)
            except AttributeError:
                pass

    def _install_instance_hooks(self) -> None:
        previous_serializer = self._save_instance_attr("_serializable_event")
        previous_resolver = self._save_instance_attr("_resolve_event_durations")
        previous_start = self._save_instance_attr("start")
        control = self

        if callable(previous_serializer):

            def serialize(event: dict[str, Any]) -> dict[str, Any]:
                result = previous_serializer(event)
                index = event.get("index")
                if isinstance(index, int):
                    with control._lock:
                        metadata = control._control_by_index.get(index)
                    if metadata is not None:
                        enriched = dict(result)
                        enriched[CONTROL_KEY] = _copy_metadata(metadata)
                        return enriched
                return result

            setattr(self.overlay, "_serializable_event", serialize)

        if callable(previous_resolver):

            def resolve(*_args: Any, **_kwargs: Any) -> Any:
                # Canonical resolution performs its existing one public XPU
                # synchronize before its own elapsed_time calls.  Resolve false
                # pairs only after that call, so there is no per-replay sync.
                try:
                    return previous_resolver()
                finally:
                    control._resolve_pending_boundaries()

            setattr(self.overlay, "_resolve_event_durations", resolve)

        if callable(previous_start):

            def start(*args: Any, **kwargs: Any) -> Any:
                result = previous_start(*args, **kwargs)
                control._reset_for_new_session()
                return result

            setattr(self.overlay, "start", start)

    def _reset_for_new_session(self) -> None:
        with self._lock:
            # Canonical start() has already stopped/flushed the prior session.
            self._pending.clear()
            self._control_by_index.clear()
            self._stop_sync_attempted = False
    @staticmethod
    def _clock_raw_ns() -> int | None:
        try:
            return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        except Exception:
            return None

    @classmethod
    def _record_boundary_call(cls, event: Any) -> dict[str, Any]:
        before = cls._clock_raw_ns()
        error: str | None = None
        try:
            event.record()
        except Exception as exc:  # instrumentation must not mask model errors
            error = f"{type(exc).__name__}:{exc}"
        after = cls._clock_raw_ns()
        return {
            "before_monotonic_raw_ns": before,
            "after_monotonic_raw_ns": after,
            "error": error,
        }

    @staticmethod
    def _classification() -> tuple[str, str, dict[str, Any]]:
        target_forward = False
        execute_model = False
        draft_anchor: str | None = None
        frame = inspect.currentframe()
        current = frame.f_back if frame is not None else None
        try:
            while current is not None:
                module = str(current.f_globals.get("__name__", ""))
                function = current.f_code.co_name
                if module == TARGET_MODULE and function == "_model_forward":
                    target_forward = True
                if module == TARGET_MODULE and function == "execute_model":
                    execute_model = True
                if function == "propose_draft_token_ids":
                    draft_anchor = f"function {function}"
                if module == DRAFT_MODULE and function == "propose":
                    draft_anchor = f"module {module} function {function}"
                current = current.f_back
        finally:
            del frame
        witness = {
            "target_model_forward": target_forward,
            "execute_model": execute_model,
            "draft_anchor": draft_anchor,
        }
        if draft_anchor is not None:
            return "unknown", f"draft anchor rejected: {draft_anchor}", witness
        if target_forward and execute_model:
            return "target", "explicit _model_forward under execute_model", witness
        return "unknown", "explicit target caller context was not complete", witness

    def _capture_state(self) -> bool | None:
        checker = getattr(self.xpu, "is_current_stream_capturing", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            return None

    def _overlay_active(self) -> bool:
        try:
            return bool(getattr(self.overlay, "active"))
        except Exception:
            return False

    def _at_native_event_bound(self) -> bool:
        try:
            count = int(getattr(self.overlay, "recorded_count"))
            maximum = int(getattr(self.overlay, "max_samples"))
        except Exception:
            return False
        return count >= maximum

    def _event_count_and_latest(self) -> tuple[int, dict[str, Any] | None]:
        try:
            events = getattr(self.overlay, "_events")
            count = len(events)
            latest = events[-1] if events else None
            return count, latest if isinstance(latest, dict) else None
        except Exception:
            try:
                return int(getattr(self.overlay, "recorded_count")), None
            except Exception:
                return 0, None

    def _base_metadata(
        self,
        classification: str,
        reason: str,
        witness: dict[str, Any],
    ) -> dict[str, Any]:
        native_thread_id: int | None
        try:
            native_thread_id = threading.get_native_id()
        except Exception:
            native_thread_id = None
        return {
            "classification": classification,
            "classification_reason": reason,
            "context_witness": dict(witness),
            "pid": os.getpid(),
            "native_thread_id": native_thread_id,
            "boundary_recorded": False,
            "replay_index": None,
            "graph_id": None,
        }

    def _attach_to_latest(
        self,
        before_count: int,
        metadata: dict[str, Any],
        latest_hint: dict[str, Any] | None = None,
    ) -> None:
        _count, latest = self._event_count_and_latest()
        if latest is None:
            latest = latest_hint
        if latest is None:
            return
        index = latest.get("index")
        if not isinstance(index, int):
            return
        if _count <= before_count and latest_hint is None:
            return
        metadata["replay_index"] = index
        provenance = latest.get("provenance")
        if isinstance(provenance, dict):
            graph_id = provenance.get("graph_id")
            if graph_id is not None:
                metadata["graph_id"] = graph_id
        with self._lock:
            self._control_by_index[index] = metadata

    def _begin_boundary(
        self,
        metadata: dict[str, Any],
    ) -> _PendingBoundary | None:
        event_class = getattr(self.xpu, "Event", None)
        if not callable(event_class):
            metadata["error"] = "torch.xpu.Event is unavailable"
            return None
        try:
            start_event = event_class(enable_timing=False)
            end_event = event_class(enable_timing=False)
        except Exception as exc:
            metadata["error"] = f"event-create:{type(exc).__name__}:{exc}"
            return None

        boundary = _PendingBoundary(start_event, end_event, metadata)
        start_record = self._record_boundary_call(start_event)
        boundary.start_record = start_record
        metadata["start_record"] = dict(start_record)
        if start_record.get("error") is not None:
            metadata["error"] = start_record["error"]
            boundary.start_event = None
            boundary.end_event = None
            return None
        with self._lock:
            self._pending.append(boundary)
        return boundary

    def _finish_boundary(
        self,
        boundary: _PendingBoundary,
        before_count: int,
    ) -> None:
        if boundary.end_event is not None:
            end_record = self._record_boundary_call(boundary.end_event)
            boundary.end_record = end_record
            boundary.metadata["end_record"] = dict(end_record)
            if end_record.get("error") is not None:
                boundary.metadata["error"] = end_record["error"]
        else:
            boundary.metadata.setdefault("error", "end event was not retained")
        boundary.metadata["boundary_recorded"] = True
        self._attach_to_latest(before_count, boundary.metadata)

    def _replay(
        self,
        original: Callable[..., Any],
        graph: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Inactive and capture-time calls are deliberately passed straight
        # through.  In particular, an auto-start performed by the inner
        # canonical hook cannot retroactively make this outer call eligible.

        if not self._overlay_active():
            return original(graph, *args, **kwargs)
        capture = self._capture_state()
        if capture is not False:
            return original(graph, *args, **kwargs)

        classification, reason, witness = self._classification()
        before_count, _ = self._event_count_and_latest()
        if classification != "target":
            metadata = self._base_metadata("unknown", reason, witness)
            try:
                return original(graph, *args, **kwargs)
            finally:
                # Draft and ambiguous calls still get a control observation in
                # the existing canonical event, but never a false target pair.
                self._attach_to_latest(before_count, metadata)

        # The canonical overlay owns the sample bound.  Do not create an extra
        # false marker after its last native event.
        if self._at_native_event_bound():
            return original(graph, *args, **kwargs)

        metadata = self._base_metadata("target", reason, witness)
        boundary = self._begin_boundary(metadata)
        if boundary is None:
            try:
                return original(graph, *args, **kwargs)
            finally:
                self._attach_to_latest(before_count, metadata)

        try:
            return original(graph, *args, **kwargs)
        finally:
            # This runs after the inner canonical wrapper has recorded its true
            # pair, including when the native graph invocation raises.
            try:
                self._finish_boundary(boundary, before_count)
            except Exception as exc:  # never mask an existing model exception
                metadata.setdefault("error", f"boundary-finish:{type(exc).__name__}:{exc}")

    def _resolve_pending_boundaries(self) -> None:
        with self._lock:
            pending = list(self._pending)
        if not pending:
            return

        # In the normal path canonical _events is non-empty and the canonical
        # resolver has already made its one public synchronize call.  A false
        # pair can exist without a native pair only if the native event setup
        # failed; make at most one stop-time fallback call in that case.
        try:
            native_events_exist = bool(getattr(self.overlay, "_events"))
        except Exception:
            native_events_exist = True
        if not native_events_exist and not self._stop_sync_attempted:
            synchronize = getattr(self.xpu, "synchronize", None)
            if callable(synchronize):
                self._stop_sync_attempted = True
                try:
                    synchronize()
                except Exception as exc:
                    for boundary in pending:
                        boundary.metadata.setdefault(
                            "error", f"synchronize:{type(exc).__name__}:{exc}"
                        )

        for boundary in pending:
            start_event = boundary.start_event
            end_event = boundary.end_event
            if start_event is not None and end_event is not None:
                try:
                    duration = float(start_event.elapsed_time(end_event))
                    if duration >= 0.0:
                        boundary.metadata["boundary_duration_ms"] = duration
                    else:
                        boundary.metadata["boundary_duration_ms"] = None
                        boundary.metadata.setdefault("error", "elapsed-time:negative")
                except Exception as exc:
                    boundary.metadata["boundary_duration_ms"] = None
                    boundary.metadata.setdefault(
                        "error", f"elapsed-time:{type(exc).__name__}:{exc}"
                    )
            else:
                boundary.metadata.setdefault("boundary_duration_ms", None)
            # Deferred cleanup: no event object survives stop resolution and no
            # object, ID, or pointer can enter the existing JSON result.
            boundary.start_event = None
            boundary.end_event = None
        with self._lock:
            self._pending.clear()

    def uninstall(self) -> None:
        """Remove the wrapper without changing the canonical source."""
        with self._lock:
            current = getattr(self.graph_class, "replay", None)
            if getattr(current, GRAPH_OWNER_ATTR, None) is self:
                if self.previous_replay is not None:
                    self.graph_class.replay = self.previous_replay  # type: ignore[assignment]
            self._restore_instance_attr("_serializable_event")
            self._restore_instance_attr("_resolve_event_durations")
            self._restore_instance_attr("start")
            self._saved_instance_attrs.clear()
            self._installed = False

    def close(self) -> None:
        """Stop the canonical window, resolve false pairs, and uninstall."""
        try:
            if bool(getattr(self.overlay, "active")):
                self.overlay.stop(reason="target_boundary_close")
        finally:
            self.uninstall()


def _copy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Copy only JSON-shaped control metadata, never event objects."""
    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            copied[key] = _copy_metadata(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            copied[key] = value
        elif isinstance(value, (list, tuple)):
            copied[key] = [
                _copy_metadata(item) if isinstance(item, dict) else item
                for item in value
            ]
    return copied


def install(overlay: Any) -> TargetBoundaryControl:
    """Install the shim around an already-installed canonical overlay."""
    if overlay is None:
        raise ValueError("target-boundary control requires a canonical overlay")
    graph_class = TargetBoundaryControl._resolve_graph_class(overlay)
    current = getattr(graph_class, "replay", None)
    existing = getattr(current, GRAPH_OWNER_ATTR, None)
    if isinstance(existing, TargetBoundaryControl):
        return existing
    control = TargetBoundaryControl(overlay)
    return control.install()


def install_active(
    module_name: str = "qwen38_step_timing_overlay",
) -> TargetBoundaryControl:
    """Install through the canonical module's active-overlay accessor.

    This is intended for a tiny disposable launcher injector after the
    canonical module has run ``install()``.  It fails loudly if that lifecycle
    point was not reached, rather than silently producing an unowned trace.
    """
    module = sys.modules.get(module_name)
    if module is None:
        raise RuntimeError(f"canonical timing module {module_name!r} is not imported")
    accessor = getattr(module, "active_overlay", None)
    if not callable(accessor):
        raise RuntimeError(f"canonical timing module {module_name!r} has no active_overlay()")
    overlay = accessor()
    if overlay is None:
        raise RuntimeError("canonical timing overlay is not active")
    return install(overlay)


__all__ = ["CONTROL_KEY", "TargetBoundaryControl", "install", "install_active"]


if __name__ == "__main__":
    raise SystemExit("import target-boundary-control.py from the disposable launcher")
