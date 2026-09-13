#!/usr/bin/env python3
"""Disposable opt-in XPU graph timing overlay for Qwen3.8 comparison cells.

This module deliberately stays outside vLLM and imports only the standard library
until :func:`install` is called.  The launcher patch imports it in each XPU worker
only when ``B70_STEP_TIMING=1``.

The graph hook records one ``torch.xpu.Event(enable_timing=True)`` pair around
``XPUGraph.replay``.  Event completion is deferred until ``stop``: there is no
per-replay synchronize.  ``host_elapsed_ns`` is the Python replay-call elapsed
time, not a kernel duration.  ``duration_ms`` is the elapsed time reported by the
XPU event on the current stream; it is not a kernel-profiler event and does not
isolate other streams.

Two bounded modes are supported:

* Native profile window (preferred): set ``B70_STEP_TIMING=1`` and use the
  existing ``/start_profile`` and ``/stop_profile`` API.  The patch wraps the
  worker's ``profile`` method and flushes after the stop call.
* Environment window: additionally set ``B70_STEP_TIMING_AUTO=1``.  The first
  non-capture replay starts the window and the overlay flushes after at most
  ``B70_STEP_TIMING_MAX_SAMPLES`` graph replays.  This does not require the
  torch kernel profiler.

The overlay is intentionally not a general profiler or a persistent hook.  It
writes one JSON artifact at stop.  Set ``B70_STEP_TIMING_DIR`` to a writable
mounted directory, or set ``B70_STEP_TIMING_OUT`` to one explicit file path.
"""
from __future__ import annotations

import atexit
import functools
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable


ENABLE_ENV = "B70_STEP_TIMING"
AUTO_ENV = "B70_STEP_TIMING_AUTO"
MAX_ENV = "B70_STEP_TIMING_MAX_SAMPLES"
OUT_ENV = "B70_STEP_TIMING_OUT"
DIR_ENV = "B70_STEP_TIMING_DIR"
DEFAULT_MAX_SAMPLES = 32
FORMAT = "b70-step-timing-overlay-v1"

_ACTIVE: "TimingOverlay | None" = None


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(value: str | None, default: int, *, minimum: int = 1, maximum: int = 4096) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _safe_name(value: str | None) -> str:
    if not value:
        return "session"
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result[:80] or "session"


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', cls.__name__)}".strip(".")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _descriptor_fields(descriptor: Any) -> dict[str, Any] | None:
    """Extract only stable scalar descriptor fields; never call arbitrary properties."""
    if descriptor is None:
        return None
    fields: dict[str, Any] = {}
    if hasattr(descriptor, "__dict__"):
        raw = vars(descriptor)
        for key, value in raw.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                fields[key] = value
            else:
                fields[key] = repr(value)
    if not fields:
        for key in (
            "cg_mode",
            "num_tokens",
            "num_reqs",
            "uniform_token_count",
            "num_active_loras",
            "max_query_len",
        ):
            try:
                value = getattr(descriptor, key)
            except Exception:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                fields[key] = value
            else:
                fields[key] = repr(value)
    return fields or None


def _classify_role(manager_name: str | None, manager_module: str | None,
                   frames: list[dict[str, Any]]) -> tuple[str, str]:
    """Classify only from observed manager/caller provenance.

    Unknown managers remain ``unknown`` rather than being inferred from a model
    or graph size.  The pinned DSpark path exposes DFlashCudaGraphManager for
    the draft graph and ModelCudaGraphManager for the target graph.
    """
    manager_name = manager_name or ""
    manager_module = manager_module or ""
    modules = [str(frame.get("module", "")) for frame in frames]
    if manager_name in {"DFlashCudaGraphManager", "DSparkCudaGraphManager"}:
        return "draft", f"manager class {manager_name}"
    if manager_module.endswith(".spec_decode.dflash.cudagraph"):
        return "draft", f"manager module {manager_module}"
    if manager_name == "ModelCudaGraphManager":
        return "target", "manager class ModelCudaGraphManager"
    if manager_name == "CudaGraphManager":
        # The base manager is target-side only when its immediate caller is a
        # known GPU model runner.  A base manager in another caller is left
        # unresolved so an old MTP layout cannot be guessed.
        if any(
            module.endswith(".gpu_model_runner")
            or module.endswith(".gpu.model_runner")
            or module.endswith(".model_runner")
            for module in modules
        ):
            return "target", "base manager called by GPU model runner"
    return "unknown", "unclassified graph manager provenance"


class TimingOverlay:
    """One bounded graph-timing session for one worker process."""

    def __init__(
        self,
        torch_module: Any,
        *,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        auto_start: bool = False,
        output_dir: str | os.PathLike[str] | None = None,
        output_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.torch = torch_module
        self.xpu = getattr(torch_module, "xpu", None)
        self.max_samples = max(1, int(max_samples))
        self.auto_start = bool(auto_start)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.output_path = Path(output_path) if output_path is not None else None

        self._original_replay: Callable[..., Any] | None = None
        self._graph_class: type[Any] | None = None
        self._installed = False
        self._worker_wrappers: list[tuple[type[Any], str, Any]] = []
        self._scope_wrappers: list[tuple[type[Any], str, Any]] = []
        self._atexit_registered = False
        self._lock = threading.RLock()
        self._scope_local = threading.local()

        self._active = False
        self._flushed = False
        self._flush_result: str | None = None
        self._session_number = 0
        self._session_kind = ""
        self._profile_prefix: str | None = None
        self._rank: int | str | None = None
        self._local_rank: int | str | None = None
        self._started_ns: int | None = None
        self._stopped_ns: int | None = None
        self._events: list[dict[str, Any]] = []
        self._host_scopes: list[dict[str, Any]] = []
        self._seen_replays = 0
        self._dropped_replays = 0
        self._auto_stop_pending = False
        self._capture_skipped = 0
        self._errors: list[str] = []
        self._provenance_cache: dict[tuple[int, str, str], dict[str, Any]] = {}

    @property
    def active(self) -> bool:
        return self._active

    @property
    def recorded_count(self) -> int:
        return len(self._events)

    def install(self) -> "TimingOverlay":
        """Install the XPUGraph replay hook; no model/vLLM import is required."""
        if self._installed:
            return self
        graph_class = getattr(self.xpu, "XPUGraph", None)
        original = getattr(graph_class, "replay", None) if graph_class is not None else None
        if graph_class is None or not callable(original):
            raise RuntimeError("torch.xpu.XPUGraph.replay is unavailable")
        existing_owner = getattr(original, "_b70_step_timing_owner", None)
        if existing_owner is not None:
            return existing_owner

        overlay = self

        @functools.wraps(original)
        def replay(graph: Any, *args: Any, **kwargs: Any) -> Any:
            return overlay._replay(original, graph, *args, **kwargs)

        setattr(replay, "_b70_step_timing_owner", self)
        setattr(replay, "_b70_step_timing_original", original)
        graph_class.replay = replay  # type: ignore[assignment]
        self._original_replay = original
        self._graph_class = graph_class
        self._installed = True
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True
        return self

    def _capture_active(self) -> bool:
        checker = getattr(self.xpu, "is_current_stream_capturing", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception as exc:  # pragma: no cover - device-specific fallback
            self._errors.append(f"capture-check:{type(exc).__name__}:{exc}")
            return False

    def _scope_stack(self) -> list[list[int]]:
        stack = getattr(self._scope_local, "stack", None)
        if stack is None:
            stack = []
            self._scope_local.stack = stack
        return stack

    def _record_scope_index(self, index: int) -> None:
        for scope in self._scope_stack():
            scope.append(index)

    def _manager_hint(self) -> tuple[str, str]:
        frame = inspect.currentframe()
        manager_name = ""
        manager_module = ""
        try:
            current = frame.f_back if frame is not None else None
            while current is not None:
                if current.f_code.co_name == "run_fullgraph":
                    manager = current.f_locals.get("self")
                    if manager is not None:
                        manager_name = type(manager).__name__
                        manager_module = type(manager).__module__
                    break
                current = current.f_back
        finally:
            del frame
        return manager_name, manager_module

    def _provenance(self, graph: Any) -> dict[str, Any]:
        graph_id = id(graph)
        manager_hint = self._manager_hint()
        cached = self._provenance_cache.get((graph_id, *manager_hint))
        if cached is not None:
            return dict(cached)
        frames: list[dict[str, Any]] = []
        manager_name: str | None = None
        manager_module: str | None = None
        descriptor: Any = None
        frame = inspect.currentframe()
        try:
            # _provenance -> _replay -> replay wrapper -> caller.
            current = frame.f_back if frame is not None else None
            depth = 0
            while current is not None and depth < 16:
                module = str(current.f_globals.get("__name__", ""))
                function = current.f_code.co_name
                self_obj = current.f_locals.get("self")
                class_name = type(self_obj).__name__ if self_obj is not None else None
                frames.append(
                    {
                        "module": module,
                        "function": function,
                        "line": current.f_lineno,
                        "class": class_name,
                    }
                )
                if function == "run_fullgraph" and manager_name is None:
                    if self_obj is not None:
                        manager_name = type(self_obj).__name__
                        manager_module = type(self_obj).__module__
                    descriptor = current.f_locals.get("desc")
                current = current.f_back
                depth += 1
        finally:
            # Do not retain frame references in the cache.
            del frame
        role, reason = _classify_role(manager_name, manager_module, frames)
        result = {
            "role": role,
            "stage": f"{role}_graph",
            "classification_reason": reason,
            "graph_type": _type_name(graph),
            "graph_id": f"0x{graph_id:x}",
            "manager_type": (
                f"{manager_module}.{manager_name}".strip(".")
                if manager_name
                else None
            ),
            "descriptor": repr(descriptor) if descriptor is not None else None,
            "descriptor_fields": _descriptor_fields(descriptor),
            "caller_stack": frames,
        }
        self._provenance_cache[(graph_id, *manager_hint)] = dict(result)
        return result

    def _replay(self, original: Callable[..., Any], graph: Any,
                *args: Any, **kwargs: Any) -> Any:
        if not self._active and self.auto_start and not self._capture_active():
            self.start(kind="env_auto", profile_prefix="auto")

        if not self._active:
            return original(graph, *args, **kwargs)

        with self._lock:
            self._seen_replays += 1
            if self._capture_active():
                self._capture_skipped += 1
                return original(graph, *args, **kwargs)
            if len(self._events) >= self.max_samples:
                self._dropped_replays += 1
                return original(graph, *args, **kwargs)

            event_class = getattr(self.xpu, "Event", None)
            if not callable(event_class):
                self._errors.append("torch.xpu.Event is unavailable")
                return original(graph, *args, **kwargs)
            try:
                start_event = event_class(enable_timing=True)
                end_event = event_class(enable_timing=True)
                host_start_ns = time.perf_counter_ns()
                start_event.record()
            except Exception as exc:
                self._errors.append(f"event-start:{type(exc).__name__}:{exc}")
                return original(graph, *args, **kwargs)
            provenance = self._provenance(graph)

        try:
            return original(graph, *args, **kwargs)
        finally:
            host_elapsed_ns = max(0, time.perf_counter_ns() - host_start_ns)
            event_error: str | None = None
            try:
                end_event.record()
            except Exception as exc:  # pragma: no cover - device-specific fallback
                event_error = f"event-end:{type(exc).__name__}:{exc}"
                with self._lock:
                    self._errors.append(event_error)
            with self._lock:
                index = len(self._events)
                self._events.append(
                    {
                        "index": index,
                        "kind": "graph_replay",
                        "provenance": provenance,
                        "start_event": start_event,
                        "end_event": end_event,
                        "event_error": event_error,
                        "host_elapsed_ns": host_elapsed_ns,
                    }
                )
                self._record_scope_index(index)
                if self.auto_start and len(self._events) >= self.max_samples:
                    self._auto_stop_pending = True
            if self.auto_start and self._auto_stop_pending and not self._scope_stack():
                self.stop(reason="auto_sample_bound")

    def start(
        self,
        *,
        kind: str = "native_profile",
        profile_prefix: str | None = None,
        rank: int | str | None = None,
        local_rank: int | str | None = None,
    ) -> None:
        with self._lock:
            if self._active:
                self.stop(reason="restart")
            self._session_number += 1
            self._session_kind = kind
            self._profile_prefix = profile_prefix
            self._rank = rank if rank is not None else os.environ.get("RANK", "0")
            self._local_rank = (
                local_rank if local_rank is not None else os.environ.get("LOCAL_RANK", "0")
            )
            self._started_ns = time.time_ns()
            self._stopped_ns = None
            self._events = []
            self._host_scopes = []
            self._seen_replays = 0
            self._dropped_replays = 0
            self._auto_stop_pending = False
            self._capture_skipped = 0
            self._errors = []
            self._provenance_cache = {}
            self._flush_result = None
            self._flushed = False
            self._active = True

    def _resolve_event_durations(self) -> None:
        """Synchronize once, then resolve all pending XPU event pairs."""
        if not self._events:
            return
        synchronize = getattr(self.xpu, "synchronize", None)
        if not callable(synchronize):
            self._errors.append("torch.xpu.synchronize is unavailable")
            return
        try:
            synchronize()
        except Exception as exc:  # pragma: no cover - device-specific fallback
            self._errors.append(f"synchronize:{type(exc).__name__}:{exc}")
            return
        for event in self._events:
            if event.get("event_error") is not None:
                event["duration_ms"] = None
                continue
            try:
                duration = float(event["start_event"].elapsed_time(event["end_event"]))
            except Exception as exc:  # pragma: no cover - device-specific fallback
                self._errors.append(f"elapsed-time:{type(exc).__name__}:{exc}")
                event["duration_ms"] = None
                continue
            event["duration_ms"] = duration if _is_finite_number(duration) and duration >= 0 else None
            if event["duration_ms"] is None:
                self._errors.append("elapsed-time:non-finite-or-negative")

    def _artifact_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        configured = os.environ.get(OUT_ENV)
        if configured:
            candidate = Path(configured)
            if candidate.suffix.lower() in {".json", ".jsonl"}:
                return candidate
            return candidate / self._default_filename()
        directory = self.output_dir
        if directory is None:
            directory = Path(os.environ.get(DIR_ENV, "/tmp"))
        return directory / self._default_filename()

    def _default_filename(self) -> str:
        prefix = _safe_name(self._profile_prefix)
        rank = _safe_name(str(self._rank if self._rank is not None else "0"))
        return f"step-timing-{prefix}-rank{rank}-session{self._session_number}.json"

    def _serializable_event(self, event: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": event["index"],
            "kind": event["kind"],
            "role": event["provenance"]["role"],
            "stage": event["provenance"]["stage"],
            "graph_id": event["provenance"]["graph_id"],
            "graph_type": event["provenance"]["graph_type"],
            "manager_type": event["provenance"]["manager_type"],
            "classification_reason": event["provenance"]["classification_reason"],
            "descriptor": event["provenance"]["descriptor"],
            "descriptor_fields": event["provenance"]["descriptor_fields"],
            "caller_stack": event["provenance"]["caller_stack"],
            "duration_ms": event.get("duration_ms"),
            "host_elapsed_ns": event["host_elapsed_ns"],
            "event_error": event.get("event_error"),
        }

    def _flush(self, *, reason: str) -> str | None:
        if self._flushed:
            return self._flush_result
        self._flushed = True
        path = self._artifact_path()
        payload: dict[str, Any] = {
            "format": FORMAT,
            "session": {
                "kind": self._session_kind,
                "profile_prefix": self._profile_prefix,
                "rank": self._rank,
                "local_rank": self._local_rank,
                "started_unix_ns": self._started_ns,
                "stopped_unix_ns": self._stopped_ns,
                "stop_reason": reason,
                "max_samples": self.max_samples,
                "timing_source": "torch.xpu.Event(enable_timing=True)",
                "duration_unit": "milliseconds",
                "host_elapsed_unit": "nanoseconds",
                "graph_only": True,
                "prefill_policy": "capture-time replays are skipped; eager/non-graph prefill is not measured",
                "concurrency_caveat": (
                    "event duration is on the current stream around XPUGraph.replay; "
                    "it is not kernel-profiler isolation or a concurrent-stream attribution"
                ),
                "native_profiler_separate": True,
                "throughput_comparison": (
                    "this overlay adds timing overhead; compare throughput only with a separate unprofiled cell"
                ),
            },
            "counts": {
                "graph_replays_seen": self._seen_replays,
                "graph_events_emitted": len(self._events),
                "graph_replays_dropped_at_bound": self._dropped_replays,
                "capture_replays_skipped": self._capture_skipped,
                "host_scopes_emitted": len(self._host_scopes),
            },
            "events": [self._serializable_event(event) for event in self._events],
            "host_scopes": list(self._host_scopes),
            "errors": list(self._errors),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")
            os.replace(temporary, path)
            self._flush_result = str(path)
        except Exception as exc:  # pragma: no cover - filesystem-specific fallback
            self._errors.append(f"write:{type(exc).__name__}:{exc}")
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
        return self._flush_result

    def stop(self, *, reason: str = "explicit") -> str | None:
        with self._lock:
            if not self._active and self._flushed:
                return self._flush_result
            self._active = False
            self._auto_stop_pending = False
            self._stopped_ns = time.time_ns()
            self._resolve_event_durations()
            return self._flush(reason=reason)

    def close(self) -> None:
        try:
            if self._active:
                self.stop(reason="close")
        except Exception as exc:  # pragma: no cover - atexit safety
            self._errors.append(f"close:{type(exc).__name__}:{exc}")
        self.uninstall()

    def uninstall(self) -> None:
        with self._lock:
            if self._graph_class is not None and self._original_replay is not None:
                current = getattr(self._graph_class, "replay", None)
                if getattr(current, "_b70_step_timing_owner", None) is self:
                    self._graph_class.replay = self._original_replay  # type: ignore[assignment]
            for cls, name, original in reversed(self._worker_wrappers + self._scope_wrappers):
                current = getattr(cls, name, None)
                owner = getattr(current, "_b70_step_timing_owner", None)
                if owner is self:
                    setattr(cls, name, original)
            self._worker_wrappers.clear()
            self._scope_wrappers.clear()
            self._installed = False

    def install_worker_profile(self, worker_class: type[Any]) -> None:
        """Wrap XPUWorker.profile and concrete XPU execute_model methods."""
        current = getattr(worker_class, "profile", None)
        if callable(current) and getattr(current, "_b70_step_timing_owner", None) is not self:
            overlay = self

            @functools.wraps(current)
            def profile(worker: Any, is_start: bool = True,
                        profile_prefix: str | None = None) -> Any:
                if is_start:
                    result = current(worker, is_start, profile_prefix)
                    overlay.start(
                        kind="native_profile",
                        profile_prefix=profile_prefix,
                        rank=getattr(worker, "rank", None),
                        local_rank=getattr(worker, "local_rank", None),
                    )
                    return result
                try:
                    return current(worker, is_start, profile_prefix)
                finally:
                    overlay.stop(reason="native_profile_stop")

            setattr(profile, "_b70_step_timing_owner", self)
            setattr(profile, "_b70_step_timing_original", current)
            worker_class.profile = profile  # type: ignore[assignment]
            self._worker_wrappers.append((worker_class, "profile", current))

        self.install_host_scope_hooks()

    def install_host_scope_hooks(self, classes: list[type[Any]] | None = None) -> None:
        """Record inclusive host elapsed time for execute_model calls containing graphs.

        This is a separate ``host_scopes`` list, never added to graph durations,
        so consumers cannot accidentally sum nested inclusive and event timings.
        """
        if classes is None:
            module = sys.modules.get("vllm.v1.worker.xpu_model_runner")
            classes = []
            if module is not None:
                for name in ("XPUModelRunner", "XPUModelRunnerV2"):
                    candidate = getattr(module, name, None)
                    if isinstance(candidate, type):
                        classes.append(candidate)
        for cls in classes:
            current = getattr(cls, "execute_model", None)
            if not callable(current) or getattr(current, "_b70_step_timing_owner", None) is self:
                continue
            overlay = self

            @functools.wraps(current)
            def execute_model(runner: Any, *args: Any, _original: Callable[..., Any] = current,
                              **kwargs: Any) -> Any:
                if not overlay._active and not overlay.auto_start:
                    return _original(runner, *args, **kwargs)
                stack = overlay._scope_stack()
                indices: list[int] = []
                stack.append(indices)
                started = time.perf_counter_ns()
                try:
                    return _original(runner, *args, **kwargs)
                finally:
                    elapsed = max(0, time.perf_counter_ns() - started)
                    stack.pop()
                    if indices and len(overlay._host_scopes) < overlay.max_samples:
                        roles = {
                            overlay._events[index]["provenance"]["role"]
                            for index in indices
                            if index < len(overlay._events)
                        }
                        role = next(iter(roles)) if len(roles) == 1 else "mixed"
                        overlay._host_scopes.append(
                            {
                                "index": len(overlay._host_scopes),
                                "kind": "host_scope",
                                "stage": "execute_model_host",
                                "role": role,
                                "graph_event_indices": list(indices),
                                "host_elapsed_ns": elapsed,
                                "nested_graph_timing": True,
                            }
                        )
                    if overlay.auto_start and overlay._auto_stop_pending and not stack:
                        overlay.stop(reason="auto_sample_bound")
            setattr(execute_model, "_b70_step_timing_owner", self)
            setattr(execute_model, "_b70_step_timing_original", current)
            setattr(cls, "execute_model", execute_model)
            self._scope_wrappers.append((cls, "execute_model", current))


def install(*, torch_module: Any | None = None, enabled: bool | None = None) -> TimingOverlay | None:
    """Install the opt-in overlay and return its process-local session object."""
    global _ACTIVE
    if enabled is None:
        enabled = _truthy(os.environ.get(ENABLE_ENV))
    if not enabled:
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    overlay = TimingOverlay(
        torch_module,
        max_samples=_bounded_int(os.environ.get(MAX_ENV), DEFAULT_MAX_SAMPLES),
        auto_start=_truthy(os.environ.get(AUTO_ENV)),
        output_dir=os.environ.get(DIR_ENV),
        output_path=os.environ.get(OUT_ENV),
    )
    overlay.install()
    _ACTIVE = overlay
    return overlay


def install_worker_profile(worker_class: type[Any]) -> None:
    if _ACTIVE is not None:
        _ACTIVE.install_worker_profile(worker_class)


def active_overlay() -> TimingOverlay | None:
    return _ACTIVE


def uninstall() -> None:
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.close()
        _ACTIVE = None


if __name__ == "__main__":
    raise SystemExit("import this module from the disposable launcher patch; it is not a profiler CLI")
