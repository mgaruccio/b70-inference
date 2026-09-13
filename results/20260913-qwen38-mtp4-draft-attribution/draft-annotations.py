#!/usr/bin/env python3
"""Bounded MTP4 draft scopes and graph-dispatch metadata.

This module is imported only by the disposable campaign patch after the native
profile boundary is installed.  It intentionally does not import torch or vLLM
at module import time: model classes, dispatcher calls, and graph replay
metadata are wrapped only for the finite ``/start_profile`` -> ``/stop_profile``
window.  The canonical step-timing overlay owns XPU events; this module adds
narrow CPU scopes and a separate attribution artifact around that overlay.
"""
from __future__ import annotations

import atexit
import contextlib
import functools
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping


ENABLE_ENV = "B70_DRAFT_ATTRIBUTION"
DIR_ENV = "B70_DRAFT_ATTRIBUTION_DIR"
OUT_ENV = "B70_DRAFT_ATTRIBUTION_OUT"
MAX_SCOPES_ENV = "B70_DRAFT_ATTRIBUTION_MAX_SCOPES"
MAX_DISPATCHES_ENV = "B70_DRAFT_ATTRIBUTION_MAX_DISPATCHES"
FORMAT = "b70-mtp4-draft-attribution-v1"
PHASE_PREFIX = "b70_draft/phase:"
DEFAULT_MAX_SCOPES = 256
DEFAULT_MAX_DISPATCHES = 256

_ACTIVE: "AttributionSession | None" = None


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(value: str | None, default: int, *, maximum: int = 4096) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, min(maximum, parsed))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_mode(value: Any) -> Any:
    """Describe an enum-like runtime mode without importing vLLM."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    raw_value = getattr(value, "value", None)
    if isinstance(name, str):
        result: dict[str, Any] = {"name": name}
        if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
            result["value"] = raw_value
        return result
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    """Serialize shapes and scalar metadata without reading tensor values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth > 3:
        return type(value).__name__

    # Tensor shape/dtype/device access does not copy a tensor to the host.
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
        try:
            shape = [int(item) for item in value.shape]
        except Exception:
            shape = None
        return {
            "kind": "tensor",
            "shape": shape,
            "dtype": str(value.dtype),
            "device": str(value.device),
        }

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in list(value.items())[:32]:
            result[str(key)] = _safe_value(child, depth=depth + 1)
        if len(value) > 32:
            result["_truncated"] = len(value) - 32
        return result
    if isinstance(value, (list, tuple)):
        result = [_safe_value(child, depth=depth + 1) for child in value[:32]]
        if len(value) > 32:
            result.append({"_truncated": len(value) - 32})
        return result
    if isinstance(value, set):
        return sorted((_safe_value(child, depth=depth + 1) for child in value), key=str)

    # BatchDescriptor is a small dataclass, but keep this duck-typed so the
    # module remains importable without vLLM in CPU fixture checks.
    fields = ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras")
    if any(hasattr(value, field) for field in fields):
        result = {}
        for field in fields:
            if hasattr(value, field):
                try:
                    result[field] = _safe_value(getattr(value, field), depth=depth + 1)
                except Exception:
                    result[field] = "<unavailable>"
        return result
    return type(value).__name__


def _descriptor(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    result: dict[str, Any] = {}
    for field in ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras"):
        try:
            child = getattr(value, field)
        except Exception:
            continue
        result[field] = _safe_value(child)
    return result or {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', cls.__name__)}".strip(".")


def _caller_stack(*, limit: int = 16) -> list[dict[str, Any]]:
    """Capture names/lines only; never retain live frames."""
    rows: list[dict[str, Any]] = []
    frame = inspect.currentframe()
    try:
        current = frame.f_back if frame is not None else None
        while current is not None and len(rows) < limit:
            self_obj = current.f_locals.get("self")
            rows.append(
                {
                    "module": str(current.f_globals.get("__name__", "")),
                    "function": current.f_code.co_name,
                    "line": current.f_lineno,
                    "class": type(self_obj).__name__ if self_obj is not None else None,
                }
            )
            current = current.f_back
    finally:
        del frame
    return rows


def classify_dispatch_call(stack: list[Mapping[str, Any]]) -> str:
    """Classify dispatcher callers conservatively from observed provenance."""
    text = " ".join(
        f"{row.get('module', '')} {row.get('function', '')} {row.get('class', '')}"
        for row in stack
    )
    # Draft dispatch is nested in SpecDecodeBaseProposer._determine... even
    # though the target runner is also present higher in the call stack.
    if "SpecDecodeBaseProposer" in text or "EagleProposer" in text or "spec_decode" in text:
        return "draft"
    if "XPUModelRunner" in text or "gpu_model_runner" in text:
        return "target"
    return "unknown"


def classify_token_stage(requested_tokens: Any, padded_tokens: Any = None) -> str:
    """Use requested (not padded) token count for the first-five/later-one split."""
    try:
        requested = int(requested_tokens)
    except (TypeError, ValueError):
        requested = None
    if requested == 5:
        return "first_five_tokens"
    if requested == 1:
        return "later_one_token"
    try:
        padded = int(padded_tokens)
    except (TypeError, ValueError):
        padded = None
    if padded == 5:
        return "five_tokens_padded_or_exact"
    if padded == 1:
        return "one_token_padded_or_exact"
    return "other"


def _phase_name(label: str) -> str:
    return PHASE_PREFIX + label


class AttributionSession:
    """One process-local, finite attribution window."""

    def __init__(self, *, max_scopes: int, max_dispatches: int) -> None:
        self.max_scopes = max(1, int(max_scopes))
        self.max_dispatches = max(1, int(max_dispatches))
        self.torch: Any | None = None
        self.overlay: Any | None = None
        self.active = False
        self.session_number = 0
        self.profile_prefix: str | None = None
        self.rank: int | str | None = None
        self.local_rank: int | str | None = None
        self.started_unix_ns: int | None = None
        self.stopped_unix_ns: int | None = None
        self.stop_reason: str | None = None
        self.scopes: list[dict[str, Any]] = []
        self.dispatches: list[dict[str, Any]] = []
        self.draft_spans: list[dict[str, Any]] = []
        self._pending_draft_spans: list[tuple[dict[str, Any], Any, Any]] = []
        self.errors: list[str] = []
        self._wrappers: list[tuple[type[Any], str, Any, bool]] = []
        self._overlay_original_replay: Any | None = None
        self._lock = threading.RLock()
        self._scope_local = threading.local()
        self._flushed_path: str | None = None
        self._flushed = False

    def install_worker_profile(self, worker_class: type[Any]) -> None:
        current = getattr(worker_class, "profile", None)
        if not callable(current):
            raise RuntimeError("XPUWorker.profile is unavailable")
        if getattr(current, "_b70_draft_attribution_owner", None) is self:
            return
        session = self

        @functools.wraps(current)
        def profile(worker: Any, is_start: bool = True, profile_prefix: str | None = None) -> Any:
            if is_start:
                result = current(worker, is_start, profile_prefix)
                try:
                    session.start(worker, profile_prefix)
                except BaseException:
                    session.close()
                    raise
                return result
            try:
                return current(worker, is_start, profile_prefix)
            finally:
                # The canonical timing wrapper normally stops its overlay in
                # this call.  stop() is idempotent and handles an early error.
                session.stop(reason="native_profile_stop")

        setattr(profile, "_b70_draft_attribution_owner", self)
        setattr(profile, "_b70_draft_attribution_original", current)
        worker_class.profile = profile  # type: ignore[assignment]
        self._wrappers.append((worker_class, "profile", current, True))

    def start(self, worker: Any, profile_prefix: str | None) -> None:
        with self._lock:
            if self.active:
                self.stop(reason="restart")
            self.session_number += 1
            self.profile_prefix = profile_prefix
            self.rank = getattr(worker, "rank", os.environ.get("RANK", "0"))
            self.local_rank = getattr(worker, "local_rank", os.environ.get("LOCAL_RANK", "0"))
            self.started_unix_ns = time.time_ns()
            self.stopped_unix_ns = None
            self.stop_reason = None
            self.scopes = []
            self.dispatches = []
            self.draft_spans = []
            self._pending_draft_spans = []
            self._flushed = False
            self._flushed_path = None
            self._scope_stack().clear()
            try:
                # Imports occur only after /start_profile, after startup graph
                # compilation/capture, so the wrappers cannot perturb capture.
                self.torch = importlib.import_module("torch")
                self.overlay = self._load_overlay()
                self._attach_dispatcher()
                self._attach_scopes()
                self._attach_overlay_replay()
                self.active = True
                self._emit_metadata_line(
                    "B70_DRAFT_ATTRIBUTION_START",
                    {
                        "profile_prefix": profile_prefix,
                        "rank": self.rank,
                        "local_rank": self.local_rank,
                        "max_scopes": self.max_scopes,
                        "max_dispatches": self.max_dispatches,
                    },
                )
            except BaseException:
                self._restore_wrappers()
                self._restore_overlay_replay()
                raise

    def _load_overlay(self) -> Any | None:
        try:
            module = importlib.import_module("qwen38_step_timing_overlay")
            return module.active_overlay()
        except Exception as exc:
            self.errors.append(f"overlay-import:{type(exc).__name__}:{exc}")
            return None

    def _scope_stack(self) -> list[int]:
        stack = getattr(self._scope_local, "stack", None)
        if stack is None:
            stack = []
            self._scope_local.stack = stack
        return stack

    def _record_context(self, name: str) -> Any:
        profiler = getattr(self.torch, "profiler", None)
        record_function = getattr(profiler, "record_function", None)
        if callable(record_function):
            try:
                return record_function(name)
            except Exception as exc:
                self.errors.append(f"record-function:{type(exc).__name__}:{exc}")
        return contextlib.nullcontext()

    def _start_draft_span(
        self, label: str, scope_index: int | None
    ) -> tuple[dict[str, Any], Any, Any] | None:
        """Queue a current-stream event pair around one draft call.

        The pair is resolved only when the finite profile window stops.  This
        avoids synchronizing the XPU stream on every proposal while still
        measuring the complete stream interval around ``EagleProposer.propose``.
        """
        xpu = getattr(self.torch, "xpu", None) if self.torch is not None else None
        event_class = getattr(xpu, "Event", None)
        if not callable(event_class):
            return None
        with self._lock:
            if len(self.draft_spans) >= self.max_scopes:
                return None
            row: dict[str, Any] = {
                "index": len(self.draft_spans),
                "phase": label,
                "scope_index": scope_index,
                "started_monotonic_ns": None,
                "finished_monotonic_ns": None,
                "host_elapsed_ns": None,
                "duration_ms": None,
                "event_error": None,
            }
            self.draft_spans.append(row)
        try:
            start_event = event_class(enable_timing=True)
            end_event = event_class(enable_timing=True)
            started = time.perf_counter_ns()
            start_event.record()
        except Exception as exc:
            message = f"draft-span-start:{type(exc).__name__}:{exc}"
            with self._lock:
                row["event_error"] = message
                self.errors.append(message)
            return None
        with self._lock:
            row["started_monotonic_ns"] = started
            self._pending_draft_spans.append((row, start_event, end_event))
        return row, start_event, end_event

    def _finish_draft_span(
        self, pending: tuple[dict[str, Any], Any, Any]
    ) -> None:
        row, _start_event, end_event = pending
        finished = time.perf_counter_ns()
        error: str | None = None
        try:
            end_event.record()
        except Exception as exc:
            error = f"draft-span-end:{type(exc).__name__}:{exc}"
        with self._lock:
            row["finished_monotonic_ns"] = finished
            started = row.get("started_monotonic_ns")
            row["host_elapsed_ns"] = (
                max(0, finished - int(started)) if isinstance(started, int) else None
            )
            if error is not None:
                row["event_error"] = error
                self.errors.append(error)

    def _resolve_draft_spans(self) -> None:
        """Resolve all queued pairs with one stop-time stream synchronization."""
        with self._lock:
            pending = list(self._pending_draft_spans)
        if not pending:
            return
        xpu = getattr(self.torch, "xpu", None) if self.torch is not None else None
        synchronize = getattr(xpu, "synchronize", None)
        sync_error: str | None = None
        if not callable(synchronize):
            sync_error = "draft-span-synchronize:torch.xpu.synchronize unavailable"
        else:
            try:
                synchronize()
            except Exception as exc:
                sync_error = f"draft-span-synchronize:{type(exc).__name__}:{exc}"
        if sync_error is not None:
            with self._lock:
                self.errors.append(sync_error)
        for row, start_event, end_event in pending:
            if row.get("event_error") is not None:
                continue
            if sync_error is not None:
                with self._lock:
                    row["event_error"] = sync_error
                continue
            try:
                duration = float(start_event.elapsed_time(end_event))
            except Exception as exc:
                error = f"draft-span-elapsed:{type(exc).__name__}:{exc}"
                with self._lock:
                    row["event_error"] = error
                    self.errors.append(error)
                continue
            if not _finite(duration) or duration < 0:
                error = f"draft-span-elapsed:non-finite-or-negative:{duration!r}"
                with self._lock:
                    row["event_error"] = error
                    self.errors.append(error)
                continue
            with self._lock:
                row["duration_ms"] = duration
        with self._lock:
            self._pending_draft_spans.clear()

    def _wrap_method(self, owner: type[Any], name: str, label: str) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            self.errors.append(f"missing-method:{_type_name(owner)}.{name}")
            return
        if getattr(original, "_b70_draft_attribution_owner", None) is self:
            return
        existed = name in getattr(owner, "__dict__", {})
        session = self

        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            if not session.active:
                return original(*args, **kwargs)
            scope_index: int | None = None
            with session._lock:
                if len(session.scopes) < session.max_scopes:
                    scope_index = len(session.scopes)
                    parent = session._scope_stack()[-1] if session._scope_stack() else None
                    session.scopes.append(
                        {
                            "index": scope_index,
                            "phase": label,
                            "phase_name": _phase_name(label),
                            "owner": _type_name(owner),
                            "method": name,
                            "thread_id": threading.get_ident(),
                            "parent_index": parent,
                            "started_monotonic_ns": time.perf_counter_ns(),
                            "args": session._scope_arguments(args, kwargs),
                        }
                    )
                    session._scope_stack().append(scope_index)
            context = session._record_context(_phase_name(label))
            pending_span = (
                session._start_draft_span(label, scope_index)
                if label == "propose"
                else None
            )
            try:
                with context:
                    return original(*args, **kwargs)
            finally:
                if pending_span is not None:
                    session._finish_draft_span(pending_span)
                if scope_index is not None:
                    finished = time.perf_counter_ns()
                    with session._lock:
                        row = session.scopes[scope_index]
                        row["finished_monotonic_ns"] = finished
                        row["host_elapsed_ns"] = max(
                            0, finished - int(row["started_monotonic_ns"])
                        )
                        stack = session._scope_stack()
                        if stack and stack[-1] == scope_index:
                            stack.pop()
                        elif scope_index in stack:
                            stack.remove(scope_index)

        setattr(call, "_b70_draft_attribution_owner", self)
        setattr(call, "_b70_draft_attribution_original", original)
        setattr(owner, name, call)
        self._wrappers.append((owner, name, original, existed))

    def _scope_arguments(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        # Keep only bounded structural metadata.  Tensor contents and token
        # values are deliberately never read.
        result: dict[str, Any] = {}
        if args:
            result["positional"] = [_safe_value(value) for value in args[:8]]
        if kwargs:
            result["keyword"] = {
                str(key): _safe_value(value) for key, value in list(kwargs.items())[:16]
            }
        return result

    def _attach_scopes(self) -> None:
        eagle_module = importlib.import_module("vllm.v1.spec_decode.eagle")
        base_module = importlib.import_module("vllm.v1.spec_decode.llm_base_proposer")
        mtp_module = importlib.import_module("vllm.model_executor.models.qwen3_5_mtp")
        eagle = getattr(eagle_module, "EagleProposer", None)
        base = getattr(base_module, "SpecDecodeBaseProposer", None)
        mtp = getattr(mtp_module, "Qwen3_5MTP", None)
        predictor = getattr(mtp_module, "Qwen3_5MultiTokenPredictor", None)
        if isinstance(eagle, type):
            self._wrap_method(eagle, "propose", "propose")
            self._wrap_method(eagle, "_sample_draft_tokens", "sample_draft")
            self._wrap_method(eagle, "_greedy_sample", "sample_greedy")
            self._wrap_method(eagle, "_sample_from_logits", "sample_logits")
        elif isinstance(base, type):
            self.errors.append("missing-class:EagleProposer")
            self._wrap_method(base, "propose", "propose")
        if isinstance(mtp, type):
            self._wrap_method(mtp, "forward", "mtp_forward")
            self._wrap_method(mtp, "compute_logits", "lm_head")
        else:
            self.errors.append("missing-class:Qwen3_5MTP")
        # The predictor is the body called by Qwen3_5MTP.forward.  It is a
        # useful narrow boundary for eager kernels; graph replay still uses the
        # outer Qwen3_5MTP/Eagle provenance and timing overlay.
        if isinstance(predictor, type):
            self._wrap_method(predictor, "forward", "mtp_predictor")

    def _attach_dispatcher(self) -> None:
        module = importlib.import_module("vllm.v1.cudagraph_dispatcher")
        dispatcher = getattr(module, "CudagraphDispatcher", None)
        if not isinstance(dispatcher, type):
            raise RuntimeError("CudagraphDispatcher is unavailable")
        self._wrap_method_dispatch(dispatcher)

    def _wrap_method_dispatch(self, owner: type[Any]) -> None:
        name = "dispatch"
        original = getattr(owner, name, None)
        if not callable(original):
            raise RuntimeError("CudagraphDispatcher.dispatch is unavailable")
        if getattr(original, "_b70_draft_attribution_owner", None) is self:
            return
        session = self

        @functools.wraps(original)
        def dispatch(dispatcher: Any, *args: Any, **kwargs: Any) -> Any:
            if not session.active:
                return original(dispatcher, *args, **kwargs)
            started = time.perf_counter_ns()
            stack = _caller_stack()
            requested = args[0] if args else kwargs.get("num_tokens")
            result: Any = None
            error: BaseException | None = None
            try:
                result = original(dispatcher, *args, **kwargs)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                finished = time.perf_counter_ns()
                with session._lock:
                    if len(session.dispatches) < session.max_dispatches:
                        mode = result[0] if isinstance(result, tuple) and len(result) >= 1 else None
                        batch = result[1] if isinstance(result, tuple) and len(result) >= 2 else None
                        padded = getattr(batch, "num_tokens", None) if batch is not None else None
                        row: dict[str, Any] = {
                            "index": len(session.dispatches),
                            "role": classify_dispatch_call(stack),
                            "requested_num_tokens": _safe_value(requested),
                            "returned_runtime_mode": _safe_mode(mode),
                            "dispatcher_runtime_mode": _safe_mode(getattr(dispatcher, "cudagraph_mode", None)),
                            "returned_batch_descriptor": _descriptor(batch),
                            "stage": classify_token_stage(requested, padded),
                            "uniform_decode": _safe_value(kwargs.get("uniform_decode", args[1] if len(args) > 1 else False)),
                            "has_lora": _safe_value(kwargs.get("has_lora", False)),
                            "num_active_loras": _safe_value(kwargs.get("num_active_loras", 0)),
                            "valid_modes": _safe_value(kwargs.get("valid_modes")),
                            "invalid_modes": _safe_value(kwargs.get("invalid_modes")),
                            "keys_initialized": bool(getattr(dispatcher, "keys_initialized", False)),
                            "available_keys": session._dispatcher_keys(dispatcher),
                            "caller_stack": stack,
                            "started_monotonic_ns": started,
                            "finished_monotonic_ns": finished,
                            "host_elapsed_ns": max(0, finished - started),
                        }
                        if error is not None:
                            row["error"] = {"type": type(error).__name__, "message": str(error)}
                        session.dispatches.append(row)

        setattr(dispatch, "_b70_draft_attribution_owner", self)
        setattr(dispatch, "_b70_draft_attribution_original", original)
        existed = name in getattr(owner, "__dict__", {})
        owner.dispatch = dispatch
        self._wrappers.append((owner, name, original, existed))

    def _dispatcher_keys(self, dispatcher: Any) -> dict[str, list[Any]]:
        keys = getattr(dispatcher, "cudagraph_keys", None)
        if not isinstance(keys, Mapping):
            return {}
        result: dict[str, list[Any]] = {}
        for mode, values in keys.items():
            try:
                rows = [_descriptor(value) for value in values]
                rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
            except Exception:
                rows = []
            result[str(_safe_mode(mode))] = rows
        return result

    def _attach_overlay_replay(self) -> None:
        overlay = self.overlay
        if overlay is None:
            return
        original = getattr(overlay, "_replay", None)
        if not callable(original):
            self.errors.append("overlay-replay-hook-unavailable")
            return
        session = self

        @functools.wraps(original)
        def replay(*args: Any, **kwargs: Any) -> Any:
            before = len(getattr(overlay, "_events", []))
            context = session._graph_call_context()
            try:
                return original(*args, **kwargs)
            finally:
                events = getattr(overlay, "_events", [])
                if len(events) > before:
                    for index in range(before, len(events)):
                        try:
                            events[index]["campaign_graph_context"] = context
                        except Exception as exc:
                            session.errors.append(
                                f"overlay-event-context:{type(exc).__name__}:{exc}"
                            )

        setattr(replay, "_b70_draft_attribution_owner", self)
        setattr(replay, "_b70_draft_attribution_original", original)
        overlay._replay = replay
        self._overlay_original_replay = original

    def _graph_call_context(self) -> dict[str, Any]:
        frame = inspect.currentframe()
        try:
            current = frame.f_back if frame is not None else None
            while current is not None:
                self_obj = current.f_locals.get("self")
                if (
                    current.f_code.co_name == "__call__"
                    and type(self_obj).__name__ == "CUDAGraphWrapper"
                ):
                    forward_context = current.f_locals.get("forward_context")
                    batch = current.f_locals.get("batch_descriptor")
                    mode = current.f_locals.get("cudagraph_runtime_mode")
                    entry = current.f_locals.get("entry")
                    if batch is None and forward_context is not None:
                        batch = getattr(forward_context, "batch_descriptor", None)
                    if mode is None and forward_context is not None:
                        mode = getattr(forward_context, "cudagraph_runtime_mode", None)
                    return {
                        "wrapper_type": _type_name(self_obj),
                        "wrapper_runtime_mode": _safe_mode(getattr(self_obj, "runtime_mode", None)),
                        "runtime_mode": _safe_mode(mode),
                        "batch_descriptor": _descriptor(batch),
                        "graph_entry_state": (
                            "capture"
                            if entry is not None and getattr(entry, "cudagraph", None) is None
                            else "replay"
                        ),
                    }
                current = current.f_back
        finally:
            del frame
        return {"wrapper_type": None, "runtime_mode": None, "batch_descriptor": None}

    def _emit_metadata_line(self, prefix: str, value: Mapping[str, Any]) -> None:
        try:
            print(prefix + ": " + json.dumps(value, sort_keys=True), flush=True)
        except Exception:
            pass

    def stop(self, *, reason: str) -> None:
        with self._lock:
            if not self.active and self._flushed:
                return
            self.active = False
            self.stopped_unix_ns = time.time_ns()
            self.stop_reason = reason
            # If the canonical profile wrapper did not reach its own stop
            # (for example because the control call raised), resolve the
            # deferred XPU event pairs exactly once here.  Never synchronize in
            # a replay or scope callback.
            if self.overlay is not None and getattr(self.overlay, "active", False):
                try:
                    self.overlay.stop(reason="draft_attribution_stop")
                except Exception as exc:
                    self.errors.append(f"overlay-stop:{type(exc).__name__}:{exc}")
            self._resolve_draft_spans()
            self._restore_overlay_replay()
            self._flush()
            self._restore_wrappers()
            self._emit_metadata_line(
                "B70_DRAFT_ATTRIBUTION_STOP",
                {
                    "reason": reason,
                    "scope_count": len(self.scopes),
                    "dispatch_count": len(self.dispatches),
                    "draft_span_count": len(self.draft_spans),
                    "artifact": self._flushed_path,
                },
            )

    def _restore_overlay_replay(self) -> None:
        overlay = self.overlay
        original = self._overlay_original_replay
        if overlay is not None and original is not None:
            current = getattr(overlay, "_replay", None)
            if getattr(current, "_b70_draft_attribution_owner", None) is self:
                overlay._replay = original
        self._overlay_original_replay = None

    def _restore_wrappers(self) -> None:
        for owner, name, original, existed in reversed(self._wrappers):
            current = getattr(owner, name, None)
            if getattr(current, "_b70_draft_attribution_owner", None) is not self:
                continue
            if existed:
                setattr(owner, name, original)
            else:
                try:
                    delattr(owner, name)
                except AttributeError:
                    pass
        self._wrappers.clear()

    def _artifact_path(self) -> Path:
        explicit = os.environ.get(OUT_ENV)
        if explicit:
            return Path(explicit)
        directory = Path(os.environ.get(DIR_ENV, "/tmp"))
        rank = str(self.rank if self.rank is not None else os.environ.get("RANK", "0"))
        return directory / f"draft-attribution-rank{rank}-session{self.session_number}.json"

    def _overlay_events(self) -> list[dict[str, Any]]:
        if self.overlay is None:
            return []
        raw_events = getattr(self.overlay, "_events", [])
        result: list[dict[str, Any]] = []
        serializer = getattr(self.overlay, "_serializable_event", None)
        for raw in raw_events:
            try:
                event = serializer(raw) if callable(serializer) else dict(raw)
            except Exception as exc:
                self.errors.append(f"overlay-event-serialize:{type(exc).__name__}:{exc}")
                continue
            event["campaign_graph_context"] = raw.get("campaign_graph_context")
            result.append(event)
        return result

    def _flush(self) -> None:
        if self._flushed:
            return
        self._flushed = True
        path = self._artifact_path()
        overlay_events = self._overlay_events()
        overlay_path = getattr(self.overlay, "_flush_result", None) if self.overlay is not None else None
        payload: dict[str, Any] = {
            "format": FORMAT,
            "session": {
                "kind": "native_profile",
                "profile_prefix": self.profile_prefix,
                "rank": self.rank,
                "local_rank": self.local_rank,
                "started_unix_ns": self.started_unix_ns,
                "stopped_unix_ns": self.stopped_unix_ns,
                "stop_reason": self.stop_reason,
                "max_scopes": self.max_scopes,
                "max_dispatches": self.max_dispatches,
                "max_draft_spans": self.max_scopes,
                "scope_timing_source": "host perf_counter_ns plus torch.profiler.record_function",
                "graph_timing_source": "canonical torch.xpu.Event overlay; deferred resolution at stop",
                "graph_events_are_separate": True,
                "cpu_scope_inclusive_not_additive": True,
                "compile_capture_policy": "wrappers attached after /start_profile; no startup/capture instrumentation",
                "canonical_overlay_artifact": overlay_path,
            },
            "counts": {
                "scope_events_emitted": len(self.scopes),
                "dispatch_events_emitted": len(self.dispatches),
                "graph_events_emitted": len(overlay_events),
                "draft_span_events_emitted": len(self.draft_spans),
            },
            "scopes": self.scopes,
            "dispatches": self.dispatches,
            "graph_replays": overlay_events,
            "draft_spans": self.draft_spans,
            "errors": self.errors,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
            self._flushed_path = str(path)
        except Exception as exc:
            self.errors.append(f"write:{type(exc).__name__}:{exc}")
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass

    def close(self) -> None:
        try:
            if self.active or (self.overlay is not None and getattr(self.overlay, "active", False)):
                self.stop(reason="close")
            else:
                self._restore_overlay_replay()
                self._restore_wrappers()
        except Exception as exc:
            self.errors.append(f"close:{type(exc).__name__}:{exc}")
            self._restore_overlay_replay()
            self._restore_wrappers()

def install_worker_profile(worker_class: type[Any]) -> None:
    """Install the profile-boundary wrapper when the campaign flag is enabled."""
    global _ACTIVE
    if not _truthy(os.environ.get(ENABLE_ENV)):
        return
    if _ACTIVE is None:
        _ACTIVE = AttributionSession(
            max_scopes=_bounded_int(os.environ.get(MAX_SCOPES_ENV), DEFAULT_MAX_SCOPES),
            max_dispatches=_bounded_int(os.environ.get(MAX_DISPATCHES_ENV), DEFAULT_MAX_DISPATCHES),
        )
        atexit.register(_ACTIVE.close)
    _ACTIVE.install_worker_profile(worker_class)


def active_session() -> AttributionSession | None:
    return _ACTIVE


def uninstall() -> None:
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.close()
        _ACTIVE = None


if __name__ == "__main__":
    raise SystemExit("import this module from the disposable campaign patch")
