#!/usr/bin/env python3
"""Bounded GDN annotations for the disposable MTP4 profile window.

The module is imported only by the campaign's opt-in worker patch.  It does not
import torch at module import time, so the selection/classification helpers stay
usable by the CPU-only fixture.  During a native profile window it uses the
public ``torch.profiler.record_function`` and ``torch.xpu.Event`` APIs.  It
writes bounded metadata to the normal server log and leaves the native profiler
trace as the machine-readable attribution artifact; it does not create a
second audit store or protocol.
"""
from __future__ import annotations

import atexit
import contextlib
import functools
import json
import math
import os
import threading
import time
from typing import Any, Mapping


ENABLE_ENV = "B70_GDN_LOCALITY"
MAX_SAMPLES_ENV = "B70_GDN_LOCALITY_MAX_SAMPLES"
MAX_METADATA_ENV = "B70_GDN_LOCALITY_MAX_METADATA"
FORMAT = "b70-gdn-locality-v1"


_ACTIVE: "GDNSession | None" = None


def _bounded_int(value: str | None, default: int, maximum: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, min(maximum, parsed))


def is_gdn_component(class_name: Any, module_name: Any = "", qualified_name: Any = "") -> bool:
    """Return whether names identify a GDN/linear-attention component.

    This deliberately does not match every class containing ``attention``.
    The pinned Qwen path uses names such as ``QwenGDNLinearAttention`` and
    ``GatedDeltaNet``; the backend path is retained as a conservative fallback.
    """
    text = " ".join(str(value or "") for value in (class_name, module_name, qualified_name)).lower()
    compact = text.replace("_", "").replace("-", "")
    return any(
        marker in compact
        for marker in (
            "gdn",
            "gateddeltanet",
            "gateddelta",
            "gdnlinearattention",
        )
    ) or "gdn_attn" in text


def classify_stage(*values: Any) -> str:
    """Classify a component as target or drafter from provenance names."""
    if values and str(values[0] or "").lower() in {"target", "draft"}:
        return str(values[0]).lower()
    text = " ".join(str(value or "") for value in values).lower()
    if any(marker in text for marker in ("draft", "speculator", "eagle", "mtp")):
        return "draft"
    return "target"


def classify_operator(name: Any, annotation: Any = "") -> str:
    """Conservatively bucket a native trace operator for the report."""
    text = f"{name or ''} {annotation or ''}".lower()
    compact = text.replace("_", "").replace("-", "")
    if any(marker in compact for marker in ("gdnattention", "gdnattn", "gateddelta")) or "gdn" in text:
        return "gdn"
    if any(marker in text for marker in ("eagle", "spec_decode", "specdecode", "draft", "mtp")):
        return "draft"
    if any(marker in text for marker in ("flash_attn", "flashattention", "varlen_fwd", "self_attn", "attention")):
        return "attention"
    if any(marker in text for marker in ("matmul", "mm.", "mm_", "::mm", "gemm", "addmm", "linear")):
        return "matmul"
    return "other"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    """Describe shapes and scalar metadata without reading tensor contents."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth > 3:
        return type(value).__name__
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            shape = [int(item) for item in value.shape]
        except Exception:
            shape = None
        result: dict[str, Any] = {
            "kind": "tensor",
            "shape": shape,
            "dtype": str(getattr(value, "dtype", "<unknown>")),
        }
        if hasattr(value, "device"):
            result["device"] = str(getattr(value, "device"))
        return result
    if isinstance(value, Mapping):
        result = {str(key): _safe_value(child, depth=depth + 1) for key, child in list(value.items())[:24]}
        if len(value) > 24:
            result["_truncated"] = len(value) - 24
        return result
    if isinstance(value, (list, tuple)):
        result = [_safe_value(child, depth=depth + 1) for child in value[:24]]
        if len(value) > 24:
            result.append({"_truncated": len(value) - 24})
        return result
    fields = (
        "num_prefills",
        "num_decodes",
        "num_spec_decodes",
        "num_actual_tokens",
        "num_accepted_tokens",
        "num_spec_decode_tokens",
    )
    if any(hasattr(value, field) for field in fields):
        result = {field: _safe_value(getattr(value, field), depth=depth + 1) for field in fields if hasattr(value, field)}
        result["type"] = f"{type(value).__module__}.{type(value).__qualname__}"
        return result
    return type(value).__name__


def _shape_arguments(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "positional": [_safe_value(value) for value in args[:12]],
        "keyword": {str(key): _safe_value(value) for key, value in list(kwargs.items())[:24]},
    }


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', cls.__name__)}".strip(".")


class GDNSession:
    """One finite profile-window annotation session."""

    def __init__(self, *, max_samples: int, max_metadata: int) -> None:
        self.max_samples = max(1, int(max_samples))
        self.max_metadata = max(1, int(max_metadata))
        self.torch: Any | None = None
        self.active = False
        self.started_unix_ns: int | None = None
        self.stopped_unix_ns: int | None = None
        self.samples: list[dict[str, Any]] = []
        self.metadata: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._restorations: list[tuple[type[Any], str, Any]] = []
        self._hooks: list[Any] = []
        self._contexts: dict[int, list[dict[str, Any]]] = {}
        self._object_stages: dict[int, str] = {}
        self._seen_metadata: set[tuple[int, str]] = set()
        self._lock = threading.RLock()
        self._flushed = False

    def install_worker_profile(self, worker_class: type[Any]) -> None:
        current = getattr(worker_class, "profile", None)
        if not callable(current):
            raise RuntimeError("XPUWorker.profile is unavailable")
        if getattr(current, "_b70_gdn_locality_owner", None) is self:
            return
        session = self

        @functools.wraps(current)
        def profile(worker: Any, is_start: bool = True, profile_prefix: str | None = None) -> Any:
            if is_start:
                result = current(worker, is_start, profile_prefix)
                try:
                    session.start(worker)
                except BaseException:
                    session.close(reason="attach_error")
                    raise
                return result
            try:
                return current(worker, is_start, profile_prefix)
            finally:
                session.close(reason="native_profile_stop")

        setattr(profile, "_b70_gdn_locality_owner", self)
        setattr(profile, "_b70_gdn_locality_original", current)
        worker_class.profile = profile  # type: ignore[assignment]
        self._restorations.append((worker_class, "profile", current))

    def start(self, worker: Any) -> None:
        with self._lock:
            if self.active:
                self.close(reason="restart")
            try:
                import torch  # type: ignore[import-not-found]
            except Exception as exc:
                self.errors.append(f"torch-import:{type(exc).__name__}:{exc}")
                self._emit_summary(reason="torch_import_error")
                return
            self.torch = torch
            self.active = True
            self.started_unix_ns = time.time_ns()
            self.stopped_unix_ns = None
            self.samples = []
            self.metadata = []
            self._contexts = {}
            self._object_stages = {}
            self._seen_metadata = set()
            self._flushed = False
            try:
                self._attach(worker)
            except BaseException as exc:
                self.errors.append(f"attach:{type(exc).__name__}:{exc}")
                self._emit_summary(reason="attach_error")

    def _roots(self, worker: Any) -> list[tuple[str, Any]]:
        runner = getattr(worker, "model_runner", None)
        roots: list[tuple[str, Any]] = []
        target = getattr(runner, "model", None)
        if target is not None:
            roots.append(("target", target))
        speculator = getattr(runner, "speculator", None)
        draft = getattr(speculator, "model", None)
        if draft is not None and draft is not target:
            roots.append(("draft", draft))
        return roots

    def _modules(self, root: Any) -> list[tuple[str, Any]]:
        named_modules = getattr(root, "named_modules", None)
        if not callable(named_modules):
            return [("", root)]
        try:
            return [(str(name), module) for name, module in named_modules()]
        except Exception as exc:
            self.errors.append(f"named-modules:{type(exc).__name__}:{exc}")
            return []

    def _attach(self, worker: Any) -> None:
        candidates: list[tuple[str, str, Any]] = []
        for root_stage, root in self._roots(worker):
            for name, module in self._modules(root):
                cls = type(module)
                class_name = cls.__name__
                module_name = getattr(cls, "__module__", "")
                if not is_gdn_component(class_name, module_name, name):
                    continue
                stage = classify_stage(root_stage, name, class_name, module_name)
                self._object_stages[id(module)] = stage
                candidates.append((stage, name, module))

        if not candidates:
            self.errors.append("no-gdn-component-found-in-model-runners")
            print("B70_GDN_LOCALITY_ERROR: no GDN component found in model runners", flush=True)
            return

        wrapped_owners: set[type[Any]] = set()
        for stage, name, module in candidates:
            owner = type(module)
            method_name = "_forward_core" if callable(getattr(owner, "_forward_core", None)) else "forward"
            method = getattr(owner, method_name, None)
            if callable(method) and owner not in wrapped_owners:
                self._wrap_method(owner, method_name)
                wrapped_owners.add(owner)

        # If a candidate class exposes no callable core/forward method, retain
        # a module-level scope.  The fallback still gives the profiler a
        # shape-labelled GDN range without guessing at a kernel name.
        for stage, name, module in candidates:
            if type(module) not in wrapped_owners:
                self._install_module_hooks(stage, name, module)

        print(
            "B70_GDN_LOCALITY_INSTALLED: "
            + json.dumps(
                {
                    "components": len(candidates),
                    "wrapped_classes": [f"{cls.__module__}.{cls.__qualname__}" for cls in wrapped_owners],
                    "scope": "native_profile_window_only",
                    "operator": "torch.ops._xpu_C.gdn_attention",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _record_context(self, label: str) -> Any:
        profiler = getattr(self.torch, "profiler", None)
        record_function = getattr(profiler, "record_function", None)
        if callable(record_function):
            try:
                return record_function(label)
            except Exception as exc:
                self.errors.append(f"record-function:{type(exc).__name__}:{exc}")
        return contextlib.nullcontext()

    def _begin(self, label: str, args: tuple[Any, ...], kwargs: Mapping[str, Any], instance: Any) -> dict[str, Any]:
        sample_enabled = len(self.samples) < self.max_samples
        token: dict[str, Any] = {
            "label": label,
            "host_start_ns": time.perf_counter_ns(),
            "sample_enabled": sample_enabled,
        }
        if sample_enabled:
            xpu = getattr(self.torch, "xpu", None)
            event_type = getattr(xpu, "Event", None)
            if callable(event_type):
                try:
                    start = event_type(enable_timing=True)
                    end = event_type(enable_timing=True)
                    start.record()
                    token["xpu_start"] = start
                    token["xpu_end"] = end
                except Exception as exc:
                    self.errors.append(f"xpu-event:{type(exc).__name__}:{exc}")
        context = self._record_context(label)
        try:
            context.__enter__()
        except Exception as exc:
            self.errors.append(f"record-enter:{type(exc).__name__}:{exc}")
        token["context"] = context
        key = (id(instance), label)
        if key not in self._seen_metadata and len(self.metadata) < self.max_metadata:
            self._seen_metadata.add(key)
            row = {
                "stage": self._object_stages.get(id(instance), "target"),
                "operator": "torch.ops._xpu_C.gdn_attention",
                "label": label,
                "class": _type_name(instance),
                "arguments": _shape_arguments(args, kwargs),
            }
            self.metadata.append(row)
            print("B70_GDN_METADATA: " + json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        return token

    def _end(self, token: dict[str, Any]) -> None:
        end = token.get("xpu_end")
        if end is not None:
            try:
                end.record()
            except Exception as exc:
                self.errors.append(f"xpu-event-end:{type(exc).__name__}:{exc}")
        context = token.get("context")
        if context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception as exc:
                self.errors.append(f"record-exit:{type(exc).__name__}:{exc}")
        token["host_end_ns"] = time.perf_counter_ns()
        if token.get("sample_enabled"):
            self.samples.append(token)

    def _wrap_method(self, owner: type[Any], method_name: str) -> None:
        original = getattr(owner, method_name)
        session = self

        @functools.wraps(original)
        def call(instance: Any, *args: Any, **kwargs: Any) -> Any:
            stage = session._object_stages.get(id(instance), "target")
            label = f"b70_gdn/{stage}/operator:gdn_attention"
            token = session._begin(label, args, kwargs, instance)
            try:
                return original(instance, *args, **kwargs)
            finally:
                session._end(token)

        setattr(call, "_b70_gdn_locality_owner", self)
        setattr(owner, method_name, call)
        self._restorations.append((owner, method_name, original))

    def _install_module_hooks(self, stage: str, name: str, module: Any) -> None:
        label = f"b70_gdn/{stage}/module:{name or type(module).__name__}"
        session = self

        def before(mod: Any, args: tuple[Any, ...]) -> None:
            token = session._begin(label, args, {}, mod)
            session._contexts.setdefault(id(mod), []).append(token)

        def after(mod: Any, args: tuple[Any, ...], output: Any) -> Any:
            stack = session._contexts.get(id(mod))
            if stack:
                session._end(stack.pop())
            return output

        try:
            self._hooks.append(module.register_forward_pre_hook(before))
            try:
                self._hooks.append(module.register_forward_hook(after, always_call=True))
            except TypeError:
                self._hooks.append(module.register_forward_hook(after))
        except Exception as exc:
            self.errors.append(f"module-hook:{type(exc).__name__}:{exc}")

    def close(self, *, reason: str) -> None:
        with self._lock:
            if not self.active and self._flushed:
                return
            for hook in self._hooks:
                try:
                    hook.remove()
                except Exception as exc:
                    self.errors.append(f"hook-remove:{type(exc).__name__}:{exc}")
            self._hooks.clear()
            for owner, name, original in reversed(self._restorations):
                try:
                    setattr(owner, name, original)
                except Exception as exc:
                    self.errors.append(f"restore:{type(exc).__name__}:{exc}")
            self._restorations.clear()
            if self.active and self.torch is not None:
                try:
                    xpu = getattr(self.torch, "xpu", None)
                    synchronize = getattr(xpu, "synchronize", None)
                    if callable(synchronize):
                        synchronize()
                except Exception as exc:
                    self.errors.append(f"xpu-synchronize:{type(exc).__name__}:{exc}")
            self.active = False
            self.stopped_unix_ns = time.time_ns()
            self._emit_summary(reason=reason)

    def _emit_summary(self, *, reason: str) -> None:
        if self._flushed:
            return
        self._flushed = True
        durations: list[float] = []
        host_durations: list[float] = []
        for token in self.samples:
            host_start = token.get("host_start_ns")
            host_end = token.get("host_end_ns")
            if _finite(host_start) and _finite(host_end) and float(host_end) >= float(host_start):
                host_durations.append((float(host_end) - float(host_start)) / 1_000_000.0)
            start = token.get("xpu_start")
            end = token.get("xpu_end")
            if start is None or end is None:
                continue
            try:
                duration = float(start.elapsed_time(end))
            except Exception as exc:
                self.errors.append(f"xpu-elapsed:{type(exc).__name__}:{exc}")
                continue
            if _finite(duration) and duration >= 0:
                durations.append(duration)
        result = {
            "format": FORMAT,
            "reason": reason,
            "operator": "torch.ops._xpu_C.gdn_attention",
            "started_unix_ns": self.started_unix_ns,
            "stopped_unix_ns": self.stopped_unix_ns,
            "metadata": self.metadata,
            "sample_count": len(self.samples),
            "finite_xpu_event_count": len(durations),
            "xpu_event_durations_ms": durations,
            "host_elapsed_durations_ms": host_durations,
            "max_samples": self.max_samples,
            "graph_inner_ops_may_be_hidden": True,
            "eager_profile_is_required_for_inner_operator_kernel_attribution": True,
            "errors": self.errors,
        }
        print("B70_GDN_LOCALITY_SUMMARY: " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)

    def close_at_exit(self) -> None:
        try:
            self.close(reason="process_exit")
        except Exception:
            pass


def install_worker_profile(worker_class: type[Any]) -> None:
    """Install the finite profile wrapper when the opt-in environment is set."""
    global _ACTIVE
    if os.environ.get(ENABLE_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    if _ACTIVE is not None:
        _ACTIVE.install_worker_profile(worker_class)
        return
    _ACTIVE = GDNSession(
        max_samples=_bounded_int(os.environ.get(MAX_SAMPLES_ENV), 128, 4096),
        max_metadata=_bounded_int(os.environ.get(MAX_METADATA_ENV), 32, 512),
    )
    atexit.register(_ACTIVE.close_at_exit)
    _ACTIVE.install_worker_profile(worker_class)


def active_session() -> GDNSession | None:
    return _ACTIVE


def uninstall() -> None:
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.close(reason="uninstall")
        _ACTIVE = None


if __name__ == "__main__":
    print(json.dumps({"format": FORMAT, "enabled": os.environ.get(ENABLE_ENV) == "1"}, sort_keys=True))
