"""Opt-in disposable serving hook for the qualified grouped-query operator.

The canonical ``qwen38_step_timing_patch.py`` import shim loads this module in
an XPU worker.  Importing the module is deliberately inert; only
``B70_GROUPED_SERVING=1`` installs the candidate.  The hook loads the separately
mounted custom-op library, then wraps the native pre-expansion helper.  It does
not replace ``_vllm_fa2_C`` or alter the numerical seam in ``grouped_verify``.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable


ENABLE_ENV = "B70_GROUPED_SERVING"
LIBRARY_ENV = "B70_GROUPED_SERVING_LIBRARY"
DEFAULT_LIBRARY = "/candidate/libb70_grouped_verify.so"
EXPECTED_LIBRARY_SHA256 = (
    "e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec"
)

# These are process-local.  A worker process can import the shim more than once,
# but the operator registration and helper replacement must each happen once.
_LIBRARY_LOADED = False
_LIBRARY_PATH: str | None = None
_INSTALLED = False
_ELIGIBLE_LOGGED = False
_UNSUPPORTED_LOGGED = False
_Q5_PROBES = 0


def _enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1"


def install(*, enabled: bool | None = None) -> bool:
    """Return whether the serving hook is enabled, without importing ML code."""
    return _enabled() if enabled is None else bool(enabled)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_library(torch_module: Any) -> None:
    global _LIBRARY_LOADED, _LIBRARY_PATH
    if _LIBRARY_LOADED:
        return

    path = Path(os.environ.get(LIBRARY_ENV, DEFAULT_LIBRARY))
    if not path.is_file():
        raise RuntimeError(f"grouped serving library is missing: {path}")
    observed = _sha256(path)
    if observed != EXPECTED_LIBRARY_SHA256:
        raise RuntimeError(
            "grouped serving library hash mismatch: "
            f"expected {EXPECTED_LIBRARY_SHA256}, observed {observed}, path={path}"
        )

    ops = getattr(torch_module, "ops", None)
    loader = getattr(ops, "load_library", None)
    if not callable(loader):
        raise RuntimeError("torch.ops.load_library is unavailable")
    loader(str(path))
    _LIBRARY_LOADED = True
    _LIBRARY_PATH = str(path)


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def _shape(value: Any) -> list[int] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return [int(size) for size in raw]
    except (TypeError, ValueError):
        return None


def _stride(value: Any) -> list[int] | None:
    getter = getattr(value, "stride", None)
    if not callable(getter):
        return None
    try:
        return [int(stride) for stride in getter()]
    except (TypeError, ValueError):
        return None


def _tensor_snapshot(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for index, name in enumerate(("q", "k", "v")):
        value = _argument(args, kwargs, index, name)
        snapshot[f"{name}_shape"] = _shape(value)
        snapshot[f"{name}_stride"] = _stride(value)
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        snapshot[f"{name}_dtype"] = str(dtype) if dtype is not None else None
        snapshot[f"{name}_device"] = str(device) if device is not None else None
    return snapshot


def _is_q5(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    shape = _shape(_argument(args, kwargs, 0, "q"))
    return shape is not None and len(shape) > 0 and shape[0] == 5


def _capture_state(torch_module: Any) -> str:
    xpu = getattr(torch_module, "xpu", None)
    checker = getattr(xpu, "is_current_stream_capturing", None)
    if not callable(checker):
        return "unavailable"
    try:
        return "capturing" if bool(checker()) else "not-capturing"
    except Exception as exc:  # pragma: no cover - XPU-version-specific
        return f"error:{type(exc).__name__}"


def _emit(event: str, fields: dict[str, Any]) -> None:
    payload = {"event": event, **fields}
    print(
        "[B70_GROUPED_SERVING] "
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _log_eligible(torch_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    global _ELIGIBLE_LOGGED
    if _ELIGIBLE_LOGGED:
        return
    fields = _tensor_snapshot(args, kwargs)
    capture_state = _capture_state(torch_module)
    fields.update(
        {
            "library_sha256": EXPECTED_LIBRARY_SHA256,
            "capture_state": capture_state,
            "graph_capture_state": capture_state,
        }
    )
    _emit("eligible-dispatch", fields)
    _ELIGIBLE_LOGGED = True


def _log_unsupported(args: tuple[Any, ...], kwargs: dict[str, Any], reason: Any) -> None:
    global _UNSUPPORTED_LOGGED
    if _UNSUPPORTED_LOGGED:
        return
    fields = _tensor_snapshot(args, kwargs)
    fields["reason"] = str(reason)
    _emit("unsupported-q5", fields)
    _UNSUPPORTED_LOGGED = True


def _make_wrapper(
    native: Callable[..., Any],
    dispatch: Callable[..., Any],
    unsupported_reason: Callable[..., Any],
    torch_module: Any,
) -> Callable[..., Any]:
    @functools.wraps(native)
    def grouped(*args: Any, **kwargs: Any) -> Any:
        # Probe only the first one or two five-row calls.  The seam itself owns
        # routing; this bounded probe only makes a geometry rejection visible.
        global _Q5_PROBES
        if _is_q5(args, kwargs) and not _ELIGIBLE_LOGGED and _Q5_PROBES < 2:
            _Q5_PROBES += 1
            reason = unsupported_reason(*args, **kwargs)
            if reason is None:
                _log_eligible(torch_module, args, kwargs)
            else:
                _log_unsupported(args, kwargs, reason)

        # Do not catch the candidate path: eligible compiler/launch failures
        # must reach the serving process instead of silently becoming native.
        return dispatch(native, *args, **kwargs)

    setattr(grouped, "_b70_grouped_serving", True)
    setattr(grouped, "_b70_grouped_serving_native", native)
    return grouped


def install_worker_profile(worker_class: type[Any]) -> bool:
    """Install the helper wrapper before the worker captures any graphs."""
    del worker_class  # The canonical shim supplies this lifecycle boundary.
    global _INSTALLED
    if not _enabled():
        return False
    if _INSTALLED:
        return True

    # Import the native interface first so its original _vllm_fa2_C remains the
    # loaded implementation.  Only the separate b70 namespace is loaded below.
    import torch
    from vllm_xpu_kernels import flash_attn_interface as fa

    native = getattr(fa, "_spec_decode_varlen_fwd", None)
    if not callable(native):
        raise RuntimeError("vllm_xpu_kernels flash_attn_interface lacks _spec_decode_varlen_fwd")
    existing_owner = getattr(native, "_b70_grouped_serving", False)
    if existing_owner:
        _INSTALLED = True
        return True

    _load_library(torch)
    seam = importlib.import_module("grouped_verify")
    dispatch = getattr(seam, "dispatch", None)
    unsupported_reason = getattr(seam, "unsupported_reason", None)
    if not callable(dispatch) or not callable(unsupported_reason):
        raise RuntimeError("grouped_verify seam lacks callable dispatch/unsupported_reason")

    fa._spec_decode_varlen_fwd = _make_wrapper(
        native, dispatch, unsupported_reason, torch
    )
    _INSTALLED = True
    _emit(
        "installed",
        {
            "library": _LIBRARY_PATH,
            "library_sha256": EXPECTED_LIBRARY_SHA256,
            "helper": "_spec_decode_varlen_fwd",
        },
    )
    return True
