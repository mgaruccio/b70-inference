"""Observational P0 greedy-divergence tracer for Qwen B70 vLLM.

Install-time wrappers observe ``GPUModelRunner._sample`` context and
``rejection_sample`` without mutating sampler inputs or outputs. Apply via
``patch_p0_tracer.py`` after the existing MTP/boundary/metadata overlays.
"""
from __future__ import annotations

import builtins
import inspect
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

SCHEMA_VERSION = "qwen38-b70-p0/v1"
TRACE_ENV = "B70_P0_TRACE_PATH"
TRACE_ARM_ENV = "B70_P0_TRACE_ARM"
_logger = logging.getLogger(__name__)

_original_sample: Callable[..., Any] | None = None
_original_rejection_sample: Callable[..., Any] | None = None
_original_rejection_forward: Callable[..., Any] | None = None
_hooks_installed = False
_import_hook_installed = False
_installing = False
_step_counter = 0
_step_lock = threading.Lock()
_original_import = builtins.__import__

@dataclass
class SampleContext:
    """Per-``_sample`` call context used to gate tracer emission."""

    req_ids: Sequence[str]
    context_lens: Sequence[int]
    step: int


_active_sample_context: ContextVar[SampleContext | None] = ContextVar(
    "b70_p0_active_sample_context", default=None
)


def row_draft_ids(
    draft_token_ids: Sequence[int],
    num_draft_tokens: Sequence[int],
    cu_num_draft_tokens: Sequence[int],
    row: int,
) -> list[int]:
    """Return draft token IDs for one batch row, preserving ``-1`` padding."""
    n = int(num_draft_tokens[row])
    if n <= 0:
        return []
    end = int(cu_num_draft_tokens[row])
    start = int(cu_num_draft_tokens[row - 1]) if row > 0 else 0
    return [int(token_id) for token_id in draft_token_ids[start : start + n]]


def accepted_prefix_len(drafted_ids: Sequence[int], target_ids: Sequence[int]) -> int:
    """Count contiguous accepted draft positions before the first mismatch."""
    for index, (draft_id, target_id) in enumerate(zip(drafted_ids, target_ids)):
        if draft_id != target_id:
            return index
    return len(drafted_ids)


def first_divergence_index(
    drafted_ids: Sequence[int], target_ids: Sequence[int]
) -> int | None:
    """Return the first draft/target mismatch index, or ``None`` on full match."""
    for index, (draft_id, target_id) in enumerate(zip(drafted_ids, target_ids)):
        if draft_id != target_id:
            return index
    return None


def draft_target_pairs(
    drafted_ids: Sequence[int], target_ids: Sequence[int]
) -> list[dict[str, int]]:
    """Build per-position draft/target records for the JSONL schema."""
    return [
        {
            "index": index,
            "draft_token_id": int(draft_id),
            "target_token_id": int(target_id),
        }
        for index, (draft_id, target_id) in enumerate(zip(drafted_ids, target_ids))
    ]


def bonus_used(drafted_ids: Sequence[int], target_ids: Sequence[int]) -> bool:
    """Return whether every drafted position matched its target."""
    return bool(drafted_ids) and all(
        draft_id == target_id for draft_id, target_id in zip(drafted_ids, target_ids)
    )


def derive_bonus_token(
    drafted_ids: Sequence[int],
    target_ids: Sequence[int],
    bonus_token_id: int | None,
) -> int | None:
    """Return the bonus token only when every draft position matched."""
    if bonus_used(drafted_ids, target_ids):
        return None if bonus_token_id is None else int(bonus_token_id)
    return None


def gdn_state_num_accepted_tokens(output_row: Sequence[int]) -> int:
    """Count committed output positions (``output != -1``)."""
    return sum(1 for token_id in output_row if int(token_id) != -1)


def build_event(
    *,
    request_id: str,
    step: int,
    context_len: int,
    drafted_ids: Sequence[int],
    target_ids: Sequence[int],
    output_row: Sequence[int],
    bonus_token_id: int | None = None,
    run_id: str | None = None,
    variant: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``qwen38-b70-p0/v1`` JSONL event from pure inputs."""
    prefix_len = accepted_prefix_len(drafted_ids, target_ids)
    event: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "request_id": request_id,
        "step": int(step),
        "context_len": int(context_len),
        "drafted_ids": [int(token_id) for token_id in drafted_ids],
        "target_ids": [int(token_id) for token_id in target_ids],
        "draft_target_pairs": draft_target_pairs(drafted_ids, target_ids),
        "accepted_prefix_len": prefix_len,
        "bonus_token": derive_bonus_token(drafted_ids, target_ids, bonus_token_id),
        "first_divergence_index": first_divergence_index(drafted_ids, target_ids),
        "num_accepted_tokens": prefix_len,
        "gdn_state_num_accepted_tokens": gdn_state_num_accepted_tokens(output_row),
        "bonus_used": bonus_used(drafted_ids, target_ids),
    }
    if run_id is not None:
        event["run_id"] = run_id
    if variant is not None:
        event["variant"] = variant
    return event


def _tensor_to_list(values: Any) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [int(values)]
    return [int(value) for value in values]


def _arg_value(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int | None = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if position is not None and position < len(args):
        return args[position]
    return None


def _next_step() -> int:
    global _step_counter
    with _step_lock:
        step = _step_counter
        _step_counter += 1
    return step


def _trace_path() -> str | None:
    return os.environ.get(TRACE_ENV)


def _trace_armed() -> bool:
    if os.environ.get(TRACE_ARM_ENV) == "1":
        return True
    trace_path = _trace_path()
    if not trace_path:
        return False
    return Path(trace_path).with_name("arm").exists()


def _write_heartbeat(note: str) -> None:
    event = {
        "schema": SCHEMA_VERSION,
        "event": "hooks_installed",
        "note": note,
        "pid": os.getpid(),
    }
    print(f"B70 P0 tracer: {note} pid={os.getpid()}", file=sys.stderr, flush=True)
    _emit_event(event)


def _emit_event(event: dict[str, Any]) -> None:
    trace_path = _trace_path()
    if not trace_path:
        return
    try:
        with open(trace_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":"), sort_keys=True))
            stream.write("\n")
    except OSError as exc:
        print(f"B70 P0 tracer failed to write {trace_path}: {exc}", file=sys.stderr, flush=True)
        _logger.warning("B70 P0 tracer failed to write %s: %s", trace_path, exc)

def _bound_call_args(
    original: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Extract named arguments from a positional or keyword call."""
    try:
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return dict(kwargs)
    return dict(bound.arguments)


def _build_events_from_call(
    *,
    context: SampleContext,
    original: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output_tensor: Any,
) -> list[dict[str, Any]]:
    bound = _bound_call_args(original, args, kwargs)
    draft_token_ids = bound.get("draft_token_ids")
    target_logits = bound.get("target_logits")
    bonus_token_ids = bound.get("bonus_token_ids")
    num_draft_tokens = bound.get("num_draft_tokens")
    cu_num_draft_tokens = bound.get("cu_num_draft_tokens")

    if draft_token_ids is None or target_logits is None:
        return []

    draft_flat = _tensor_to_list(draft_token_ids)
    target_argmax = _tensor_to_list(target_logits.argmax(dim=-1))
    num_drafts = _tensor_to_list(num_draft_tokens)
    cu_drafts = _tensor_to_list(cu_num_draft_tokens)
    bonus_ids = _tensor_to_list(bonus_token_ids)
    output_rows = output_tensor.detach().cpu().tolist()
    if output_rows and not isinstance(output_rows[0], list):
        output_rows = [output_rows]

    events: list[dict[str, Any]] = []
    batch_size = max(
        len(context.req_ids),
        len(output_rows),
        len(num_drafts),
    )
    for row in range(batch_size):
        if row >= len(num_drafts) or row >= len(cu_drafts):
            continue
        drafted_ids = row_draft_ids(draft_flat, num_drafts, cu_drafts, row)
        if not drafted_ids:
            continue
        n = len(drafted_ids)
        end = int(cu_drafts[row])
        start = int(cu_drafts[row - 1]) if row > 0 else 0
        target_ids = target_argmax[start : start + n]
        bonus_token_id = bonus_ids[row] if row < len(bonus_ids) else None
        output_row = output_rows[row] if row < len(output_rows) else []
        request_id = (
            str(context.req_ids[row])
            if row < len(context.req_ids)
            else f"row-{row}"
        )
        context_len = (
            int(context.context_lens[row])
            if row < len(context.context_lens)
            else 0
        )
        events.append(
            build_event(
                request_id=request_id,
                step=context.step,
                context_len=context_len,
                drafted_ids=drafted_ids,
                target_ids=target_ids,
                output_row=output_row,
                bonus_token_id=bonus_token_id,
            )
        )
    return events


def _context_from_output(output_tensor: Any) -> SampleContext:
    output_rows = output_tensor.detach().cpu().tolist()
    if output_rows and not isinstance(output_rows[0], list):
        output_rows = [output_rows]
    n_rows = max(len(output_rows), 1)
    return SampleContext(
        req_ids=[f"row-{index}" for index in range(n_rows)],
        context_lens=[0] * n_rows,
        step=_next_step(),
    )


def _make_rejection_sample_wrapper(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        context = _active_sample_context.get()
        result = original(*args, **kwargs)
        if context is None and not _trace_armed():
            return result
        if context is None:
            context = _context_from_output(result)
        try:
            for event in _build_events_from_call(
                context=context,
                original=original,
                args=args,
                kwargs=kwargs,
                output_tensor=result,
            ):
                _emit_event(event)
        except Exception as exc:  # noqa: BLE001 - observability must not change serving
            print(f"B70 P0 tracer failed to record rejection_sample: {exc}", file=sys.stderr, flush=True)
            _logger.warning("B70 P0 tracer failed to record rejection_sample: %s", exc)
        return result

    return wrapper


def _make_sample_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        req_ids = list(getattr(self.input_batch, "req_ids", []) or [])
        num_computed = getattr(self.input_batch, "num_computed_tokens_cpu", None)
        if num_computed is None:
            context_lens = [0] * len(req_ids)
        else:
            context_lens = _tensor_to_list(num_computed)
            if len(context_lens) < len(req_ids):
                context_lens.extend([0] * (len(req_ids) - len(context_lens)))
        context = SampleContext(
            req_ids=req_ids,
            context_lens=context_lens,
            step=_next_step(),
        )
        token = _active_sample_context.set(context)
        try:
            return original(self, *args, **kwargs)
        finally:
            _active_sample_context.reset(token)

    return wrapper


def _make_rejection_forward_wrapper(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = _active_sample_context.get()
        result = original(self, *args, **kwargs)
        if context is None and not _trace_armed():
            return result
        sampled = getattr(result, "sampled_token_ids", result)
        if context is None:
            context = _context_from_output(sampled)
        try:
            metadata = _arg_value(args, kwargs, "metadata", 0)
            if metadata is None:
                return result
            events = _build_events_from_metadata(
                context=context,
                metadata=metadata,
                output_tensor=sampled,
            )
            for event in events:
                _emit_event(event)
        except Exception as exc:  # noqa: BLE001
            print(f"B70 P0 tracer failed to record forward: {exc}", file=sys.stderr, flush=True)
            _logger.warning("B70 P0 tracer failed to record forward: %s", exc)
        return result

    return wrapper


def _build_events_from_metadata(
    *,
    context: SampleContext,
    metadata: Any,
    output_tensor: Any,
) -> list[dict[str, Any]]:
    draft_flat = _tensor_to_list(getattr(metadata, "draft_token_ids", None))
    num_drafts = _tensor_to_list(getattr(metadata, "num_draft_tokens", None))
    cu_drafts = _tensor_to_list(getattr(metadata, "cu_num_draft_tokens", None))
    output_rows = output_tensor.detach().cpu().tolist()
    if output_rows and not isinstance(output_rows[0], list):
        output_rows = [output_rows]
    events: list[dict[str, Any]] = []
    batch_size = max(len(context.req_ids), len(output_rows), len(num_drafts))
    for row in range(batch_size):
        if row >= len(num_drafts) or row >= len(cu_drafts):
            continue
        drafted_ids = row_draft_ids(draft_flat, num_drafts, cu_drafts, row)
        if not drafted_ids:
            continue
        output_row = output_rows[row] if row < len(output_rows) else []
        # Without processed target logits, recover target ids from the greedy
        # output prefix: accepted drafts plus the first replacement token.
        target_ids = [
            int(token)
            for token in output_row[: len(drafted_ids)]
            if int(token) != -1
        ]
        if len(target_ids) < len(drafted_ids):
            target_ids.extend(drafted_ids[len(target_ids) :])
        else:
            target_ids = target_ids[: len(drafted_ids)]
        request_id = (
            str(context.req_ids[row])
            if row < len(context.req_ids)
            else f"row-{row}"
        )
        context_len = (
            int(context.context_lens[row])
            if row < len(context.context_lens)
            else 0
        )
        events.append(
            build_event(
                request_id=request_id,
                step=context.step,
                context_len=context_len,
                drafted_ids=drafted_ids,
                target_ids=target_ids,
                output_row=output_row,
                bonus_token_id=None,
            )
        )
    return events


def _install_hooks_impl(
    gpu_model_runner_module: Any,
    rejection_sampler_module: Any,
) -> None:
    global _original_sample, _original_rejection_sample, _original_rejection_forward, _hooks_installed

    if _hooks_installed:
        return

    runner_cls = gpu_model_runner_module.GPUModelRunner
    original_sample = runner_cls._sample
    original_rejection_sample = rejection_sampler_module.rejection_sample
    original_forward = rejection_sampler_module.RejectionSampler.forward

    runner_cls._sample = _make_sample_wrapper(original_sample)
    rejection_sampler_module.rejection_sample = _make_rejection_sample_wrapper(
        original_rejection_sample
    )
    rejection_sampler_module.RejectionSampler.forward = _make_rejection_forward_wrapper(
        original_forward
    )

    _original_sample = original_sample
    _original_rejection_sample = original_rejection_sample
    _original_rejection_forward = original_forward
    _hooks_installed = True
    _write_heartbeat("hooks installed")


def install_hooks() -> None:
    """Install wrappers once vLLM modules are importable."""
    global _installing
    if _hooks_installed or _installing:
        return
    _installing = True
    try:
        from vllm.v1.sample import rejection_sampler
        from vllm.v1.worker import gpu_model_runner
    except ImportError as exc:
        _installing = False
        raise RuntimeError(
            "B70 P0 tracer requires vLLM; install the pinned image/source before "
            "calling install_hooks()"
        ) from exc
    try:
        _install_hooks_impl(gpu_model_runner, rejection_sampler)
    finally:
        _installing = False


def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if not _hooks_installed and (
        name.startswith("vllm")
        or (level != 0 and fromlist)
    ):
        try:
            if "vllm.v1.worker.gpu_model_runner" in sys.modules and (
                "vllm.v1.sample.rejection_sampler" in sys.modules
            ):
                install_hooks()
        except Exception as exc:  # noqa: BLE001
            print(f"B70 P0 tracer import-hook install failed: {exc}", file=sys.stderr, flush=True)
    return module


def install_import_hook() -> None:
    """Patch vLLM when EngineCore imports it, not during sitecustomize."""
    global _import_hook_installed
    if _import_hook_installed:
        return
    builtins.__import__ = _hooked_import
    _import_hook_installed = True
    _write_heartbeat("import hook armed")


def uninstall_hooks() -> None:
    """Remove installed wrappers and restore original callables."""
    global _original_sample, _original_rejection_sample, _original_rejection_forward
    global _hooks_installed, _import_hook_installed

    if _import_hook_installed:
        builtins.__import__ = _original_import
        _import_hook_installed = False

    if not _hooks_installed:
        return

    try:
        from vllm.v1.sample import rejection_sampler
        from vllm.v1.worker import gpu_model_runner
    except ImportError:
        _original_sample = None
        _original_rejection_sample = None
        _original_rejection_forward = None
        _hooks_installed = False
        return

    if _original_sample is not None:
        gpu_model_runner.GPUModelRunner._sample = _original_sample
    if _original_rejection_sample is not None:
        rejection_sampler.rejection_sample = _original_rejection_sample
    if _original_rejection_forward is not None:
        rejection_sampler.RejectionSampler.forward = _original_rejection_forward

    _original_sample = None
    _original_rejection_sample = None
    _original_rejection_forward = None
    _hooks_installed = False


@contextmanager
def testing_sample_context(context: SampleContext) -> Iterator[None]:
    """Test helper to activate tracer context without a real ``_sample`` call."""
    token = _active_sample_context.set(context)
    try:
        yield
    finally:
        _active_sample_context.reset(token)
