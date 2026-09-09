#!/usr/bin/env python3
"""Cold long-context sweep for an already-running OpenAI-compatible Qwen server.

The cell deliberately owns no server lifecycle.  It renders Qwen's actual
non-thinking chat framing through ``/tokenize``, inserts deterministic,
engineering-oriented content, and sends the resulting token IDs through the
streaming ``/v1/completions`` API.  All network evidence is retained under
``--out`` so an unsuccessful run is still useful evidence.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "qwen38"
DEFAULT_LENGTHS = (512, 8192, 16384, 32768, 65536, 120000, 160000)
OUTPUT_TOKENS = 128
SEED = 42
TEMPERATURE = 0
IM_END_ID = 248046
DEFAULT_TIMEOUT_S = 1800.0
REQUIRED_MEASUREMENTS = 6


class BenchmarkError(RuntimeError):
    """An expected benchmark/setup failure that must be retained in artifacts."""


class APIError(BenchmarkError):
    """An HTTP or malformed-response error from the public API."""

    def __init__(
        self,
        path: str,
        message: str,
        *,
        status: int | None = None,
        body: bytes | str = b"",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.status = status
        self.body = body if isinstance(body, bytes) else body.encode("utf-8", "replace")
        self.headers = dict(headers or {})


class ServerLost(APIError):
    """The running service can no longer be reached."""


class UnhealthyServer(BenchmarkError):
    """The service responded as unhealthy while a run was in progress."""


class HTTPResponse:
    def __init__(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
        self.status = status
        self.headers = dict(headers)
        self.body = body

    def json(self, path: str) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIError(path, f"invalid JSON response: {exc}", status=self.status, body=self.body,
                           headers=self.headers) from exc


class PublicAPI:
    """Small stdlib-only transport for the public HTTP boundary."""

    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> urllib.request.Request:
        if not path.startswith("/"):
            path = "/" + path
        body = None if payload is None else json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST" if payload is not None else "GET",
        )

    def get(self, path: str) -> HTTPResponse:
        return self._open(path, None)

    def post(self, path: str, payload: Mapping[str, Any]) -> HTTPResponse:
        return self._open(path, payload)

    def _open(self, path: str, payload: Mapping[str, Any] | None) -> HTTPResponse:
        request = self._request(path, payload)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return HTTPResponse(response.status, response.headers, response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read()
            raise APIError(path, f"HTTP {exc.code} from {path}", status=exc.code, body=body,
                           headers=exc.headers) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServerLost(path, f"server connection failed for {path}: {exc}") from exc

    def stream(self, path: str, payload: Mapping[str, Any]) -> Iterable[tuple[float, bytes]]:
        """Yield ``(monotonic_timestamp, raw_line)`` for a streaming response."""
        request = self._request(path, payload)
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            raise APIError(path, f"HTTP {exc.code} from {path}", status=exc.code, body=body,
                           headers=exc.headers) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServerLost(path, f"server connection failed for {path}: {exc}") from exc

        try:
            with response:
                for line in response:
                    yield time.monotonic(), line
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServerLost(path, f"stream connection failed for {path}: {exc}") from exc


class ArtifactStore:
    """Write only beneath the run's newly-created output directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)

    def path(self, relative: str | Path) -> Path:
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def write_json(self, relative: str | Path, value: Any) -> Path:
        destination = self.path(relative)
        destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    def write_text(self, relative: str | Path, value: str) -> Path:
        destination = self.path(relative)
        destination.write_text(value, encoding="utf-8")
        return destination

    def write_bytes(self, relative: str | Path, value: bytes) -> Path:
        destination = self.path(relative)
        destination.write_bytes(value)
        return destination


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def jsonable_error(error: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, APIError):
        result.update({
            "path": error.path,
            "status": error.status,
            "body": error.body.decode("utf-8", "replace"),
            "headers": error.headers,
        })
    return result


def save_http_json(store: ArtifactStore, stem: str, response: HTTPResponse) -> Any:
    """Retain both the response bytes and a convenient parsed JSON copy."""
    store.write_bytes(stem + ".raw", response.body)
    value = response.json(stem)
    store.write_json(stem + ".json", value)
    return value


def recursive_values(value: Any, wanted_keys: set[str]) -> list[Any]:
    """Find exact normalized keys in a small JSON response."""
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in wanted_keys:
                found.append(child)
            found.extend(recursive_values(child, wanted_keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_values(child, wanted_keys))
    return found


def extract_model_and_max_len(models: Mapping[str, Any], model: str) -> tuple[Mapping[str, Any], int]:
    entries = models.get("data") if isinstance(models, Mapping) else None
    if not isinstance(entries, list):
        raise BenchmarkError("/v1/models response has no data list")
    selected: Mapping[str, Any] | None = None
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("id") == model:
            selected = entry
            break
    if selected is None:
        raise BenchmarkError(f"requested model {model!r} is not present in /v1/models")

    values = recursive_values(selected, {"max_model_len", "max_context_len", "max_context_length"})
    if not values:
        values = recursive_values(models, {"max_model_len", "max_context_len", "max_context_length"})
    numeric = []
    for value in values:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            continue
        if candidate > 0:
            numeric.append(candidate)
    if not numeric:
        raise BenchmarkError(f"could not find a positive max_model_len for model {model!r}")
    return selected, numeric[0]


def select_lengths(requested: Sequence[int], configured_max: int) -> tuple[list[int], int]:
    """Add the standard near-limit point and return one ascending sweep."""
    validate_lengths(requested)
    if configured_max <= OUTPUT_TOKENS:
        raise BenchmarkError(f"configured max_model_len={configured_max} leaves no room for output")
    near_limit = 190000 if configured_max > 200000 else configured_max - OUTPUT_TOKENS
    if near_limit <= 0:
        raise BenchmarkError(f"invalid near-limit length derived from max_model_len={configured_max}")
    return sorted(set(int(length) for length in requested) | {near_limit}), near_limit


def validate_lengths(lengths: Sequence[int]) -> None:
    if not lengths:
        raise BenchmarkError("at least one context length is required")
    for length in lengths:
        if isinstance(length, bool) or int(length) != length or int(length) <= 0:
            raise BenchmarkError(f"context lengths must be positive integers: {length!r}")


def classify_length(requested_length: int, configured_max: int, output_tokens: int = OUTPUT_TOKENS) -> dict[str, Any]:
    """Classify capacity without making a fake measurement.

    A prompt at ``max_model_len`` cannot also request the standard 128 output
    tokens, so the effective supported boundary is ``max_model_len - 128``.
    """
    if requested_length > configured_max:
        return {
            "supported": False,
            "reason": "requested_prompt_exceeds_max_model_len",
            "requested_length": requested_length,
            "configured_max_model_len": configured_max,
            "output_tokens": output_tokens,
        }
    if requested_length + output_tokens > configured_max:
        return {
            "supported": False,
            "reason": "prompt_plus_output_exceeds_max_model_len",
            "requested_length": requested_length,
            "configured_max_model_len": configured_max,
            "output_tokens": output_tokens,
        }
    return {
        "supported": True,
        "reason": "within_prompt_and_output_capacity",
        "requested_length": requested_length,
        "configured_max_model_len": configured_max,
        "output_tokens": output_tokens,
    }


def inclusive_quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    """Return median, Q1, Q3 using the inclusive quartile convention."""
    if not values:
        raise ValueError("cannot calculate quartiles for an empty sequence")
    ordered = sorted(float(value) for value in values)
    median = statistics.median(ordered)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 3
    return median, quartiles[0], quartiles[2]


def inclusive_iqr(values: Sequence[float]) -> float:
    """Return Q3-Q1 with the inclusive quartile method."""
    _, q1, q3 = inclusive_quartiles(values)
    return q3 - q1


def summarize_measurements(rows: Sequence[Mapping[str, Any]], required: int = REQUIRED_MEASUREMENTS) -> dict[str, Any]:
    """Summarize exactly the valid measured rows; warmups never enter this set."""
    measured = [row for row in rows if row.get("kind") == "measured"]
    valid = [row for row in measured if row.get("valid") is True]
    invalid = [row for row in measured if row.get("valid") is not True]
    summary: dict[str, Any] = {
        "required_valid_measurements": required,
        "measured_count": len(measured),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "complete": len(valid) == required,
        "warmups_excluded": True,
        "statistics": "median and inclusive IQR over six valid measured rows only",
        "median": {},
        "iqr_inclusive": {},
    }
    if len(valid) != required:
        return summary

    metric_keys = (
        "total_time_s",
        "ttft_s",
        "decode_elapsed_s",
        "decode_tps_post_first",
        "e2e_tps",
        "prefill_proxy_input_tps",
    )
    for key in metric_keys:
        values = [float(row[key]) for row in valid if row.get(key) is not None]
        if len(values) != required:
            summary["complete"] = False
            continue
        median, q1, q3 = inclusive_quartiles(values)
        summary["median"][key] = median
        summary["iqr_inclusive"][key] = q3 - q1
    return summary


def validate_token_counts(usage: Mapping[str, Any] | None, requested_length: int,
                          output_tokens: int = OUTPUT_TOKENS) -> dict[str, Any]:
    """Validate the two hard count contracts for a successful stream."""
    usage = usage or {}
    issues: list[str] = []
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens != requested_length:
        issues.append(f"prompt_tokens={prompt_tokens!r} != requested_length={requested_length}")
    if completion_tokens != output_tokens:
        issues.append(f"completion_tokens={completion_tokens!r} != output_tokens={output_tokens}")
    return {
        "valid": not issues,
        "issues": issues,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def deterministic_nonce(length: int, trial: str | int) -> str:
    material = f"qwen38-cold-long-context-v1|length={length}|trial={trial}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def content_segments(length: int, trial: str | int) -> tuple[str, str, str, str]:
    """Return deterministic coherent source segments and their nonce."""
    nonce = deterministic_nonce(length, trial)
    prefix = (
        "Deployment note for the Qwen long-context cold sweep.\n"
        f"Run nonce {nonce} identifies context length {length} and trial {trial}; "
        "it is deterministic so baseline and candidate receive the same text.\n"
        "The service validates inputs, bounds queue sizes, retries only idempotent "
        "operations, records failures, and tests rollback behavior.\n"
    )
    body = (
        "Engineering record: keep configuration changes reviewable, capture the "
        "active model and cache policy, monitor admission and memory pressure, "
        "and preserve enough operational detail for a second engineer to reproduce "
        "the deployment. Do not infer capacity from a successful short request.\n"
    )
    tail = (
        "Question for the deployment review: state the recovery action, verify the "
        "health boundary, and retain the exact request evidence.\n"
    )
    return prefix, body, tail, nonce


def repeat_tokens(tokens: Sequence[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if not tokens:
        raise BenchmarkError("content tokenizer returned no body tokens")
    repeats, remainder = divmod(count, len(tokens))
    return list(tokens) * repeats + list(tokens[:remainder])


def compose_content_tokens(content_length: int, prefix_tokens: Sequence[int], body_tokens: Sequence[int],
                            tail_tokens: Sequence[int]) -> list[int]:
    """Fill a content region exactly while retaining coherent tokenized text."""
    if content_length < 0:
        raise BenchmarkError("chat template framing is longer than requested prompt")
    if content_length == 0:
        return []
    if len(prefix_tokens) >= content_length:
        return list(prefix_tokens[:content_length])

    # Keep the question tail at the end whenever the requested point has room.
    available_after_prefix = content_length - len(prefix_tokens)
    tail_count = min(len(tail_tokens), available_after_prefix)
    body_count = available_after_prefix - tail_count
    result = list(prefix_tokens)
    result.extend(repeat_tokens(body_tokens, body_count))
    result.extend(tail_tokens[:tail_count])
    if len(result) != content_length:
        raise AssertionError(f"content fill produced {len(result)} tokens, wanted {content_length}")
    return result


def compose_prompt(template_tokens: Sequence[int], prefix_tokens: Sequence[int], body_tokens: Sequence[int],
                   tail_tokens: Sequence[int], requested_length: int, im_end_id: int = IM_END_ID) -> list[int]:
    """Preserve the actual chat header/footer and produce an exact token count."""
    try:
        end_user = max(index for index, token in enumerate(template_tokens) if token == im_end_id)
    except ValueError as exc:
        raise BenchmarkError(f"template has no Qwen im_end token {im_end_id}") from exc
    header = list(template_tokens[:end_user])
    footer = list(template_tokens[end_user:])
    content_length = requested_length - len(header) - len(footer)
    content = compose_content_tokens(content_length, prefix_tokens, body_tokens, tail_tokens)
    prompt = header + content + footer
    if len(prompt) != requested_length:
        raise AssertionError(f"rendered prompt has {len(prompt)} tokens, wanted {requested_length}")
    return prompt


def _decode_sse_line(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace").rstrip("\r\n")
    return raw.rstrip("\r\n")


def _choice_text(chunk: Mapping[str, Any]) -> str:
    text_parts: list[str] = []
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        text = choice.get("text")
        if isinstance(text, str):
            text_parts.append(text)
            continue
        delta = choice.get("delta")
        if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
            text_parts.append(delta["content"])
    return "".join(text_parts)


def parse_sse_events(events: Iterable[tuple[float, bytes | str]]) -> dict[str, Any]:
    """Parse OpenAI SSE data while retaining burst/chunk timing semantics."""
    parsed: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    usage: Mapping[str, Any] | None = None
    finish_reason: Any = None
    first_nonempty: float | None = None
    done_at: float | None = None
    text_parts: list[str] = []

    for monotonic_s, raw in events:
        line = _decode_sse_line(raw)
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done_at = monotonic_s
            parsed.append({"monotonic_s": monotonic_s, "raw": line, "done": True})
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            parsed.append({"monotonic_s": monotonic_s, "raw": line, "parse_error": str(exc)})
            continue
        if not isinstance(chunk, Mapping):
            parse_errors.append("SSE JSON data was not an object")
            parsed.append({"monotonic_s": monotonic_s, "raw": line, "parse_error": "not an object"})
            continue
        text = _choice_text(chunk)
        if text:
            text_parts.append(text)
            if first_nonempty is None:
                first_nonempty = monotonic_s
        if isinstance(chunk.get("usage"), Mapping):
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, Mapping) and choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")
        parsed.append({
            "monotonic_s": monotonic_s,
            "raw": line,
            "text": text,
            "usage": chunk.get("usage"),
            "finish_reason": finish_reason,
        })

    return {
        "events": parsed,
        "text": "".join(text_parts),
        "usage": dict(usage) if usage is not None else None,
        "finish_reason": finish_reason,
        "first_nonempty_monotonic": first_nonempty,
        "done_monotonic": done_at,
        "nonempty_chunk_count": sum(1 for event in parsed if event.get("text")),
        "parse_errors": parse_errors,
    }


def compute_stream_metrics(parsed: Mapping[str, Any], request_started_monotonic: float,
                           stream_end_monotonic: float, prompt_tokens: int | None = None) -> dict[str, Any]:
    """Compute TTFT, post-first burst rate, E2E rate, and the prefill proxy."""
    usage = parsed.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    output_tokens = usage.get("completion_tokens")
    ttft_start = parsed.get("first_nonempty_monotonic")
    total_time = max(0.0, stream_end_monotonic - request_started_monotonic)
    result: dict[str, Any] = {
        "total_time_s": total_time,
        "ttft_s": None,
        "decode_elapsed_s": None,
        "decode_tps_post_first": None,
        "e2e_tps": None,
        "prefill_proxy_input_tps": None,
        "prefill_proxy_definition": "input_tokens / TTFT; TTFT is a prefill proxy, not a direct prefill measurement",
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else usage.get("prompt_tokens"),
        "first_nonempty_monotonic": ttft_start,
        "stream_end_monotonic": stream_end_monotonic,
        "rate_formula": "(output_tokens - 1) / (stream_end - first_nonempty); speculative bursts are not per-token ITL",
    }
    if isinstance(ttft_start, (int, float)):
        ttft = max(0.0, float(ttft_start) - request_started_monotonic)
        decode_elapsed = max(0.0, stream_end_monotonic - float(ttft_start))
        result["ttft_s"] = ttft
        result["decode_elapsed_s"] = decode_elapsed
        if isinstance(output_tokens, (int, float)):
            if decode_elapsed > 0:
                result["decode_tps_post_first"] = max(0.0, float(output_tokens) - 1.0) / decode_elapsed
            elif output_tokens <= 1:
                result["decode_tps_post_first"] = 0.0
            if total_time > 0:
                result["e2e_tps"] = float(output_tokens) / total_time
            input_count = result["prompt_tokens"]
            if isinstance(input_count, (int, float)) and ttft > 0:
                result["prefill_proxy_input_tps"] = float(input_count) / ttft
    return result


def completion_payload(model: str, prompt_tokens: Sequence[int]) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": list(prompt_tokens),
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def metric_spec_lines(body: bytes) -> list[str]:
    """Return raw lines carrying speculative/position counters for easy review."""
    text = body.decode("utf-8", "replace")
    needles = ("spec", "draft", "accept", "position")
    return [line for line in text.splitlines() if not line.startswith("#") and any(needle in line.lower() for needle in needles)]


def _attempt_name(kind: str, index: int) -> str:
    return f"{kind}-{index:02d}"


class LongContextBenchmark:
    def __init__(self, args: argparse.Namespace, store: ArtifactStore, client: PublicAPI) -> None:
        self.args = args
        self.store = store
        self.client = client
        self.errors: list[dict[str, Any]] = []
        self.models: Mapping[str, Any] | None = None
        self.model_record: Mapping[str, Any] | None = None
        self.configured_max_model_len: int | None = None
        self.template_tokens: list[int] | None = None
        self.body_tokens: list[int] | None = None
        self.prefix_cache: dict[str, Any] = {}
        self.point_results: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        self._capture_health("health-start.json")
        self._capture_models()
        self._validate_prefix_cache()
        assert self.configured_max_model_len is not None
        lengths, near_limit = select_lengths(self.args.lengths, self.configured_max_model_len)
        self._write_protocol(lengths, near_limit)
        self._prepare_renderer()

        for length in lengths:
            classification = classify_length(length, self.configured_max_model_len)
            if classification["supported"]:
                point = self._run_supported_point(length, classification)
            else:
                point = self._run_unsupported_point(length, classification)
            self.point_results.append(point)

        self._capture_health("health-end.json")
        failed = [point for point in self.point_results if point.get("status") == "failed"]
        status = "completed_with_failures" if failed else "completed"
        summary = {
            "status": status,
            "base_url": self.args.base_url,
            "model": self.args.model,
            "configured_max_model_len": self.configured_max_model_len,
            "near_limit_length": near_limit,
            "prefix_cache": self.prefix_cache,
            "protocol": {
                "output_tokens": OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "ignore_eos": True,
                "stream_options": {"include_usage": True},
                "warmups_per_supported_point": 1,
                "measured_trials_per_supported_point": REQUIRED_MEASUREMENTS,
                "unsupported_points_get_one_real_probe": True,
                "statistics": "median and inclusive IQR over six valid measured rows only",
            },
            "points": self.point_results,
            "errors": self.errors,
        }
        self.store.write_json("summary.json", summary)
        return summary

    def _write_protocol(self, lengths: Sequence[int], near_limit: int) -> None:
        self.store.write_json("protocol.json", {
            "standard": "B70 + Qwen 3.8 cold long-context sweep",
            "requested_lengths": list(self.args.lengths),
            "selected_lengths": list(lengths),
            "near_limit_length": near_limit,
            "configured_max_model_len": self.configured_max_model_len,
            "prompt_rendering": {
                "template_endpoint": "/tokenize",
                "empty_chat_messages": [{"role": "user", "content": ""}],
                "chat_template_kwargs": {"enable_thinking": False},
                "qwen_im_end_id": IM_END_ID,
                "completion_endpoint": "/v1/completions",
                "prompt_transport": "rendered token IDs",
            },
            "nonce": "sha256(qwen38-cold-long-context-v1|length|trial)[:24]; same across baseline/candidate",
        })

    def _capture_health(self, filename: str) -> dict[str, Any]:
        try:
            response = self.client.get("/health")
            result = {"status": response.status, "headers": response.headers}
            self.store.write_bytes(filename.replace(".json", ".raw"), response.body)
            self.store.write_json(filename, result)
            if response.status >= 500:
                raise UnhealthyServer(f"/health returned HTTP {response.status}")
            return result
        except APIError as error:
            if isinstance(error, ServerLost):
                raise
            self.store.write_bytes(filename.replace(".json", ".error-body"), error.body)
            result = {"status": error.status, "error": jsonable_error(error)}
            self.store.write_json(filename, result)
            if error.status is not None and error.status >= 500:
                raise UnhealthyServer(f"/health returned HTTP {error.status}") from error
            # Some compatible servers do not expose /health; /v1/models remains
            # the authoritative startup reachability check.
            return result

    def _capture_models(self) -> None:
        try:
            response = self.client.get("/v1/models")
            models = save_http_json(self.store, "v1-models", response)
        except APIError as error:
            self._record_fatal(error, "v1-models-error.json")
            raise
        self.models = models
        self.model_record, self.configured_max_model_len = extract_model_and_max_len(models, self.args.model)
        self.store.write_json("model-selection.json", {
            "requested_model": self.args.model,
            "selected_model": self.model_record,
            "configured_max_model_len": self.configured_max_model_len,
        })

    def _validate_prefix_cache(self) -> None:
        try:
            response = self.client.get("/server_info")
            server_info = save_http_json(self.store, "server-info", response)
            values = recursive_values(server_info, {"enable_prefix_caching", "enable_prefix_cache"})
            bool_values = [value for value in values if isinstance(value, bool)]
            if any(value is True for value in bool_values):
                raise BenchmarkError("server_info reports prefix caching enabled; cold sweep requires it disabled")
            if any(value is False for value in bool_values):
                self.prefix_cache = {
                    "mode": "verified_disabled",
                    "source": "/server_info",
                    "server_info_accessible": True,
                    "reported_values": bool_values,
                }
            else:
                if not self.args.confirm_prefix_cache_disabled:
                    raise BenchmarkError(
                        "/server_info was accessible but did not expose enable_prefix_caching; "
                        "pass --confirm-prefix-cache-disabled for a user assertion"
                    )
                self.prefix_cache = {
                    "mode": "user_asserted_disabled",
                    "source": "--confirm-prefix-cache-disabled",
                    "server_info_accessible": True,
                    "reported_values": bool_values,
                }
        except APIError as error:
            if isinstance(error, ServerLost):
                raise
            self.store.write_bytes("server-info.error-body", error.body)
            if error.status not in (404, 405, 501):
                raise
            if not self.args.confirm_prefix_cache_disabled:
                raise BenchmarkError(
                    "/server_info is unavailable; pass --confirm-prefix-cache-disabled "
                    "to record the required user assertion"
                ) from error
            self.prefix_cache = {
                "mode": "user_asserted_disabled",
                "source": "--confirm-prefix-cache-disabled",
                "server_info_accessible": False,
                "server_info_status": error.status,
            }
        except BenchmarkError as error:
            self.store.write_json("prefix-cache-error.json", jsonable_error(error))
            raise
        self.store.write_json("prefix-cache.json", self.prefix_cache)

    def _tokenize(self, label: str, payload: Mapping[str, Any], artifact_dir: str = "rendering") -> list[int]:
        request_path = f"{artifact_dir}/{label}-request.json"
        response_path = f"{artifact_dir}/{label}-response"
        self.store.write_json(request_path, payload)
        try:
            response = self.client.post("/tokenize", payload)
            value = save_http_json(self.store, response_path, response)
        except APIError as error:
            self.store.write_bytes(f"{artifact_dir}/{label}-error-body", error.body)
            self.store.write_json(f"{artifact_dir}/{label}-error.json", jsonable_error(error))
            raise
        tokens = value.get("tokens") if isinstance(value, Mapping) else None
        if tokens is None and isinstance(value, Mapping):
            tokens = value.get("prompt_token_ids")
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            error = BenchmarkError(f"/tokenize {label} response has no integer token list")
            self.store.write_json(f"{artifact_dir}/{label}-error.json", jsonable_error(error))
            raise error
        return list(tokens)

    def _prepare_renderer(self) -> None:
        template_request = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": ""}],
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        self.template_tokens = self._tokenize("template", template_request)
        if not any(token == IM_END_ID for token in self.template_tokens):
            raise BenchmarkError(f"template did not contain Qwen im_end token {IM_END_ID}")
        _, body_text, _, _ = content_segments(512, "body")
        body_request = {
            "model": self.args.model,
            "prompt": body_text,
            "add_special_tokens": False,
        }
        self.body_tokens = self._tokenize("content-body", body_request)
        self.store.write_json("rendering/template-framing.json", {
            "im_end_id": IM_END_ID,
            "template_token_count": len(self.template_tokens),
            "final_im_end_index": max(index for index, token in enumerate(self.template_tokens) if token == IM_END_ID),
        })

    def _render(self, length: int, trial: str | int) -> dict[str, Any]:
        if self.template_tokens is None or self.body_tokens is None:
            raise BenchmarkError("renderer was not prepared")
        prefix_text, body_text, tail_text, nonce = content_segments(length, trial)
        attempt_dir = f"rendering/length-{length}/{_attempt_name('trial', int(trial) if isinstance(trial, int) else 0)}"
        prefix_request = {"model": self.args.model, "prompt": prefix_text, "add_special_tokens": False}
        tail_request = {"model": self.args.model, "prompt": tail_text, "add_special_tokens": False}
        prefix_tokens = self._tokenize("content-prefix", prefix_request, attempt_dir)
        # The body is fixed and retained once under rendering/content-body.
        tail_tokens = self._tokenize("content-tail", tail_request, attempt_dir)
        prompt = compose_prompt(self.template_tokens, prefix_tokens, self.body_tokens, tail_tokens, length)
        result = {
            "requested_length": length,
            "trial": trial,
            "nonce": nonce,
            "header_token_count": len(self.template_tokens[:max(i for i, token in enumerate(self.template_tokens) if token == IM_END_ID)]),
            "footer_token_count": len(self.template_tokens[max(i for i, token in enumerate(self.template_tokens) if token == IM_END_ID):]),
            "content_token_count": length - len(self.template_tokens[:max(i for i, token in enumerate(self.template_tokens) if token == IM_END_ID)]) - len(self.template_tokens[max(i for i, token in enumerate(self.template_tokens) if token == IM_END_ID):]),
            "prompt_token_count": len(prompt),
            "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "prompt": prompt,
            "source": {
                "prefix": prefix_text,
                "body": body_text,
                "tail": tail_text,
            },
        }
        self.store.write_json(f"{attempt_dir}/rendered-prompt.json", result)
        return result

    def _capture_metrics(self, relative_dir: str, phase: str) -> dict[str, Any]:
        started = time.monotonic()
        body = b""
        try:
            response = self.client.get("/metrics")
            body = response.body
            status = response.status
            headers = response.headers
            error = None
        except APIError as exc:
            if isinstance(exc, ServerLost):
                raise
            body = exc.body
            status = exc.status
            headers = exc.headers
            error = jsonable_error(exc)
        ended = time.monotonic()
        raw_path = f"{relative_dir}/metrics-{phase}.raw"
        self.store.write_bytes(raw_path, body)
        result: dict[str, Any] = {
            "path": "/metrics",
            "phase": phase,
            "status": status,
            "started_monotonic": started,
            "ended_monotonic": ended,
            "raw_path": raw_path,
            "headers": headers,
            "speculative_position_counter_lines": metric_spec_lines(body),
        }
        if error is not None:
            result["error"] = error
        self.store.write_json(f"{relative_dir}/metrics-{phase}.json", result)
        return result

    def _write_sse(self, relative_dir: str, events: Sequence[tuple[float, bytes]]) -> str:
        relative = f"{relative_dir}/sse.jsonl"
        lines = []
        for monotonic_s, raw in events:
            lines.append(json.dumps({
                "monotonic_s": monotonic_s,
                "raw": _decode_sse_line(raw),
            }, ensure_ascii=False))
        self.store.write_text(relative, "\n".join(lines) + ("\n" if lines else ""))
        return relative

    def _execute_attempt(self, length: int, trial: str | int, kind: str, index: int) -> dict[str, Any]:
        rendering = self._render(length, trial)
        payload = completion_payload(self.args.model, rendering["prompt"])
        relative_dir = f"points/length-{length}/{_attempt_name(kind, index)}"
        self.store.write_json(f"{relative_dir}/request.json", payload)
        metrics_before = self._capture_metrics(relative_dir, "before")
        request_started = time.monotonic()
        self.store.write_json(f"{relative_dir}/request-meta.json", {
            "requested_length": length,
            "trial": trial,
            "kind": kind,
            "prompt_token_count": len(rendering["prompt"]),
            "nonce": rendering["nonce"],
            "request_started_monotonic": request_started,
        })
        raw_events: list[tuple[float, bytes]] = []
        try:
            for monotonic_s, raw in self.client.stream("/v1/completions", payload):
                raw_events.append((monotonic_s, raw))
            stream_end = raw_events[-1][0] if raw_events else time.monotonic()
            sse_path = self._write_sse(relative_dir, raw_events)
            parsed = parse_sse_events(raw_events)
            metrics = compute_stream_metrics(parsed, request_started, stream_end,
                                             parsed.get("usage", {}).get("prompt_tokens") if isinstance(parsed.get("usage"), Mapping) else None)
            validation = validate_token_counts(parsed.get("usage"), length)
            valid = validation["valid"] and not parsed["parse_errors"] and parsed.get("first_nonempty_monotonic") is not None
            result: dict[str, Any] = {
                "status": "ok" if valid else "failed",
                "valid": valid,
                "kind": kind,
                "trial": trial,
                "requested_length": length,
                "request_path": f"{relative_dir}/request.json",
                "sse_path": sse_path,
                "metrics_before": metrics_before,
                "stream": parsed,
                "timing": metrics,
                "total_time_s": metrics.get("total_time_s"),
                "ttft_s": metrics.get("ttft_s"),
                "decode_elapsed_s": metrics.get("decode_elapsed_s"),
                "decode_tps_post_first": metrics.get("decode_tps_post_first"),
                "e2e_tps": metrics.get("e2e_tps"),
                "prefill_proxy_input_tps": metrics.get("prefill_proxy_input_tps"),
                "validation": validation,
            }
        except APIError as error:
            if isinstance(error, ServerLost):
                # The request and any partial stream are written before aborting the run.
                if raw_events:
                    self._write_sse(relative_dir, raw_events)
                self.store.write_json(f"{relative_dir}/error.json", jsonable_error(error))
                self.store.write_bytes(f"{relative_dir}/error-body", error.body)
                raise
            result = {
                "status": "http_error",
                "valid": False,
                "kind": kind,
                "trial": trial,
                "requested_length": length,
                "request_path": f"{relative_dir}/request.json",
                "error": jsonable_error(error),
            }
            self.store.write_json(f"{relative_dir}/error.json", result)
            self.store.write_bytes(f"{relative_dir}/error-body", error.body)
            if error.status is not None and error.status >= 500:
                self._ensure_healthy_after_error()
        finally:
            metrics_after = self._capture_metrics(relative_dir, "after")
            if "result" in locals():
                result["metrics_after"] = metrics_after
                self.store.write_json(f"{relative_dir}/result.json", result)
        return result

    def _ensure_healthy_after_error(self) -> None:
        health = self._capture_health("health-after-error.json")
        if health.get("status", 200) >= 500:
            raise UnhealthyServer(f"server remained unhealthy after request error: {health}")

    def _run_supported_point(self, length: int, classification: Mapping[str, Any]) -> dict[str, Any]:
        point: dict[str, Any] = {
            "requested_length": length,
            "configured_max_model_len": self.configured_max_model_len,
            "classification": dict(classification),
            "status": "failed",
            "warmup": None,
            "measurements": [],
        }
        try:
            warmup = self._execute_attempt(length, "warmup", "warmup", 0)
            point["warmup"] = warmup
            if not warmup.get("valid"):
                point["failure"] = "warmup_failed"
            else:
                for index in range(1, REQUIRED_MEASUREMENTS + 1):
                    measurement = self._execute_attempt(length, index, "measured", index)
                    point["measurements"].append(measurement)
                point["summary"] = summarize_measurements(point["measurements"])
                point["status"] = "complete" if point["summary"]["complete"] else "failed"
                if point["status"] == "failed":
                    point["failure"] = "one_or_more_measured_rows_invalid"
        except (APIError, BenchmarkError) as error:
            if isinstance(error, ServerLost) or isinstance(error, UnhealthyServer):
                raise
            point["failure"] = jsonable_error(error)
            self.errors.append({"point": length, **jsonable_error(error)})
        return point

    def _run_unsupported_point(self, length: int, classification: Mapping[str, Any]) -> dict[str, Any]:
        point: dict[str, Any] = {
            "requested_length": length,
            "configured_max_model_len": self.configured_max_model_len,
            "classification": dict(classification),
            "status": "unsupported",
            "warmup": None,
            "measurements": [],
            "unsupported_probe": None,
        }
        try:
            probe = self._execute_attempt(length, "unsupported", "unsupported", 0)
            probe["expected_http_rejection"] = probe.get("status") == "http_error"
            point["unsupported_probe"] = probe
            if not probe["expected_http_rejection"]:
                point["probe_note"] = "configured capacity classified this point unsupported; server did not return HTTP error"
        except (APIError, BenchmarkError) as error:
            if isinstance(error, ServerLost) or isinstance(error, UnhealthyServer):
                raise
            point["unsupported_probe"] = {"status": "probe_failed", "error": jsonable_error(error)}
            self.errors.append({"point": length, "unsupported_probe": True, **jsonable_error(error)})
        return point

    def _record_fatal(self, error: BaseException, filename: str) -> None:
        self.errors.append(jsonable_error(error))
        self.store.write_json(filename, jsonable_error(error))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"running OpenAI-compatible server (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--out", type=Path, required=True, help="new directory for raw benchmark artifacts")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS),
                        help="prompt lengths to sweep; the near-limit point is added automatically")
    parser.add_argument("--confirm-prefix-cache-disabled", action="store_true",
                        help="record a user assertion when /server_info cannot verify prefix caching is disabled")
    args = parser.parse_args(argv)
    try:
        validate_lengths(args.lengths)
    except BenchmarkError as error:
        parser.error(str(error))
    if args.out.exists():
        parser.error(f"--out must name a new directory: {args.out}")
    return args


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(args.out)
    run_record: dict[str, Any] = {
        "status": "running",
        "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]),
        "started_utc": utc_now(),
        "base_url": args.base_url,
        "model": args.model,
        "requested_lengths": list(args.lengths),
        "output_tokens": OUTPUT_TOKENS,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    store.write_json("run.json", run_record)
    benchmark = LongContextBenchmark(args, store, PublicAPI(args.base_url))
    exit_code = 0
    try:
        summary = benchmark.run()
        exit_code = 1 if summary["status"] != "completed" else 0
        run_record.update({
            "status": summary["status"],
            "finished_utc": utc_now(),
            "summary_path": "summary.json",
            "error_count": len(summary.get("errors", [])),
        })
    except BaseException as error:
        exit_code = 1
        benchmark.errors.append(jsonable_error(error))
        store.write_json("fatal-error.json", jsonable_error(error))
        run_record.update({
            "status": "aborted",
            "finished_utc": utc_now(),
            "error_count": len(benchmark.errors),
        })
        # Retain partial point evidence even when a lost server aborts the sweep.
        store.write_json("summary.json", {
            "status": "aborted",
            "configured_max_model_len": benchmark.configured_max_model_len,
            "prefix_cache": benchmark.prefix_cache,
            "points": benchmark.point_results,
            "errors": benchmark.errors,
        })
    finally:
        store.write_json("run.json", run_record)
    return exit_code


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
