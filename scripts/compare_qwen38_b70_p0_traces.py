#!/usr/bin/env python3
"""Compare two Qwen B70 P0 JSONL traces offline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

JOIN_KEYS = ("request_id", "step", "context_len")
COMPARE_FIELDS = (
    "drafted_ids",
    "target_ids",
    "accepted_prefix_len",
    "bonus_token",
    "num_accepted_tokens",
)


def _load_trace(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if "drafted_ids" not in event:
                continue
            events.append(event)
    return events


def _index_events(events: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        key = tuple(event.get(field) for field in JOIN_KEYS)
        if None in key:
            raise ValueError(f"event missing join key fields {JOIN_KEYS}: {event!r}")
        if key in indexed:
            raise ValueError(f"duplicate join key {key}")
        indexed[key] = event
    return indexed


def _iter_joined_keys(
    left: dict[tuple[Any, ...], dict[str, Any]],
    right: dict[tuple[Any, ...], dict[str, Any]],
) -> Iterator[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]]:
    shared = sorted(set(left) & set(right), key=lambda item: (item[0], item[1], item[2]))
    for key in shared:
        yield key, left[key], right[key]


def _first_list_divergence(
    left_values: list[Any], right_values: list[Any]
) -> int | None:
    limit = max(len(left_values), len(right_values))
    for index in range(limit):
        left_value = left_values[index] if index < len(left_values) else None
        right_value = right_values[index] if index < len(right_values) else None
        if left_value != right_value:
            return index
    return None


def compare_traces(
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the first divergence summary, or ``None`` when traces match."""
    left = _index_events(left_events)
    right = _index_events(right_events)

    all_keys = sorted(set(left) | set(right), key=lambda item: (item[0], item[1], item[2]))
    for key in all_keys:
        if key not in left or key not in right:
            request_id, step, context_len = key
            return {
                "kind": "missing_join",
                "request_id": request_id,
                "step": step,
                "context_len": context_len,
                "missing_from": "left" if key not in left else "right",
            }

    for key, left_event, right_event in _iter_joined_keys(left, right):
        request_id, step, context_len = key
        for field in COMPARE_FIELDS:
            left_value = left_event.get(field)
            right_value = right_event.get(field)
            if field in {"drafted_ids", "target_ids"}:
                index = _first_list_divergence(
                    list(left_value or []),
                    list(right_value or []),
                )
                if index is not None:
                    return {
                        "kind": "field_divergence",
                        "field": field,
                        "request_id": request_id,
                        "step": step,
                        "context_len": context_len,
                        "index": index,
                        "left_value": left_value,
                        "right_value": right_value,
                        "left_drafted_ids": left_event.get("drafted_ids"),
                        "right_drafted_ids": right_event.get("drafted_ids"),
                        "left_target_ids": left_event.get("target_ids"),
                        "right_target_ids": right_event.get("target_ids"),
                    }
            elif left_value != right_value:
                return {
                    "kind": "field_divergence",
                    "field": field,
                    "request_id": request_id,
                    "step": step,
                    "context_len": context_len,
                    "index": None,
                    "left_value": left_value,
                    "right_value": right_value,
                    "left_drafted_ids": left_event.get("drafted_ids"),
                    "right_drafted_ids": right_event.get("drafted_ids"),
                    "left_target_ids": left_event.get("target_ids"),
                    "right_target_ids": right_event.get("target_ids"),
                }
    return None


def format_divergence(result: dict[str, Any]) -> str:
    if result["kind"] == "missing_join":
        return (
            f"missing join at request_id={result['request_id']!r} "
            f"step={result['step']} context_len={result['context_len']} "
            f"({result['missing_from']} trace)"
        )
    drafted_left = result.get("left_drafted_ids")
    drafted_right = result.get("right_drafted_ids")
    target_left = result.get("left_target_ids")
    target_right = result.get("right_target_ids")
    index = result.get("index")
    index_text = "n/a" if index is None else str(index)
    return (
        f"divergence at step={result['step']} context_len={result['context_len']} "
        f"index={index_text} field={result['field']!r} "
        f"left={result['left_value']!r} right={result['right_value']!r} "
        f"draft_left={drafted_left!r} draft_right={drafted_right!r} "
        f"target_left={target_left!r} target_right={target_right!r}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="reference JSONL trace")
    parser.add_argument("right", type=Path, help="candidate JSONL trace")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path to write the first divergence JSON summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        left_events = _load_trace(args.left)
        right_events = _load_trace(args.right)
        divergence = compare_traces(left_events, right_events)
    except (OSError, ValueError) as exc:
        print(f"compare failed: {exc}", file=sys.stderr)
        return 2

    if divergence is None:
        print("traces match on required fields")
        return 0

    message = format_divergence(divergence)
    print(message)
    if args.output is not None:
        args.output.write_text(
            json.dumps(divergence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
