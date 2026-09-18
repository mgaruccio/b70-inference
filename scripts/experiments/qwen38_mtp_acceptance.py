#!/usr/bin/env python3
"""Bounded, evaluation-only joint-prefix scoring for native Qwen MTP4.

This diagnostic consumes existing native captures and MTP-only BF16 exports.  It
never trains, exports, or modifies a checkpoint.  The frozen LM head is the
runtime-effective RTN head used by the existing native training helpers; the
post-RTN dense stage is the primary comparison, but BF16 export scores are
reported alongside it.

Run in the pinned inference-host ML environment, not in the interactive Pi
runtime.  Captures and prompts may contain private development data; the JSON
report stores token and prompt identities as digests rather than raw values.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
from typing import Any

try:
    import qwen38_train_mtp as trainer
except ModuleNotFoundError:  # Imported from the repository root by focused tests.
    import importlib.util

    _TRAINER_PATH = Path(__file__).with_name("qwen38_train_mtp.py")
    _SPEC = importlib.util.spec_from_file_location("qwen38_train_mtp", _TRAINER_PATH)
    if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard.
        raise ImportError(f"Cannot load {_TRAINER_PATH}")
    trainer = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(trainer)


DEPTH = 4
SEED = 42
DEFAULT_ROOTS = 64
DEFAULT_LOGITS_CHUNK = 64
STAGES = ("BF16_export", "RTN_effective_dense")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class AcceptanceError(trainer.TrainingError):
    """Invalid diagnostic input or an unsafe evaluation-only configuration."""


def _length(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    return len(value)


def _item(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _bool_at(values: Any, index: int) -> bool:
    return bool(_item(values[index]))


def _ids_digest(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in ids:
        digest.update(struct.pack("<q", int(token)))
    return digest.hexdigest()


def _sequence_digest(prompt_id: str) -> str:
    return hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()


def _stable_seed(prompt_id: str, seed: int = SEED) -> int:
    material = f"qwen38-mtp-acceptance\0{seed}\0{prompt_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _all_mask(mask: Any, start: int, end: int) -> bool:
    if start < 0 or end > _length(mask) or start >= end:
        return False
    values = mask[start:end]
    if hasattr(values, "all"):
        return bool(values.all().item())
    return all(bool(value) for value in values)


def _terminal_failure(mask: Any, start: int, end: int) -> bool:
    """Distinguish a contiguous terminal tail from an interior mask hole."""
    if end > _length(mask):
        return True
    failed = [index for index in range(start, end) if not _bool_at(mask, index)]
    if not failed:
        return False
    first = failed[0]
    return not any(_bool_at(mask, index) for index in range(first + 1, _length(mask)))


def root_eligibility(record: Mapping[str, Any], depth: int = DEPTH) -> dict[str, Any]:
    """Return full-horizon roots and mutually exclusive exclusion counts.

    ``t`` is the aligned base row.  A four-label root consumes x[t+1] as its
    observed response boundary and scores x[t+2:t+6].  The boundary check is
    deliberately separate from the four labels: accepting t when only x[t+2]
    is supervised would admit the prompt-2 root and shift the diagnostic.
    """
    if depth != DEPTH:
        raise AcceptanceError("The native acceptance diagnostic requires depth 4")
    try:
        ids = record["input_ids"]
        mask = record["loss_mask"]
        hidden = record["target_last_hidden_states"]
        prompt_id = record["prompt_id"]
    except (KeyError, TypeError) as exc:
        raise AcceptanceError("Capture record is missing root-planning fields") from exc
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise AcceptanceError("Capture record requires a nonempty prompt_id")
    length = _length(ids)
    aligned_length = max(0, length - 2)
    observed = int(hidden.shape[0]) if hasattr(hidden, "shape") else _length(hidden)
    counts = {
        "candidate_roots": aligned_length,
        "hidden_excluded": 0,
        "terminal_excluded": 0,
        "boundary_excluded": 0,
        "masked_excluded": 0,
    }
    eligible: list[int] = []
    for root in range(aligned_length):
        if root >= observed:
            counts["hidden_excluded"] += 1
            continue
        # c_1..c_4 are x[t+2]..x[t+5].
        if root + depth + 1 >= length:
            counts["terminal_excluded"] += 1
            continue
        # x[t+1] must already be a response token, not merely the first label.
        if not _bool_at(mask, root + 1):
            counts["boundary_excluded"] += 1
            continue
        label_start, label_end = root + 2, root + depth + 2
        if not _all_mask(mask, label_start, label_end):
            if _terminal_failure(mask, label_start, label_end):
                counts["terminal_excluded"] += 1
            else:
                counts["masked_excluded"] += 1
            continue
        eligible.append(root)
    counts["eligible_roots"] = len(eligible)
    return {"eligible": eligible, "counts": counts}


def eligible_roots(record: Mapping[str, Any], depth: int = DEPTH) -> list[int]:
    """Return roots with a response boundary and a complete four-label horizon."""
    return list(root_eligibility(record, depth)["eligible"])


def select_roots(record: Mapping[str, Any], count: int = DEFAULT_ROOTS, seed: int = SEED) -> list[int]:
    """Select a stable, sorted, without-replacement root sample per sequence."""
    if type(count) is not int or not 1 <= count <= DEFAULT_ROOTS:
        raise AcceptanceError(f"roots must be an integer in 1..{DEFAULT_ROOTS}")
    if type(seed) is not int:
        raise AcceptanceError("seed must be an integer")
    roots = eligible_roots(record)
    rng = random.Random(_stable_seed(str(record["prompt_id"]), seed))
    return sorted(rng.sample(roots, min(count, len(roots))))


def _record_labels(record: Mapping[str, Any], roots: Sequence[int]) -> list[list[int]]:
    ids = record["input_ids"]
    return [[int(_item(ids[root + depth + 1])) for depth in range(1, DEPTH + 1)] for root in roots]


def make_sequence_plan(
    path: Path,
    record: Mapping[str, Any],
    *,
    family: str,
    roots: int = DEFAULT_ROOTS,
) -> dict[str, Any]:
    path = Path(path)
    eligibility = root_eligibility(record)
    selected = select_roots(record, roots, SEED)
    sequence_id = _sequence_digest(record["prompt_id"])
    labels = _record_labels(record, selected)
    labels_by_depth = [
        [row[depth - 1] for row in labels] for depth in range(1, DEPTH + 1)
    ]
    root_identities = [
        {
            "root": root,
            "identity": f"{sequence_id}:{root}",
            "label_token_ids_sha256": _ids_digest(label_row),
        }
        for root, label_row in zip(selected, labels)
    ]
    plan_material = {
        "sequence_id": sequence_id,
        "roots": selected,
        "labels_sha256_by_depth": [_ids_digest(values) for values in labels_by_depth],
    }
    plan_digest = hashlib.sha256(
        json.dumps(plan_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": path,
        "capture": path.name,
        "sequence_id": sequence_id,
        "source_group": family,
        "root_order": selected,
        "root_identities": root_identities,
        "labels_sha256_by_depth": plan_material["labels_sha256_by_depth"],
        "plan_sha256": plan_digest,
        "eligible_roots": eligibility["counts"]["eligible_roots"],
        "exclusions": eligibility["counts"],
    }


def load_prompt_families(path: Path | None) -> dict[str, str]:
    """Read only explicit id/source_group mappings; never discover prompt files."""
    if path is None:
        return {}
    path = Path(path).resolve()
    if not path.is_file():
        raise AcceptanceError(f"Prompt mapping JSONL is not a file: {path}")
    result: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise AcceptanceError(f"Cannot read prompt mapping: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AcceptanceError(f"Invalid prompt JSONL at line {line_number}") from exc
        if not isinstance(row, Mapping):
            raise AcceptanceError(f"Prompt mapping line {line_number} is not an object")
        prompt_id = row.get("id", row.get("prompt_id"))
        family = row.get("source_group")
        if (not isinstance(prompt_id, str) or not prompt_id.strip()
                or not isinstance(family, str) or not family.strip()):
            raise AcceptanceError(f"Prompt mapping line {line_number} needs id and source_group")
        if prompt_id in result:
            raise AcceptanceError(f"Duplicate prompt mapping id: {prompt_id}")
        result[prompt_id] = family
    if not result:
        raise AcceptanceError(f"Prompt mapping JSONL is empty: {path}")
    return result


def source_group(record: Mapping[str, Any], prompt_families: Mapping[str, str]) -> str:
    metadata = record.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise AcceptanceError("Capture metadata must be an object")
    captured = metadata.get("source_group", record.get("source_group"))
    mapped = prompt_families.get(record["prompt_id"])
    for value in (captured, mapped):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise AcceptanceError("source_group must be a nonempty string")
    if captured is not None and mapped is not None and captured != mapped:
        raise AcceptanceError(f"Capture/prompt source_group mismatch for {record['prompt_id']}")
    return str(captured or mapped or "unknown")


def load_sequence_plans(
    captures: Path,
    checkpoint: Any,
    *,
    prompt_families: Mapping[str, str],
    max_length: int,
    roots: int,
) -> list[dict[str, Any]]:
    captures = Path(captures).resolve()
    if not captures.is_dir():
        raise AcceptanceError(f"Captures must be an existing directory: {captures}")
    paths = sorted(captures.glob("*.pt"))
    if not paths:
        raise AcceptanceError(f"No complete-sequence .pt captures in {captures}")
    plans = []
    for path in paths:
        record = trainer.load_record(path, checkpoint.config, max_length)
        plans.append(make_sequence_plan(
            path,
            record,
            family=source_group(record, prompt_families),
            roots=roots,
        ))
    return plans


def parse_candidates(values: Sequence[str] | None) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values or ():
        if not isinstance(value, str) or "=" not in value:
            raise AcceptanceError("Each --candidate must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not _NAME.fullmatch(name) or name == "stock":
            raise AcceptanceError("Candidate names must be safe and must not be 'stock'")
        if name in seen:
            raise AcceptanceError(f"Duplicate candidate name: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise AcceptanceError(f"Candidate file is not readable: {path}")
        seen.add(name)
        result.append((name, path))
    return result


def _load_state(path: Path | None, checkpoint: Any) -> dict[str, Any]:
    if path is None:
        state = checkpoint.mtp_state()
    else:
        try:
            with trainer.runtime().safe_open(path, framework="pt", device="cpu") as handle:
                state = {key: handle.get_tensor(key) for key in handle.keys()}
        except (OSError, KeyError, RuntimeError) as exc:
            raise AcceptanceError(f"Cannot read candidate MTP export: {path}") from exc
    trainer.validate_mtp_state(state, checkpoint.shapes, stock=True)
    return state


def _new_accumulator() -> dict[str, Any]:
    return {
        "eligible_roots": 0,
        "evaluated_roots": 0,
        "depths": [
            {"loss_sum": 0.0, "tokens": 0, "correct": 0} for _ in range(DEPTH)
        ],
        "survival_counts": [0] * DEPTH,
        "length_histogram": [0] * (DEPTH + 1),
        "sum_j": 0,
    }


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise AcceptanceError(f"Nonfinite {label}")
    return value


def score_logits(hidden: Any, head: Any, labels: Any, *, chunk_tokens: int) -> dict[str, Any]:
    """Score a bounded hidden-row slice without allocating root*vocab logits."""
    torch = trainer.runtime().torch
    if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 256:
        raise AcceptanceError("logits chunk must be an integer in 1..256")
    if hidden.ndim != 2 or labels.ndim != 1 or hidden.shape[0] != labels.numel():
        raise AcceptanceError("Hidden/logit label row count mismatch")
    count = int(labels.numel())
    if count == 0:
        return {"loss_sum": 0.0, "tokens": 0, "correct": 0, "correct_by_root": []}
    total = 0.0
    correct_by_root: list[bool] = []
    with torch.no_grad():
        for offset in range(0, count, chunk_tokens):
            rows = hidden[offset:offset + chunk_tokens]
            target = labels[offset:offset + chunk_tokens]
            if not torch.isfinite(rows).all().item():
                raise AcceptanceError("Nonfinite draft hidden state")
            with trainer.autocast(hidden.device):
                logits = head(rows)
            if logits.ndim != 2 or logits.shape[0] != target.numel():
                raise AcceptanceError("LM head returned an invalid logits shape")
            if not torch.isfinite(logits).all().item():
                raise AcceptanceError("Nonfinite draft logits")
            loss = torch.nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            if not torch.isfinite(loss).item():
                raise AcceptanceError("Nonfinite draft cross entropy")
            total += float(loss.detach().item())
            correct_by_root.extend((logits.argmax(-1) == target).detach().cpu().tolist())
    return {
        "loss_sum": _finite(total, "cross entropy"),
        "tokens": count,
        "correct": sum(bool(value) for value in correct_by_root),
        "correct_by_root": correct_by_root,
    }


def summarize_scores(
    depth_scores: Sequence[Mapping[str, Any]],
    *,
    eligible_roots: int,
) -> dict[str, Any]:
    if len(depth_scores) != DEPTH:
        raise AcceptanceError("A joint-prefix score requires exactly four depths")
    root_count = int(depth_scores[0]["tokens"])
    if root_count < 0 or eligible_roots < root_count:
        raise AcceptanceError("Invalid eligible/evaluated root counts")
    correctness: list[list[bool]] = []
    for depth, score in enumerate(depth_scores, start=1):
        if int(score["tokens"]) != root_count:
            raise AcceptanceError(f"Depth {depth} evaluated a different root set")
        values = [bool(value) for value in score.get("correct_by_root", [])]
        if len(values) != root_count:
            raise AcceptanceError(f"Depth {depth} is missing per-root predictions")
        correctness.append(values)
    survival_counts = [
        sum(all(correctness[depth][root] for depth in range(index + 1))
            for root in range(root_count))
        for index in range(DEPTH)
    ]
    histogram = [0] * (DEPTH + 1)
    for root in range(root_count):
        accepted = sum(
            all(correctness[depth][root] for depth in range(index + 1))
            for index in range(DEPTH)
        )
        histogram[accepted] += 1
    depths = []
    survival = []
    for index, score in enumerate(depth_scores):
        denominator = int(score["tokens"])
        loss_sum = _finite(score["loss_sum"], f"depth {index + 1} loss sum")
        correct = int(score["correct"])
        if correct < 0 or correct > denominator:
            raise AcceptanceError(f"Invalid top-token count at depth {index + 1}")
        depths.append({
            "depth": index + 1,
            "ce_sum": loss_sum,
            "ce_denominator": denominator,
            "ce": loss_sum / denominator if denominator else None,
            "top_token_correct": correct,
            "top_token_denominator": denominator,
            "top_token_agreement": correct / denominator if denominator else None,
            # Compatibility with the trainer's metric spelling.
            "argmax_agreement": correct / denominator if denominator else None,
            "survival_count": survival_counts[index],
            "survival_denominator": root_count,
        })
        survival.append({
            "depth": index + 1,
            "count": survival_counts[index],
            "denominator": root_count,
            "rate": survival_counts[index] / root_count if root_count else None,
        })
    sum_j = sum(survival_counts)
    return {
        "eligible_roots": int(eligible_roots),
        "evaluated_roots": root_count,
        "root_denominator": root_count,
        "depths": depths,
        "survival": survival,
        "length_histogram": histogram,
        "accepted_prefix_length_sum": sum_j,
        "mean_sum_j": sum_j / root_count if root_count else None,
        "mean_accepted_prefix_length": sum_j / root_count if root_count else None,
    }


def _add_summary(acc: dict[str, Any], summary: Mapping[str, Any]) -> None:
    acc["eligible_roots"] += int(summary["eligible_roots"])
    acc["evaluated_roots"] += int(summary["evaluated_roots"])
    for index, row in enumerate(summary["depths"]):
        acc["depths"][index]["loss_sum"] += _finite(row["ce_sum"], "aggregate loss sum")
        acc["depths"][index]["tokens"] += int(row["ce_denominator"])
        acc["depths"][index]["correct"] += int(row["top_token_correct"])
    for index, row in enumerate(summary["survival"]):
        acc["survival_counts"][index] += int(row["count"])
    for index, count in enumerate(summary["length_histogram"]):
        acc["length_histogram"][index] += int(count)
    acc["sum_j"] += int(summary["accepted_prefix_length_sum"])


def _finalize_accumulator(acc: Mapping[str, Any]) -> dict[str, Any]:
    eligible = int(acc["eligible_roots"])
    evaluated = int(acc["evaluated_roots"])
    if evaluated < 0 or eligible < evaluated:
        raise AcceptanceError("Invalid aggregate root counts")
    depths = []
    survival = []
    for index, raw in enumerate(acc["depths"]):
        denominator = int(raw["tokens"])
        correct = int(raw["correct"])
        loss_sum = _finite(raw["loss_sum"], "aggregate loss sum")
        if denominator != evaluated or correct < 0 or correct > denominator:
            raise AcceptanceError("Aggregate depth denominator mismatch")
        agreement = correct / denominator if denominator else None
        depths.append({
            "depth": index + 1,
            "ce_sum": loss_sum,
            "ce_denominator": denominator,
            "ce": loss_sum / denominator if denominator else None,
            "top_token_correct": correct,
            "top_token_denominator": denominator,
            "top_token_agreement": agreement,
            "argmax_agreement": agreement,
            "survival_count": int(acc["survival_counts"][index]),
            "survival_denominator": evaluated,
        })
        count = int(acc["survival_counts"][index])
        survival.append({
            "depth": index + 1,
            "count": count,
            "denominator": evaluated,
            "rate": count / evaluated if evaluated else None,
        })
    sum_j = int(acc["sum_j"])
    return {
        "eligible_roots": eligible,
        "evaluated_roots": evaluated,
        "root_denominator": evaluated,
        "depths": depths,
        "survival": survival,
        "length_histogram": list(acc["length_histogram"]),
        "accepted_prefix_length_sum": sum_j,
        "mean_sum_j": sum_j / evaluated if evaluated else None,
        "mean_accepted_prefix_length": sum_j / evaluated if evaluated else None,
    }


def _score_sequence(
    model: Any,
    embedding: Any,
    head: Any,
    record: Mapping[str, Any],
    roots: Sequence[int],
    *,
    device: str,
    logits_chunk: int,
) -> dict[str, Any]:
    if not roots:
        return summarize_scores([
            {"loss_sum": 0.0, "tokens": 0, "correct": 0, "correct_by_root": []}
            for _ in range(DEPTH)
        ], eligible_roots=0)
    torch = trainer.runtime().torch
    outputs = trainer.sequence_depths(model, embedding, record, device, DEPTH, list(roots))
    root_indices = torch.tensor(list(roots), dtype=torch.long, device=device)
    depth_scores = []
    base_hidden, base_labels, base_mask = outputs[0]
    if not torch.isfinite(base_hidden).all().item():
        raise AcceptanceError("Nonfinite depth-1 draft hidden state")
    if not bool(base_mask.index_select(0, root_indices).all().item()):
        raise AcceptanceError("Selected root has an invalid depth-1 label mask")
    depth_scores.append(score_logits(
        base_hidden.index_select(0, root_indices),
        head,
        base_labels.index_select(0, root_indices),
        chunk_tokens=logits_chunk,
    ))
    for depth in range(2, DEPTH + 1):
        hidden, labels, mask = outputs[depth - 1]
        if not torch.isfinite(hidden).all().item():
            raise AcceptanceError(f"Nonfinite depth-{depth} draft hidden state")
        if hidden.shape[0] != len(roots) or labels.numel() != len(roots):
            raise AcceptanceError(f"Depth {depth} returned an invalid root batch")
        if not bool(mask.all().item()):
            raise AcceptanceError(f"Selected root has an invalid depth-{depth} label mask")
        depth_scores.append(score_logits(hidden, head, labels, chunk_tokens=logits_chunk))
    return summarize_scores(depth_scores, eligible_roots=len(roots))


def _public_sequence_summary(plan: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capture": plan["capture"],
        "sequence_id": plan["sequence_id"],
        "source_group": plan["source_group"],
        "plan_sha256": plan["plan_sha256"],
        "root_order": list(plan["root_order"]),
        "eligible_roots": plan["eligible_roots"],
        "evaluated_roots": summary["evaluated_roots"],
        "depths": summary["depths"],
        "survival": summary["survival"],
        "length_histogram": summary["length_histogram"],
        "accepted_prefix_length_sum": summary["accepted_prefix_length_sum"],
        "mean_sum_j": summary["mean_sum_j"],
        "mean_accepted_prefix_length": summary["mean_accepted_prefix_length"],
    }


def _evaluate_model(
    model: Any,
    embedding: Any,
    head: Any,
    plans: Sequence[Mapping[str, Any]],
    checkpoint: Any,
    *,
    device: str,
    max_length: int,
    logits_chunk: int,
) -> dict[str, Any]:
    model.eval()
    aggregate = _new_accumulator()
    families: dict[str, dict[str, Any]] = {}
    sequences = []
    torch = trainer.runtime().torch
    with torch.no_grad():
        for plan in plans:
            roots = list(plan["root_order"])
            if roots:
                record = trainer.load_record(plan["path"], checkpoint.config, max_length)
                # Detect a changed capture before any candidate can see a different label set.
                current = make_sequence_plan(
                    plan["path"],
                    record,
                    family=plan["source_group"],
                    roots=max(1, len(roots)),
                )
                if (current["root_order"] != roots
                        or current["plan_sha256"] != plan["plan_sha256"]):
                    raise AcceptanceError(f"Capture changed during evaluation: {plan['capture']}")
                summary = _score_sequence(
                    model,
                    embedding,
                    head,
                    record,
                    roots,
                    device=device,
                    logits_chunk=logits_chunk,
                )
            else:
                summary = summarize_scores([
                    {"loss_sum": 0.0, "tokens": 0, "correct": 0, "correct_by_root": []}
                    for _ in range(DEPTH)
                ], eligible_roots=plan["eligible_roots"])
            # The scorer evaluates the selected roots, while eligible_roots remains the
            # full unbounded denominator for coverage reporting.
            summary["eligible_roots"] = int(plan["eligible_roots"])
            _add_summary(aggregate, summary)
            family = str(plan["source_group"])
            family_acc = families.setdefault(family, _new_accumulator())
            _add_summary(family_acc, summary)
            sequences.append(_public_sequence_summary(plan, summary))
    return {
        "aggregate": _finalize_accumulator(aggregate),
        "families": {
            family: _finalize_accumulator(families[family]) for family in sorted(families)
        },
        "sequences": sequences,
    }


def _candidate_report(
    name: str,
    path: Path | None,
    checkpoint: Any,
    embedding: Any,
    head: Any,
    plans: Sequence[Mapping[str, Any]],
    *,
    device: str,
    max_length: int,
    logits_chunk: int,
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in STAGES:
        state = _load_state(path, checkpoint)
        effective = trainer.rtn_effective_core(state) if stage == "RTN_effective_dense" else state
        model = trainer.build_native_mtp(checkpoint.config, effective, device)
        del effective, state
        stages[stage] = _evaluate_model(
            model,
            embedding,
            head,
            plans,
            checkpoint,
            device=device,
            max_length=max_length,
            logits_chunk=logits_chunk,
        )
        del model
    return {
        "name": name,
        "path": str(path) if path is not None else str(checkpoint.path),
        "stages": stages,
    }


def _plan_counts(plans: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = ("candidate_roots", "hidden_excluded", "terminal_excluded",
            "boundary_excluded", "masked_excluded", "eligible_roots")
    counts = {key: 0 for key in keys}
    counts["sequences"] = len(plans)
    counts["selected_roots"] = 0
    for plan in plans:
        for key in keys:
            if key == "eligible_roots":
                counts[key] += int(plan[key])
            else:
                counts[key] += int(plan["exclusions"].get(key, 0))
        counts["selected_roots"] += len(plan["root_order"])
    return counts


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capture": plan["capture"],
        "sequence_id": plan["sequence_id"],
        "source_group": plan["source_group"],
        "root_order": list(plan["root_order"]),
        "root_identities": list(plan["root_identities"]),
        "labels_sha256_by_depth": list(plan["labels_sha256_by_depth"]),
        "plan_sha256": plan["plan_sha256"],
        "eligible_roots": plan["eligible_roots"],
        "exclusions": dict(plan["exclusions"]),
    }


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if not 3 <= args.max_length <= 2048:
        raise AcceptanceError("max-length must be in 3..2048")
    if type(args.roots) is not int or not 1 <= args.roots <= DEFAULT_ROOTS:
        raise AcceptanceError(f"roots must be in 1..{DEFAULT_ROOTS}")
    if type(args.logits_chunk) is not int or not 1 <= args.logits_chunk <= 256:
        raise AcceptanceError("logits-chunk must be in 1..256")
    if args.device not in ("cpu", "xpu"):
        raise AcceptanceError("device must be cpu or xpu")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the diagnostic and return an in-memory report; no output is written."""
    _validate_runtime_args(args)
    output = Path(args.output).expanduser().resolve()
    if output.suffix != ".json":
        raise AcceptanceError("--output must name a JSON file")
    if output.exists():
        raise AcceptanceError(f"Refusing to overwrite existing output: {output}")
    checkpoint = trainer.Checkpoint(args.model)
    if output.is_relative_to(checkpoint.path):
        raise AcceptanceError("Output must not be inside the stock model directory")
    candidate_specs = parse_candidates(args.candidate)
    if any(output == path for _, path in candidate_specs):
        raise AcceptanceError("Output must differ from every candidate export")
    prompt_families = load_prompt_families(args.prompts)
    plans = load_sequence_plans(
        Path(args.captures),
        checkpoint,
        prompt_families=prompt_families,
        max_length=args.max_length,
        roots=args.roots,
    )
    device = args.device
    torch = trainer.runtime().torch
    torch_device = torch.device(device)
    if torch_device.type == "xpu" and not torch.xpu.is_available():
        raise AcceptanceError("Requested XPU is not available")
    embedding, head = trainer.frozen_heads(checkpoint, device)
    candidates = [("stock", None)] + candidate_specs
    candidate_reports = [
        _candidate_report(
            name,
            path,
            checkpoint,
            embedding,
            head,
            plans,
            device=device,
            max_length=args.max_length,
            logits_chunk=args.logits_chunk,
        )
        for name, path in candidates
    ]
    return {
        "mode": "evaluation_only_joint_prefix",
        "primary_stage": "RTN_effective_dense",
        "model": str(checkpoint.path),
        "captures": str(Path(args.captures).expanduser().resolve()),
        "prompts": str(Path(args.prompts).expanduser().resolve()) if args.prompts else None,
        "protocol": {
            "depth": DEPTH,
            "seed": SEED,
            "roots_per_sequence": args.roots,
            "max_length": args.max_length,
            "logits_chunk": args.logits_chunk,
            "alignment": "depth1 x[t+1], h[t], p[t] -> x[t+2]; depth d x[t+d], previous draft hidden, p[t]+d-1 -> x[t+d+1]",
            "root_boundary": "loss_mask[t+1] must be supervised; labels are x[t+2:t+6]",
            "acceptance": "J_d is the cumulative product of c_1..c_d; later matches after a mismatch do not add tokens",
            "precision": "BF16_export and dense RTN_effective_core; post-RTN is primary; dense RTN is not INT4-kernel proof",
            "writes": "evaluation only; no optimizer, export, or weight mutation",
        },
        "counts": _plan_counts(plans),
        "root_plan": [_public_plan(plan) for plan in plans],
        "candidates": candidate_reports,
    }


def write_exclusive(path: Path, report: Mapping[str, Any]) -> None:
    """Write only the requested JSON report, never replacing an existing file."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise AcceptanceError(f"Refusing to overwrite existing output: {path}") from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True, help="Local stock GPTQ model directory; no downloads")
    result.add_argument("--captures", required=True, help="Held-out native capture directory (*.pt)")
    result.add_argument("--prompts", help="Explicit JSONL id/source_group mapping when captures omit family")
    result.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH",
                        help="Saved BF16 MTP export; may be repeated (stock is automatic)")
    result.add_argument("--output", required=True, help="Exclusive JSON report path")
    result.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    result.add_argument("--max-length", type=int, default=2048,
                        help="Reject, never truncate, longer sequences (3..2048)")
    result.add_argument("--roots", type=int, default=DEFAULT_ROOTS,
                        help="Maximum deterministic roots per sequence (1..64)")
    result.add_argument("--logits-chunk", type=int, default=DEFAULT_LOGITS_CHUNK,
                        help="Root rows per LM-head logits allocation (default 64; max 256)")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run(args)
        output = Path(args.output).expanduser().resolve()
        write_exclusive(output, report)
        print(json.dumps({"output": str(output), "candidates": len(report["candidates"])}, sort_keys=True), flush=True)
    except (AcceptanceError, trainer.TrainingError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Qwen38 MTP acceptance: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
