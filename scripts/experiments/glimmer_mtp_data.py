"""Streaming prompt selection and bounded frozen-Glimmer teacher captures.

This module deliberately keeps its import surface standard-library-only.  The
``datasets`` and ``torch`` imports are made inside the operations that need
them so prompt validation and module discovery remain usable in the small
runtime used by the repository's CPU tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import unicodedata
from collections.abc import Iterable, Mapping


MODEL = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
STATE_KIND = "target_final_norm"
WIDTH = 6656
DTYPE = "bfloat16"
CAPTURE_SCHEMA = "glimmer-mtp-capture-v1"
PROMPT_SCHEMA = "glimmer-mtp-prompts-v1"
SPLITS = ("train", "validation", "test")
MAX_CONTEXT = 1024
STATE_WINDOW = 9
TOKEN_WINDOW = 10
SHARD_STATE_TOKEN_BUDGET = 64 * 1024
FULL_TRAIN_TOKEN_BUDGET = 3_000_000
FULL_TRAIN_ROOT_GATE = 1_500_000


class CaptureFormatError(ValueError):
    """Raised when a capture is unsafe, incomplete, or contract-incompatible."""


def _require(condition: bool, message: str, error_type=ValueError) -> None:
    if not condition:
        raise error_type(message)


def _canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _dedup_key(text: str) -> str:
    # Case-folding is only used for duplicate detection.  The stored prompt is
    # canonicalized but otherwise retains the source's spelling.
    return _canonical(text).casefold()


def _arg(args, name: str, default=None):
    return getattr(args, name, default)


def _json_value(value):
    """Return a bounded, JSON-serializable representation of an argument."""
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return repr(value)


def _arguments(args, excluded=()):
    values = vars(args) if hasattr(args, "__dict__") else {}
    return {key: _json_value(value) for key, value in values.items() if key not in excluded}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            _require(isinstance(row, Mapping), f"prompt row {line_number} is not an object")
            rows.append(dict(row))
    return rows


def _safe_relative(root: Path, relative: str, what: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{what} path is missing", CaptureFormatError)
    candidate = Path(relative)
    _require(not candidate.is_absolute(), f"{what} path must be relative", CaptureFormatError)
    _require(".." not in candidate.parts, f"{what} path traversal is refused", CaptureFormatError)
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    _require(resolved == resolved_root or resolved_root in resolved.parents,
             f"{what} path escapes the capture directory", CaptureFormatError)
    return resolved


def _metadata_path(output: Path) -> Path:
    return Path(str(output) + ".meta.json")


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return value[:80] or "row"


def _stable_row_id(source: str, source_id: str, ordinal: int) -> str:
    # The digest is only an ordinary collision-resistant identifier for a
    # source row, not a receipt, seal, or provenance mechanism.
    digest = hashlib.sha256(f"{source}\0{source_id}\0{ordinal}".encode()).hexdigest()[:12]
    return f"{_slug(source)}-{_slug(source_id)}-{digest}"


def _first_user_prompt(row: Mapping):
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")).lower() == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def _prompt_text(source: str, row: Mapping):
    if source == "coding":
        value = row.get("input")
        return value if isinstance(value, str) else None
    return _first_user_prompt(row)


def _source_id(source: str, row: Mapping, ordinal: int) -> str:
    keys = ("id", "prompt_id") if source == "coding" else ("prompt_id", "id")
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"row-{ordinal}"


def _dataset_info_sha(info) -> str | None:
    if isinstance(info, Mapping):
        value = info.get("sha")
    else:
        value = getattr(info, "sha", None)
    if value is None:
        return None
    value = str(value)
    return value if re.fullmatch(r"[0-9a-fA-F]{40}", value) else None


def _resolve_revision(api, repo: str, requested: str) -> str:
    _require(isinstance(requested, str) and requested.strip(),
             f"missing immutable revision for {repo}")
    requested = requested.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        return requested.lower()
    try:
        info = api.dataset_info(repo, revision=requested)
    except Exception as exc:
        raise RuntimeError(f"could not resolve dataset revision for {repo}") from exc
    resolved = _dataset_info_sha(info)
    _require(resolved is not None, f"dataset revision for {repo} did not resolve to an immutable SHA")
    return resolved.lower()


def _load_stream(load_dataset, repo: str, split: str, revision: str):
    # Keep this call intentionally small: streaming avoids materializing the
    # multi-million-row OpenCodeInstruct corpus in the capture runtime.
    try:
        return load_dataset(repo, split=split, revision=revision, streaming=True)
    except TypeError:
        # A test double or an older datasets runtime may not expose every
        # keyword.  Do not drop streaming or revision, which are the safety
        # properties this path relies on.
        return load_dataset(repo, split=split, revision=revision, streaming=True,
                            trust_remote_code=False)


def _split_families(rows, seed: int):
    families = {}
    for row in rows:
        families.setdefault(row["family"], []).append(row)
    family_names = list(families)
    random.Random(seed).shuffle(family_names)
    count = len(family_names)
    if count == 0:
        return []
    if count == 1:
        targets = {"train": 1, "validation": 0, "test": 0}
    elif count == 2:
        targets = {"train": 1, "validation": 1, "test": 0}
    else:
        validation = max(1, int(round(count * 0.05)))
        test = max(1, int(round(count * 0.05)))
        train = count - validation - test
        if train < 1:
            train = 1
            if validation >= test:
                validation = max(0, count - train - test)
            else:
                test = max(0, count - train - validation)
        targets = {"train": train, "validation": validation, "test": test}
    assignments = {}
    cursor = 0
    for split in SPLITS:
        for family in family_names[cursor:cursor + targets[split]]:
            assignments[family] = split
        cursor += targets[split]
    result = []
    for row in rows:
        copied = dict(row)
        copied["split"] = assignments[row["family"]]
        result.append(copied)
    # Make output stable within each split while retaining the seeded family
    # order.  The family assignment, not an individual generated root, is the
    # unit of held-out separation.
    family_order = {family: index for index, family in enumerate(family_names)}
    result.sort(key=lambda row: (SPLITS.index(row["split"]), family_order[row["family"]], row["id"]))
    return result


def prepare_prompts(args):
    """Stream, deduplicate, split, and write frozen-target prompt records.

    ``args`` needs ``output``, ``seed``, ``coding_fraction`` and
    ``max_prompts``.  Optional dataset/revision/split attributes are accepted
    for isolated runtime pinning.  The return value is a small selection
    summary; the JSONL and ``.meta.json`` files are the durable interface.
    """
    output = Path(_arg(args, "output"))
    seed = int(_arg(args, "seed", 0))
    max_prompts = int(_arg(args, "max_prompts", 0))
    coding_fraction = float(_arg(args, "coding_fraction", 0.4))
    _require(max_prompts > 0, "max_prompts must be positive")
    _require(0.0 <= coding_fraction <= 1.0, "coding_fraction must be between zero and one")
    _require(not output.exists(), f"prompt output already exists: {output}")

    coding_repo = str(_arg(args, "coding_dataset", "nvidia/OpenCodeInstruct"))
    general_repo = str(_arg(args, "general_dataset", "HuggingFaceH4/ultrachat_200k"))
    coding_requested = str(_arg(args, "coding_revision", "main"))
    general_requested = str(_arg(args, "general_revision", "main"))
    coding_split = str(_arg(args, "coding_split", "train"))
    general_split = str(_arg(args, "general_split", "train_sft"))
    coding_license = str(_arg(args, "coding_license", "CC BY 4.0"))
    general_license = str(_arg(args, "general_license", "MIT"))

    coding_count = int(math.floor(max_prompts * coding_fraction + 0.5))
    coding_count = min(max_prompts, max(0, coding_count))
    general_count = max_prompts - coding_count

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("prepare_prompts requires datasets and huggingface_hub in the isolated runtime") from exc

    api = HfApi()
    coding_revision = _resolve_revision(api, coding_repo, coding_requested)
    general_revision = _resolve_revision(api, general_repo, general_requested)
    seen = set()
    selected = []
    source_specs = (
        ("coding", coding_repo, coding_revision, coding_requested, coding_split, coding_license, coding_count),
        ("general", general_repo, general_revision, general_requested, general_split, general_license, general_count),
    )
    source_summaries = []
    scan_limit_multiplier = max(10, int(_arg(args, "scan_limit_multiplier", 100)))

    for family_kind, repo, revision, requested, split, license_name, target_count in source_specs:
        summary = {
            "family": family_kind,
            "dataset": repo,
            "requested_revision": requested,
            "revision": revision,
            "split": split,
            "license": license_name,
            "requested_count": target_count,
            "scanned_rows": 0,
            "selected_count": 0,
            "skipped_rows": 0,
            "status": "disabled" if target_count == 0 else "exhausted",
        }
        if target_count == 0:
            source_summaries.append(summary)
            continue
        stream = _load_stream(load_dataset, repo, split, revision)
        scan_limit = max(target_count * scan_limit_multiplier, target_count + 100)
        scan_limited = False
        for ordinal, raw in enumerate(stream):
            summary["scanned_rows"] += 1
            if summary["scanned_rows"] > scan_limit:
                scan_limited = True
                break
            if not isinstance(raw, Mapping):
                summary["skipped_rows"] += 1
                continue
            text = _prompt_text(family_kind, raw)
            if not isinstance(text, str) or not text.strip():
                summary["skipped_rows"] += 1
                continue
            text = _canonical(text)
            key = _dedup_key(text)
            if not key or key in seen:
                summary["skipped_rows"] += 1
                continue
            source_id = _source_id(family_kind, raw, ordinal)
            family = f"{family_kind}:{source_id}"
            seen.add(key)
            selected.append({
                "id": _stable_row_id(family_kind, source_id, ordinal),
                "family": family,
                "split": "pending",
                "text": text,
                "source": repo,
                "revision": revision,
                "source_id": source_id,
                "license": license_name,
                "selection": "input" if family_kind == "coding" else "first_user_from_train_sft",
            })
            summary["selected_count"] += 1
            if summary["selected_count"] >= target_count:
                summary["status"] = "complete"
                break
        if scan_limited and summary["status"] != "complete":
            summary["status"] = "scan_limit_reached"
        source_summaries.append(summary)

    rows = _split_families(selected, seed)
    split_counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    metadata = {
        "schema": PROMPT_SCHEMA,
        "seed": seed,
        "coding_fraction_requested": coding_fraction,
        "requested_count": max_prompts,
        "selected_count": len(rows),
        "split_counts": split_counts,
        "source_selections": source_summaries,
        "split_policy": "seeded family-level train/validation/test assignment before generation",
        "deduplication": "NFKC whitespace canonicalization plus case-folded prompt key across both sources",
        "generation_policy": "responses are regenerated by the frozen Glimmer target; source assistant outputs are not used",
    }
    _write_jsonl(output, rows)
    _write_json(_metadata_path(output), metadata)
    return metadata


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("capture/sampler operations require torch in the isolated runtime") from exc
    return torch


def _as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return value


def _single_token_row(value):
    value = _as_list(value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    _require(isinstance(value, list), "tokenizer did not return a token sequence")
    return [int(token) for token in value]


def _render_prompt(tokenizer, text: str, max_prompt_tokens: int):
    messages = [{"role": "user", "content": text}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        # This fallback is only for a compatible test double/older tokenizer;
        # it still uses the chat template and never tokenizes its rendered text
        # a second time (which would risk a doubled BOS).
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    if isinstance(rendered, Mapping):
        input_ids = rendered.get("input_ids")
        attention = rendered.get("attention_mask")
    else:
        input_ids, attention = rendered, None
    tokens = _single_token_row(input_ids)
    if attention is not None:
        mask = _single_token_row(attention)
        _require(len(mask) == len(tokens), "chat template attention mask length mismatch")
        tokens = [token for token, keep in zip(tokens, mask) if int(keep)]
    _require(tokens, "chat template produced an empty prompt")
    _require(len(tokens) <= max_prompt_tokens,
             f"prompt has {len(tokens)} tokens, exceeding --max-prompt-tokens {max_prompt_tokens}")
    return tokens


def _target_eos(target, tokenizer):
    values = getattr(target, "eos", None)
    if values is None:
        values = getattr(tokenizer, "eos_token_id", None)
    if values is None:
        config = getattr(target, "config", None)
        values = getattr(config, "eos_token_id", None)
    if values is None:
        return set()
    if isinstance(values, int):
        return {values}
    return {int(value) for value in values}


def _target_device(target, torch):
    device = getattr(target, "device", None)
    if device is not None:
        return device
    model = getattr(target, "model", None)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _padding_id(target, tokenizer, eos):
    value = getattr(tokenizer, "pad_token_id", None)
    if value is not None:
        return int(value)
    if eos:
        return int(sorted(eos)[0])
    config = getattr(target, "config", None)
    value = getattr(config, "pad_token_id", None)
    return int(value) if value is not None else 0


def _make_batch(token_rows, pad_id, device, torch):
    width = max(len(row) for row in token_rows)
    input_ids = torch.full((len(token_rows), width), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros((len(token_rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(token_rows):
        values = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[index, -len(row):] = values
        attention[index, -len(row):] = 1
    # Left padding is represented by a true position for every non-pad token;
    # pad positions are zero and are ignored by the attention mask.
    position_ids = attention.cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention == 0, 0)
    return input_ids, attention, position_ids


def _generate_batch(target, token_rows, max_new_tokens, pad_id, eos, torch):
    tokenizer = target.tokenizer
    device = _target_device(target, torch)
    input_ids, attention, position_ids = _make_batch(token_rows, pad_id, device, torch)
    model = target.model
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention,
        "position_ids": position_ids,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": False,
        "pad_token_id": int(pad_id),
    }
    if eos:
        kwargs["eos_token_id"] = sorted(eos) if len(eos) > 1 else next(iter(eos))
    with torch.inference_mode():
        try:
            generated = model.generate(**kwargs)
        except TypeError as exc:
            # Keep the required true-position and attention arguments.  Only
            # optional generation conveniences are removed for small fakes or
            # older Transformers builds.
            optional = ("return_dict_in_generate", "use_cache", "num_beams")
            reduced = dict(kwargs)
            for key in optional:
                reduced.pop(key, None)
            try:
                generated = model.generate(**reduced)
            except TypeError:
                raise exc
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    elif isinstance(generated, Mapping) and "sequences" in generated:
        generated = generated["sequences"]
    rows = _as_list(generated)
    if rows and isinstance(rows[0], int):
        rows = [rows]
    _require(isinstance(rows, list) and len(rows) == len(token_rows),
             "model.generate returned the wrong batch size")
    result = []
    padded_rows = []
    width = max(len(row) for row in token_rows)
    for row, prompt in zip(rows, token_rows):
        row = [int(token) for token in row]
        padded = [int(pad_id)] * (width - len(prompt)) + prompt
        if len(row) >= width and row[:width] == padded:
            response = row[width:]
        elif len(row) >= len(prompt) and row[:len(prompt)] == prompt:
            response = row[len(prompt):]
        else:
            # Some small test doubles return only newly generated IDs.
            response = row
        result.append(response)
        padded_rows.append(padded)
    del input_ids, attention, position_ids, padded_rows
    return result


def _trim_response(prompt, response, eos):
    response = [int(token) for token in response]
    if eos:
        for index, token in enumerate(response):
            if token in eos:
                response = response[:index + 1]
                break
    return prompt + response


def _extract_states(target, token_ids, torch):
    with torch.inference_mode():
        output = target.forward(token_ids, None, logits_to_keep=1)
    states = output[0] if isinstance(output, tuple) else getattr(output, "states", None)
    if states is None:
        raise ValueError("target.forward did not return final-normalized states")
    if getattr(states, "ndim", 0) == 3:
        states = states[0]
    _require(getattr(states, "ndim", 0) == 2, "target final states must be [sequence,width]")
    _require(int(states.shape[0]) == len(token_ids), "target state/token sequence alignment mismatch")
    _require(int(states.shape[1]) == WIDTH, f"target state width must be {WIDTH}")
    return states.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def _assert_target_frozen(target):
    checker = getattr(target, "assert_frozen", None)
    if callable(checker):
        checker()
        return
    model = getattr(target, "model", None)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            _require(not bool(getattr(parameter, "requires_grad", False)),
                     "target parameter is trainable")
            _require(getattr(parameter, "grad", None) is None,
                     "target parameter has a gradient")

def _eligible_root_positions(token_ids, prompt_length):
    start = max(0, prompt_length - 1)
    return list(range(start, len(token_ids) - 9))


class _ShardBuilder:
    """Bounded CPU shard writer for full-sequence states, not duplicated windows."""

    def __init__(self, output_dir: Path, split: str, torch, capacity: int):
        self.output_dir = output_dir
        self.split = split
        self.torch = torch
        self.capacity = max(1, int(capacity))
        self.states = []
        self.tokens = []
        self.record_indices = []
        self.record_offsets = []
        self.pending = []
        self.token_count = 0
        self.root_count = 0
        self.shards = []
        self.next_shard = 0

    def _flush(self):
        if not self.states:
            return
        states = self.torch.cat(self.states, dim=0).to(
            device="cpu", dtype=self.torch.bfloat16).contiguous()
        tokens = self.torch.cat(self.tokens, dim=0).to(
            device="cpu", dtype=self.torch.long).contiguous()
        record_indices = self.torch.tensor(self.record_indices, dtype=self.torch.long)
        record_offsets = self.torch.tensor(self.record_offsets, dtype=self.torch.long)
        _require(states.ndim == 2 and int(states.shape[0]) == self.token_count,
                 "internal shard state count mismatch")
        _require(tokens.ndim == 1 and int(tokens.shape[0]) == self.token_count,
                 "internal shard token count mismatch")
        name = f"shard-{self.split}-{self.next_shard:05d}.pt"
        self.next_shard += 1
        path = self.output_dir / name
        temporary = path.with_name(path.name + ".tmp")
        self.torch.save({
            "states": states,
            "tokens": tokens,
            "record_indices": record_indices,
            "record_offsets": record_offsets,
            "split": self.split,
            "state_kind": STATE_KIND,
            "width": WIDTH,
            "dtype": DTYPE,
        }, temporary)
        temporary.replace(path)
        shard = {
            "path": name,
            "split": self.split,
            "root_count": int(self.root_count),
            "token_count": int(states.shape[0]),
            "state_token_count": int(states.shape[0]),
            "token_id_count": int(tokens.shape[0]),
            "record_count": len(self.record_indices),
        }
        self.shards.append(shard)
        for record_index, token_offset, token_count, root_offset, root_count in self.pending:
            self._record_chunks[record_index].append({
                "shard": name,
                "token_offset": token_offset,
                "token_count": token_count,
                "root_offset": root_offset,
                "root_count": root_count,
            })
        self.states = []
        self.tokens = []
        self.record_indices = []
        self.record_offsets = []
        self.pending = []
        self.token_count = 0
        self.root_count = 0

    def start_records(self, record_chunks):
        self._record_chunks = record_chunks

    def add(self, record_index: int, states, tokens, root_count: int):
        total = int(states.shape[0])
        _require(states.ndim == 2 and tokens.ndim == 1,
                 "full-sequence shard tensors have invalid rank")
        _require(total == int(tokens.shape[0]), "state/token sequence count mismatch")
        _require(total > 0 and int(root_count) >= 0, "empty sequence cannot be captured")
        if self.states and self.token_count + total > self.capacity:
            self._flush()
        # A single valid sequence can be larger than an intentionally tiny
        # test shard.  Keep it intact and report the bounded oversize rather
        # than splitting a state window across shard boundaries.
        _require(not self.states or self.token_count + total <= self.capacity,
                 "internal shard capacity handling failed")
        token_offset = self.token_count
        root_offset = self.root_count
        self.states.append(states.detach().to(device="cpu", dtype=self.torch.bfloat16).contiguous())
        self.tokens.append(tokens.detach().to(device="cpu", dtype=self.torch.long).contiguous())
        self.record_indices.append(int(record_index))
        self.record_offsets.append([token_offset, total, root_offset, int(root_count)])
        self.pending.append((int(record_index), token_offset, total, root_offset, int(root_count)))
        self.token_count += total
        self.root_count += int(root_count)
        if self.token_count >= self.capacity:
            self._flush()

    def finish(self):
        self._flush()
        return self.shards


def _prompt_rows(source):
    if isinstance(source, (str, os.PathLike, Path)):
        return _read_jsonl(Path(source))
    _require(isinstance(source, Iterable), "prompts must be a JSONL path or iterable of records")
    return [dict(row) for row in source]


def _validate_prompt_rows(rows):
    seen_ids = set()
    seen_texts = set()
    family_splits = {}
    result = []
    for index, row in enumerate(rows, 1):
        _require(all(isinstance(row.get(key), str) and row[key].strip()
                     for key in ("id", "family", "split", "text")),
                 f"prompt row {index} has invalid id/family/split/text")
        _require(row["split"] in SPLITS, f"prompt row {index} has unsupported split")
        _require(row["id"] not in seen_ids, f"duplicate prompt id {row['id']}")
        seen_ids.add(row["id"])
        key = _dedup_key(row["text"])
        _require(key not in seen_texts, "duplicate prompt text across source records")
        seen_texts.add(key)
        previous = family_splits.setdefault(row["family"], row["split"])
        _require(previous == row["split"], "prompt family crosses capture splits")
        copied = dict(row)
        copied["text"] = _canonical(copied["text"])
        result.append(copied)
    return result


def _budget(args, split):
    value = _arg(args, f"{split}_token_budget", 0)
    _require(value is not None and int(value) >= 0, f"{split}_token_budget must be non-negative")
    return int(value)


def _capture_metadata(args, target, prompt_metadata, budgets):
    environment = getattr(target, "environment", {})
    if not isinstance(environment, Mapping):
        environment = {}
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "state_kind": STATE_KIND,
        "width": WIDTH,
        "dtype": DTYPE,
        "context_limit": MAX_CONTEXT,
        "state_window": STATE_WINDOW,
        "token_window": TOKEN_WINDOW,
        "attention": _arg(args, "attention", None),
        "budgets": budgets,
        "arguments": _arguments(args, excluded=("target", "model")),
        "target_environment": _json_value(environment),
        "prompt_source": str(_arg(args, "prompts", "")),
        "prompt_selection": prompt_metadata,
        "generation_path": "frozen target model.generate, greedy batched with left padding and true position_ids",
        "capture_path": "same frozen target target.forward(full_tokens, None, logits_to_keep=1)",
        "fidelity_note": "generation and capture share the mathematical target; BF16 batching/slicing need not be bitwise identical",
    }
    return metadata


def _load_prompt_metadata(prompts):
    path = Path(prompts) if isinstance(prompts, (str, os.PathLike, Path)) else None
    if path is None:
        return None
    companion = _metadata_path(path)
    if not companion.is_file():
        return None
    try:
        value = json.loads(companion.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def capture_generated(args, target):
    """Generate frozen-target continuations and write a bounded tensor capture."""
    torch = _import_torch()
    output_dir = Path(_arg(args, "output_dir"))
    index_path = output_dir / "index.json"
    _require(not index_path.exists(), f"capture index already exists: {index_path}")
    _require(not output_dir.exists() or not any(output_dir.iterdir()),
             f"capture output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _validate_prompt_rows(_prompt_rows(_arg(args, "prompts")))
    max_prompt_tokens = int(_arg(args, "max_prompt_tokens", 512))
    max_new_tokens = int(_arg(args, "max_new_tokens", 512))
    batch_size = int(_arg(args, "generation_batch_size", 4))
    shard_budget = int(_arg(args, "shard_token_budget", SHARD_STATE_TOKEN_BUDGET))
    _require(0 < max_prompt_tokens <= MAX_CONTEXT, "max_prompt_tokens exceeds the capture context")
    _require(0 < max_new_tokens <= MAX_CONTEXT, "max_new_tokens must be positive and bounded")
    _require(batch_size > 0, "generation_batch_size must be positive")
    _require(shard_budget > 0, "shard_token_budget must be positive")
    budgets = {split: _budget(args, split) for split in SPLITS}
    prompt_metadata = _load_prompt_metadata(_arg(args, "prompts"))
    metadata = _capture_metadata(args, target, prompt_metadata, budgets)
    eos = _target_eos(target, target.tokenizer)
    pad_id = _padding_id(target, target.tokenizer, eos)
    _assert_target_frozen(target)

    records = []
    shards = []
    split_summaries = {}
    next_record_index = 0
    try:
        for split in SPLITS:
            split_rows = [row for row in prompts if row["split"] == split]
            budget = budgets[split]
            summary = {
                "budget": budget,
                "status": "disabled" if budget == 0 else "exhausted",
                "prompt_candidates": len(split_rows),
                "prompts_attempted": 0,
                "prompts_captured": 0,
                "prompts_skipped": 0,
                "sequence_token_count": 0,
                "root_count": 0,
                "state_token_count": 0,
                "budget_overshoot": 0,
                "skip_reasons": {},
            }
            builder = _ShardBuilder(output_dir, split, torch,
                                    max(1, shard_budget // STATE_WINDOW))
            record_chunks = {}
            builder.start_records(record_chunks)
            if budget == 0:
                split_summaries[split] = summary
                continue
            cursor = 0
            while cursor < len(split_rows) and summary["sequence_token_count"] < budget:
                batch_rows = []
                token_rows = []
                while cursor < len(split_rows) and len(batch_rows) < batch_size:
                    row = split_rows[cursor]
                    cursor += 1
                    summary["prompts_attempted"] += 1
                    try:
                        tokens = _render_prompt(target.tokenizer, row["text"], max_prompt_tokens)
                    except Exception as exc:
                        summary["prompts_skipped"] += 1
                        reason = str(exc).split(":", 1)[0]
                        summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1
                        continue
                    batch_rows.append(row)
                    token_rows.append(tokens)
                if not token_rows:
                    continue
                batch_prompt_width = max(len(tokens) for tokens in token_rows)
                generation_limit = min(max_new_tokens, MAX_CONTEXT - batch_prompt_width)
                if generation_limit <= 0:
                    for row in batch_rows:
                        summary["prompts_skipped"] += 1
                        summary["skip_reasons"]["prompt_leaves_no_context"] = summary["skip_reasons"].get("prompt_leaves_no_context", 0) + 1
                    continue
                responses = _generate_batch(target, token_rows, generation_limit, pad_id, eos, torch)
                for row, prompt_tokens, response in zip(batch_rows, token_rows, responses):
                    full_tokens = _trim_response(prompt_tokens, response, eos)
                    if len(full_tokens) > MAX_CONTEXT:
                        summary["prompts_skipped"] += 1
                        summary["skip_reasons"]["generated_context_exceeds_1024"] = summary["skip_reasons"].get("generated_context_exceeds_1024", 0) + 1
                        continue
                    if len(full_tokens) <= len(prompt_tokens):
                        summary["prompts_skipped"] += 1
                        summary["skip_reasons"]["empty_response"] = summary["skip_reasons"].get("empty_response", 0) + 1
                        continue
                    sequence_tokens = len(full_tokens)
                    if summary["sequence_token_count"] >= budget:
                        summary["skip_reasons"]["generated_after_budget"] = summary["skip_reasons"].get("generated_after_budget", 0) + 1
                        continue
                    states = _extract_states(target, full_tokens, torch)
                    root_positions = _eligible_root_positions(full_tokens, len(prompt_tokens))
                    record_index = next_record_index
                    next_record_index += 1
                    record = {
                        "id": row["id"],
                        "family": row["family"],
                        "split": split,
                        "text": row["text"],
                        "source": row.get("source"),
                        "revision": row.get("revision"),
                        "source_id": row.get("source_id"),
                        "license": row.get("license"),
                        "prompt_token_count": len(prompt_tokens),
                        "root_start": max(0, len(prompt_tokens) - 1),
                        "response_token_count": sequence_tokens - len(prompt_tokens),
                        "sequence_token_count": sequence_tokens,
                        "root_count": len(root_positions),
                        "chunks": [],
                    }
                    records.append(record)
                    record_chunks[record_index] = record["chunks"]
                    if root_positions:
                        full_token_tensor = torch.tensor(full_tokens, dtype=torch.long)
                        builder.add(record_index, states, full_token_tensor, len(root_positions))
                    summary["prompts_captured"] += 1
                    summary["sequence_token_count"] += sequence_tokens
                    summary["root_count"] += len(root_positions)
                    summary["state_token_count"] += sequence_tokens if root_positions else 0
                    if not root_positions:
                        summary["skip_reasons"]["no_eligible_roots"] = summary["skip_reasons"].get("no_eligible_roots", 0) + 1
                    if summary["sequence_token_count"] >= budget:
                        summary["budget_overshoot"] = summary["sequence_token_count"] - budget
                    del states, root_positions
            if cursor >= len(split_rows) and summary["sequence_token_count"] < budget:
                summary["status"] = "budget_insufficient_prompts_exhausted"
            elif summary["sequence_token_count"] >= budget:
                summary["status"] = "budget_met"
            shards.extend(builder.finish())
            split_summaries[split] = summary
            _assert_target_frozen(target)

        train_budget = budgets["train"]
        expected_root_gate = FULL_TRAIN_ROOT_GATE if train_budget >= FULL_TRAIN_TOKEN_BUDGET else 0
        train_roots = split_summaries["train"]["root_count"]
        if expected_root_gate and train_roots < expected_root_gate:
            overall_status = "incomplete_train_root_gate"
        elif any(summary["status"] == "budget_insufficient_prompts_exhausted"
                 for summary in split_summaries.values() if summary["budget"] > 0):
            overall_status = "incomplete_budget"
        else:
            overall_status = "complete"
        index = {
            "schema": CAPTURE_SCHEMA,
            "schema_version": 1,
            "status": overall_status,
            "complete": overall_status == "complete",
            "model": MODEL,
            "revision": REVISION,
            "state_kind": STATE_KIND,
            "width": WIDTH,
            "dtype": DTYPE,
            "metadata": metadata,
            "budgets": budgets,
            "root_gate": {"full_run_threshold_tokens": FULL_TRAIN_TOKEN_BUDGET,
                          "full_run_minimum_train_roots": FULL_TRAIN_ROOT_GATE,
                          "applied_minimum_train_roots": expected_root_gate,
                          "observed_train_roots": train_roots},
            "root_count": sum(summary["root_count"] for summary in split_summaries.values()),
            "token_count": sum(summary["sequence_token_count"] for summary in split_summaries.values()),
            "state_token_count": sum(summary["state_token_count"] for summary in split_summaries.values()),
            "split_summaries": split_summaries,
            "records": records,
            "shards": shards,
        }
        _write_json(index_path, index)
        return index
    except BaseException as exc:
        failure = {
            "schema": CAPTURE_SCHEMA,
            "schema_version": 1,
            "status": "failed",
            "complete": False,
            "model": MODEL,
            "revision": REVISION,
            "state_kind": STATE_KIND,
            "width": WIDTH,
            "dtype": DTYPE,
            "metadata": metadata,
            "budgets": budgets,
            "records": records,
            "shards": shards,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(index_path, failure)
        raise
    finally:
        _assert_target_frozen(target)


def _torch_load(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise CaptureFormatError("capture loading requires torch.load(weights_only=True)") from exc


def _validate_capture_index(index_path: Path, index: Mapping, split: str):
    _require(index.get("schema") == CAPTURE_SCHEMA, "unsupported capture index schema", CaptureFormatError)
    _require(int(index.get("schema_version", -1)) == 1, "unsupported capture index version", CaptureFormatError)
    _require(index.get("status") == "complete" and index.get("complete") is True,
             "capture is incomplete or failed", CaptureFormatError)
    for key, expected in (("model", MODEL), ("revision", REVISION),
                          ("state_kind", STATE_KIND), ("width", WIDTH), ("dtype", DTYPE)):
        _require(index.get(key) == expected, f"capture {key} mismatch", CaptureFormatError)
    metadata = index.get("metadata")
    _require(isinstance(metadata, Mapping), "capture metadata is missing", CaptureFormatError)
    for key, expected in (("model", MODEL), ("revision", REVISION),
                          ("state_kind", STATE_KIND), ("width", WIDTH), ("dtype", DTYPE)):
        _require(metadata.get(key) == expected, f"capture metadata {key} mismatch", CaptureFormatError)
    _require(split in SPLITS, "unsupported capture split", CaptureFormatError)
    records = index.get("records")
    shards = index.get("shards")
    _require(isinstance(records, list) and isinstance(shards, list),
             "capture records/shards are missing", CaptureFormatError)
    seen_ids, seen_texts, families = set(), set(), {}
    for record_index, record in enumerate(records):
        _require(isinstance(record, Mapping), "capture record is not an object", CaptureFormatError)
        for key in ("id", "family", "split", "text"):
            _require(isinstance(record.get(key), str) and record[key].strip(),
                     f"capture record {record_index} has invalid {key}", CaptureFormatError)
        _require(record["split"] in SPLITS, "capture record has unsupported split", CaptureFormatError)
        _require(record["id"] not in seen_ids, "duplicate capture record ID", CaptureFormatError)
        seen_ids.add(record["id"])
        key = _dedup_key(record["text"])
        _require(key not in seen_texts, "duplicate capture prompt text", CaptureFormatError)
        seen_texts.add(key)
        old_split = families.setdefault(record["family"], record["split"])
        _require(old_split == record["split"], "capture family crosses splits", CaptureFormatError)
        root_count = int(record.get("root_count", -1))
        sequence_count = int(record.get("sequence_token_count", -1))
        prompt_count = int(record.get("prompt_token_count", -1))
        root_start = int(record.get("root_start", -1))
        _require(0 < prompt_count < sequence_count and root_count >= 0 and
                 root_start == max(0, prompt_count - 1),
                 "capture record counts are invalid", CaptureFormatError)
        maximum_roots = max(0, sequence_count - 9 - max(0, prompt_count - 1))
        _require(root_count == maximum_roots, "capture record root mask is invalid", CaptureFormatError)
        chunks = record.get("chunks", [])
        _require(isinstance(chunks, list) and all(isinstance(chunk, Mapping) for chunk in chunks),
                 "capture record chunks are invalid", CaptureFormatError)
        _require(sum(int(chunk.get("root_count", -1)) for chunk in chunks) == root_count,
                 "capture record/chunk root count mismatch", CaptureFormatError)
        chunk_tokens = sum(int(chunk.get("token_count", -1)) for chunk in chunks)
        _require(chunk_tokens == sequence_count if chunks else root_count == 0,
                 "capture record/chunk token count mismatch", CaptureFormatError)
    shard_by_path = {}
    for shard in shards:
        _require(isinstance(shard, Mapping), "capture shard is not an object", CaptureFormatError)
        path = shard.get("path")
        _require(isinstance(path, str) and path not in shard_by_path,
                 "capture shard path is invalid or duplicated", CaptureFormatError)
        resolved_shard = _safe_relative(index_path.parent, path, "capture shard")
        _require(resolved_shard.is_file(), f"capture shard is missing: {path}", CaptureFormatError)
        root_count = int(shard.get("root_count", -1))
        token_count = int(shard.get("token_count", -1))
        _require(root_count > 0 and token_count > 0, "capture shard is empty", CaptureFormatError)
        _require(int(shard.get("state_token_count", -1)) == token_count,
                 "capture shard state-token count mismatch", CaptureFormatError)
        _require(int(shard.get("token_id_count", -1)) == token_count,
                 "capture shard token count mismatch", CaptureFormatError)
        _require(int(shard.get("record_count", -1)) > 0, "capture shard has no records", CaptureFormatError)
        _require(shard.get("split") in SPLITS, "capture shard split is invalid", CaptureFormatError)
        shard_by_path[path] = shard
    references = set()
    root_ranges = {path: [] for path in shard_by_path}
    token_ranges = {path: [] for path in shard_by_path}
    chunk_counts = {path: 0 for path in shard_by_path}
    for record_index, record in enumerate(records):
        for chunk in record.get("chunks", []):
            shard_path = chunk.get("shard")
            _require(shard_path in shard_by_path, "capture chunk references unknown shard", CaptureFormatError)
            shard = shard_by_path[shard_path]
            _require(shard["split"] == record["split"], "capture chunk split mismatch", CaptureFormatError)
            token_offset = int(chunk.get("token_offset", -1))
            token_count = int(chunk.get("token_count", -1))
            root_offset = int(chunk.get("root_offset", -1))
            root_count = int(chunk.get("root_count", -1))
            _require(token_offset >= 0 and token_count > 0 and token_offset + token_count <= int(shard["token_count"]),
                     "capture chunk token range is invalid", CaptureFormatError)
            _require(root_offset >= 0 and root_count > 0 and root_offset + root_count <= int(shard["root_count"]),
                     "capture chunk root range is invalid", CaptureFormatError)
            root_ranges[shard_path].append((root_offset, root_offset + root_count, record_index))
            token_ranges[shard_path].append((token_offset, token_offset + token_count, record_index))
            chunk_counts[shard_path] += 1
            references.add(shard_path)
    _require(references == set(shard_by_path), "capture has unreferenced or missing shards", CaptureFormatError)
    for shard_path, shard in shard_by_path.items():
        for ranges, expected, label in ((root_ranges[shard_path], int(shard["root_count"]), "root"),
                                         (token_ranges[shard_path], int(shard["token_count"]), "token")):
            cursor = 0
            for start, end, _record_index in sorted(ranges):
                _require(start == cursor, f"capture shard {shard_path} has overlapping/missing {label}s", CaptureFormatError)
                cursor = end
            _require(cursor == expected, f"capture shard {shard_path} is not fully covered by {label}s", CaptureFormatError)
        _require(chunk_counts[shard_path] == int(shard["record_count"]),
                 f"capture shard {shard_path} record count mismatch", CaptureFormatError)
    split_records = [record for record in records if record["split"] == split]
    split_roots = sum(int(record["root_count"]) for record in split_records)
    split_tokens = sum(int(record["sequence_token_count"]) for record in split_records if int(record["root_count"]) > 0)
    _require(split_roots > 0, f"capture split {split} has no eligible roots", CaptureFormatError)
    _require(split_tokens > 0, f"capture split {split} has no captured sequence tokens", CaptureFormatError)
    return shard_by_path, split_records, split_roots, split_tokens


class CapturedDataset:
    """Lazy, shard-local view of a completed Glimmer capture."""

    def __init__(self, path, split="train"):
        torch = _import_torch()
        supplied = Path(path)
        if supplied.is_dir():
            supplied = supplied / "index.json"
        _require(supplied.name == "index.json",
                 "CapturedDataset accepts only a new index.json", CaptureFormatError)
        _require(supplied.is_file(), f"capture index does not exist: {supplied}", CaptureFormatError)
        try:
            index = json.loads(supplied.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CaptureFormatError("capture index is not valid JSON") from exc
        _require(isinstance(index, Mapping), "capture index is not an object", CaptureFormatError)
        shard_by_path, split_records, root_count, token_count = _validate_capture_index(
            supplied, index, split)
        self._torch = torch
        self.index_path = supplied.resolve()
        self._root = self.index_path.parent
        self._index = dict(index)
        self._shard_by_path = dict(shard_by_path)
        self._split = split
        self._records = [dict(record) for record in split_records]
        self._shards = [dict(shard) for shard in shard_by_path.values()
                       if shard["split"] == split]
        self._shards.sort(key=lambda shard: str(shard["path"]))
        self.root_count = int(root_count)
        self.token_count = int(token_count)
        self.metadata = dict(index["metadata"])
        self.metadata.update({
            "model": index["model"],
            "revision": index["revision"],
            "state_kind": index["state_kind"],
            "width": index["width"],
            "dtype": index["dtype"],
            "split": split,
            "index_path": str(self.index_path),
        })

    def manifest(self):
        return [{key: record[key] for key in ("id", "family", "split", "text")}
                for record in self._records]

    def new_sampler(self, seed):
        return _ShardSampler(self, int(seed))

    def iter_batches(self, batch_size, limit_roots=None, seed=0):
        _require(int(batch_size) > 0, "batch_size must be positive")
        limit = self.root_count if limit_roots is None else int(limit_roots)
        _require(limit >= 0, "limit_roots must be non-negative")
        limit = min(limit, self.root_count)
        if limit == 0:
            return
        sampler = self.new_sampler(seed)
        remaining = limit
        while remaining:
            count = min(int(batch_size), remaining)
            yield sampler.next_batch(count)
            remaining -= count


class _ShardSampler:
    def __init__(self, dataset: CapturedDataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.shard_order = []
        self.shard_cursor = 0
        self.root_order = []
        self.root_cursor = 0
        self.current_shard = None
        self._payload = None
        self._generator = dataset._torch.Generator(device="cpu")
        self._root_generator = dataset._torch.Generator(device="cpu")
        self._start_epoch()

    def _mix_seed(self, epoch, shard_index=0):
        value = self.seed & 0x7FFF_FFFF_FFFF_FFFF
        value = (value + (epoch + 1) * 0x9E3779B97F4A7C15
                 + (shard_index + 1) * 0xBF58476D1CE4E5B9) & 0x7FFF_FFFF_FFFF_FFFF
        return value

    def _start_epoch(self):
        _require(self.dataset._shards, "capture split has no shards", CaptureFormatError)
        self._generator.manual_seed(self._mix_seed(self.epoch))
        count = len(self.dataset._shards)
        self.shard_order = self.dataset._torch.randperm(
            count, generator=self._generator).tolist()
        self.shard_cursor = 0
        self.root_order = []
        self.root_cursor = 0
        self.current_shard = None
        self._payload = None

    def _prepare_shard(self):
        while self.root_cursor >= len(self.root_order):
            if self.shard_cursor >= len(self.shard_order):
                self.epoch += 1
                self._start_epoch()
                continue
            shard_index = self.shard_order[self.shard_cursor]
            self.shard_cursor += 1
            shard = self.dataset._shards[shard_index]
            self.current_shard = shard_index
            self._root_generator.manual_seed(self._mix_seed(self.epoch, shard_index))
            self.root_order = self.dataset._torch.randperm(
                int(shard["root_count"]), generator=self._root_generator).tolist()
            self.root_cursor = 0
            self._payload = self.dataset._load_shard(shard)

    def next_batch(self, batch_size):
        _require(int(batch_size) > 0, "batch_size must be positive")
        state_batches, token_batches = [], []
        remaining = int(batch_size)
        while remaining:
            self._prepare_shard()
            if self._payload is None:
                _require(self.current_shard is not None, "sampler current shard is missing")
                self._payload = self.dataset._load_shard(
                    self.dataset._shards[self.current_shard])
            take = min(remaining, len(self.root_order) - self.root_cursor)
            indices = self.root_order[self.root_cursor:self.root_cursor + take]
            self.root_cursor += take
            payload_states, payload_tokens, spans = self._payload
            state_windows, token_windows = [], []
            for root in indices:
                for root_start, root_count, root_token_start, sequence_token_start, sequence_token_count, _record_index in spans:
                    if root_start <= root < root_start + root_count:
                        offset = root_token_start + root - root_start
                        _require(offset + TOKEN_WINDOW <= sequence_token_start + sequence_token_count,
                                 "capture root future exceeds its sequence", CaptureFormatError)
                        state_windows.append(payload_states[offset:offset + STATE_WINDOW])
                        token_windows.append(payload_tokens[offset:offset + TOKEN_WINDOW])
                        break
                else:
                    raise CaptureFormatError("capture root does not map to a shard sequence")
            state_batches.append(self.dataset._torch.stack(state_windows, dim=0))
            token_batches.append(self.dataset._torch.stack(token_windows, dim=0))
            remaining -= take
        return (self.dataset._torch.cat(state_batches, dim=0).to(device="cpu").contiguous(),
                self.dataset._torch.cat(token_batches, dim=0).to(device="cpu").contiguous())

    def state_dict(self):
        return {
            "schema": "glimmer-mtp-sampler-v1",
            "seed": self.seed,
            "index_path": str(self.dataset.index_path),
            "split": self.dataset._split,
            "epoch": self.epoch,
            "shard_order": list(self.shard_order),
            "shard_cursor": self.shard_cursor,
            "current_shard": self.current_shard,
            "root_order": list(self.root_order),
            "root_cursor": self.root_cursor,
            "generator_state": self._generator.get_state().clone(),
            "root_generator_state": self._root_generator.get_state().clone(),
        }

    def load_state_dict(self, state):
        _require(isinstance(state, Mapping), "sampler state must be a mapping")
        _require(state.get("schema") == "glimmer-mtp-sampler-v1", "sampler state schema mismatch")
        _require(int(state.get("seed")) == self.seed, "sampler seed mismatch")
        _require(str(state.get("index_path")) == str(self.dataset.index_path), "sampler index path mismatch")
        _require(state.get("split") == self.dataset._split, "sampler split mismatch")
        epoch = int(state.get("epoch", -1))
        shard_order = [int(value) for value in state.get("shard_order", [])]
        shard_cursor = int(state.get("shard_cursor", -1))
        root_order = [int(value) for value in state.get("root_order", [])]
        root_cursor = int(state.get("root_cursor", -1))
        _require(epoch >= 0 and sorted(shard_order) == list(range(len(self.dataset._shards))),
                 "sampler shard permutation is invalid")
        _require(0 <= shard_cursor <= len(shard_order), "sampler shard cursor is invalid")
        current = state.get("current_shard")
        if root_order:
            current = int(current)
            _require(0 <= current < len(self.dataset._shards), "sampler current shard is invalid")
            _require(sorted(root_order) == list(range(int(self.dataset._shards[current]["root_count"]))),
                     "sampler root permutation is invalid")
            _require(0 <= root_cursor <= len(root_order), "sampler root cursor is invalid")
        else:
            _require(root_cursor in (0, -1), "sampler empty-root cursor is invalid")
            current = None
        self.epoch = epoch
        self.shard_order = shard_order
        self.shard_cursor = shard_cursor
        self.current_shard = current
        self.root_order = root_order
        self.root_cursor = max(0, root_cursor)
        self._payload = None
        generator_state = state.get("generator_state")
        root_generator_state = state.get("root_generator_state")
        if generator_state is not None:
            self._generator.set_state(generator_state)
        if root_generator_state is not None:
            self._root_generator.set_state(root_generator_state)


def _validate_payload(dataset: CapturedDataset, shard, payload):
    torch = dataset._torch
    _require(isinstance(payload, Mapping), "capture shard payload is not a mapping", CaptureFormatError)
    for key in ("states", "tokens", "record_indices", "record_offsets"):
        _require(key in payload, f"capture shard payload lacks {key}", CaptureFormatError)
    states = payload["states"]
    tokens = payload["tokens"]
    record_indices = payload["record_indices"]
    record_offsets = payload["record_offsets"]
    states_device = getattr(getattr(states, "device", None), "type", None)
    tokens_device = getattr(getattr(tokens, "device", None), "type", None)
    ids_device = getattr(getattr(record_indices, "device", None), "type", None)
    offsets_device = getattr(getattr(record_offsets, "device", None), "type", None)
    _require(states_device == "cpu", "capture states must be CPU tensors", CaptureFormatError)
    _require(getattr(states, "dtype", None) == torch.bfloat16 and getattr(states, "ndim", None) == 2,
             "capture states must be CPU BF16 [tokens,6656]", CaptureFormatError)
    _require(tuple(states.shape[1:]) == (WIDTH,), "capture state layout mismatch", CaptureFormatError)
    _require(tokens_device == "cpu" and getattr(tokens, "dtype", None) == torch.long and getattr(tokens, "ndim", None) == 1,
             "capture tokens must be CPU int64 [tokens]", CaptureFormatError)
    _require(ids_device == "cpu" and getattr(record_indices, "dtype", None) == torch.long and getattr(record_indices, "ndim", None) == 1,
             "capture record IDs must be CPU int64", CaptureFormatError)
    _require(offsets_device == "cpu" and getattr(record_offsets, "dtype", None) == torch.long and getattr(record_offsets, "ndim", None) == 2 and tuple(record_offsets.shape[1:]) == (4,),
             "capture record offsets must be CPU int64 [records,4]", CaptureFormatError)
    _require(int(states.shape[0]) == int(tokens.shape[0]) == int(shard["token_count"]),
             "capture shard token count mismatch", CaptureFormatError)
    _require(int(record_indices.shape[0]) == int(record_offsets.shape[0]) == int(shard["record_count"]),
             "capture shard record count mismatch", CaptureFormatError)
    _require(payload.get("split") == shard["split"] and
             payload.get("state_kind") == STATE_KIND and
             payload.get("dtype") == DTYPE and int(payload.get("width", -1)) == WIDTH,
             "capture shard model layout mismatch", CaptureFormatError)
    _require(bool(torch.isfinite(states).all().item()),
             "capture states contain non-finite values", CaptureFormatError)
    _require(bool(((tokens >= 0) & (tokens < 202048)).all().item()),
             "capture token IDs are outside the Glimmer vocabulary", CaptureFormatError)
    expected_records = []
    expected_offsets = []
    for record_index, record in enumerate(dataset._index["records"]):
        for chunk in record.get("chunks", []):
            if chunk.get("shard") == shard["path"]:
                expected_records.append(record_index)
                expected_offsets.append([int(chunk["token_offset"]), int(chunk["token_count"]),
                                        int(chunk["root_offset"]), int(chunk["root_count"])])
    order = sorted(range(len(expected_offsets)), key=lambda index: expected_offsets[index][0])
    expected_records = [expected_records[index] for index in order]
    expected_offsets = [expected_offsets[index] for index in order]
    _require(record_indices.tolist() == expected_records,
             "capture shard record IDs do not match index", CaptureFormatError)
    _require(record_offsets.tolist() == expected_offsets,
             "capture shard record offsets do not match index", CaptureFormatError)
    spans = []
    for record_index, record in enumerate(dataset._index["records"]):
        for chunk in record.get("chunks", []):
            if chunk.get("shard") == shard["path"]:
                root_token_start = int(chunk["token_offset"]) + int(record["root_start"])
                spans.append((int(chunk["root_offset"]), int(chunk["root_count"]),
                              root_token_start, int(chunk["token_offset"]),
                              int(chunk["token_count"]), record_index))
    spans.sort(key=lambda span: span[0])
    return states, tokens, spans


def _load_shard(self, shard):
    path = _safe_relative(self._root, shard["path"], "capture shard")
    payload = _torch_load(self._torch, path)
    return _validate_payload(self, shard, payload)


# Keep the loader as a method assigned after the class body to keep the class
# definition's public surface compact and make the lazy torch boundary obvious.
CapturedDataset._load_shard = _load_shard


__all__ = [
    "MODEL", "REVISION", "STATE_KIND", "WIDTH", "DTYPE", "CAPTURE_SCHEMA",
    "prepare_prompts", "capture_generated", "CapturedDataset", "CaptureFormatError",
]
