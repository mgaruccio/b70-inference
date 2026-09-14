#!/usr/bin/env python3
"""Bounded, experiment-only quality harness for the Qwen3.8 grouped trial.

The module itself is stdlib-only at import time.  Dataset preparation and
scoring deliberately load the pinned official packages lazily so that no ML or
inference runtime is needed for local syntax/contract checks.  Generation uses
only urllib against an already-running OpenAI-compatible HTTP server.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as _datetime
import gzip
import zlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable
import urllib.error
import urllib.request


HARNESS_SCHEMA = "qwen38-grouped-quality/v1"
MODEL_NAME = "qwen38"
OUTPUT_BUDGET = 4096
SEED = 42
CACHE_SALT = "C1"

# These are immutable source pins used by prepare.  Hashes are of the exact
# downloaded bytes, not a mutable branch name or a parsed/re-serialized file.
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFEVAL_URL = (
    "https://huggingface.co/datasets/google/IFEval/resolve/"
    f"{IFEVAL_REVISION}/ifeval_input_data.jsonl?download=true"
)
IFEVAL_SHA256 = "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"
IFEVAL_COUNT = 541

GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM_TEST_URL = (
    "https://huggingface.co/datasets/openai/gsm8k/resolve/"
    f"{GSM_REVISION}/main/test-00000-of-00001.parquet?download=true"
)
GSM_TEST_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
GSM_TEST_COUNT = 1319
GSM_TRAIN_URL = (
    "https://huggingface.co/datasets/openai/gsm8k/resolve/"
    f"{GSM_REVISION}/main/train-00000-of-00001.parquet?download=true"
)
GSM_TRAIN_SHA256 = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"
GSM_TRAIN_COUNT = 7473
GSM_FEWSHOT_COUNT = 5
GSM_FEWSHOT_POLICY = "first five rows of the pinned main/train split"

EVALPLUS_VERSION = "0.3.1"
HUMANEVALPLUS_RELEASE = "v0.1.10"
HUMANEVALPLUS_URL = (
    "https://github.com/evalplus/humanevalplus_release/releases/download/"
    f"{HUMANEVALPLUS_RELEASE}/HumanEvalPlus.jsonl.gz"
)
HUMANEVALPLUS_SHA256 = (
    "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
)
HUMANEVALPLUS_COUNT = 164
MBPPPLUS_RELEASE = "v0.2.0"
MBPPPLUS_URL = (
    "https://github.com/evalplus/mbppplus_release/releases/download/"
    f"{MBPPPLUS_RELEASE}/MbppPlus.jsonl.gz"
)
MBPPPLUS_SHA256 = (
    "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"
)
MBPPPLUS_COUNT = 378

# Official task-format references frozen alongside the data manifest.
LMEVAL_IFEVAL_URL = (
    "https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13/"
    "lm_eval/tasks/ifeval"
)
LMEVAL_IFEVAL_UTILS_URL = (
    "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/"
    "v0.4.13/lm_eval/tasks/ifeval/utils.py"
)
LMEVAL_GSM8K_URL = (
    "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/"
    "v0.4.13/lm_eval/tasks/gsm8k/gsm8k.yaml"
)
EVALPLUS_CLI_URL = "https://github.com/evalplus/evalplus/blob/v0.3.1/docs/cli.md"
EVALPLUS_EXECUTION_URL = (
    "https://github.com/evalplus/evalplus/blob/v0.3.1/docs/execution.md"
)
VLLM_OPENAI_URL = (
    "https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/"
)
VLLM_REPRO_URL = "https://docs.vllm.ai/en/stable/usage/reproducibility.html"

EXPECTED_PRIMARY_COUNTS = {
    "ifeval": IFEVAL_COUNT,
    "gsm8k": GSM_TEST_COUNT,
    "humanevalplus": HUMANEVALPLUS_COUNT,
    "mbppplus": MBPPPLUS_COUNT,
}
EXPECTED_PRIMARY_TOTAL = sum(EXPECTED_PRIMARY_COUNTS.values())
REPEAT_COUNT = 32 * 2
EXPECTED_REQUEST_TOTAL = EXPECTED_PRIMARY_TOTAL + REPEAT_COUNT

_IFEVAL_ID_RE = re.compile(r"^ifeval/(\d+)$")
_GSM_STRICT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_GSM_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


class HarnessError(RuntimeError):
    """A clean, fail-closed harness or input-contract error."""


_MISSING = object()


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _jsonable(value: Any) -> Any:
    """Convert dataset-library scalar/container values without changing data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise HarnessError(f"dataset value is not JSON-serializable: {type(value)!r}")


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HarnessError(f"cannot serialize JSON value: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _rows_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, data: bytes, *, force: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise HarnessError(f"refusing to overwrite existing file: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any, *, force: bool = True) -> None:
    _write_bytes(
        path,
        (json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        force=force,
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, force: bool = True) -> str:
    data = b"".join(_json_bytes(row) + b"\n" for row in rows)
    _write_bytes(path, data, force=force)
    return _sha256_bytes(data)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"JSON object expected in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise HarnessError(f"cannot read JSONL {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(raw_lines, 1):
        if not raw.strip():
            raise HarnessError(f"blank JSONL line is not allowed: {path}:{line_no}")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HarnessError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise HarnessError(f"JSON object expected at {path}:{line_no}")
        rows.append(value)
    return rows


def _download_verified(url: str, path: Path, expected_sha256: str, timeout: float) -> dict[str, Any]:
    """Download one pinned source; never retry or accept a hash mismatch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.download-{os.getpid()}")
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "qwen38-grouped-quality/1", "Accept": "*/*"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open(
            "wb"
        ) as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise HarnessError(
                f"pinned source hash mismatch for {url}: expected {expected_sha256}, observed {observed}"
            )
        os.replace(temporary, path)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise HarnessError(f"download failed for {url}: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"url": url, "sha256": expected_sha256, "bytes": size, "path": str(path)}


def _parse_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HarnessError(f"invalid source JSON at {label}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise HarnessError(f"source object expected at {label}:{line_no}")
        rows.append(_jsonable(value))
    return rows


def _require_distribution(distribution: str, expected: str) -> str:
    try:
        observed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise HarnessError(
            f"missing pinned evaluator prerequisite {distribution}=={expected}; "
            "run this command in the supplied CPU evaluator image"
        ) from exc
    if observed != expected:
        raise HarnessError(
            f"wrong {distribution} version: expected {expected}, observed {observed}"
        )
    return observed


def _load_parquet_rows(path: Path, label: str) -> list[dict[str, Any]]:
    # This is intentionally lazy.  The local checker and generation path must
    # not import datasets, pyarrow, torch, or any inference/ML package.
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise HarnessError(
            f"{label} preparation requires the pinned datasets dependency in the remote image"
        ) from exc
    old_offline = os.environ.get("HF_DATASETS_OFFLINE")
    old_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        try:
            dataset = load_dataset(
                "parquet", data_files={"data": str(path)}, split="data"
            )
        except Exception as exc:  # datasets wraps pyarrow errors in several types
            raise HarnessError(f"cannot parse pinned {label} parquet: {exc}") from exc
        return [_jsonable(row) for row in dataset]
    finally:
        if old_offline is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = old_offline
        if old_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hub_offline


def _evalplus_release_rows(
    compressed_path: Path,
    uncompressed_path: Path,
    package_name: str,
    expected_count: int,
    release: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_distribution("evalplus", EVALPLUS_VERSION)
    try:
        raw = gzip.decompress(compressed_path.read_bytes())
    except (OSError, EOFError, zlib.error) as exc:
        raise HarnessError(f"cannot decompress {compressed_path}: {exc}") from exc
    _write_bytes(uncompressed_path, raw)
    rows = _parse_jsonl_bytes(raw, str(uncompressed_path))
    if len(rows) != expected_count:
        raise HarnessError(
            f"{package_name} count mismatch: expected {expected_count}, observed {len(rows)}"
        )

    # Use EvalPlus 0.3.1's own loader against the verified release file.  The
    # override avoids a second mutable download and lets the frozen JSONL be
    # the only scoring dataset later as well.
    old_human = os.environ.get("HUMANEVAL_OVERRIDE_PATH")
    old_mbpp = os.environ.get("MBPP_OVERRIDE_PATH")
    if package_name == "humanevalplus":
        os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(uncompressed_path)
        loaded_module = sys.modules.get("evalplus.data.humaneval")
        if loaded_module is not None:
            loaded_module.HUMANEVAL_OVERRIDE_PATH = str(uncompressed_path)
    else:
        os.environ["MBPP_OVERRIDE_PATH"] = str(uncompressed_path)
        loaded_module = sys.modules.get("evalplus.data.mbpp")
        if loaded_module is not None:
            loaded_module.MBPP_OVERRIDE_PATH = str(uncompressed_path)
    try:
        try:
            if package_name == "humanevalplus":
                from evalplus.data import get_human_eval_plus  # type: ignore

                loaded = get_human_eval_plus(err_incomplete=True, version=release)
            else:
                from evalplus.data import get_mbpp_plus  # type: ignore

                loaded = get_mbpp_plus(err_incomplete=True, version=release)
        except Exception as exc:
            raise HarnessError(
                f"EvalPlus {EVALPLUS_VERSION} could not load verified {package_name}: {exc}"
            ) from exc
    finally:
        if old_human is None:
            os.environ.pop("HUMANEVAL_OVERRIDE_PATH", None)
        else:
            os.environ["HUMANEVAL_OVERRIDE_PATH"] = old_human
        if old_mbpp is None:
            os.environ.pop("MBPP_OVERRIDE_PATH", None)
        else:
            os.environ["MBPP_OVERRIDE_PATH"] = old_mbpp

    loaded_by_id = {str(key): value for key, value in loaded.items()}
    raw_by_id = {str(row.get("task_id")): row for row in rows}
    if set(raw_by_id) != set(loaded_by_id):
        raise HarnessError(f"EvalPlus {package_name} loader changed task IDs")
    # MBPP's loader intentionally deserializes some JSON arrays to tuples or
    # sets.  Compare invariant source fields and rely on the verified release
    # hash for the test-input payload instead of falsely treating that shape
    # conversion as dataset drift.
    for task_id, raw_row in raw_by_id.items():
        loaded_row = loaded_by_id[task_id]
        for key in ("task_id", "prompt", "entry_point", "canonical_solution", "contract", "atol"):
            if _jsonable(raw_row.get(key)) != _jsonable(loaded_row.get(key)):
                raise HarnessError(f"EvalPlus {package_name} loader drift at {task_id} field {key}")
    return rows, {
        "package": "evalplus",
        "package_version": EVALPLUS_VERSION,
        "release": release,
        "count": expected_count,
        "uncompressed_sha256": _sha256_bytes(raw),
        "task_ids_sha256": _sha256_json(sorted(raw_by_id)),
    }


def _make_prompt_row(
    *,
    task: str,
    item_id: str,
    source_id: str,
    source_index: int,
    source: dict[str, Any],
    prompt: str,
    divergence_sample: bool = False,
    stop_sequences: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # EvalPlus includes IEEE infinity in Mbpp/404 test inputs. Preserve the
    # complete native JSON as a string; never coerce/drop those test values.
    source_json = None
    if task in {"humanevalplus", "mbppplus"}:
        source_json = json.dumps(source, sort_keys=True, ensure_ascii=False, allow_nan=True)
        source = {k: v for k, v in source.items() if k not in {"base_input", "plus_input"}}
    messages = [{"role": "user", "content": prompt}]
    row: dict[str, Any] = {
        "id": item_id,
        "source_id": source_id,
        "source_index": source_index,
        "task": task,
        "sample_role": "primary",
        "repeat_of": None,
        "divergence_sample": divergence_sample,
        "prompt": prompt,
        "messages": messages,
        "source": _jsonable(source),
        "source_sha256": _sha256_json(source),
        "prompt_sha256": _sha256_json(prompt),
        "messages_sha256": _sha256_json(messages),
        "output_budget": OUTPUT_BUDGET,
        "eos_mode": "normal",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if source_json is not None:
        row["source_json"] = source_json
        row["source_sha256"] = _sha256_bytes(source_json.encode("utf-8"))
    if stop_sequences:
        row["stop_sequences"] = list(stop_sequences)
    if extra:
        row.update(extra)
    return row


def _gsm_prompt(question: str, shots: list[dict[str, Any]]) -> str:
    # This is the exact doc_to_text/doc_to_target shape from gsm8k.yaml:
    # "Question: {{question}}\\nAnswer:" plus the complete target answer.
    examples = [
        f"Question: {shot['question']}\nAnswer:{shot['answer']}" for shot in shots
    ]
    examples.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(examples)


def _code_prompt(source_prompt: str) -> str:
    # EvalPlus's official codegen uses this instruction prefix.  We ask for
    # source without fences so the scorer receives the exact completion; no
    # sanitizer or repair is applied after generation.
    return (
        "Please provide a self-contained Python script that solves the following "
        "problem. Output only Python source, without markdown fences:\n\n"
        + source_prompt.strip()
        + "\n"
    )


def _natural_task_sort(task_id: str) -> tuple[str, int | str]:
    prefix, _, number = task_id.partition("/")
    try:
        return prefix, int(number)
    except ValueError:
        return prefix, task_id


def prepare(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not args.force:
        raise HarnessError(f"refusing non-empty prepare output; use --force: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sources = out / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    timeout = float(args.timeout)

    # IFEval is a JSONL source in the pinned Hugging Face dataset revision.
    ifeval_file = sources / "ifeval_input_data.jsonl"
    ifeval_download = _download_verified(
        IFEVAL_URL, ifeval_file, IFEVAL_SHA256, timeout
    )
    ifeval_source = _parse_jsonl_bytes(ifeval_file.read_bytes(), str(ifeval_file))
    if len(ifeval_source) != IFEVAL_COUNT:
        raise HarnessError(
            f"IFEval count mismatch: expected {IFEVAL_COUNT}, observed {len(ifeval_source)}"
        )
    ifeval_entries: list[tuple[int, dict[str, Any]]] = []
    for index, source in enumerate(ifeval_source):
        try:
            key = int(source["key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HarnessError(f"IFEval row {index} has no integer key") from exc
        if not isinstance(source.get("prompt"), str):
            raise HarnessError(f"IFEval row {index} has no string prompt")
        ifeval_entries.append((key, source))
    if len({key for key, _ in ifeval_entries}) != IFEVAL_COUNT:
        raise HarnessError("IFEval keys are not unique")
    ifeval_entries.sort(key=lambda pair: pair[0])

    # GSM8K is downloaded as the exact HF parquet at the pinned revision.  The
    # datasets package is used only to decode the local parquet, never to pick
    # a mutable remote revision implicitly.
    gsm_test_file = sources / "gsm8k-main-test.parquet"
    gsm_train_file = sources / "gsm8k-main-train.parquet"
    gsm_test_download = _download_verified(
        GSM_TEST_URL, gsm_test_file, GSM_TEST_SHA256, timeout
    )
    gsm_train_download = _download_verified(
        GSM_TRAIN_URL, gsm_train_file, GSM_TRAIN_SHA256, timeout
    )
    gsm_test = _load_parquet_rows(gsm_test_file, "GSM8K test")
    gsm_train = _load_parquet_rows(gsm_train_file, "GSM8K train")
    if len(gsm_test) != GSM_TEST_COUNT:
        raise HarnessError(
            f"GSM8K test count mismatch: expected {GSM_TEST_COUNT}, observed {len(gsm_test)}"
        )
    if len(gsm_train) != GSM_TRAIN_COUNT:
        raise HarnessError(
            f"GSM8K train count mismatch: expected {GSM_TRAIN_COUNT}, observed {len(gsm_train)}"
        )
    if any(
        not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str)
        for row in gsm_test + gsm_train
    ):
        raise HarnessError("GSM8K rows must contain string question and answer")
    gsm_shots = gsm_train[:GSM_FEWSHOT_COUNT]
    if len(gsm_shots) != GSM_FEWSHOT_COUNT:
        raise HarnessError("GSM8K fixed five-shot selection is incomplete")

    # EvalPlus 0.3.1 supplies and validates the two pinned release files.
    human_gz = sources / "HumanEvalPlus.jsonl.gz"
    mbpp_gz = sources / "MbppPlus.jsonl.gz"
    human_jsonl = sources / "HumanEvalPlus.jsonl"
    mbpp_jsonl = sources / "MbppPlus.jsonl"
    human_download = _download_verified(
        HUMANEVALPLUS_URL, human_gz, HUMANEVALPLUS_SHA256, timeout
    )
    mbpp_download = _download_verified(
        MBPPPLUS_URL, mbpp_gz, MBPPPLUS_SHA256, timeout
    )
    human_source, human_meta = _evalplus_release_rows(
        human_gz,
        human_jsonl,
        "humanevalplus",
        HUMANEVALPLUS_COUNT,
        HUMANEVALPLUS_RELEASE,
    )
    mbpp_source, mbpp_meta = _evalplus_release_rows(
        mbpp_gz, mbpp_jsonl, "mbppplus", MBPPPLUS_COUNT, MBPPPLUS_RELEASE
    )

    # Mark exactly the first 500 numerically sorted IFEval IDs for divergence.
    divergence_ids = {
        f"ifeval/{key}" for key, _ in ifeval_entries[: min(500, len(ifeval_entries))]
    }
    if len(divergence_ids) < 500:
        raise HarnessError("IFEval divergence selection is smaller than 500")

    prepared: list[dict[str, Any]] = []
    for source_index, (key, source) in enumerate(ifeval_entries):
        item_id = f"ifeval/{key}"
        prepared.append(
            _make_prompt_row(
                task="ifeval",
                item_id=item_id,
                source_id=item_id,
                source_index=source_index,
                source=source,
                prompt=source["prompt"],
                divergence_sample=item_id in divergence_ids,
                extra={
                    "scoring": "lm-eval-ifeval-strict-loose",
                    "instruction_ids": list(source["instruction_id_list"]),
                },
            )
        )
    gsm_stop = ["Question:", "</s>", "<|im_end|>"]
    for source_index, source in enumerate(gsm_test):
        item_id = f"gsm8k/{source_index:04d}"
        prepared.append(
            _make_prompt_row(
                task="gsm8k",
                item_id=item_id,
                source_id=item_id,
                source_index=source_index,
                source=source,
                prompt=_gsm_prompt(source["question"], gsm_shots),
                stop_sequences=gsm_stop,
                extra={
                    "scoring": "lm-eval-gsm8k-strict-flexible",
                    "fewshot_ids": [f"gsm8k/train/{i:04d}" for i in range(5)],
                    "fewshot_policy": GSM_FEWSHOT_POLICY,
                },
            )
        )
    for source_index, source in enumerate(sorted(human_source, key=lambda x: _natural_task_sort(str(x["task_id"])) )):
        task_id = str(source["task_id"])
        prepared.append(
            _make_prompt_row(
                task="humanevalplus",
                item_id=task_id,
                source_id=task_id,
                source_index=source_index,
                source=source,
                prompt=_code_prompt(str(source["prompt"])),
                extra={
                    "evaluation_prefix": str(source["prompt"]),
                    "entry_point": source.get("entry_point"),
                    "scoring": "evalplus-base-extra-pass-at-1",
                },
            )
        )
    for source_index, source in enumerate(sorted(mbpp_source, key=lambda x: _natural_task_sort(str(x["task_id"])) )):
        task_id = str(source["task_id"])
        prepared.append(
            _make_prompt_row(
                task="mbppplus",
                item_id=task_id,
                source_id=task_id,
                source_index=source_index,
                source=source,
                prompt=_code_prompt(str(source["prompt"])),
                extra={
                    "evaluation_prefix": str(source["prompt"]),
                    "entry_point": source.get("entry_point"),
                    "scoring": "evalplus-base-extra-pass-at-1",
                },
            )
        )

    if len(prepared) != EXPECTED_PRIMARY_TOTAL:
        raise HarnessError(
            f"prepared primary count mismatch: expected {EXPECTED_PRIMARY_TOTAL}, observed {len(prepared)}"
        )

    # Repeated rows are additional requests, never additional scored items.
    first32 = sorted(
        (row for row in prepared if row["task"] == "ifeval"),
        key=lambda row: int(_IFEVAL_ID_RE.match(row["id"]).group(1)),  # type: ignore[union-attr]
    )[:32]
    for repeat_index in (1, 2):
        for base in first32:
            repeat = dict(base)
            repeat["id"] = f"{base['id']}/repeat-{repeat_index}"
            repeat["sample_role"] = "repeat"
            repeat["repeat_of"] = base["id"]
            repeat["repeat_index"] = repeat_index
            repeat["prompt_sha256"] = base["prompt_sha256"]
            repeat["messages_sha256"] = base["messages_sha256"]
            prepared.append(repeat)
    if len(prepared) != EXPECTED_REQUEST_TOTAL:
        raise HarnessError(
            f"prepared request count mismatch: expected {EXPECTED_REQUEST_TOTAL}, observed {len(prepared)}"
        )
    if len({row["id"] for row in prepared}) != len(prepared):
        raise HarnessError("prepared item IDs are not unique")

    prepared_path = out / "prepared.jsonl"
    prepared_sha256 = _write_jsonl(prepared_path, prepared, force=True)
    manifest: dict[str, Any] = {
        "schema": HARNESS_SCHEMA,
        "created_at_utc": _utc_now(),
        "model": args.model,
        "pinned_evaluator": {
            "lm_eval": "0.4.13",
            "evalplus": EVALPLUS_VERSION,
            "no_local_execution": True,
        },
        "counts": {
            "primary": EXPECTED_PRIMARY_COUNTS,
            "primary_total": EXPECTED_PRIMARY_TOTAL,
            "repeat": REPEAT_COUNT,
            "requests_total": EXPECTED_REQUEST_TOTAL,
        },
        "divergence_sample": {
            "task": "ifeval",
            "count": len(divergence_ids),
            "selection": "first 500 numerically sorted IFEval IDs",
            "output_budget": OUTPUT_BUDGET,
        },
        "repeatability": {
            "base_task": "ifeval",
            "base_count": 32,
            "repeats_per_base": 2,
            "repeat_ids": "<base-id>/repeat-1 and <base-id>/repeat-2",
        },
        "request_contract": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": SEED,
            "stream": False,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_salt": CACHE_SALT,
            "max_tokens": OUTPUT_BUDGET,
            "eos": "normal (ignore_eos is deliberately absent)",
        },
        "datasets": {
            "ifeval": {
                "dataset": "google/IFEval",
                "split": "train",
                "revision": IFEVAL_REVISION,
                "count": IFEVAL_COUNT,
                "source": ifeval_download,
                "canonical_rows_sha256": _rows_hash(ifeval_source),
                "scoring_source": LMEVAL_IFEVAL_UTILS_URL,
            },
            "gsm8k": {
                "dataset": "openai/gsm8k",
                "config": "main",
                "test_split": "test",
                "train_split": "train",
                "revision": GSM_REVISION,
                "test_count": GSM_TEST_COUNT,
                "train_count": GSM_TRAIN_COUNT,
                "test_source": gsm_test_download,
                "train_source": gsm_train_download,
                "test_canonical_rows_sha256": _rows_hash(gsm_test),
                "train_canonical_rows_sha256": _rows_hash(gsm_train),
                "fixed_five_shot": {
                    "policy": GSM_FEWSHOT_POLICY,
                    "count": GSM_FEWSHOT_COUNT,
                    "source_indices": list(range(GSM_FEWSHOT_COUNT)),
                    "rows_sha256": _rows_hash(gsm_shots),
                },
                "scoring_source": LMEVAL_GSM8K_URL,
            },
            "humanevalplus": {
                **human_meta,
                "source": human_download,
                "release_url": HUMANEVALPLUS_URL,
            },
            "mbppplus": {
                **mbpp_meta,
                "source": mbpp_download,
                "release_url": MBPPPLUS_URL,
            },
        },
        "prepared_jsonl": {
            "path": str(prepared_path),
            "sha256": prepared_sha256,
            "rows": len(prepared),
        },
        "preservation": {
            "original_data": "each row includes source and source_sha256; raw source files are under sources/",
            "prompt_freeze": "each row includes prompt, messages, prompt_sha256, and messages_sha256",
        },
    }
    _write_json(out / "manifest.json", manifest, force=True)
    print(
        json.dumps(
            {
                "prepared": str(prepared_path),
                "manifest": str(out / "manifest.json"),
                "primary": EXPECTED_PRIMARY_TOTAL,
                "repeats": REPEAT_COUNT,
                "requests": EXPECTED_REQUEST_TOTAL,
                "divergence": len(divergence_ids),
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve_data(data_arg: str | os.PathLike[str]) -> tuple[Path, dict[str, Any] | None]:
    supplied = Path(data_arg).expanduser().resolve()
    if supplied.is_dir():
        prepared = supplied / "prepared.jsonl"
        manifest_path = supplied / "manifest.json"
    else:
        prepared = supplied
        manifest_path = supplied.parent / "manifest.json" if supplied.name == "prepared.jsonl" else Path()
    if not prepared.is_file():
        raise HarnessError(f"prepared data JSONL is missing: {prepared}")
    manifest: dict[str, Any] | None = None
    if manifest_path and manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("schema") != HARNESS_SCHEMA:
            raise HarnessError(f"unsupported prepared data schema in {manifest_path}")
        expected = manifest.get("prepared_jsonl", {}).get("sha256")
        observed = _sha256_file(prepared)
        if expected and observed != expected:
            raise HarnessError(
                f"prepared JSONL drift: expected {expected}, observed {observed}: {prepared}"
            )
    rows = _read_jsonl(prepared)
    ids: set[str] = set()
    for line_no, row in enumerate(rows, 1):
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise HarnessError(f"prepared row {line_no} has no string id")
        if item_id in ids:
            raise HarnessError(f"duplicate prepared id: {item_id}")
        ids.add(item_id)
        prompt = row.get("prompt")
        messages = row.get("messages")
        if prompt is not None and not isinstance(prompt, str):
            raise HarnessError(f"prepared row {item_id} has a non-string prompt")
        if (not isinstance(messages, list) or not messages) and not isinstance(prompt, str):
            raise HarnessError(f"prepared row {item_id} lacks prompt/messages")
        if row.get("prompt_sha256") and row["prompt_sha256"] != _sha256_json(prompt):
            raise HarnessError(f"prompt drift at {item_id}")
        if row.get("messages_sha256") and row["messages_sha256"] != _sha256_json(messages):
            raise HarnessError(f"messages drift at {item_id}")
    if manifest:
        expected_rows = manifest.get("counts", {}).get("requests_total")
        if expected_rows is not None and len(rows) != expected_rows:
            raise HarnessError(
                f"prepared row count drift: expected {expected_rows}, observed {len(rows)}"
            )
    return prepared, manifest


def _generation_meta_path(path: Path) -> Path:
    return Path(str(path) + ".meta.json")


def _load_generation(path_arg: str | os.PathLike[str]) -> tuple[Path, list[dict[str, Any]], dict[str, Any] | None]:
    path = Path(path_arg).expanduser().resolve()
    if not path.is_file():
        raise HarnessError(f"generation JSONL is missing: {path}")
    rows = _read_jsonl(path)
    ids: set[str] = set()
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or item_id in ids:
            raise HarnessError(f"generation IDs are missing or duplicated in {path}")
        ids.add(item_id)
    meta_path = _generation_meta_path(path)
    meta = _read_json(meta_path) if meta_path.is_file() else None
    return path, rows, meta


def _row_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise HarnessError(f"row {row.get('id')} has no messages or prompt")
        messages = [{"role": "user", "content": prompt}]
    if any(not isinstance(message, dict) for message in messages):
        raise HarnessError(f"row {row.get('id')} has invalid messages")
    return messages


def _build_request(row: dict[str, Any], model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _row_messages(row),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": SEED,
        "max_tokens": OUTPUT_BUDGET,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": CACHE_SALT,
        "return_token_ids": True,
    }
    stops = row.get("stop_sequences")
    if stops:
        if not isinstance(stops, list) or any(not isinstance(stop, str) for stop in stops):
            raise HarnessError(f"row {row.get('id')} has invalid stop_sequences")
        payload["stop"] = list(stops)
    return payload


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _id_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        return None
    return list(value)


def _extract_response_fields(
    parsed: dict[str, Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    fields: dict[str, Any] = {
        "content": None,
        "reasoning_content": None,
        "output_token_ids": None,
        "prompt_token_ids": None,
        "finish_reason": None,
        "stop_reason": None,
        "usage": parsed.get("usage"),
        "response_id": parsed.get("id"),
    }
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        errors.append({"kind": "missing_choices", "message": "response choices is empty or absent"})
        return fields, errors
    if len(choices) != 1:
        errors.append(
            {
                "kind": "unexpected_choice_count",
                "message": f"expected one choice, observed {len(choices)}; no choices are filtered",
            }
        )
        return fields, errors
    choice = choices[0]
    if not isinstance(choice, dict):
        errors.append({"kind": "invalid_choice", "message": "choice[0] is not an object"})
        return fields, errors
    message = choice.get("message")
    if not isinstance(message, dict):
        errors.append({"kind": "missing_message", "message": "choice[0].message is absent"})
    else:
        fields["content"] = message.get("content")
        fields["reasoning_content"] = message.get("reasoning_content")
        if not isinstance(fields["content"], str):
            errors.append({"kind": "missing_content", "message": "message.content is not a string"})
    output_ids_value = choice.get("token_ids", _MISSING)
    if output_ids_value is _MISSING:
        output_ids_value = parsed.get("token_ids", _MISSING)
    if output_ids_value is _MISSING:
        output_ids_value = parsed.get("output_token_ids", _MISSING)
    output_ids = None if output_ids_value is _MISSING else _id_list(output_ids_value)
    fields["output_token_ids"] = output_ids
    if output_ids is None:
        errors.append(
            {
                "kind": "missing_output_token_ids",
                "message": "return_token_ids=true did not yield an integer token_ids list; no IDs are inferred from text",
            }
        )
    prompt_ids_value = parsed.get("prompt_token_ids", _MISSING)
    if prompt_ids_value is _MISSING:
        prompt_ids_value = choice.get("prompt_token_ids", _MISSING)
    fields["prompt_token_ids"] = (
        None if prompt_ids_value is _MISSING else _id_list(prompt_ids_value)
    )
    if prompt_ids_value is _MISSING:
        errors.append(
            {
                "kind": "missing_prompt_token_ids",
                "message": "return_token_ids=true did not yield prompt_token_ids; no prompt IDs are inferred",
            }
        )
    elif fields["prompt_token_ids"] is None:
        errors.append(
            {"kind": "invalid_prompt_token_ids", "message": "prompt_token_ids is not an integer list"}
        )
    finish_reason = choice.get("finish_reason", _MISSING)
    if finish_reason is _MISSING or finish_reason is None:
        errors.append({"kind": "missing_finish_reason", "message": "finish_reason is absent"})
    elif not isinstance(finish_reason, str):
        errors.append({"kind": "invalid_finish_reason", "message": "finish_reason is not a string"})
    else:
        fields["finish_reason"] = finish_reason
    fields["stop_reason"] = choice.get("stop_reason")
    usage = fields["usage"]
    if not isinstance(usage, dict):
        errors.append({"kind": "missing_usage", "message": "response usage object is absent"})
    else:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        counts = (prompt_tokens, completion_tokens, total_tokens)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            errors.append({"kind": "invalid_usage_counts", "message": "usage token counts must be nonnegative integers"})
        else:
            if output_ids is not None and completion_tokens != len(output_ids):
                errors.append(
                    {
                        "kind": "completion_token_count_mismatch",
                        "message": f"usage completion_tokens={completion_tokens} but token_ids={len(output_ids)}",
                    }
                )
            if fields["prompt_token_ids"] is not None and prompt_tokens != len(fields["prompt_token_ids"]):
                errors.append(
                    {
                        "kind": "prompt_token_count_mismatch",
                        "message": f"usage prompt_tokens={prompt_tokens} but prompt_token_ids={len(fields['prompt_token_ids'])}",
                    }
                )
            if total_tokens != prompt_tokens + completion_tokens:
                errors.append(
                    {
                        "kind": "total_token_count_mismatch",
                        "message": f"usage total_tokens={total_tokens} but prompt+completion={prompt_tokens + completion_tokens}",
                    }
                )
    return fields, errors


def _error_body(raw: bytes) -> dict[str, Any]:
    return {
        "raw_response": raw.decode("utf-8", errors="replace"),
        "raw_response_b64": base64.b64encode(raw).decode("ascii"),
        "raw_response_sha256": _sha256_bytes(raw),
    }


def generate(args: argparse.Namespace) -> int:
    data_path, data_manifest = _resolve_data(args.data)
    rows = _read_jsonl(data_path)
    if args.limit is not None:
        if args.limit < 1:
            raise HarnessError("--limit must be a positive integer")
        selected = rows[: args.limit]
        if len(selected) != args.limit:
            raise HarnessError(f"--limit {args.limit} exceeds data rows {len(rows)}")
    else:
        selected = rows
    if not selected:
        raise HarnessError("no rows selected for generation")

    out = Path(args.out).expanduser().resolve()
    if out.exists() and not args.force:
        raise HarnessError(f"refusing to overwrite generation output; use --force: {out}")
    endpoint = _chat_endpoint(args.base_url)
    model = args.model
    records: list[dict[str, Any]] = []
    for row in selected:
        item_id = str(row["id"])
        payload = _build_request(row, model)
        body = _json_bytes(payload)
        request_started = time.monotonic()
        started_at = _utc_now()
        record: dict[str, Any] = {
            "schema": HARNESS_SCHEMA,
            "id": item_id,
            "source_id": row.get("source_id", item_id),
            "task": row.get("task", "external"),
            "sample_role": row.get("sample_role", "primary"),
            "repeat_of": row.get("repeat_of"),
            "repeat_index": row.get("repeat_index"),
            "divergence_sample": bool(row.get("divergence_sample", False)),
            "prompt": row.get("prompt"),
            "prompt_sha256": row.get("prompt_sha256") or _sha256_json(row.get("prompt")),
            "messages_sha256": row.get("messages_sha256") or _sha256_json(payload["messages"]),
            "request_payload": payload,
            "request_body": body.decode("utf-8"),
            "request_body_b64": base64.b64encode(body).decode("ascii"),
            "request_sha256": _sha256_bytes(body),
            "endpoint": endpoint,
            "started_at_utc": started_at,
            "http_status": None,
            "raw_response": None,
            "raw_response_b64": None,
            "raw_response_sha256": None,
            "response": None,
            "content": None,
            "reasoning_content": None,
            "output_token_ids": None,
            "token_ids": None,
            "prompt_token_ids": None,
            "finish_reason": None,
            "stop_reason": None,
            "usage": None,
            "truncated": False,
            "errors": [],
        }
        raw = b""
        try:
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    **(
                        {"Authorization": f"Bearer {args.api_key}"}
                        if args.api_key
                        else {}
                    ),
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=float(args.timeout)) as response:
                record["http_status"] = getattr(response, "status", None)
                raw = response.read()
            record.update(_error_body(raw))
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                record["errors"].append(
                    {"kind": "invalid_json_response", "message": str(exc)}
                )
                parsed = None
            if isinstance(parsed, dict):
                record["response"] = parsed
                fields, errors = _extract_response_fields(parsed, payload)
                record.update(fields)
                record["token_ids"] = record.get("output_token_ids")
                record["errors"].extend(errors)
                finish = record.get("finish_reason")
                usage = record.get("usage")
                record["truncated"] = finish == "length" or (
                    isinstance(usage, dict)
                    and usage.get("completion_tokens") == OUTPUT_BUDGET
                    and finish != "stop"
                )
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read()
            except OSError:
                raw = b""
            record["http_status"] = exc.code
            record.update(_error_body(raw))
            record["errors"].append(
                {"kind": "http_error", "message": str(exc), "status": exc.code}
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            record["errors"].append({"kind": "transport_error", "message": str(exc)})
            if raw:
                record.update(_error_body(raw))
        finally:
            record["elapsed_s"] = time.monotonic() - request_started
            record["finished_at_utc"] = _utc_now()
        record["ok"] = not bool(record["errors"])
        records.append(record)

    _write_jsonl(out, records, force=True)
    failed_ids = [record["id"] for record in records if not record["ok"]]
    meta = {
        "schema": HARNESS_SCHEMA,
        "created_at_utc": _utc_now(),
        "arm": args.arm,
        "model": model,
        "base_url": args.base_url,
        "endpoint": endpoint,
        "data": str(data_path),
        "data_manifest_sha256": (
            _sha256_file(data_path.parent / "manifest.json")
            if data_manifest and (data_path.parent / "manifest.json").is_file()
            else None
        ),
        "prepared_sha256": data_manifest.get("prepared_jsonl", {}).get("sha256")
        if data_manifest
        else None,
        "input_ids": [record["id"] for record in records],
        "requested": len(records),
        "failed": len(failed_ids),
        "failed_ids": failed_ids,
        "complete": not failed_ids,
        "full_run": args.limit is None,
        "explicit_limit": args.limit,
        "request_contract": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": SEED,
            "stream": False,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_salt": CACHE_SALT,
            "max_tokens": OUTPUT_BUDGET,
            "eos": "normal",
        },
        "no_retry_or_repair": True,
    }
    _write_json(_generation_meta_path(out), meta, force=True)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "output": str(out),
                "requested": len(records),
                "failed": len(failed_ids),
                "complete": not failed_ids,
            },
            sort_keys=True,
        )
    )
    return 0 if not failed_ids else 1


def _normalise_task_name(name: str) -> str:
    aliases = {
        "ifeval": "ifeval",
        "gsm8k": "gsm8k",
        "humaneval": "humanevalplus",
        "humanevalplus": "humanevalplus",
        "mbpp": "mbppplus",
        "mbppplus": "mbppplus",
    }
    if name not in aliases:
        raise HarnessError(f"unsupported task name: {name}")
    return aliases[name]


def _generation_for_rows(
    data_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    generation_meta: dict[str, Any] | None,
    task: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = {
        str(row["id"]): row
        for row in data_rows
        if row.get("sample_role", "primary") == "primary"
        and _normalise_task_name(str(row.get("task", "external"))) == task
    }
    selected: list[dict[str, Any]] = []
    unknown: list[str] = []
    for generated in generation_rows:
        item_id = str(generated.get("id"))
        if generated.get("sample_role", "primary") != "primary":
            continue
        if item_id not in primary:
            if _normalise_task_name(str(generated.get("task", "external"))) == task:
                unknown.append(item_id)
            continue
        selected.append(generated)
        expected = primary[item_id]
        if generated.get("prompt_sha256") != expected.get("prompt_sha256"):
            raise HarnessError(f"prompt identity mismatch for {item_id}")
        if generated.get("messages_sha256") != expected.get("messages_sha256"):
            raise HarnessError(f"message identity mismatch for {item_id}")
    if unknown:
        raise HarnessError(f"generation contains unknown {task} IDs: {unknown[:3]}")
    if not selected:
        raise HarnessError(f"no generated rows available for task {task}")
    if generation_meta and generation_meta.get("full_run"):
        expected_count = EXPECTED_PRIMARY_COUNTS.get(task)
        if expected_count is not None and len(selected) != expected_count:
            raise HarnessError(
                f"full generation for {task} is incomplete: expected {expected_count}, observed {len(selected)}"
            )
    for generated in selected:
        if not generated.get("ok"):
            raise HarnessError(
                f"cannot score failed generation row {generated.get('id')}: "
                f"{generated.get('errors')}"
            )
        if not isinstance(generated.get("content"), str):
            raise HarnessError(f"generation content missing for {generated.get('id')}")
        if not isinstance(generated.get("output_token_ids"), list):
            raise HarnessError(f"generation token IDs missing for {generated.get('id')}")
    return selected, [primary[str(row["id"])] for row in selected]


def _require_lm_eval() -> str:
    return _require_distribution("lm-eval", "0.4.13")


def _score_ifeval(generated: list[dict[str, Any]], data: list[dict[str, Any]]) -> dict[str, Any]:
    _require_lm_eval()
    try:
        from lm_eval.tasks.ifeval import utils as ifeval_utils  # type: ignore
    except ImportError as exc:
        raise HarnessError(
            "IFEval scoring requires lm-eval==0.4.13 in the remote evaluator image"
        ) from exc
    items: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {
        "prompt_level_strict_acc": [],
        "inst_level_strict_acc": [],
        "prompt_level_loose_acc": [],
        "inst_level_loose_acc": [],
    }
    for record, row in zip(generated, data):
        source = row.get("source")
        if not isinstance(source, dict):
            raise HarnessError(f"IFEval source missing for {row.get('id')}")
        try:
            result = ifeval_utils.process_results(source, [record["content"]])
        except Exception as exc:
            raise HarnessError(f"official IFEval scorer failed at {row.get('id')}: {exc}") from exc
        strict_inst = list(result["inst_level_strict_acc"])
        loose_inst = list(result["inst_level_loose_acc"])
        item = {
            "id": row["id"],
            "source_id": row.get("source_id"),
            "prompt_level_strict_acc": bool(result["prompt_level_strict_acc"]),
            "inst_level_strict_acc": [bool(value) for value in strict_inst],
            "inst_level_strict_all": all(strict_inst),
            "inst_level_strict_fraction": (
                sum(strict_inst) / len(strict_inst) if strict_inst else 0.0
            ),
            "prompt_level_loose_acc": bool(result["prompt_level_loose_acc"]),
            "inst_level_loose_acc": [bool(value) for value in loose_inst],
            "inst_level_loose_all": all(loose_inst),
            "inst_level_loose_fraction": (
                sum(loose_inst) / len(loose_inst) if loose_inst else 0.0
            ),
        }
        items.append(item)
        metric_values["prompt_level_strict_acc"].append(float(item["prompt_level_strict_acc"]))
        metric_values["inst_level_strict_acc"].extend(float(value) for value in strict_inst)
        metric_values["prompt_level_loose_acc"].append(float(item["prompt_level_loose_acc"]))
        metric_values["inst_level_loose_acc"].extend(float(value) for value in loose_inst)
    return {
        "task": "ifeval",
        "count": len(items),
        "metrics": {
            key: (sum(values) / len(values) if values else None)
            for key, values in metric_values.items()
        },
        "items": items,
        "scoring": {
            "package": "lm-eval",
            "version": "0.4.13",
            "strict_loose": True,
            "source": LMEVAL_IFEVAL_UTILS_URL,
        },
    }


def _gsm_normalize(value: str) -> str:
    value = value.lower()
    for pattern, replacement in (
        (",", ""),
        (r"\$", ""),
        (r"(?s).*#### ", ""),
        (r"\.$", ""),
    ):
        value = re.sub(pattern, replacement, value)
    return value


def _gsm_extract_strict(value: str) -> str | None:
    match = _GSM_STRICT_RE.search(value)
    return match.group(1).strip() if match else None


def _gsm_extract_flexible(value: str) -> str | None:
    matches = _GSM_FLEXIBLE_RE.findall(value)
    if not matches:
        return None
    match = matches[-1]
    if isinstance(match, tuple):
        nonempty = [part for part in match if part]
        return nonempty[0].strip() if nonempty else None
    return str(match).strip()


def _score_gsm8k(generated: list[dict[str, Any]], data: list[dict[str, Any]]) -> dict[str, Any]:
    _require_lm_eval()
    items: list[dict[str, Any]] = []
    strict_values: list[float] = []
    flexible_values: list[float] = []
    for record, row in zip(generated, data):
        source = row.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("answer"), str):
            raise HarnessError(f"GSM8K source answer missing for {row.get('id')}")
        gold = _gsm_normalize(source["answer"])
        strict_pred = _gsm_extract_strict(record["content"])
        flexible_pred = _gsm_extract_flexible(record["content"])
        strict = strict_pred is not None and _gsm_normalize(strict_pred) == gold
        flexible = flexible_pred is not None and _gsm_normalize(flexible_pred) == gold
        items.append(
            {
                "id": row["id"],
                "source_id": row.get("source_id"),
                "strict_prediction": strict_pred,
                "flexible_prediction": flexible_pred,
                "gold_normalized": gold,
                "strict_exact": strict,
                "flexible_exact": flexible,
            }
        )
        strict_values.append(float(strict))
        flexible_values.append(float(flexible))
    return {
        "task": "gsm8k",
        "count": len(items),
        "metrics": {
            "strict_exact": sum(strict_values) / len(strict_values),
            "flexible_exact": sum(flexible_values) / len(flexible_values),
        },
        "items": items,
        "scoring": {
            "package": "lm-eval",
            "version": "0.4.13",
            "source": LMEVAL_GSM8K_URL,
            "strict_regex": _GSM_STRICT_RE.pattern,
            "flexible_regex": _GSM_FLEXIBLE_RE.pattern,
        },
    }


def _require_code_sandbox(input_paths: Iterable[Path]) -> None:
    if os.environ.get("QUALITY_EVAL_SANDBOX") != "1":
        raise HarnessError(
            "EvalPlus code execution requires the supplied sandbox Docker image; "
            "QUALITY_EVAL_SANDBOX=1 is absent"
        )
    if not (Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()):
        raise HarnessError("refusing to execute generated code outside a container")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise HarnessError("refusing EvalPlus code execution as root")
    input_root = os.environ.get("QUALITY_INPUT_ROOT")
    if not input_root:
        raise HarnessError("QUALITY_INPUT_ROOT is absent; refusing unbounded code inputs")
    root = Path(input_root).resolve()
    for path in input_paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise HarnessError(
                f"generated code input is not under read-only sandbox root {root}: {resolved}"
            )

def _score_evalplus(
    generated: list[dict[str, Any]],
    data: list[dict[str, Any]],
    dataset_name: str,
    output: Path,
    force: bool,
) -> dict[str, Any]:
    _require_distribution("evalplus", EVALPLUS_VERSION)
    if dataset_name not in {"humanevalplus", "mbppplus"}:
        raise HarnessError(f"invalid EvalPlus dataset {dataset_name}")
    for row in data:
        source = row.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("task_id"), str):
            raise HarnessError(f"EvalPlus source/task_id missing for {row.get('id')}")
    if len(generated) != len(data):
        raise HarnessError("EvalPlus generated/data row count mismatch")
    with tempfile.TemporaryDirectory(prefix="qwen38-evalplus-") as temporary:
        temporary_dir = Path(temporary)
        override = temporary_dir / f"{dataset_name}.jsonl"
        # Restore the exact native JSON encoding consumed by EvalPlus's loader.
        override.write_text("".join(row["source_json"] + "\n" for row in data), encoding="utf-8")
        samples = temporary_dir / f"{dataset_name}.samples.jsonl"
        sample_rows = [
            {"task_id": row["source"]["task_id"], "solution": record["content"]}
            for record, row in zip(generated, data)
        ]
        _write_jsonl(samples, sample_rows, force=True)
        old_human = os.environ.get("HUMANEVAL_OVERRIDE_PATH")
        old_mbpp = os.environ.get("MBPP_OVERRIDE_PATH")
        os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(override) if dataset_name == "humanevalplus" else (old_human or "")
        os.environ["MBPP_OVERRIDE_PATH"] = str(override) if dataset_name == "mbppplus" else (old_mbpp or "")
        loaded_module = sys.modules.get(
            "evalplus.data.humaneval" if dataset_name == "humanevalplus" else "evalplus.data.mbpp"
        )
        if loaded_module is not None:
            if dataset_name == "humanevalplus":
                loaded_module.HUMANEVAL_OVERRIDE_PATH = str(override)
            else:
                loaded_module.MBPP_OVERRIDE_PATH = str(override)
        try:
            try:
                if dataset_name == "humanevalplus":
                    from evalplus.data import get_human_eval_plus  # type: ignore

                    loaded = get_human_eval_plus(err_incomplete=True, version=HUMANEVALPLUS_RELEASE)
                    eval_dataset = "humaneval"
                else:
                    from evalplus.data import get_mbpp_plus  # type: ignore

                    loaded = get_mbpp_plus(err_incomplete=True, version=MBPPPLUS_RELEASE)
                    eval_dataset = "mbpp"
                from evalplus.evaluate import evaluate  # type: ignore
                from evalplus.eval import estimate_pass_at_k  # type: ignore
            except ImportError as exc:
                raise HarnessError(
                    "EvalPlus scoring requires evalplus==0.3.1 in the remote evaluator image"
                ) from exc
            if set(loaded) != {row["source"]["task_id"] for row in data}:
                raise HarnessError("EvalPlus override task IDs do not match prepared rows")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    evaluate(
                        dataset=eval_dataset,
                        samples=str(samples),
                        base_only=False,
                        parallel=1,
                        test_details=False,
                        version=(
                            HUMANEVALPLUS_RELEASE
                            if dataset_name == "humanevalplus"
                            else MBPPPLUS_RELEASE
                        ),
                    )
                except Exception as exc:
                    raise HarnessError(f"official EvalPlus scorer failed: {exc}") from exc
            result_path = Path(str(samples).replace(".jsonl", "_eval_results.json"))
            if not result_path.is_file():
                raise HarnessError("EvalPlus completed without an eval_results JSON")
            official = _read_json(result_path)
            eval_map = official.get("eval")
            if not isinstance(eval_map, dict):
                raise HarnessError("EvalPlus result has no eval map")
            items: list[dict[str, Any]] = []
            for row in data:
                task_id = row["source"]["task_id"]
                values = eval_map.get(task_id)
                if not isinstance(values, list) or len(values) != 1:
                    raise HarnessError(f"EvalPlus result missing exactly one sample for {task_id}")
                value = values[0]
                if not isinstance(value, dict):
                    raise HarnessError(f"EvalPlus result row is invalid for {task_id}")
                base_status = value.get("base_status")
                plus_status = value.get("plus_status")
                if base_status not in {"pass", "fail", "timeout"} or plus_status not in {"pass", "fail", "timeout"}:
                    raise HarnessError(f"EvalPlus result has an invalid status for {task_id}")
                items.append(
                    {
                        "id": row["id"],
                        "source_id": row.get("source_id"),
                        "task_id": task_id,
                        "base_status": base_status,
                        "plus_status": plus_status,
                        "base_pass": base_status == "pass",
                        "plus_pass": plus_status == "pass",
                    }
                )
            base_correct = [int(item["base_pass"]) for item in items]
            plus_correct = [
                int(item["base_pass"] and item["plus_pass"]) for item in items
            ]
            # n=1 for each task; use EvalPlus's official estimator rather than
            # presenting a bespoke pass@1 definition.
            base_pass_at_1 = float(
                estimate_pass_at_k([1] * len(items), base_correct, 1).mean()
            )
            plus_pass_at_1 = float(
                estimate_pass_at_k([1] * len(items), plus_correct, 1).mean()
            )
            preserved = output.with_name(
                f"{output.stem}.{dataset_name}.eval_results.json"
            )
            _write_bytes(preserved, result_path.read_bytes(), force=force)
            return {
                "task": dataset_name,
                "count": len(items),
                "metrics": {
                    "base_pass@1": base_pass_at_1,
                    "plus_pass@1": plus_pass_at_1,
                },
                "items": items,
                "scoring": {
                    "package": "evalplus",
                    "version": EVALPLUS_VERSION,
                    "base_extra": True,
                    "official_result": str(preserved),
                    "official_result_sha256": _sha256_file(preserved),
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "source": [EVALPLUS_CLI_URL, EVALPLUS_EXECUTION_URL],
                },
            }
        finally:
            if old_human is None:
                os.environ.pop("HUMANEVAL_OVERRIDE_PATH", None)
            else:
                os.environ["HUMANEVAL_OVERRIDE_PATH"] = old_human
            if old_mbpp is None:
                os.environ.pop("MBPP_OVERRIDE_PATH", None)
            else:
                os.environ["MBPP_OVERRIDE_PATH"] = old_mbpp


def score(args: argparse.Namespace) -> int:
    data_path, manifest = _resolve_data(args.data)
    data_rows = _read_jsonl(data_path)
    generation_path, generation_rows, generation_meta = _load_generation(args.generation)
    requested = args.task
    if requested == "all":
        tasks = []
        seen: set[str] = set()
        for row in generation_rows:
            if row.get("sample_role", "primary") != "primary":
                continue
            task = _normalise_task_name(str(row.get("task", "external")))
            if task not in seen:
                tasks.append(task)
                seen.add(task)
    elif requested == "evalplus":
        tasks = ["humanevalplus", "mbppplus"]
    else:
        tasks = [_normalise_task_name(requested)]
    if not tasks:
        raise HarnessError("generation has no scorable primary tasks")
    if any(task in {"humanevalplus", "mbppplus"} for task in tasks):
        if not args.allow_code_execution:
            raise HarnessError(
                "EvalPlus scoring executes generated code; pass --allow-code-execution "
                "and run inside the supplied sandbox Docker image"
            )
        _require_code_sandbox([generation_path, data_path])

    out = (
        Path(args.out).expanduser().resolve()
        if args.out
        else generation_path.with_suffix(".scores.json")
    )
    if out.exists() and not args.force:
        raise HarnessError(f"refusing to overwrite score output; use --force: {out}")
    task_results: dict[str, Any] = {}
    for task in tasks:
        generated, data = _generation_for_rows(
            data_rows, generation_rows, generation_meta, task
        )
        if task == "ifeval":
            task_results[task] = _score_ifeval(generated, data)
        elif task == "gsm8k":
            task_results[task] = _score_gsm8k(generated, data)
        else:
            task_results[task] = _score_evalplus(
                generated, data, task, out, args.force
            )
    result = {
        "schema": HARNESS_SCHEMA,
        "created_at_utc": _utc_now(),
        "data": str(data_path),
        "generation": str(generation_path),
        "generation_meta": generation_meta,
        "task_results": task_results,
        "code_execution": any(
            task in {"humanevalplus", "mbppplus"} for task in tasks
        ),
        "no_fake_scores": True,
    }
    _write_json(out, result, force=True)
    print(
        json.dumps(
            {
                "score": str(out),
                "tasks": tasks,
                "counts": {task: value["count"] for task, value in task_results.items()},
            },
            sort_keys=True,
        )
    )
    return 0


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _score_file(path_arg: str | None, generation_path: Path) -> Path | None:
    if path_arg:
        path = Path(path_arg).expanduser().resolve()
    else:
        path = generation_path.with_suffix(".scores.json")
    return path if path.is_file() else None


def _read_score(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = _read_json(path)
    task_results = value.get("task_results")
    if not isinstance(task_results, dict):
        raise HarnessError(f"score file has no task_results: {path}")
    return value


def _score_metrics(value: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    if value is None:
        return {}
    result: dict[str, dict[str, float]] = {}
    for task, task_result in value.get("task_results", {}).items():
        metrics = task_result.get("metrics") if isinstance(task_result, dict) else None
        if not isinstance(metrics, dict):
            continue
        result[task] = {
            str(metric): number
            for metric, raw in metrics.items()
            if (number := _numeric(raw)) is not None
        }
    return result


def _score_items(value: dict[str, Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    if value is None:
        return {}
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for task, task_result in value.get("task_results", {}).items():
        items = task_result.get("items") if isinstance(task_result, dict) else None
        if not isinstance(items, list):
            continue
        result[task] = {
            str(item["id"]): item
            for item in items
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    return result


def _paired_uncertainty(
    target_items: dict[str, dict[str, Any]],
    arm_items: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    keys = sorted(
        {
            key
            for item_id in set(target_items) & set(arm_items)
            for key in set(target_items[item_id]) & set(arm_items[item_id])
            if isinstance(target_items[item_id][key], bool)
            and isinstance(arm_items[item_id][key], bool)
        }
    )
    for key in keys:
        differences: list[float] = []
        for item_id in sorted(set(target_items) & set(arm_items)):
            left = target_items[item_id].get(key)
            right = arm_items[item_id].get(key)
            if isinstance(left, bool) and isinstance(right, bool):
                differences.append(float(right) - float(left))
        if not differences:
            continue
        mean = statistics.fmean(differences)
        sample_sd = statistics.stdev(differences) if len(differences) > 1 else 0.0
        standard_error = sample_sd / math.sqrt(len(differences))
        metrics[key] = {
            "n": len(differences),
            "mean_delta": mean,
            "improved": sum(value > 0 for value in differences),
            "regressed": sum(value < 0 for value in differences),
            "tied": sum(value == 0 for value in differences),
            "descriptive_normal_95ci": [
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
            ],
        }
    return metrics


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index + 1
    if len(left) != len(right):
        return min(len(left), len(right)) + 1
    return None


def _divergence_summary(
    target: dict[str, dict[str, Any]],
    arm: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    exact = 0
    divergent = 0
    unavailable = 0
    positions: list[int] = []
    text_exact = 0
    for item_id in ids:
        left = target.get(item_id)
        right = arm.get(item_id)
        if not left or not right or not left.get("ok") or not right.get("ok"):
            unavailable += 1
            continue
        left_ids = left.get("output_token_ids")
        right_ids = right.get("output_token_ids")
        if not isinstance(left_ids, list) or not isinstance(right_ids, list):
            unavailable += 1
            continue
        if left_ids == right_ids:
            exact += 1
        else:
            divergent += 1
            first = _first_divergence(left_ids, right_ids)
            if first is not None:
                positions.append(first)
        if left.get("content") == right.get("content"):
            text_exact += 1
    distribution: dict[str, int] = {}
    for position in positions:
        key = str(position)
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "n_requested": len(ids),
        "n_available": exact + divergent,
        "exact_token_output_matches": exact,
        "exact_token_output_match_rate": (
            exact / (exact + divergent) if exact + divergent else None
        ),
        "requests_with_token_divergence": divergent,
        "request_divergence_rate": (
            divergent / (exact + divergent) if exact + divergent else None
        ),
        "unavailable": unavailable,
        "first_divergence_position_indexed_from_one": {
            "median": statistics.median(positions) if positions else None,
            "distribution": dict(sorted(distribution.items(), key=lambda pair: int(pair[0]))),
        },
        "exact_text_matches": text_exact,
        "output_budget": OUTPUT_BUDGET,
    }


def _repeatability_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bases = sorted(
        {
            str(row.get("repeat_of"))
            for row in records.values()
            if row.get("sample_role") == "repeat" and row.get("repeat_of")
        }
    )
    comparisons = {"base_vs_repeat_1": [], "base_vs_repeat_2": [], "repeat_1_vs_repeat_2": []}
    unavailable = 0
    for base_id in bases:
        base = records.get(base_id)
        repeats = [
            record
            for record in records.values()
            if record.get("repeat_of") == base_id
        ]
        repeats.sort(key=lambda record: int(record.get("repeat_index", 0)))
        by_index = {record.get("repeat_index"): record for record in repeats}
        for left_name, left, right_name, right in (
            ("base_vs_repeat_1", base, 1, by_index.get(1)),
            ("base_vs_repeat_2", base, 2, by_index.get(2)),
            ("repeat_1_vs_repeat_2", by_index.get(1), 1, by_index.get(2)),
        ):
            if not left or not right or not left.get("ok") or not right.get("ok"):
                unavailable += 1
                continue
            left_ids = left.get("output_token_ids")
            right_ids = right.get("output_token_ids")
            if not isinstance(left_ids, list) or not isinstance(right_ids, list):
                unavailable += 1
                continue
            comparisons[left_name].append(left_ids == right_ids)
    return {
        "base_count": len(bases),
        "unavailable_comparisons": unavailable,
        "comparisons": {
            name: {
                "n": len(values),
                "exact_token_matches": sum(values),
                "exact_match_rate": (sum(values) / len(values) if values else None),
            }
            for name, values in comparisons.items()
        },
    }

def _identity_summary(target: dict[str, dict[str, Any]], arm: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    request_matches = 0
    request_available = 0
    prompt_matches = 0
    prompt_available = 0
    messages_matches = 0
    messages_available = 0
    prompt_token_matches = 0
    prompt_token_available = 0
    text_matches = 0
    for item_id in ids:
        left, right = target.get(item_id), arm.get(item_id)
        if not left or not right:
            continue
        left_request = left.get("request_sha256")
        right_request = right.get("request_sha256")
        if isinstance(left_request, str) and isinstance(right_request, str):
            request_available += 1
            request_matches += left_request == right_request
        left_prompt = left.get("prompt_sha256")
        right_prompt = right.get("prompt_sha256")
        if isinstance(left_prompt, str) and isinstance(right_prompt, str):
            prompt_available += 1
            prompt_matches += left_prompt == right_prompt
        left_messages = left.get("messages_sha256")
        right_messages = right.get("messages_sha256")
        if isinstance(left_messages, str) and isinstance(right_messages, str):
            messages_available += 1
            messages_matches += left_messages == right_messages
        if isinstance(left.get("prompt_token_ids"), list) and isinstance(right.get("prompt_token_ids"), list):
            prompt_token_available += 1
            prompt_token_matches += left["prompt_token_ids"] == right["prompt_token_ids"]
        if isinstance(left.get("content"), str) and isinstance(right.get("content"), str):
            text_matches += left["content"] == right["content"]
    return {
        "n_common": len(ids),
        "request_payload_hash_comparisons": request_available,
        "request_payload_hash_matches": request_matches,
        "prompt_hash_comparisons": prompt_available,
        "prompt_hash_matches": prompt_matches,
        "messages_hash_comparisons": messages_available,
        "messages_hash_matches": messages_matches,
        "prompt_token_id_comparisons": prompt_token_available,
        "prompt_token_id_matches": prompt_token_matches,
        "output_text_comparisons": sum(
            isinstance(target.get(item_id, {}).get("content"), str)
            and isinstance(arm.get(item_id, {}).get("content"), str)
            for item_id in ids
        ),
        "output_text_matches": text_matches,
    }


def compare(args: argparse.Namespace) -> int:
    target_path, target_rows, target_meta = _load_generation(args.target)
    native_path, native_rows, native_meta = _load_generation(args.native)
    candidate_path, candidate_rows, candidate_meta = _load_generation(args.candidate)
    by_arm = {
        "target": {str(row["id"]): row for row in target_rows},
        "native": {str(row["id"]): row for row in native_rows},
        "candidate": {str(row["id"]): row for row in candidate_rows},
    }
    issues: list[str] = []
    target_ids = set(by_arm["target"])
    for arm in ("native", "candidate"):
        missing = sorted(target_ids - set(by_arm[arm]))
        extra = sorted(set(by_arm[arm]) - target_ids)
        if missing or extra:
            issues.append(
                f"{arm} input ID set differs (missing={len(missing)}, extra={len(extra)})"
            )
    common_ids = sorted(target_ids & set(by_arm["native"]) & set(by_arm["candidate"]))
    if not common_ids:
        raise HarnessError("no common generation IDs to compare")

    divergence_ids = sorted(
        item_id
        for item_id in common_ids
        if by_arm["target"][item_id].get("task") == "ifeval"
        and by_arm["target"][item_id].get("sample_role", "primary") == "primary"
        and by_arm["target"][item_id].get("divergence_sample")
    )
    if target_meta and target_meta.get("full_run") and len(divergence_ids) < 500:
        issues.append(
            f"full compare has only {len(divergence_ids)} marked divergence prompts; required >=500"
        )
    for arm, meta in (("target", target_meta), ("native", native_meta), ("candidate", candidate_meta)):
        if any(not row.get("ok", False) for row in by_arm[arm].values()):
            issues.append(f"{arm} generation contains failed requests")
        elif meta and not meta.get("complete", False):
            issues.append(f"{arm} generation reports failed requests")

    target_score_path = _score_file(args.target_score, target_path)
    native_score_path = _score_file(args.native_score, native_path)
    candidate_score_path = _score_file(args.candidate_score, candidate_path)
    score_paths = {
        "target": target_score_path,
        "native": native_score_path,
        "candidate": candidate_score_path,
    }
    missing_scores = [arm for arm, path in score_paths.items() if path is None]
    if missing_scores:
        issues.append(f"missing score files for: {', '.join(missing_scores)}")
    scores = {
        arm: _read_score(path)
        for arm, path in score_paths.items()
    }
    metric_maps = {arm: _score_metrics(value) for arm, value in scores.items()}
    all_tasks = sorted({task for metrics in metric_maps.values() for task in metrics})
    score_comparison: dict[str, Any] = {}
    for task in all_tasks:
        score_comparison[task] = {}
        metric_names = sorted(
            {
                metric
                for arm in metric_maps
                for metric in metric_maps[arm].get(task, {})
            }
        )
        for metric in metric_names:
            values = {
                arm: metric_maps[arm].get(task, {}).get(metric) for arm in metric_maps
            }
            target_value = values.get("target")
            score_comparison[task][metric] = {
                "target": target_value,
                "native": values.get("native"),
                "candidate": values.get("candidate"),
                "native_minus_target": (
                    values["native"] - target_value
                    if values.get("native") is not None and target_value is not None
                    else None
                ),
                "candidate_minus_target": (
                    values["candidate"] - target_value
                    if values.get("candidate") is not None and target_value is not None
                    else None
                ),
            }

    item_maps = {arm: _score_items(value) for arm, value in scores.items()}
    paired: dict[str, Any] = {}
    for task in sorted(set(item_maps["target"]) | set(item_maps["native"]) | set(item_maps["candidate"])):
        target_items = item_maps["target"].get(task, {})
        paired[task] = {
            "native_vs_target": _paired_uncertainty(
                target_items, item_maps["native"].get(task, {})
            ),
            "candidate_vs_target": _paired_uncertainty(
                target_items, item_maps["candidate"].get(task, {})
            ),
        }
    identity = {
        "native_vs_target": _identity_summary(by_arm["target"], by_arm["native"], common_ids),
        "candidate_vs_target": _identity_summary(by_arm["target"], by_arm["candidate"], common_ids),
    }
    for arm, summary in identity.items():
        if summary["request_payload_hash_comparisons"] != len(common_ids):
            issues.append(f"{arm} request payload hashes are unavailable for some common IDs")
        elif summary["request_payload_hash_matches"] != len(common_ids):
            issues.append(f"{arm} request payload hashes differ from target")
        if summary["prompt_hash_comparisons"] != len(common_ids):
            issues.append(f"{arm} prompt hashes are unavailable for some common IDs")
        elif summary["prompt_hash_matches"] != len(common_ids):
            issues.append(f"{arm} prompt hashes differ from target")
        if summary["messages_hash_comparisons"] != len(common_ids):
            issues.append(f"{arm} message hashes are unavailable for some common IDs")
        elif summary["messages_hash_matches"] != len(common_ids):
            issues.append(f"{arm} message hashes differ from target")

    output = {
        "schema": HARNESS_SCHEMA,
        "created_at_utc": _utc_now(),
        "interpretation": (
            "Descriptive paired differences only; no broad quality-equivalence, "
            "non-inferiority, or deployment claim is made."
        ),
        "arms": {
            arm: {
                "generation": str(path),
                "generation_meta": meta,
                "score": str(score_paths[arm]) if score_paths[arm] else None,
                "score_sha256": _sha256_file(score_paths[arm]) if score_paths[arm] else None,
                "requests": len(rows),
                "failed": sum(not row.get("ok", False) for row in rows),
                "truncated": {
                    "count": sum(bool(row.get("truncated")) for row in rows),
                    "ids": [row["id"] for row in rows if row.get("truncated")],
                },
            }
            for arm, path, rows, meta in (
                ("target", target_path, target_rows, target_meta),
                ("native", native_path, native_rows, native_meta),
                ("candidate", candidate_path, candidate_rows, candidate_meta),
            )
        },
        "alignment": {
            "target_ids": len(target_ids),
            "common_ids": len(common_ids),
            "divergence_sample_ids": len(divergence_ids),
            "divergence_sample_requirement_met": len(divergence_ids) >= 500,
            "issues": issues,
        },
        "identity": identity,
        "token_divergence": {
            "sample": "marked primary IFEval rows; first 500 sorted IDs in a full prepared run",
            "native_vs_target": _divergence_summary(
                by_arm["target"], by_arm["native"], divergence_ids
            ),
            "candidate_vs_target": _divergence_summary(
                by_arm["target"], by_arm["candidate"], divergence_ids
            ),
        },
        "repeatability": {
            arm: _repeatability_summary(by_arm[arm])
            for arm in ("target", "native", "candidate")
        },
        "scores": score_comparison,
        "descriptive_paired_uncertainty": paired,
        "score_files": {arm: str(path) if path else None for arm, path in score_paths.items()},
    }
    out = Path(args.out).expanduser().resolve()
    if out.exists() and not args.force:
        raise HarnessError(f"refusing to overwrite compare output; use --force: {out}")
    _write_json(out, output, force=True)
    print(json.dumps({"compare": str(out), "issues": len(issues), "common": len(common_ids)}, sort_keys=True))
    return 0 if not issues else 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment-only Qwen3.8 grouped quality harness; no inference runtime"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="download and freeze pinned benchmark data")
    prep.add_argument("--out", required=True, help="new prepared-data directory")
    prep.add_argument("--model", default=MODEL_NAME)
    prep.add_argument("--timeout", type=float, default=120.0)
    prep.add_argument("--force", action="store_true")
    prep.set_defaults(function=prepare)

    gen = subparsers.add_parser("generate", help="issue one HTTP request per frozen row")
    gen.add_argument("--arm", required=True, choices=("target", "native", "candidate"))
    gen.add_argument("--data", required=True, help="prepared directory or JSONL")
    gen.add_argument("--out", required=True, help="generation JSONL")
    gen.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    gen.add_argument("--model", default=MODEL_NAME)
    gen.add_argument("--timeout", type=float, default=120.0)
    gen.add_argument("--api-key")
    gen.add_argument("--limit", type=int, help="explicit nonfull prefix smoke run")
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(function=generate)

    scoring = subparsers.add_parser("score", help="score saved outputs offline")
    scoring.add_argument(
        "--task",
        default="all",
        choices=("all", "ifeval", "gsm8k", "evalplus", "humaneval", "humanevalplus", "mbpp", "mbppplus"),
    )
    scoring.add_argument("--data", required=True)
    scoring.add_argument("--generation", "--input", dest="generation", required=True)
    scoring.add_argument("--out")
    scoring.add_argument("--allow-code-execution", action="store_true")
    scoring.add_argument("--force", action="store_true")
    scoring.set_defaults(function=score)

    cmp = subparsers.add_parser("compare", help="compare the three saved arms")
    cmp.add_argument("--target", required=True)
    cmp.add_argument("--native", required=True)
    cmp.add_argument("--candidate", required=True)
    cmp.add_argument("--target-score", "--target-scores")
    cmp.add_argument("--native-score", "--native-scores")
    cmp.add_argument("--candidate-score", "--candidate-scores")
    cmp.add_argument("--out", required=True)
    cmp.add_argument("--force", action="store_true")
    cmp.set_defaults(function=compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except HarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: interrupted; no retry or repair was performed", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
