#!/usr/bin/env python3
"""Read-only Qwen3.8 teacher execution-path diagnostic. No optimizer and no training.

offline: retained HF rows vs native aligned rows and in-bounds ±1 neighbors.
hf: actual Checkpoint/FrozenTarget replay vs uncropped hybrid-cache decode.
Numeric gates remain parity.errors. Parent owns GPU publication.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import qwen38_mtp_live_parity as parity
import qwen38_train_mtp as trainer

MAX_ENTRIES, MAX_TOKENS, SCORED = 128, 1_048_576, 4
KERNELS = ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
           "torch_recurrent_gated_delta_rule")


def fail(reason):
    raise ValueError(reason)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_entries(path):
    try:
        entries = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest is not a JSON list: {path}") from exc
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_ENTRIES:
        fail(f"manifest must contain 1..{MAX_ENTRIES} entries")
    seen, tokens, rows = set(), 0, []
    for entry in entries:
        if not isinstance(entry, dict):
            fail("manifest entry must be an object")
        name, ids = entry.get("name"), entry.get("input_ids")
        if not isinstance(name, str) or not name or name in (".", "..") or any(c in name for c in "/\\"):
            fail("unsafe entry name")
        if name in seen:
            fail(f"duplicate entry name: {name}")
        seen.add(name)
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            fail("manifest input_ids must be nonnegative ints")
        if len(ids) < SCORED + 1:
            fail("prefix shorter than 5")
        if entry.get("scored_teacher_positions") != list(range(len(ids) - SCORED, len(ids))):
            fail("scored positions are not the last four")
        tokens += len(ids)
        if tokens > MAX_TOKENS:
            fail(f"manifest exceeds {MAX_TOKENS} tokens")
        rows.append((entry, ids))
    return rows


def as_long(tensor, what):
    if not torch.is_tensor(tensor) or tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        fail(f"{what} ids must be a 1D integer tensor")
    return tensor.long()


def require_bf16(tensor, what, rows):
    finite = torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all())
    if not finite or tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or tensor.shape[0] != rows:
        fail(f"{what} must be finite BF16")


def load_pt(path):
    path = Path(path)
    if not path.is_file():
        fail(f"missing tensor file: {path.name}")
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot read tensor file: {path.name}") from exc


def pair_metrics(actual, reference):
    left, right = actual.detach().float().cpu().reshape(-1), reference.detach().float().cpu().reshape(-1)
    if left.shape != right.shape or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        fail("nonfinite neighbor comparison")
    left_norm, right_norm = left.norm(), right.norm()
    if left_norm == 0 or right_norm == 0:
        fail("zero-norm neighbor row")
    return {"cosine": float(torch.dot(left, right) / (left_norm * right_norm)),
            "relative_l2": float((left - right).norm() / right_norm)}


def neighbor_row(hf_row, hidden, position):
    """One existing HF row vs native[position±1]. Out of range stays absent; no wrap, no invented onset."""
    choices = {0: pair_metrics(hf_row, hidden[position])}
    item = {"position": position, "aligned": choices[0], "minus": None, "plus": None}
    for offset, label in ((-1, "minus"), (1, "plus")):
        index = position + offset
        if 0 <= index < hidden.shape[0]:
            choices[offset] = item[label] = pair_metrics(hf_row, hidden[index])
    def winner(key, higher):
        best = 0
        for offset, metrics in choices.items():
            better = metrics[key] > choices[best][key] if higher else metrics[key] < choices[best][key]
            if offset and better:
                best = offset
        return best
    cosine, rel = winner("cosine", True), winner("relative_l2", False)
    item.update(aligned_best_cosine=cosine == 0, neighbor_better_cosine=cosine != 0,
                aligned_best_relative_l2=rel == 0, neighbor_better_relative_l2=rel != 0,
                best_cosine_offset=cosine, best_relative_l2_offset=rel)
    return item


def verifier_block(path, ids, hf_rows, native_rows, positions):
    if not Path(path).is_file():
        return None
    payload = load_pt(path)
    rows = payload.get("teacher_rows") if isinstance(payload, dict) else None
    require_bf16(rows, "verifier teacher_rows", SCORED)
    if rows.shape != hf_rows.shape:
        fail("verifier teacher_rows shape mismatch")
    if "prefix" in payload and not torch.equal(as_long(payload["prefix"], "verifier"), ids):
        fail("verifier prefix mismatch")
    if "positions" in payload and torch.as_tensor(payload["positions"]).reshape(-1).tolist() != positions:
        fail("verifier positions are not the scored positions")
    return {"native_prefill_errors": parity.errors(native_rows, rows), "hf_errors": parity.errors(hf_rows, rows)}


def offline_report(manifest, hf_dir, native_dir, verifier_dir=None):
    rows = []
    for entry, ids in load_entries(manifest):
        expected = torch.tensor(ids, dtype=torch.long)
        hf, native = load_pt(Path(hf_dir) / f"{entry['name']}.pt"), load_pt(Path(native_dir) / f"{entry['name']}.pt")
        if not isinstance(hf, dict) or not isinstance(native, dict):
            fail("tensor file must be a dict")
        if not torch.equal(as_long(hf.get("prefix"), "hf"), expected) or not torch.equal(
                as_long(native.get("input_ids"), "native"), expected):
            fail("exact ID/prefix mismatch")
        positions = native.get("positions")
        if not torch.is_tensor(positions) or positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64) or not torch.equal(positions.long(), torch.arange(len(ids))):
            fail("gapped or noncontiguous positions")
        hidden, hf_rows = native.get("target_last_hidden_states"), hf.get("teacher_rows")
        require_bf16(hidden, "native hidden", len(ids))
        require_bf16(hf_rows, "hf teacher_rows", SCORED)
        if hf_rows.shape[1] != hidden.shape[1]:
            fail("hf teacher_rows must be 4xH")
        native_rows = hidden[-SCORED:]
        neighbors = [neighbor_row(hf_rows[i], hidden, len(ids) - SCORED + i) for i in range(SCORED)]
        row = {"name": entry["name"], "prefix_tokens": len(ids),
               "scored_positions": list(range(len(ids) - SCORED, len(ids))),
               "errors": parity.errors(hf_rows, native_rows), "neighbors": neighbors,
               "aligned_best": all(n["aligned_best_cosine"] and n["aligned_best_relative_l2"] for n in neighbors),
               "neighbor_better": any(n["neighbor_better_cosine"] or n["neighbor_better_relative_l2"] for n in neighbors)}
        if verifier_dir is not None:
            block = verifier_block(Path(verifier_dir) / f"{entry['name']}.pt", expected, hf_rows, native_rows,
                                   row["scored_positions"])
            if block:
                row["verifier"] = block
        rows.append(row)
    verifier_rows = [row["verifier"] for row in rows if "verifier" in row]
    if verifier_dir is not None and not verifier_rows:
        fail("verifier histories required")
    numeric = all(row["errors"]["pass"] for row in rows)
    flat = [item for row in rows for item in row["neighbors"]]
    report = {"status": "aligned_numeric_pass" if numeric else "teacher_parity_gate_open", "mode": "offline",
              "optimizer_updates": 0, "optimizer": None, "training_allowed": False, "entries": len(rows),
              "tokens": sum(row["prefix_tokens"] for row in rows), "tolerance": parity.TOLERANCE,
              "numeric_passed": numeric,
              "aligned_best_cosine_rows": sum(item["aligned_best_cosine"] for item in flat),
              "neighbor_better_cosine_rows": sum(item["neighbor_better_cosine"] for item in flat),
              "aligned_best_relative_l2_rows": sum(item["aligned_best_relative_l2"] for item in flat),
              "neighbor_better_relative_l2_rows": sum(item["neighbor_better_relative_l2"] for item in flat),
              "aligned_best_histories": sum(row["aligned_best"] for row in rows),
              "neighbor_better_histories": sum(row["neighbor_better"] for row in rows), "rows": rows,
              "scope": "HF scored rows vs native all-N rows; ±1 neighbors in bounds only; no invented HF onset"}
    if verifier_rows:
        report["verifier_histories"] = len(verifier_rows)
        report["verifier_numeric_passed"] = all(item["native_prefill_errors"]["pass"] and item["hf_errors"]["pass"]
                                                for item in verifier_rows)
    return report


def assert_frozen(teacher):
    model = getattr(teacher, "model", None)
    if model is not None and (model.training or any(p.requires_grad or p.grad is not None for p in model.parameters())):
        fail("teacher must stay frozen with zero gradients")


def replay_full_rows(teacher, tokens):
    proposals = torch.cat([tokens[-3:], tokens.new_zeros(1)]).reshape(1, SCORED)
    rows = teacher.replay({"input_ids": tokens[:-3]}, [tokens.numel() - 5], proposals)
    if rows.ndim != 3 or tuple(rows.shape[:2]) != (1, SCORED):
        fail("teacher replay did not return one depth-four row")
    return rows[0]


def cached_text_rows(teacher, tokens):
    """Prefill prefix[:-4], then four decode steps. Chain the model cache; do not crop it."""
    length = tokens.numel()
    first = teacher.text(input_ids=tokens[:-SCORED].unsqueeze(0), past_key_values=None, use_cache=True,
                         position_ids=torch.arange(length - SCORED, device=tokens.device).unsqueeze(0))
    past, rows = first.past_key_values, []
    if past is None:
        fail("teacher prefill did not return a hybrid cache")
    for position in range(length - SCORED, length):
        step = teacher.text(input_ids=tokens[position:position + 1].unsqueeze(0), past_key_values=past, use_cache=True,
                            position_ids=torch.tensor([[position]], device=tokens.device))
        past = step.past_key_values
        if past is None:
            fail("teacher decode did not return a hybrid cache")
        rows.append(step.last_hidden_state[0, -1])
    return torch.stack(rows), past


def shared_argmax(teacher, rows):
    with trainer.autocast(teacher.device, dtype=torch.bfloat16):
        return teacher.head(rows).float().argmax(-1).detach().cpu().tolist()


def run_hf_prefix(teacher, prefix):
    tokens = torch.as_tensor(prefix, dtype=torch.long, device=getattr(teacher, "device", "cpu"))
    assert_frozen(teacher)
    with torch.no_grad():
        started = trainer.synchronized_time(teacher.device)
        full = replay_full_rows(teacher, tokens)
        full_seconds = trainer.synchronized_time(teacher.device) - started
        started = trainer.synchronized_time(teacher.device)
        cached, past = cached_text_rows(teacher, tokens)
        cached_seconds = trainer.synchronized_time(teacher.device) - started
        if cached.shape != full.shape or not bool(torch.isfinite(cached).all() and torch.isfinite(full).all()):
            fail("nonfinite or mismatched cached/full rows")
        require_bf16(full, "HF full rows", SCORED)
        require_bf16(cached, "HF cached rows", SCORED)
        errors = parity.errors(cached.detach().cpu(), full.detach().cpu())
        full_ids, cached_ids = shared_argmax(teacher, full), shared_argmax(teacher, cached)
    assert_frozen(teacher)
    return {"prefix": tokens.detach().cpu(), "full_rows": full.detach().cpu(), "cached_rows": cached.detach().cpu(),
            "past": past, "errors": errors, "full_argmax": full_ids, "cached_argmax": cached_ids,
            "argmax_equal": full_ids == cached_ids, "full_seconds": full_seconds, "cached_seconds": cached_seconds}


def _file(obj):
    try:
        return inspect.getfile(obj)
    except (TypeError, OSError):
        return None


def describe(fn):
    return {"name": getattr(fn, "__qualname__", type(fn).__name__),
            "module": getattr(fn, "__module__", None), "file": _file(fn)}


def bound_name(fn):
    """Resolve the callable a decorator actually invokes, not only the wrapper."""
    impl, seen = fn, set()
    while id(impl) not in seen:
        seen.add(id(impl))
        found = []
        for cell in getattr(impl, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) and id(value) not in seen:
                found.append(value)
        if len(found) == 1:
            impl = found[0]
            continue
        wrapped = getattr(impl, "__wrapped__", None)
        if callable(wrapped) and id(wrapped) not in seen:
            impl = wrapped
            continue
        break
    info = describe(fn)
    runtime = describe(impl)
    info.update(runtime=runtime["name"], runtime_module=runtime["module"], runtime_file=runtime["file"])
    return info


def cache_facts(cache):
    if cache is None:
        return None
    facts = {"class": type(cache).__name__, "module": type(cache).__module__, "seq_length": None, "layer_classes": []}
    length = getattr(cache, "get_seq_length", None)
    if callable(length):
        try:
            facts["seq_length"] = length()
        except Exception:
            pass
    layers = list(getattr(cache, "layers", None) or [])
    facts["layer_classes"] = sorted({type(layer).__name__ for layer in layers})
    buckets = {"conv": set(), "recurrent": set(), "attention_key": set()}
    for layer in layers:
        for key, attr in (("conv", "conv_states"), ("recurrent", "recurrent_states")):
            states = getattr(layer, attr, None)
            values = [states] if torch.is_tensor(states) else states.values() if isinstance(states, dict) else [] if states is None else states
            buckets[key].update(str(state.dtype).removeprefix("torch.") for state in values if torch.is_tensor(state))
        if torch.is_tensor(getattr(layer, "keys", None)):
            buckets["attention_key"].add(str(layer.keys.dtype).removeprefix("torch."))
    facts["state_dtypes"] = {key: sorted(value) for key, value in buckets.items()}
    return facts


def kernel_facts(teacher, cache):
    # Read bound GDN callables and cache dtypes. Do not install FLA or change USE_HUB_KERNELS.
    facts = {"use_hub_kernels": os.environ.get("USE_HUB_KERNELS"), "cache": cache_facts(cache),
             "linear_modules": [], "functions": []}
    for layer in getattr(getattr(teacher, "text", None), "layers", None) or []:
        module = getattr(layer, "linear_attn", None)
        if module is None or any(item["class"] == type(module).__name__ for item in facts["linear_modules"]):
            continue
        facts["linear_modules"].append({"class": type(module).__name__, "module": type(module).__module__,
                                        "file": _file(type(module)), "forward": bound_name(module.forward)})
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
        facts["functions"] = [{"symbol": name, **bound_name(getattr(modeling, name))}
                              for name in KERNELS if hasattr(modeling, name)]
    except ImportError:
        pass
    return facts


def hf_report(model, manifest, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    checkpoint = trainer.Checkpoint(model)
    teacher = trainer.FrozenTarget(checkpoint, device)
    if getattr(teacher, "model", None) is not None:
        teacher.model.requires_grad_(False).eval()
    rows, past = [], None
    for entry, ids in load_entries(manifest):
        result = run_hf_prefix(teacher, ids)
        past = result["past"]
        torch.save({key: result[key] for key in ("prefix", "full_rows", "cached_rows")}, output_dir / f"{entry['name']}.pt")
        rows.append({"name": entry["name"], "prefix_tokens": len(ids), "errors": result["errors"],
                     "argmax_equal": result["argmax_equal"], "full_argmax": result["full_argmax"],
                     "cached_argmax": result["cached_argmax"], "full_seconds": result["full_seconds"],
                     "cached_seconds": result["cached_seconds"]})
    numeric, argmax = all(row["errors"]["pass"] for row in rows), all(row["argmax_equal"] for row in rows)
    raw = getattr(checkpoint, "raw", None)
    return {"status": "hf_full_cached_internal_pass" if numeric and argmax else "hf_full_cached_internal_gate_open",
            "mode": "hf_cached", "optimizer_updates": 0, "optimizer": None, "training_allowed": False,
            "device": str(device), "checkpoint_revision": raw.get("_commit_hash") if isinstance(raw, dict) else None,
            "entries": len(rows), "tokens": sum(row["prefix_tokens"] for row in rows), "tolerance": parity.TOLERANCE,
            "numeric_passed": numeric, "argmax_passed": argmax, "kernels": kernel_facts(teacher, past), "rows": rows,
            "scope": "HF-internal FrozenTarget.replay versus teacher.text hybrid-cache decode; cache chained, not cropped. Not native serving parity."}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("offline")
    for flag in ("--manifest", "--hf-dir", "--native-dir", "--output"):
        offline.add_argument(flag, type=Path, required=True)
    offline.add_argument("--verifier-dir", type=Path)
    hf = commands.add_parser("hf")
    for flag in ("--model", "--manifest", "--output-dir"):
        hf.add_argument(flag, type=Path, required=True)
    hf.add_argument("--device", default="cuda")
    return cli


def main(argv=None):
    os.umask(0o077)
    args = parser().parse_args(argv)
    if args.command == "offline":
        report = offline_report(args.manifest, args.hf_dir, args.native_dir, args.verifier_dir)
        write_json(args.output, report)
        ok = report["numeric_passed"]
    else:
        report = hf_report(args.model, args.manifest, args.output_dir, args.device)
        write_json(args.output_dir / "fidelity.json", report)
        ok = report["numeric_passed"] and report["argmax_passed"]
    print(f"TEACHER_FIDELITY_STATUS={report['status']} ZERO_UPDATES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
