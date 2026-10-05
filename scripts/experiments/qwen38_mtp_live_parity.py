#!/usr/bin/env python3
r"""Disposable LIVE native-MTP parity diagnostic; not a benchmark/training run.

Lead-only E2E: no rental/start/stop commands are executed by this program.
Baseline: pinned stock BF16 native MTP D4/D8. Controls: complete stock overlay
D4/D8 and no-spec. Differences: eager, C1, cold prefix, public train/dev only,
<=1024 prompt tokens, 64 generated tokens, first 16 rounds, <=128 MiB trace bytes.
The existing D4 native capture adds <=2200 BF16 target rows (~22 MiB).
Keep the original nine serving cells separate, capture OFF. No timing claims.

On the authorized idle H100, create a NEW external root per cell:
  mkdir -m 700 /external/run/parity-stock4
In a disposable external copy of run-pilot.py (NEVER its archived original):
  * Point HOME/CODE/RUN/MODEL at the new run, unchanged trainer and exact
    snapshot 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
  * Use Server(..., capture=False), preserving the existing /profile mount.
    Additionally mount /external/run/parity-stock4:/profile/parity (RW).
    Mount existing patches dir, including qwen38_mtp_parity.py, /patches:ro.
  * Retain --dtype bfloat16 --kv-cache-dtype auto --max-model-len 8192
    --max-num-seqs 1 --max-num-batched-tokens 2048 --no-async-scheduling
    --language-model-only. Add --enforce-eager, REMOVE --compilation-config,
    use --no-enable-prefix-caching.
  * Retain the pinned image, VLLM_USE_V2_MODEL_RUNNER=0 and
    VLLM_ENABLE_CUDA_COMPATIBILITY=1; add docker environment:
      -e QWEN38_MTP_PARITY_DIR=/profile/parity
      -e QWEN38_MTP_PARITY_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
      -e QWEN38_MTP_PARITY_IMAGE=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
  * Retain both existing patch_mtp_training.py / patch_mtp_native_capture.py
    setup commands, then insert: python3 /patches/qwen38_mtp_parity.py
    before exec vllm serve. No PYTHONPATH change is needed.
  * At D4 ONLY, also add (the client writes its existing control files):
      -e B70_MTP_NATIVE_CAPTURE_DIR=/profile/parity/native
      -e B70_MTP_NATIVE_MAX_TOKENS=2200 -e B70_MTP_NATIVE_MAX_REQUESTS=2
    This exercises the UNCHANGED original capture at the same public boundary.
    D8 cannot use that D4-only collector; its row mapping is measured separately.
  * DO NOT call evaluate(), canary or warmup: unselected requests fail closed.
    /health and /v1/models are safe. Use request below.
    For no-spec omit all capture/parity env and parity installer; same client.
    For stock-overlay use Server(weights=complete_stock_export), produced by:
      python3 qwen38_train_mtp.py --model /external/model --export-stock \
        --output /external/run/stock.safetensors

Client/replay commands (lead-provided paths; no implicit historical root):
  python3 qwen38_mtp_live_parity.py request --root /external/run/parity-stock4 \
    --hook /external/code/patches/qwen38_mtp_parity.py \
    --requests /external/run/train-requests.jsonl --split train --prompt-id ID
  # Optionally a second dev request; same selected IDs in every control cell.
  # Stop the owned serving container first, then in the isolated CUDA ML env:
  python3 qwen38_mtp_live_parity.py check --root /external/run/parity-stock4 \
    --hook /external/code/patches/qwen38_mtp_parity.py \
    --trainer /external/code/qwen38_train_mtp.py --model /external/model

Expected: exact token/position/target-row/feedback/slot-prefix agreement; fixed
BF16 tolerance AND identical argmax at every sampled state. Inspect raw errors
and unobserved classes. Repeat D8/stock-overlay controls; compare response choice
token_ids with no-spec and report nonidentity. Retain actual argv, container
digest, snapshot provenance, collect_env and logs through the existing workflow.
This tool retains only selected requests/responses, installed source, raw tensors
and parity-report.json; never pass private signed configs. Retain failures; after
artifact readback, clean up only owned disposable files/containers. No-spec has
no proposer trace and cannot pass check. CPU tests do NOT close live Gate A.

Research (official sources inspected 2026-10-05):
https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/spec_decode/llm_base_proposer.py
https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/qwen3_5_mtp.py
Unchanged first-pass positions, embed-first separate norms, final norm, sampled
recurrence, padded rejected suffix then slot reuse. HF replay uses the unchanged
trainer's native model, frozen_heads, aligned_inputs and prefix_cache APIs.
It rebuilds KV from COMPLETE prompt-origin live inputs, cropping at each actual
write position. It never invents an empty prefix or equates sampled histories
with ground-truth teacher rows. Slot identity is checked, not direct KV values.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import hashlib
import sys
import urllib.request

# Declared before live results. BF16 unit roundoff is 2^-8 (~0.0039); these
# conservative cross-kernel gates allow accumulation/SDPA differences (~8 u in
# relative L2), but never tolerate an index/position error or argmax disagreement.
# Max-abs is also bounded in units of each reference row's RMS, not hidden scale.
TOLERANCE = {"relative_l2": 0.03, "cosine": 0.999, "max_abs_over_rms": 0.25}
CLASSES = {"prefill", "full_acceptance", "partial_rejection", "zero_acceptance"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def positions(tensor):
    import torch
    require(tensor.dtype in (torch.int32, torch.int64), "noninteger positions")
    if tensor.ndim == 2:
        require(tensor.shape[0] == 3 and torch.equal(tensor, tensor[0:1].expand_as(tensor)),
                "non-text rotary positions")
        tensor = tensor[0]
    require(tensor.ndim == 1, "invalid positions rank")
    return tensor.tolist()


def valid_samples(tensor):
    values = tensor.tolist()
    require(len(values) == 1 and isinstance(values[0], list), "expected C1 sampler matrix")
    row = values[0]
    result = [v for v in row if 0 <= v < 248320]
    require(result and row == result + [-1] * (len(row) - len(result)), "noncontiguous/invalid sampler output")
    return result


def round_layout(trace, prompt, cursor, stream, previous_drafts):
    """Strict discrete contract, independent of floating point tolerances."""
    ids, pos = trace["target_ids"].tolist(), positions(trace["target_positions"])
    n, outputs, drafts = len(ids), valid_samples(trace["sampled"]), trace["scheduled_drafts"]
    require(pos == list(range(cursor, cursor + n)), "target position gap/rebase/wrong row")
    require(trace["depth"] in (4, 8), "unsupported draft depth")
    if cursor == 0:
        require(ids == prompt and not drafts and len(outputs) == 1, "missing complete cold prefill")
        keep, kind = n, "prefill"
        stream.extend(prompt + outputs)
    else:
        require(n == len(drafts) + 1 and 1 <= len(outputs) <= n, "invalid verification rows/count")
        require(ids == [stream[cursor]] + drafts, "target input/scheduled draft mismatch")
        require(drafts == previous_drafts[:len(drafts)], "scheduled drafts not previous live proposal")
        require(ids[1:len(outputs)] == outputs[:-1], "accepted target rows do not match sampler")
        keep = len(outputs)
        kind = ("no_drafts" if not drafts else "zero_acceptance" if keep == 1 else
                "full_acceptance" if keep == n else "partial_rejection")
        stream.extend(outputs)
    first = trace["first"]
    require(first["selected"].tolist() == [keep - 1], "wrong proposer-selected target row")
    require(first["next_token_ids"].tolist() == [outputs[-1]], "wrong bonus/next input")
    rejected = first["num_rejected_tokens_gpu"]
    require((rejected is None and cursor == 0) or
            (rejected is not None and rejected.tolist() == [n - keep]), "wrong rejected suffix length")
    require(first["target_token_ids"].tolist() == ids and positions(first["target_positions"]) == pos,
            "proposer target token/position selection differs from runner")
    base = trace["calls"][0]
    require(base["input_ids"][:keep].tolist() == ids[1:keep] + [outputs[-1]], "wrong first-pass token shift")
    require(positions(base["positions"]) == pos, "first-pass position shifted/rebased")
    return keep, kind


def check_slots(call, valid, block_size, historical_slots):
    """Verify logical prefix continuity against actual forward-context slots."""
    pos = positions(call["positions"])
    require(pos and pos == list(range(pos[0], pos[0] + len(pos))), "branch positions not contiguous")
    require(call["query_start"].tolist() == [0, len(pos)] and
            call["seq_lens"].tolist() == [pos[0] + len(pos)], "wrong attention query/sequence boundary")
    table = call["block_table"].tolist()
    require(len(table) == 1 and block_size > 0, "unsupported block table")
    def slot(p):
        require(p // block_size < len(table[0]), "missing prefix block")
        block = table[0][p // block_size]
        require(block >= 0, "invalid prefix block")
        return block * block_size + p % block_size
    for p in range(pos[0]):
        require(p in historical_slots and historical_slots[p] == slot(p), "missing/moved prefix KV slot")
    expected = [slot(p) for p in pos[:valid]]
    require(call["slots"][:valid].tolist() == expected, "wrong live slot mapping")
    for p in list(historical_slots):
        if p >= pos[0]:
            del historical_slots[p]
    historical_slots.update(zip(pos[:valid], expected))


def errors(actual, reference):
    import torch
    a, r = actual.detach().float().cpu(), reference.detach().float().cpu()
    require(a.shape == r.shape and a.ndim == 2 and a.numel() > 0, "output shape mismatch")
    require(bool(torch.isfinite(a).all() and torch.isfinite(r).all()), "nonfinite parity tensors")
    an, rn = a.norm(dim=-1), r.norm(dim=-1)
    require(bool((rn > 0).all() and (an > 0).all()), "zero-norm parity row")
    diff = (a - r).norm(dim=-1)
    maxabs = (a - r).abs().amax(dim=-1)
    cosine = (a * r).sum(-1) / (an * rn)
    rel = diff / rn
    scaled_max = maxabs / (rn / r.shape[-1] ** 0.5)
    return {
        "reference_norm": rn.tolist(), "actual_norm": an.tolist(), "error_norm": diff.tolist(),
        "norm_error": (an - rn).tolist(), "relative_l2": rel.tolist(), "cosine": cosine.tolist(),
        "max_abs": maxabs.tolist(), "max_abs_over_rms": scaled_max.tolist(),
        "pass": bool((rel <= TOLERANCE["relative_l2"]).all() and
                     (cosine >= TOLERANCE["cosine"]).all() and
                     (scaled_max <= TOLERANCE["max_abs_over_rms"]).all()),
    }


def compare_native_step(trace, native):
    """Compare ORIGINAL native-capture artifacts from the same real API call."""
    import torch
    keep = trace["first"]["selected"].item() + 1
    require(native["request_id"] == trace["request_id"] and native["step"] == trace["step"],
            "native capture request/round mismatch")
    require(native["input_ids"].tolist() == trace["target_ids"][:keep].tolist() and
            positions(native["positions"]) == positions(trace["target_positions"])[:keep],
            "original native capture token/position row selection mismatch")
    require(torch.equal(native["target_last_hidden_states"], trace["target_hidden"][:keep]),
            "original native capture hidden row selection mismatch")
    require(native["output_ids"].tolist() == valid_samples(trace["sampled"]),
            "original native capture sampler output mismatch")
    return keep


def replay(traces, control, response_ids, trainer, model, embedding, head, device):
    import torch
    require(traces and [t["step"] for t in traces] == list(range(len(traces))), "missing/reordered trace rounds")
    cache, recurrent_cache = trainer.runtime().Cache(), trainer.runtime().Cache()
    stream, hidden_rows, base_ids, base_positions, historical_slots = [], [], [], [], {}
    cursor, previous_drafts, observed, rows = 0, [], set(), []
    for trace in traces:
        require(trace["request_id"] == control["request_id"], "mixed requests")
        require(len(trace["calls"]) == len(trace["samples"]) == trace["depth"], "missing live calls/samples")
        keep, kind = round_layout(trace, control["prompt_token_ids"], cursor, stream, previous_drafts)
        observed.add(kind)
        require(torch.equal(trace["first"]["target_hidden_states"], trace["target_hidden"]),
                "proposer not using captured post-final-norm target rows")
        hidden_rows.append(trace["target_hidden"][:keep])
        base_ids.extend(trace["calls"][0]["input_ids"][:keep].tolist())
        base_positions.extend(positions(trace["calls"][0]["positions"])[:keep])
        previous_live = previous_hf = None
        for depth, (call, sampled) in enumerate(zip(trace["calls"], trace["samples"])):
            valid = keep if depth == 0 else 1
            pos = positions(call["positions"])[:valid]
            ids = call["input_ids"][:valid].long().to(device)
            live_input = call["hidden"][:valid].to(device)
            require(live_input.dtype == torch.bfloat16 and call["output"].dtype == torch.bfloat16, "non-BF16 live boundary")
            require(bool(torch.isfinite(live_input).all()), "nonfinite live inputs")
            if depth == 0:
                require(torch.equal(call["hidden"], trace["target_hidden"]), "first-pass hidden row mismatch")
            else:
                require(pos == [cursor + keep - 1 + depth], "wrong recursive position")
                require(ids.tolist() == trace["samples"][depth - 1]["ids"].tolist(), "not the previous sampled token")
                require(torch.equal(call["hidden"], previous_live), "wrong live recursive feedback state")
            check_slots(call, valid, trace["block_size"], historical_slots)
            # All prefix rows come from observed real calls, including the whole
            # prefill. Cropping discards rejected padding and stale sibling KV.
            require(pos[0] <= cache.get_seq_length() and pos[0] <= recurrent_cache.get_seq_length(), "missing HF KV prefix")
            cache = trainer.prefix_cache(cache, pos[0]) if pos[0] else trainer.runtime().Cache()
            recurrent_cache = (trainer.prefix_cache(recurrent_cache, pos[0]) if pos[0]
                               else trainer.runtime().Cache())
            input_positions = torch.tensor(pos, device=device, dtype=torch.long)
            if call["inputs_embeds"] is not None:
                require(torch.equal(embedding(ids).cpu(), call["inputs_embeds"][:valid]), "shared embedding mismatch")
            with trainer.autocast(device, dtype=embedding.weight.dtype):
                offline = model(ids, live_input, input_positions, embedding, past_key_values=cache)
                recurrence = model(ids, live_input if depth == 0 else previous_hf,
                                   input_positions, embedding, past_key_values=recurrent_cache)
                offline_argmax = head(offline[-1:]).argmax(-1).tolist()
                recurrence_argmax = head(recurrence[-1:]).argmax(-1).tolist()
                live_head_argmax = head(call["output"][valid - 1:valid].to(device)).argmax(-1).tolist()
            live_output = call["output"][:valid]
            require(torch.equal(sampled["hidden"], live_output[-1:]), "sampler chose wrong MTP row")
            live_argmax = sampled["ids"].tolist()
            metrics, recursive_metrics = errors(offline, live_output), errors(recurrence, live_output)
            same_argmax = offline_argmax == recurrence_argmax == live_head_argmax == live_argmax
            rows.append(dict(round=trace["step"], depth=depth + 1, kind=kind, positions=pos,
                             live_input=metrics, recurrence=recursive_metrics, live_argmax=live_argmax,
                             hf_argmax=offline_argmax, hf_recurrence_argmax=recurrence_argmax,
                             frozen_head_on_live_argmax=live_head_argmax, argmax_equal=same_argmax))
            previous_live, previous_hf = live_output[-1:], recurrence[-1:]
        previous_drafts = [sample["ids"].item() for sample in trace["samples"]]
        require(trace["draft_ids"].tolist() == [previous_drafts], "returned proposal differs from samples")
        cursor += keep
    generated = stream[len(control["prompt_token_ids"]):]
    common = min(len(generated), len(response_ids))
    require(common > 0 and generated[:common] == response_ids[:common], "trace/API output token mismatch")
    # Exercise the actual unchanged trainer alignment API using ONLY observed
    # target rows. Exclude sampler tail beyond API cap and the terminal no-label
    # row, exactly as the trainer does; do not manufacture future teacher states.
    tokens = stream[:len(control["prompt_token_ids"]) + common]
    target_hidden = torch.cat(hidden_rows)[:len(tokens)]
    record = dict(input_ids=torch.tensor(tokens), positions=torch.arange(len(target_hidden)),
                  target_last_hidden_states=target_hidden, loss_mask=torch.ones(len(tokens), dtype=torch.bool))
    aligned_ids, _, aligned_pos, labels, _ = trainer.aligned_inputs(record, "cpu")
    count = aligned_ids.numel()
    require(aligned_ids.tolist() == base_ids[:count] and aligned_pos.tolist() == base_positions[:count] and
            labels.tolist() == tokens[2:count + 2], "unchanged trainer alignment disagrees with live first-pass rows")
    return dict(observed_classes=sorted(observed), untested_classes=sorted(CLASSES - observed),
                untested=["EOS (ignore_eos=true)", "preemption", "chunked/prefix-cached prefill", "graphs/async/batched serving"],
                trace_rounds=len(traces), traced_generated_tokens=common, api_generated_tokens=len(response_ids),
                trainer_aligned_rows=count, rows=rows,
                observed_numeric_pass=all(r["live_input"]["pass"] and r["recurrence"]["pass"] and r["argmax_equal"] for r in rows))


def post(base, path, body):
    request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def request_command(args, hook):
    require(args.root.is_absolute() and args.root.is_dir() and args.root.stat().st_mode & 0o077 == 0,
            "lead must create a new mode-0700 absolute external root")
    require(args.requests.name == args.split + "-requests.jsonl", "use the existing public train/dev request file, not private configs")
    selected = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    selected = [r for r in selected if r.get("prompt_id") == args.prompt_id]
    require(len(selected) == 1, "public prompt_id not found/ambiguous")
    messages = selected[0]["messages"]
    require(messages and all(set(m) == {"role", "content"} and m["role"] in ("user", "assistant", "system") and
                             isinstance(m["content"], str) for m in messages), "only selected public text messages")
    key = "b70-native-" + hashlib.sha256((args.split + "/" + args.prompt_id).encode()).hexdigest()[:32]
    require(len(list(args.root.glob("*.control.json"))) < hook.MAX_REQUESTS, "request bound exceeded")
    common = dict(model="qwen38", messages=messages, chat_template_kwargs={"enable_thinking": False})
    rendered = post(args.base_url, "/tokenize", dict(common, add_generation_prompt=True))
    control = hook.validate_control(dict(request_id=key, split=args.split, prompt_id=args.prompt_id,
                                         prompt_token_ids=rendered["tokens"], max_tokens=hook.MAX_OUTPUT), key)
    body = dict(common, request_id=key, max_tokens=hook.MAX_OUTPUT, temperature=0, seed=42,
                ignore_eos=True, stream=False, n=1, return_token_ids=True)
    hook.private_write(args.root / (key + ".control.json"), json.dumps(control).encode())
    native = args.root / "native"
    native.mkdir(mode=0o700, exist_ok=True)
    (native / key).mkdir(mode=0o700)
    native_control = {k: control[k] for k in ("request_id", "prompt_token_ids", "max_tokens")}
    hook.private_write(native / key / "control.json", json.dumps(native_control).encode())
    hook.private_write(args.root / (key + ".request.json"), json.dumps(body).encode())
    try:
        response = post(args.base_url, "/v1/chat/completions", body)
        hook.private_write(args.root / (key + ".response.json"), json.dumps(response).encode())
        require(response.get("prompt_token_ids") == rendered["tokens"], "API prompt IDs differ from tokenize")
        require(len(response["choices"]) == 1 and len(response["choices"][0]["token_ids"]) == hook.MAX_OUTPUT,
                "missing/truncated public output token IDs")
    except Exception as exc:
        hook.private_write(args.root / (key + ".failure.txt"), str(exc).encode())
        raise
    print(key + ": public API completed; no parity claim until check")


def check_command(args, hook):
    import torch
    report = dict(tier="development", tolerance=TOLERANCE, gate_a_closed=False, requests=[], status="incomplete")
    try:
        require(args.device == "cuda" and torch.cuda.is_available(), "real BF16 CUDA replay required; CPU tests are not live parity")
        runtime = json.loads((args.root / "runtime.json").read_text())
        require(runtime["vllm"] == "0.27.1" and runtime["declared_image"] == hook.IMAGE and
                runtime["declared_model_revision"] == hook.REVISION, "wrong trace runtime declarations")
        trainer = module_at(args.trainer, "parity_existing_trainer")
        checkpoint = trainer.Checkpoint(args.model)
        require(checkpoint.regime == "native_bf16", "no quantized teacher/head substitution")
        model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state(), args.device).to(torch.bfloat16).eval()
        model.requires_grad_(False)
        embedding, head = trainer.frozen_heads(checkpoint, args.device)
        controls = sorted(args.root.glob("*.control.json"))
        require(1 <= len(controls) <= hook.MAX_REQUESTS, "missing/too many public request controls")
        files = list(args.root.glob("*.pt"))
        require(sum(p.stat().st_size for p in files) <= hook.MAX_BYTES and
                1 <= len(files) <= hook.MAX_ROUNDS * hook.MAX_REQUESTS, "trace count/byte bound exceeded")
        with torch.inference_mode():
            for path in controls:
                key = path.name.removesuffix(".control.json")
                control = hook.validate_control(json.loads(path.read_text()), key)
                traces = [torch.load(p, map_location="cpu", weights_only=True) for p in sorted(args.root.glob(key + ".*.pt"))]
                response = json.loads((args.root / (key + ".response.json")).read_text())
                require(response["prompt_token_ids"] == control["prompt_token_ids"], "wrong API prompt")
                result = replay(traces, control, response["choices"][0]["token_ids"], trainer, model, embedding, head, args.device)
                result.update(request_id=key, round_limit_reached=len(traces) == hook.MAX_ROUNDS)
                require(traces[0]["depth"] != 4 or runtime["native_capture_enabled"],
                        "D4 requires the unchanged native capture at the same API boundary; see --help")
                if runtime["native_capture_enabled"]:
                    result["original_capture_rows_checked"] = sum(compare_native_step(trace, torch.load(
                        args.root / "native" / key / f"step-{trace['step']:06d}.pt",
                        map_location="cpu", weights_only=True)) for trace in traces)
                else:
                    result["untested"].append("original native-capture collector is D4-only (D8 not supported)")
                report["requests"].append(result)
        report["status"] = "observed_parity_pass" if all(r["observed_numeric_pass"] for r in report["requests"]) else "numeric_or_argmax_failure"
        report["note"] = "Single-cell diagnostic only; Gate A still needs D4/D8 stock/complete-overlay, no-spec identity and coverage review. Sampled recurrence is not ground-truth teacher forcing."
    except Exception as exc:
        report.update(status="blocked", error=str(exc))
    hook.private_write(args.root / "parity-report.json", json.dumps(report, indent=2, allow_nan=False).encode())
    print(json.dumps({"status": report["status"], "gate_a_closed": False, "report": str(args.root / "parity-report.json")}))
    return 0 if report["status"] == "observed_parity_pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("request", "check"):
        sub = commands.add_parser(name)
        sub.add_argument("--root", type=Path, required=True)
        sub.add_argument("--hook", type=Path, required=True)
        if name == "request":
            sub.add_argument("--requests", type=Path, required=True)
            sub.add_argument("--split", choices=("train", "dev"), required=True)
            sub.add_argument("--prompt-id", required=True)
            sub.add_argument("--base-url", default="http://127.0.0.1:8000")
        else:
            sub.add_argument("--trainer", type=Path, required=True)
            sub.add_argument("--model", type=Path, required=True)
            sub.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    hook = module_at(args.hook, "parity_disposable_hook")
    return request_command(args, hook) if args.command == "request" else check_command(args, hook)


if __name__ == "__main__":
    raise SystemExit(main())
