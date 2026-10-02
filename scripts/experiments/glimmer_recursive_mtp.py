#!/usr/bin/env python3
"""Bounded, development-tier frozen Muse Glimmer recursive-MTP pilot.

Heavy dependencies are imported only by GPU commands / the head factory. See
`glimmer_recursive_mtp.md` for the model, cache, alignment and measurement contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time
import unicodedata

MODEL = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
TRANSFORMERS_COMMIT = "35dff0957a99d50eaf85d7852a96fd29e59052d6"
WEIGHTS = (1.0, 1.0, .8, .8, .5, .5, .5, .5)
DEPTHS = (1, 2, 4, 8)
VARIANTS = ("fixed-ce", "shared-ce", "shared-state")
CATEGORIES = {"code", "prose", "reasoning", "structured", "repetitive", "high-entropy"}
# This bound includes prompt, generated output AND every uncommitted proposal.
MAX_CONTEXT = 1792
FIXTURES = Path(__file__).with_name("glimmer_recursive_mtp.jsonl")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def validate_records(records):
    require(bool(records), "empty corpus")
    ids, texts, tokens, families = set(), {}, {}, {}
    for row in records:
        require(all(isinstance(row.get(k), str) and row[k].strip()
                    for k in ("id", "family", "split", "text")), "invalid record fields")
        require(row["split"] in ("train", "eval"), "split must be train or eval")
        require(row["id"] not in ids, "duplicate sequence id")
        ids.add(row["id"])
        text = canonical(row["text"])
        require(text not in texts, "duplicate/overlapping full sequence text")
        texts[text] = row["split"]
        split = families.setdefault(row["family"], row["split"])
        require(split == row["split"], "prompt family overlaps train/eval")
        if "token_ids" in row:
            key = tuple(row["token_ids"])
            require(key and all(isinstance(t, int) and 0 <= t < 202048 for t in key),
                    "invalid token IDs")
            require(key not in tokens, "duplicate/overlapping tokenized sequence")
            tokens[key] = row["split"]
        if row["split"] == "eval":
            require(row.get("category") in CATEGORIES, "unknown heldout category")
    require({r["split"] for r in records} == {"train", "eval"}, "need train and eval sequences")
    return records


def read_records(path):
    with Path(path).open() as handle:
        return validate_records([json.loads(line) for line in handle if line.strip()])


def check_heldout(training, evaluation):
    families = {r["family"] for r in training}
    texts = {canonical(r["text"]) for r in training}
    tokens = {tuple(r["token_ids"]) for r in training}
    for row in evaluation:
        require(row["family"] not in families, "checkpoint training family in evaluation")
        require(canonical(row["text"]) not in texts, "checkpoint training text in evaluation")
        require(tuple(row["token_ids"]) not in tokens, "checkpoint training tokens in evaluation")


def context_guard(prompt_length, max_new_tokens, depth, limit=MAX_CONTEXT):
    require(prompt_length > 0 and max_new_tokens > 0, "empty prompt or output budget")
    require(depth in (0, *DEPTHS), "unsupported depth")
    require(limit <= MAX_CONTEXT and prompt_length + max_new_tokens + depth + 1 <= limit,
            "prompt + output + speculative margin must fit below SWA 2048 (pilot limit 1792)")


def make_head(hidden_size, rank, shared, max_depth=8):
    """Only the small blocks are registered: no target weights in the checkpoint."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(2 * hidden_size, rank, bias=False)
            self.up = nn.Linear(rank, hidden_size, bias=False)
            self.gate = nn.Parameter(torch.full((hidden_size,), -2.0))
            nn.init.normal_(self.up.weight, std=.01)

        def forward(self, state, embedding, norm):
            z = torch.cat((state.float(), embedding.float()), dim=-1)
            delta = self.up(torch.nn.functional.silu(self.down(z)))
            # Use the actual frozen target final RMSNorm, including its learned scale.
            return norm(state.float() + self.gate.sigmoid() * delta).to(state.dtype)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(1 if shared else max_depth)])

        def step(self, state, embedding, depth, norm):
            require(1 <= depth <= max_depth, "head depth out of range")
            return self.blocks[0 if shared else depth - 1](state, embedding, norm)

    return Head()


def project(lm_head, state, text_config):
    """Match the official Glimmer forward, including BF16 scale/softcap order."""
    import torch
    logits = lm_head(state.to(lm_head.weight.dtype))
    logits = logits * text_config.output_multiplier
    logits = logits / text_config.final_logit_softcapping
    logits = torch.tanh(logits)
    return logits * text_config.final_logit_softcapping


def state_errors(predicted, teacher):
    import torch.nn.functional as F
    p, t = predicted.float(), teacher.float()
    p = p * (p.square().mean(-1, keepdim=True) + 1e-8).rsqrt()
    t = t * (t.square().mean(-1, keepdim=True) + 1e-8).rsqrt()
    return (p - t).square().mean(), (1 - F.cosine_similarity(p, t, dim=-1)).mean()


def training_loss(head, states, tokens, target, state_weight=0., kl_weight=0.):
    """At step d: consume x[t+d], predict x[t+d+1], supervise h[t+d]."""
    import torch
    import torch.nn.functional as F
    state = states[:, 0]  # No teacher hidden state is fed after this root.
    loss, metrics = 0., []
    for d, weight in enumerate(WEIGHTS, 1):
        with torch.no_grad():
            embedding = target.embedding(tokens[:, d])
        state = head.step(state, embedding, d, target.norm)
        logits = project(target.lm_head, state, target.config)
        ce = F.cross_entropy(logits.float(), tokens[:, d + 1])
        mse, cosine = state_errors(state, states[:, d])
        kl = logits.new_zeros((), dtype=torch.float32)
        if kl_weight:
            with torch.no_grad():
                teacher_logits = project(target.lm_head, states[:, d], target.config).float()
            kl = F.kl_div(F.log_softmax(logits.float(), dim=-1),
                          F.softmax(teacher_logits, dim=-1), reduction="batchmean")
        loss = loss + weight * (ce + state_weight * (mse + cosine) + kl_weight * kl)
        metrics.append({"depth": d, "ce": float(ce.detach()), "normalized_mse": float(mse.detach()),
                        "cosine_distance": float(cosine.detach()), "teacher_kl": float(kl.detach())})
    return loss / sum(WEIGHTS), metrics


def check_cache(cache, expected, layer_types, window=2048):
    from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
    require(type(cache) is DynamicCache, "only an ordinary Transformers DynamicCache is supported")
    require(0 <= expected < window, "cannot roll back a saturated sliding-window cache")
    require(len(cache.layers) == len(layer_types), "cache layer count mismatch")
    require(cache.get_seq_length() == expected, "cache sequence length mismatch")
    for layer, kind in zip(cache.layers, layer_types):
        expected_type = DynamicSlidingWindowLayer if kind == "sliding_attention" else DynamicLayer
        require(kind in ("sliding_attention", "full_attention") and type(layer) is expected_type,
                "unsupported cache layer type")
        require(layer.get_seq_length() == expected, "per-layer cache length mismatch")
        require(layer.keys.shape[-2] == expected and layer.values.shape[-2] == expected,
                "cache already discarded part of the prefix")
        if kind == "sliding_attention":
            require(layer.sliding_window == window, "cache window mismatch")


def crop_cache(cache, keep, layer_types, window=2048):
    current = cache.get_seq_length()
    check_cache(cache, current, layer_types, window)
    require(0 < keep <= current, "invalid rollback length")
    if keep < current:
        # Negative removal count works with both historical and current HF crop APIs.
        cache.crop(-(current - keep))
    check_cache(cache, keep, layer_types, window)


class GlimmerTarget:
    def __init__(self, attention="sdpa"):
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        require(torch.cuda.is_available(), "GPU commands require a dedicated CUDA runtime")
        require(torch.cuda.is_bf16_supported(), "BF16 CUDA support required")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation=attention)
        require(type(self.model).__name__ == "MuseGlimmerForConditionalGeneration",
                "unsupported model class; no substitute target is allowed")
        self.model.eval().requires_grad_(False)
        self.config = self.model.config.text_config
        c = self.config
        require((c.hidden_size, c.vocab_size, c.num_hidden_layers, c.sliding_window)
                == (6656, 202048, 52, 2048), "official Glimmer text config mismatch")
        require(len(c.layer_types) == 52 and set(c.layer_types) == {"sliding_attention", "full_attention"},
                "unsupported attention layout")
        self.embedding = self.model.get_input_embeddings()
        self.lm_head = self.model.get_output_embeddings()
        self.norm = self.model.model.language_model.norm
        require(self.embedding.weight.shape == self.lm_head.weight.shape == (202048, 6656),
                "unexpected embedding/LM-head dimensions")
        self.assert_frozen()
        self.eos = {c.eos_token_id} if isinstance(c.eos_token_id, int) else set(c.eos_token_id)
        self.environment = {"model": MODEL, "revision": REVISION, "torch": str(torch.__version__),
                            "transformers": transformers.__version__, "cuda": torch.version.cuda,
                            "recommended_transformers_commit": TRANSFORMERS_COMMIT,
                            "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16",
                            "attention": attention, "python": sys.version, "platform": platform.platform()}
        try:
            hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                                       "--format=csv"], capture_output=True, text=True, timeout=5)
            self.environment["nvidia_smi"] = hardware.stdout + hardware.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.environment["nvidia_smi"] = str(exc)
        # Validate the actual public output's final-normalized hidden state contract.
        with torch.inference_mode():
            states, logits, _ = self.forward([c.bos_token_id], None, logits_to_keep=1)
            reconstructed = project(self.lm_head, states[-1:], c)
            require(torch.allclose(reconstructed, logits, atol=.01, rtol=.005),
                    "hidden_states[-1] is not the expected final-normalized LM-head input")
        self.sync()

    def assert_frozen(self):
        require(not any(p.requires_grad or p.grad is not None for p in self.model.parameters()),
                "a target/embedding/LM-head parameter is trainable or has a gradient")

    def sync(self):
        self.torch.cuda.synchronize(self.device)

    def reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self):
        return {"peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device)}

    def forward(self, token_ids, cache, logits_to_keep=0):
        torch = self.torch
        before = 0 if cache is None else cache.get_seq_length()
        require(before + len(token_ids) <= MAX_CONTEXT, "target forward exceeds safe pilot context")
        if cache is not None:
            check_cache(cache, before, self.config.layer_types)
        out = self.model(input_ids=torch.tensor([token_ids], device=self.device),
                         past_key_values=cache, use_cache=True, output_hidden_states=True,
                         return_dict=True, logits_to_keep=logits_to_keep)
        check_cache(out.past_key_values, before + len(token_ids), self.config.layer_types)
        states = out.hidden_states[-1][0]
        require(states.shape == (len(token_ids), self.config.hidden_size), "hidden state alignment mismatch")
        return states, out.logits[0], out.past_key_values

    def prefill(self, token_ids):
        states, logits, cache = self.forward(token_ids, None, logits_to_keep=1)
        return states[-1].clone(), int(logits[-1].argmax()), cache

    def verify(self, token_ids, cache):
        states, logits, cache = self.forward(token_ids, cache)
        return states, logits.argmax(-1).tolist(), cache

    def crop(self, cache, keep):
        crop_cache(cache, keep, self.config.layer_types)

    @staticmethod
    def cache_length(cache):
        return cache.get_seq_length()


class Drafter:
    def __init__(self, head, target):
        self.head, self.target = head, target

    def draft(self, root, anchor, depth, eos):
        state, token = root, anchor
        drafts, states = [], []
        for d in range(1, depth + 1):
            token_tensor = self.target.torch.tensor(token, device=self.target.device)
            state = self.head.step(state, self.target.embedding(token_tensor), d, self.target.norm)
            states.append(state)
            token = int(project(self.target.lm_head, state, self.target.config).argmax())
            drafts.append(token)
            if token in eos:
                break
        return drafts, states


def accepted_prefix(drafts, verified_next):
    require(len(verified_next) == len(drafts) + 1, "verification must include anchor + every draft")
    accepted = 0
    for draft, correct in zip(drafts, verified_next):
        if draft != correct:
            break
        accepted += 1
    return accepted


def decode(target, prompt, max_new_tokens, depth=0, drafter=None, diagnostics=False):
    """One cache, no full-prefix fallback. Anchor is correct but not yet consumed.

    Row i of verify([anchor, d1, ...]) predicts draft i+1. After k accepts,
    keep anchor + k drafts, restore hidden row k, carry row-k argmax as the
    next (unconsumed) correct anchor. This also handles rejection at position 1.
    """
    context_guard(len(prompt), max_new_tokens, depth)
    require((depth == 0) == (drafter is None), "depth/drafter mismatch")
    target.reset_peak()
    target.sync()
    started = time.perf_counter()
    root, anchor, cache = target.prefill(prompt)
    target.sync()
    prefill_s = time.perf_counter() - started
    started = time.perf_counter()
    output, passes, drift = [], [], []
    proposed, accepted_at = [0] * depth, [0] * depth
    draft_s = verify_s = 0.
    while len(output) < max_new_tokens:
        remaining = max_new_tokens - len(output)
        # Neither an EOS anchor nor the final budget token needs to enter KV.
        if anchor in target.eos or remaining == 1:
            output.append(anchor)
            break
        n = min(depth, remaining - 1)
        target.sync()
        tick = time.perf_counter()
        drafts, predicted_states = drafter.draft(root, anchor, n, target.eos) if n else ([], [])
        target.sync()
        draft_s += time.perf_counter() - tick if n else 0.
        require(len(drafts) <= n, "drafter exceeded output budget")
        before = target.cache_length(cache)
        tick = time.perf_counter()
        teacher_states, verified, cache = target.verify([anchor, *drafts], cache)
        k = accepted_prefix(drafts, verified)
        committed = [anchor, *drafts[:k]]
        if any(t in target.eos for t in committed):
            committed = committed[:next(i for i, t in enumerate(committed) if t in target.eos) + 1]
            k = len(committed) - 1
        target.crop(cache, before + 1 + k)
        root = teacher_states[k].clone()
        next_anchor = verified[k]
        target.sync()
        verify_s += time.perf_counter() - tick
        for i in range(len(drafts)):
            proposed[i] += 1
            accepted_at[i] += int(i < k)
        passes.append({"seed": anchor, "drafts": drafts, "accepted_drafts": k,
                       "next_correct_token": next_anchor, "cache_before": before,
                       "cache_after": target.cache_length(cache)})
        if diagnostics:
            # State at step i+1 consumes anchor + i drafts. It still follows the
            # actual chosen path at the first rejected prediction (i == k).
            for i, predicted in enumerate(predicted_states):
                mse, cosine = state_errors(predicted, teacher_states[i])
                drift.append({"depth": i + 1, "chosen_prefix": i <= k,
                              "normalized_mse": float(mse), "cosine_distance": float(cosine)})
        output.extend(committed)
        anchor = next_anchor
        if output[-1] in target.eos:
            break
    target.sync()
    decode_s = time.perf_counter() - started
    accepted = sum(p["accepted_drafts"] for p in passes)
    proposals = sum(proposed)
    return {"token_ids": output, "generated_tokens": len(output), "prefill_s": prefill_s,
            "decode_s": decode_s, "total_generation_s": prefill_s + decode_s,
            "decode_tokens_per_s": len(output) / decode_s, "draft_s": draft_s, "verify_s": verify_s,
            "target_calls_decode": len(passes), "target_calls_total": len(passes) + 1,
            "target_calls_per_generated_token": (len(passes) + 1) / len(output),
            "verification_passes": len(passes) if depth else 0,
            "proposed_drafts": proposals, "accepted_drafts": accepted,
            "mean_accepted_drafts_per_pass": accepted / len(passes) if depth and passes else 0.,
            "draft_acceptance_rate": accepted / proposals if proposals else 0.,
            "proposed_by_depth": proposed, "accepted_by_depth": accepted_at,
            "acceptance_histogram": dict(Counter(p["accepted_drafts"] for p in passes)) if depth else {},
            "passes": passes, "drift": drift, "diagnostic_only": diagnostics, **target.memory()}


def summarize_drift(rows):
    result = {}
    for scope, chosen in (("chosen_token_prefix", True), ("conditional_after_rejection", False)):
        result[scope] = []
        for d in range(1, 9):
            group = [r for r in rows if r["depth"] == d and r["chosen_prefix"] == chosen]
            if group:
                result[scope].append({"depth": d, "count": len(group),
                    **{key: statistics.mean(r[key] for r in group)
                       for key in ("normalized_mse", "cosine_distance")}})
    return result


def chosen_path_drift(target, head, prompt, generated, depth, root_offsets):
    """Separate diagnostic, never a generation fallback or part of live timing.

    Materialize genuine teacher states for the *chosen* full prefix once. Roll
    the head on chosen token inputs even beyond its first incorrect prediction;
    only the initial hidden state is teacher-provided at each verification root.
    """
    tokens = [*prompt, *generated]
    teacher, _, _ = target.forward(tokens, None, logits_to_keep=1)
    rows = []
    for offset in root_offsets:
        start = len(prompt) + offset
        require(0 <= offset < len(generated), "diagnostic root outside generated path")
        state = teacher[start - 1]
        for d in range(1, min(depth, len(tokens) - start) + 1):
            token = target.torch.tensor(tokens[start + d - 1], device=target.device)
            state = head.step(state, target.embedding(token), d, target.norm)
            mse, cosine = state_errors(state, teacher[start + d - 1])
            rows.append({"depth": d, "root_generated_offset": offset, "chosen_prefix": True,
                         "normalized_mse": float(mse), "cosine_distance": float(cosine)})
    return rows


def command_metadata(args):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
    return {"command": shlex.join([sys.executable, *sys.argv]), "arguments": vars(args),
            "code_commit": git.stdout.strip(), "model": MODEL, "revision": REVISION,
            "tier": "development", "time_unix": time.time()}


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def load_capture(path):
    import torch
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    require(bundle.get("model") == MODEL and bundle.get("revision") == REVISION,
            "capture model/revision mismatch")
    require(bundle.get("state_kind") == "target_final_norm", "unsupported capture states")
    validate_records(bundle["records"])
    for row in bundle["records"]:
        require(row["states"].shape == (len(row["token_ids"]), 6656), "capture state dimensions mismatch")
        require(row["states"].device.type == "cpu" and torch.isfinite(row["states"]).all().item(),
                "capture must contain finite CPU states")
    return bundle


def manifest(records):
    return [{k: v for k, v in r.items() if k != "states"} for r in records]


def capture_command(args):
    import torch
    records = read_records(args.data)
    require(not Path(args.output).exists(), "capture output already exists")
    target = GlimmerTarget(args.attention)
    for row in records:
        row["token_ids"] = target.tokenizer.encode(row["text"], add_special_tokens=True)
        require(10 <= len(row["token_ids"]) <= args.max_sequence_tokens <= 1024,
                "sequences must contain 10..max_sequence_tokens tokens; no silent truncation")
    validate_records(records)
    with torch.inference_mode():
        for row in records:
            states, _, _ = target.forward(row["token_ids"], None, logits_to_keep=1)
            row["states"] = states.detach().to(device="cpu", dtype=torch.bfloat16).clone()
            print(f"captured {row['id']}: {len(row['token_ids'])} tokens", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({**command_metadata(args), "environment": target.environment,
                "state_kind": "target_final_norm", "records": records}, args.output)


def train_command(args):
    import torch
    require(args.updates > 0 and args.batch_size > 0 and args.lr > 0, "invalid training budget")
    require(args.state_weight > 0 and args.kl_weight >= 0, "invalid auxiliary loss weights")
    require(not Path(args.output_dir).exists(), "training directory already exists")
    bundle = load_capture(args.capture)
    records = [r for r in bundle["records"] if r["split"] == "train"]
    roots = [(i, t) for i, r in enumerate(records) for t in range(len(r["token_ids"]) - 9)]
    require(bool(roots), "no full-depth training windows")
    # Exactly the same ordered full-depth windows and update counts for every variant.
    rng = random.Random(args.seed)
    schedule = [[rng.choice(roots) for _ in range(args.batch_size)] for _ in range(args.updates)]
    target = GlimmerTarget(args.attention)
    output = Path(args.output_dir)
    output.mkdir(parents=True)
    for variant in args.variants:
        torch.manual_seed(args.seed)
        head = make_head(6656, args.rank, variant != "fixed-ce").to(target.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
        log = []
        target.reset_peak()
        target.sync()
        started = time.perf_counter()
        for update, selections in enumerate(schedule, 1):
            states = torch.stack([records[i]["states"][t:t + 9] for i, t in selections]).to(target.device)
            tokens = torch.tensor([records[i]["token_ids"][t:t + 10] for i, t in selections], device=target.device)
            optimizer.zero_grad(set_to_none=True)
            loss, per_depth = training_loss(head, states, tokens, target,
                args.state_weight if variant == "shared-state" else 0., args.kl_weight)
            require(torch.isfinite(loss).item(), "nonfinite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            log.append({"update": update, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                        "per_depth": per_depth})
            if update == 1 or update % 10 == 0 or update == args.updates:
                print(f"{variant} update={update} loss={float(loss.detach()):.5f}", flush=True)
        target.sync()
        target.assert_frozen()
        metadata = {**command_metadata(args), "environment": target.environment, "variant": variant,
                    "rank": args.rank, "max_depth": 8, "depth_weights": WEIGHTS,
                    "state_weight": args.state_weight if variant == "shared-state" else 0.,
                    "kl_weight": args.kl_weight, "head_parameters": sum(p.numel() for p in head.parameters()),
                    "training_s": time.perf_counter() - started, "selection": "last update; no heldout selection",
                    "train_manifest": manifest(records), "schedule": schedule, **target.memory()}
        path = output / f"{variant}.pt"
        torch.save({**metadata, "head": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, path)
        json_write(output / f"{variant}.json", {**metadata, "checkpoint_bytes": path.stat().st_size, "updates": log})
        del optimizer, head, states, tokens, loss


def evaluate_command(args):
    import torch
    require(not Path(args.output).exists(), "evaluation output already exists")
    require(args.repeats > 0, "repeats must be positive")
    bundle = load_capture(args.capture)
    evaluation = [r for r in bundle["records"] if r["split"] == "eval"]
    require(all(sum(r.get("category") == c for r in evaluation) >= 2 for c in CATEGORIES),
            "pilot requires at least two heldout prompts in each of six categories")
    for row in evaluation:
        context_guard(len(row["token_ids"]), args.max_new_tokens, 8)
    checkpoints = []
    for path in args.heads:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        require(ckpt.get("model") == MODEL and ckpt.get("revision") == REVISION and ckpt.get("max_depth") == 8,
                "head model/revision/depth mismatch")
        require(ckpt.get("variant") in VARIANTS, "unsupported head variant")
        check_heldout(ckpt["train_manifest"], evaluation)
        checkpoints.append((path, ckpt))
    target = GlimmerTarget(args.attention)
    report = {**command_metadata(args), "environment": target.environment, "status": "running",
              "baseline": "same HF target, no speculation, same BF16/backend/prompt/output/EOS policy",
              "intentional_differences": "shared/unshared blocks, state loss, draft depth; equal data/updates NOT parameters",
              "performance_scope": "standalone synchronized HF development pilot; not production serving or DFlash comparison",
              "pairs": [], "heads": []}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for path, ckpt in checkpoints:
                head = make_head(6656, ckpt["rank"], ckpt["variant"] != "fixed-ce")
                head.load_state_dict(ckpt["head"], strict=True)
                head.eval()
                report["heads"].append({"path": path, "variant": ckpt["variant"],
                    "parameters": sum(p.numel() for p in head.parameters()), "checkpoint_bytes": Path(path).stat().st_size,
                    "training_arguments": ckpt["arguments"]})
                for depth in DEPTHS:
                    # Both paths are warmed; head transfers are outside measured regions.
                    head.cpu()
                    decode(target, evaluation[0]["token_ids"], 10)
                    head.to(target.device)
                    decode(target, evaluation[0]["token_ids"], 10, depth, Drafter(head, target))
                    for repeat in range(args.repeats):
                        for index, row in enumerate(evaluation):
                            pair = {"id": row["id"], "category": row["category"], "repeat": repeat,
                                    "variant": ckpt["variant"], "depth": depth, "prompt_token_ids": row["token_ids"],
                                    "order": ["baseline", "candidate"] if (repeat + index) % 2 == 0 else ["candidate", "baseline"]}
                            report["pairs"].append(pair)
                            for mode in pair["order"]:
                                head.to("cpu" if mode == "baseline" else target.device)
                                pair[mode] = decode(target, row["token_ids"], args.max_new_tokens,
                                    0 if mode == "baseline" else depth, None if mode == "baseline" else Drafter(head, target))
                                pair[mode]["text"] = target.tokenizer.decode(pair[mode]["token_ids"])
                            pair["exact_token_identity"] = pair["baseline"]["token_ids"] == pair["candidate"]["token_ids"]
                            require(pair["exact_token_identity"], "greedy identity failed; no retry or fallback permitted")
                            pair["decode_speedup"] = pair["baseline"]["decode_s"] / pair["candidate"]["decode_s"]
                            if repeat == 0:
                                head.to(target.device)
                                diagnostic = decode(target, row["token_ids"], args.max_new_tokens,
                                                    depth, Drafter(head, target), diagnostics=True)
                                require(diagnostic["token_ids"] == pair["baseline"]["token_ids"], "diagnostic replay identity failed")
                                pair["drift_separate_untimed_replay"] = summarize_drift(diagnostic["drift"])
                                pair["drift_raw"] = diagnostic["drift"]
                                roots = [p["cache_before"] - len(row["token_ids"])
                                         for p in pair["candidate"]["passes"]] or [0]
                                chosen_drift = chosen_path_drift(target, head, row["token_ids"],
                                    pair["candidate"]["token_ids"], depth, roots)
                                pair["chosen_path_drift_separate_diagnostic"] = summarize_drift(chosen_drift)
                                pair["chosen_path_drift_raw"] = chosen_drift
                            json_write(args.output, report)
                            print(f"{ckpt['variant']} d={depth} {row['id']} identity=OK accepted/pass="
                                  f"{pair['candidate']['mean_accepted_drafts_per_pass']:.3f} "
                                  f"decode_ratio={pair['decode_speedup']:.3f}", flush=True)
                del head
        report["summary"] = []
        for variant in {p["variant"] for p in report["pairs"]}:
            for depth in DEPTHS:
                group = [p for p in report["pairs"] if p["variant"] == variant and p["depth"] == depth]
                accepted = sum(p["candidate"]["accepted_drafts"] for p in group)
                passes = sum(p["candidate"]["verification_passes"] for p in group)
                report["summary"].append({"variant": variant, "depth": depth, "pairs": len(group),
                    "median_paired_decode_speedup": statistics.median(p["decode_speedup"] for p in group),
                    "accepted_drafts_per_pass": accepted / passes if passes else 0., "exact_token_identity": True})
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        json_write(args.output, report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="dependency-free sequence/family split check")
    validate.add_argument("--data", default=str(FIXTURES))
    capture = sub.add_parser("capture", help="capture official frozen Glimmer final-normalized states")
    capture.add_argument("--data", default=str(FIXTURES))
    capture.add_argument("--output", required=True)
    capture.add_argument("--max-sequence-tokens", type=int, default=512)
    train = sub.add_parser("train", help="equal-data/equal-update three-variant training")
    train.add_argument("--capture", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    train.add_argument("--rank", type=int, choices=(64, 128), default=64)
    train.add_argument("--updates", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--state-weight", type=float, default=.2)
    train.add_argument("--kl-weight", type=float, default=0., help="optional teacher KL, common to every variant")
    train.add_argument("--seed", type=int, default=20261002)
    evaluate = sub.add_parser("evaluate", help="real cached greedy A/B decode and separate drift replay")
    evaluate.add_argument("--capture", required=True)
    evaluate.add_argument("--heads", nargs="+", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--max-new-tokens", type=int, choices=(64, 128), default=64)
    evaluate.add_argument("--repeats", type=int, default=1)
    for command in (capture, train, evaluate):
        command.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    return p


def main():
    args = parser().parse_args()
    if args.command == "validate":
        records = read_records(args.data)
        print(json.dumps({"sequences": len(records), "splits": dict(Counter(r["split"] for r in records)),
                          "heldout_categories": dict(Counter(r["category"] for r in records if r["split"] == "eval"))}))
    else:
        {"capture": capture_command, "train": train_command, "evaluate": evaluate_command}[args.command](args)


if __name__ == "__main__":
    main()
