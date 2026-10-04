#!/usr/bin/env python3
"""Bounded, development-tier frozen Muse Glimmer recursive-MTP pilot.

Heavy dependencies are imported only by GPU commands / the head factory. See
`glimmer_recursive_mtp.md` for the model, cache, alignment and measurement contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
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
STATE_TRANSITION = "postnorm-gated-residual-v1"
WEIGHTS = (1.0, 1.0, .8, .8, .5, .5, .5, .5)
DEPTHS = (1, 2, 4, 8)
VARIANTS = ("fixed-ce", "shared-ce", "shared-state", "shared-state-norm")
CATEGORIES = {"code", "prose", "reasoning", "structured", "repetitive", "high-entropy"}
# This bound includes prompt, generated output AND every uncommitted proposal.
MAX_CONTEXT = 1792
FIXTURES = Path(__file__).with_name("glimmer_recursive_mtp.jsonl")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_head_transition(checkpoint):
    require(checkpoint.get("state_transition") == STATE_TRANSITION,
            "head state-transition mismatch; repeated-norm pilot checkpoints must be retrained")


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
    tokens = {tuple(r["token_ids"]) for r in training if "token_ids" in r}
    ids = {r["id"] for r in training if "id" in r}
    for row in evaluation:
        require(row.get("id") not in ids, "checkpoint training id in evaluation")
        require(row["family"] not in families, "checkpoint training family in evaluation")
        require(canonical(row["text"]) not in texts, "checkpoint training text in evaluation")
        if "token_ids" in row:
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
            # Input/output live in the target's already-final-normalized space.
            # Reapplying its signed learned norm breaks the zero-update identity.
            return (state.float() + self.gate.sigmoid() * delta).to(state.dtype)

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


def state_rms_error(predicted, teacher):
    """Relative RMS error: scale-sensitive, FP32, with detached teacher targets."""
    p, t = predicted.float(), teacher.detach().float()
    ratio = ((p.square().mean(-1) + 1e-8) / (t.square().mean(-1) + 1e-8)).sqrt()
    return (ratio - 1).square().mean()

def training_loss(head, states, tokens, target, state_weight=0., kl_weight=0.,
                  ce_weight=1., temperature=1., depth=8, teacher_argmax_diagnostic=False,
                  state_norm_weight=0.):
    """Teacher-forced tokens, predicted hidden feedback; sequence CE + forward KL."""
    import torch
    import torch.nn.functional as F
    require(depth in DEPTHS and temperature > 0, "invalid loss depth/temperature")
    require(math.isfinite(state_norm_weight) and state_norm_weight >= 0, "invalid state norm weight")
    states = states.detach()
    state = states[:, 0]  # No teacher hidden state is fed after this root.
    loss, metrics = 0., []
    for d, weight in enumerate(WEIGHTS[:depth], 1):
        with torch.no_grad():
            embedding = target.embedding(tokens[:, d])
        state = head.step(state, embedding, d, target.norm)
        logits = project(target.lm_head, state, target.config).float()
        teacher_logits = None
        if kl_weight or teacher_argmax_diagnostic:
            with torch.no_grad():
                teacher_logits = project(target.lm_head, states[:, d], target.config).float()
        labels = teacher_logits.argmax(-1) if teacher_argmax_diagnostic else tokens[:, d + 1]
        ce = F.cross_entropy(logits, labels)
        mse, cosine = state_errors(state, states[:, d])
        norm_error = state_rms_error(state, states[:, d])
        kl = logits.new_zeros(())
        if kl_weight:
            kl = F.kl_div(F.log_softmax(logits / temperature, dim=-1),
                          F.softmax(teacher_logits / temperature, dim=-1),
                          reduction="batchmean") * temperature ** 2
        loss = loss + weight * (ce_weight * ce + state_weight * (mse + cosine)
                                + kl_weight * kl + state_norm_weight * norm_error)
        metrics.append({"depth": d, "ce": float(ce.detach()), "normalized_mse": float(mse.detach()),
                        "cosine_distance": float(cosine.detach()), "teacher_kl": float(kl.detach()),
                        "rms_ratio_error": float(norm_error.detach())})
    return loss / sum(WEIGHTS[:depth]), metrics


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
                       for key in ("normalized_mse", "cosine_distance", "teacher_argmax_agreement",
                                   "teacher_top5_recall", "teacher_kl", "predicted_rms", "teacher_rms", "state_mse",
                                   "rms_ratio_error")
                       if all(key in r for r in group)}})
    return result


def chosen_path_drift(target, head, prompt, generated, depth, root_offsets, distribution_metrics=False):
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
            if distribution_metrics:
                import torch.nn.functional as F
                actual = teacher[start + d - 1]
                logits = project(target.lm_head, state, target.config).float().reshape(1, -1)
                teacher_logits = project(target.lm_head, actual, target.config).float().reshape(1, -1)
                label = teacher_logits.argmax(-1)
                rows[-1].update({
                    "teacher_argmax_agreement": float((logits.argmax(-1) == label).float().mean()),
                    "teacher_top5_recall": float((logits.topk(min(5, logits.shape[-1]), -1).indices == label[:, None]).any()),
                    "teacher_kl": float(F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher_logits, -1), reduction="batchmean")),
                    "predicted_rms": float(state.float().square().mean().sqrt()),
                    "teacher_rms": float(actual.float().square().mean().sqrt()),
                    "state_mse": float((state.float() - actual.float()).square().mean()),
                    "rms_ratio_error": float(state_rms_error(state, actual)),
                })
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


def data_helper():
    # Script execution puts this sibling on sys.path; tests inject the agreed API.
    import glimmer_mtp_data
    return glimmer_mtp_data


def split_manifest(dataset, split):
    # The helper API permits metadata for the full capture; select the actual root split.
    return [row for row in dataset.manifest() if row["split"] == split]


def prepare_prompts_command(args):
    return data_helper().prepare_prompts(args)


def capture_generated_command(args):
    return data_helper().capture_generated(args, GlimmerTarget(args.attention))


class LegacyCapture:
    """Compatibility adapter for the original small .pt fixture only."""
    def __init__(self, path, split):
        bundle = load_capture(path)
        self.metadata = {k: bundle[k] for k in ("model", "revision", "state_kind")}
        self.records = [r for r in bundle["records"] if r["split"] == split]
        self.roots = [(i, t) for i, r in enumerate(self.records) for t in range(len(r["token_ids"]) - 9)]
        self.root_count = len(self.roots)
        self.token_count = sum(len(r["token_ids"]) for r in self.records)

    def manifest(self):
        return manifest(self.records)

    def batch(self, roots):
        import torch
        return (torch.stack([self.records[i]["states"][t:t + 9] for i, t in roots]),
                torch.tensor([self.records[i]["token_ids"][t:t + 10] for i, t in roots]))

    def new_sampler(self, seed):
        return LegacySampler(self, seed)

    def iter_batches(self, batch_size, limit_roots, seed):
        # Independent, unique validation sample; no replacement and no global RNG.
        roots = random.Random(seed).sample(self.roots, min(limit_roots, self.root_count))
        for start in range(0, len(roots), batch_size):
            yield self.batch(roots[start:start + batch_size])


class LegacySampler:
    def __init__(self, dataset, seed):
        self.dataset, self.rng, self.cursor = dataset, random.Random(seed), 0

    def next_batch(self, batch_size):
        roots = [self.rng.choice(self.dataset.roots) for _ in range(batch_size)]
        self.cursor += batch_size
        return self.dataset.batch(roots)

    def state_dict(self):
        return {"rng": self.rng.getstate(), "cursor": self.cursor}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.cursor = state["cursor"]


def captured_dataset(path, split="train"):
    path = Path(path)
    if path.suffix == ".pt":
        return LegacyCapture(path, split)
    require(path.name == "index.json", "unsupported capture layout; expected legacy .pt or index.json")
    return data_helper().CapturedDataset(path, split=split)


def read_prompt_records(path, split):
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    require(bool(rows), "empty heldout prompt file")
    seen_ids, seen_texts = set(), set()
    for row in rows:
        require(all(isinstance(row.get(k), str) and row[k].strip()
                    for k in ("id", "family", "text")), "invalid heldout prompt fields")
        require(row.get("split") == split, f"heldout prompts must have split={split}")
        require(row.get("category") in CATEGORIES, "unknown heldout category")
        require(row.get("prompt_format") in ("chat", "raw"), "declare prompt_format=chat or raw")
        require("token_ids" not in row, "external prompts must be tokenized by the official target")
        require(row["id"] not in seen_ids and canonical(row["text"]) not in seen_texts,
                "duplicate heldout prompt")
        seen_ids.add(row["id"])
        seen_texts.add(canonical(row["text"]))
    require(all(sum(r["category"] == c for r in rows) >= 2 for c in CATEGORIES),
            "need at least two heldout prompts in every category")
    return rows


def tokenize_prompts(target, rows):
    result = []
    for row in rows:
        if row["prompt_format"] == "chat":
            rendered = target.tokenizer.apply_chat_template(
                [{"role": "user", "content": row["text"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True)
            ids = rendered["input_ids"]
        else:
            ids = target.tokenizer.encode(row["text"], add_special_tokens=True)
        require(isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids),
                "tokenizer must return a nonempty token list")
        result.append({**row, "token_ids": ids})
    return result


def parse_curriculum(value):
    if not value:
        return []
    try:
        stages = [tuple(int(x) for x in stage.split(":")) for stage in value.split(",")]
        require(all(len(s) == 2 and s[0] in DEPTHS and s[1] > 0 for s in stages), "invalid curriculum")
        require(all(a[0] < b[0] for a, b in zip(stages, stages[1:])), "curriculum depths must increase")
        return stages
    except (ValueError, TypeError) as exc:
        raise ValueError("curriculum must be increasing depth:updates pairs, e.g. 2:10000,4:10000,8:40000") from exc


def active_depth(config, update):
    require(update > 0, "update must be positive")
    stages = parse_curriculum(config["curriculum"])
    if not stages:
        return config["train_depth"]
    for depth, count in stages:
        if update <= count:
            return depth
        update -= count
    raise ValueError("update exceeds curriculum")


def planned_depth(config):
    stages = parse_curriculum(config["curriculum"])
    return stages[-1][0] if stages else config["train_depth"]


def learning_rate(config, update):
    warmup, horizon = config["warmup_updates"], config["schedule_updates"]
    if warmup and update <= warmup:
        return config["lr"] * update / warmup
    if not horizon:
        return config["lr"]  # Legacy constant-LR path.
    progress = (update - warmup) / (horizon - warmup)
    return config["lr"] * .5 * (1 + math.cos(math.pi * min(1., max(0., progress))))


TRAIN_DEFAULTS = {
    "variants": list(VARIANTS[:3]), "rank": 64, "batch_size": 4, "lr": 3e-4,
    "weight_decay": .01, "grad_clip": 1., "state_weight": .2, "ce_weight": 1.,
    "kl_weight": 0., "temperature": 1., "seed": 20261002, "train_depth": 8,
    "curriculum": None, "schedule_updates": 0, "warmup_updates": 0,
    "checkpoint_every": 1000, "validation_every": 0, "probe_every": 0,
    "validation_roots": 1024, "validation_batch_size": 8, "validation_seed": 314159,
    "validation_prompts": None, "probe_new_tokens": 64, "attention": "sdpa",
    "init_head": None, "overfit_roots": 0, "teacher_argmax_diagnostic": False,
    "record_divergence": False,
}
TRAINING_FORMAT = "glimmer-mtp-training-v1"
SELECTION = "minimum mean validation teacher KL (T=1), equal weight at fixed planned depths; earliest tie"


def training_config(args, checkpoint=None):
    supplied = {k: v for k, v in vars(args).items() if k in TRAIN_DEFAULTS}
    if checkpoint is not None:
        require(checkpoint.get("training_format") == TRAINING_FORMAT, "checkpoint is not resumable training state")
        config = checkpoint["training_config"].copy()
        require(set(config) == set(TRAIN_DEFAULTS), "unknown or incomplete saved training options")
        for key, value in supplied.items():
            require(value == config[key], f"incompatible resume override: --{key.replace('_', '-')}")
        updates = getattr(args, "updates", checkpoint["planned_updates"])
    else:
        config = {**TRAIN_DEFAULTS, **supplied}
        updates = getattr(args, "updates", 100)
    require(updates > 0 and config["batch_size"] > 0 and config["lr"] > 0, "invalid training budget")
    require(config["rank"] in (64, 128, 256), "unsupported head rank")
    require(config["train_depth"] in DEPTHS and config["temperature"] > 0, "invalid depth/temperature")
    require(all(config[k] >= 0 for k in ("ce_weight", "kl_weight", "state_weight", "weight_decay"))
            and config["ce_weight"] + config["kl_weight"] > 0 and config["grad_clip"] > 0,
            "invalid objective/optimizer weights")
    require(all(math.isfinite(config[k]) for k in ("lr", "ce_weight", "kl_weight", "state_weight",
                                                  "temperature", "weight_decay", "grad_clip")),
            "nonfinite training option")
    require(config["variants"] and len(set(config["variants"])) == len(config["variants"])
            and set(config["variants"]) <= set(VARIANTS), "unknown/duplicate variants")
    horizon, warmup = config["schedule_updates"], config["warmup_updates"]
    require(horizon >= 0 and warmup >= 0 and (not horizon or warmup < horizon), "invalid LR horizon/warmup")
    require(not horizon or updates <= horizon, "updates exceed the fixed LR horizon")
    stages = parse_curriculum(config["curriculum"])
    require(not stages or updates <= sum(n for _, n in stages), "updates exceed curriculum")
    require(not stages or "train_depth" not in supplied or config["train_depth"] == TRAIN_DEFAULTS["train_depth"],
            "use curriculum or explicit train-depth, not both")
    require(config["checkpoint_every"] > 0 and config["validation_every"] >= 0 and config["probe_every"] >= 0,
            "invalid checkpoint/validation/probe intervals")
    require(config["validation_roots"] > 0 and config["validation_batch_size"] > 0, "invalid validation sample")
    require(config["probe_new_tokens"] > 1, "probes require at least two generated tokens")
    require(not config["probe_every"] or config["validation_prompts"], "probes require validation-prompts")
    require(0 <= config["overfit_roots"] <= 64, "overfit diagnostic is bounded to 64 unique roots")
    require(not config["teacher_argmax_diagnostic"] or
            (0 < config["overfit_roots"] <= 64 and planned_depth(config) == 1 and updates <= 1000),
            "teacher-argmax diagnostic is only for <=64 fixed roots, depth 1, <=1000 updates")
    return config, updates


def resume_environment_matches(saved, current):
    """Same recorded stack/hardware; a replacement GPU may have a new UUID.

    This is checkpoint portability, not a cross-host bitwise-identity guarantee.
    Preserve the full unmodified environment in every checkpoint.
    """
    import csv

    def comparable(environment):
        result = dict(environment)
        smi = result.get("nvidia_smi")
        if isinstance(smi, str) and smi.strip():
            rows = list(csv.reader(smi.strip().splitlines(), skipinitialspace=True))
            require(rows and len(rows[0]) == 5 and rows[0][1].strip() == "uuid"
                    and all(len(row) == 5 for row in rows), "unexpected recorded nvidia-smi schema")
            result["nvidia_smi"] = [[value.strip() for i, value in enumerate(row) if i != 1]
                                     for row in rows]
        return result

    return comparable(saved) == comparable(current)

def rng_state():
    import torch
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        require(torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count(),
                "resume CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def load_head_checkpoint(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint.get("model") == MODEL and checkpoint.get("revision") == REVISION,
            "head model/revision mismatch")
    check_head_transition(checkpoint)
    require(checkpoint.get("variant") in VARIANTS and checkpoint.get("max_depth") in DEPTHS,
            "unsupported head variant/depth")
    require(isinstance(checkpoint.get("rank"), int) and checkpoint["rank"] > 0, "invalid head rank")
    return checkpoint


def head_from_checkpoint(checkpoint, target, evaluation_depth=None):
    width = getattr(target.config, "hidden_size", target.lm_head.weight.shape[1])
    require(checkpoint.get("hidden_size", 6656) == width, "head representation width mismatch")
    depth = checkpoint["max_depth"] if evaluation_depth is None else evaluation_depth
    require(depth in DEPTHS, "invalid evaluation depth")
    shared = checkpoint["variant"] != "fixed-ce"
    require(shared or depth <= checkpoint["max_depth"], "cannot extend untrained fixed-depth blocks")
    # Extending a shared head changes only its call bound, never weights/training metadata.
    head = make_head(width, checkpoint["rank"], shared, max(depth, checkpoint["max_depth"]))
    head.load_state_dict(checkpoint["head"], strict=True)
    return head


def initialize_head(head, checkpoint, rank, width):
    check_head_transition(checkpoint)
    require(checkpoint["rank"] == rank, "init-head rank mismatch")
    require(checkpoint.get("hidden_size", 6656) == width, "init-head representation width mismatch")
    require(checkpoint["max_depth"] == 1, "init-head must be a one-step checkpoint")
    block = {k[len("blocks.0."):]: v for k, v in checkpoint["head"].items() if k.startswith("blocks.0.")}
    require(len(block) == len(checkpoint["head"]), "init-head must contain exactly one block")
    for destination in head.blocks:
        destination.load_state_dict(block, strict=True)


def checkpoint_manifest(checkpoint):
    if "train_manifest" in checkpoint:
        return checkpoint["train_manifest"]  # Historical small checkpoints.
    require(checkpoint.get("train_manifest_path"), "checkpoint lacks training manifest")
    return json.loads(Path(checkpoint["train_manifest_path"]).read_text())


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def dataset_identity(path, dataset):
    return {**file_identity(path), "root_count": dataset.root_count, "token_count": dataset.token_count}


def save_checkpoint(path, checkpoint):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_training_result(output, variant, checkpoint):
    path = output / f"{variant}.pt"
    save_checkpoint(path, checkpoint)
    summary = {k: v for k, v in checkpoint.items() if k not in ("head", "optimizer", "rng", "sampler")}
    json_write(output / f"{variant}.json", {**summary, "checkpoint_bytes": path.stat().st_size,
        "updates_completed": checkpoint["update"], "training_log": str(output / f"{variant}-training.jsonl")})


def offline_validation(target, head, dataset, depths, limit_roots=1024, batch_size=8, seed=314159):
    """Fixed unique heldout roots, teacher-forced tokens; not a decode acceptance proxy."""
    import torch
    import torch.nn.functional as F
    require(limit_roots > 0 and batch_size > 0 and dataset.root_count > 0, "empty/invalid validation sample")
    require(depths and all(d in DEPTHS for d in depths), "unsupported validation depths")
    count, sums = 0, {d: Counter() for d in depths}
    was_training = head.training
    started = time.perf_counter()
    try:
        with isolated_rng(), torch.inference_mode():
            head.eval()
            for states, tokens in dataset.iter_batches(batch_size, min(limit_roots, dataset.root_count), seed):
                states, tokens = states.to(target.device), tokens.to(target.device)
                n = len(states)
                count += n
                require(count <= min(limit_roots, dataset.root_count), "validation iterator exceeded unique-root limit")
                state = states[:, 0]
                for d in range(1, max(depths) + 1):
                    state = head.step(state, target.embedding(tokens[:, d]), d, target.norm)
                    if d not in sums:
                        continue
                    logits = project(target.lm_head, state, target.config).float()
                    teacher = project(target.lm_head, states[:, d], target.config).float()
                    teacher_argmax = teacher.argmax(-1)
                    mse, cosine = state_errors(state, states[:, d])
                    metrics = {
                        "sequence_ce": F.cross_entropy(logits, tokens[:, d + 1]),
                        "teacher_argmax_agreement": (logits.argmax(-1) == teacher_argmax).float().mean(),
                        "teacher_top5_recall": (logits.topk(min(5, logits.shape[-1]), dim=-1).indices ==
                                                teacher_argmax[:, None]).any(-1).float().mean(),
                        "teacher_kl": F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher, -1), reduction="batchmean"),
                        "normalized_mse": mse, "cosine_distance": cosine,
                        "state_mse": (state.float() - states[:, d].float()).square().mean(),
                        "rms_ratio_error": state_rms_error(state, states[:, d]),
                        "predicted_rms": state.float().square().mean(-1).sqrt().mean(),
                        "teacher_rms": states[:, d].float().square().mean(-1).sqrt().mean(),
                    }
                    for key, value in metrics.items():
                        require(torch.isfinite(value).item(), "nonfinite validation metric")
                        sums[d][key] += float(value) * n
            target.sync()
    finally:
        head.train(was_training)
    require(count > 0, "empty validation iterator")
    rows = [{"depth": d, **{k: v / count for k, v in sums[d].items()}} for d in depths]
    elapsed = time.perf_counter() - started
    return {"root_count": count, "available_unique_roots": dataset.root_count, "seed": seed,
            "token_policy": "teacher-forced; predicted hidden feedback after root",
            "top5_definition": "teacher argmax in head top-5", "per_depth": rows,
            "selection_definition": SELECTION,
            "selection_score": statistics.mean(r["teacher_kl"] for r in rows),
            "elapsed_s": elapsed, "roots_per_s": count / elapsed}


def validation_probe(target, head, rows, max_new_tokens=64, record_divergence=False, depth=1):
    import torch
    require(all(any(r["category"] == c for r in rows) for c in CATEGORIES), "empty probe category")
    require(depth in DEPTHS, "invalid validation probe depth")
    diagnostic_categories = set()
    results = []
    was_training = head.training
    try:
        with isolated_rng(), torch.inference_mode():
            head.eval()
            for row in rows:
                context_guard(len(row["token_ids"]), max_new_tokens, depth)
                pair = {"id": row["id"], "category": row["category"], "prompt_token_ids": row["token_ids"]}
                pair["baseline"] = decode(target, row["token_ids"], max_new_tokens)
                pair["candidate"] = decode(target, row["token_ids"], max_new_tokens, depth, Drafter(head, target))
                check_pair_fidelity(pair, record_divergence)
                if depth > 1 and row["category"] not in diagnostic_categories:
                    # One prompt/category, outside decode timing; final test stays sealed.
                    roots = [p["cache_before"] - len(row["token_ids"])
                             for p in pair["candidate"]["passes"]] or [0]
                    drift = chosen_path_drift(target, head, row["token_ids"],
                        pair["candidate"]["token_ids"], depth, roots, distribution_metrics=True)
                    pair["chosen_path_drift_separate_diagnostic"] = summarize_drift(drift)
                    pair["chosen_path_drift_raw"] = drift
                    diagnostic_categories.add(row["category"])
                results.append(pair)
    finally:
        head.train(was_training)
    categories = []
    for category in sorted(CATEGORIES):
        group = [r["candidate"] for r in results if r["category"] == category]
        accepted, proposed = sum(r["accepted_drafts"] for r in group), sum(r["proposed_drafts"] for r in group)
        categories.append({"category": category, "prompts": len(group), "accepted_drafts": accepted,
                           "proposed_drafts": proposed, "draft_acceptance_rate": accepted / proposed if proposed else 0.})
    accepted = sum(r["accepted_drafts"] for r in categories)
    proposed = sum(r["proposed_drafts"] for r in categories)
    passes = sum(r["candidate"]["verification_passes"] for r in results)
    return {"depth": depth, "scope": "real cached validation decode; excludes anchor/correction/bonus",
            "verification_passes": passes, "mean_accepted_drafts_per_pass": accepted / passes if passes else 0.,
            "conditional_acceptance": conditional_acceptance(results, depth), "diagnostics_timed": False,
            "accepted_drafts": accepted, "proposed_drafts": proposed,
            "draft_acceptance_rate": accepted / proposed if proposed else 0.,
            "categories": categories, "pairs": results, "fidelity": fidelity_summary(results)}


class FixedRootSample:
    """Explicit Stage-0 overfit diagnostic only; never labelled unique exposures."""
    def __init__(self, dataset, limit, seed):
        import torch
        require(0 < limit <= dataset.root_count, "overfit root count exceeds actual unique roots")
        batches = list(dataset.iter_batches(min(64, limit), limit, seed))
        self.states = torch.cat([s for s, _ in batches])
        self.tokens = torch.cat([t for _, t in batches])
        require(len(self.states) == limit, "overfit iterator did not supply requested unique roots")
        self.root_count = limit
        self.token_count = dataset.token_count

    def new_sampler(self, seed):
        sample = self
        class Sampler:
            def __init__(self):
                self.rng = random.Random(seed)
            def next_batch(self, size):
                indices = [self.rng.randrange(sample.root_count) for _ in range(size)]
                return sample.states[indices], sample.tokens[indices]
            def state_dict(self):
                return {"rng": self.rng.getstate()}
            def load_state_dict(self, state):
                self.rng.setstate(state["rng"])
        return Sampler()

    def iter_batches(self, batch_size, limit_roots, seed):
        for start in range(0, min(limit_roots, self.root_count), batch_size):
            end = min(start + batch_size, limit_roots, self.root_count)
            yield self.states[start:end], self.tokens[start:end]


def train_command(args):
    import torch
    resume = load_head_checkpoint(args.resume) if args.resume else None
    config, updates = training_config(args, resume)
    dataset = captured_dataset(args.capture)
    train_manifest = split_manifest(dataset, "train")
    require(dataset.root_count > 0 and train_manifest, "no full-depth training windows/manifest")
    identity = dataset_identity(args.capture, dataset)
    output = Path(args.output_dir).resolve()
    if resume:
        require(resume["capture_identity"] == identity, "resume capture changed")
        require(resume["output_dir"] == str(output), "resume must use original output-dir")
    else:
        require(not output.exists(), "training directory already exists; use --resume")
    validation, validation_manifest = None, []
    if config["validation_every"]:
        require(Path(args.capture).name == "index.json", "legacy capture has no validation split")
        validation = captured_dataset(args.capture, "validation")
        validation_manifest = split_manifest(validation, "validation")
        check_heldout(train_manifest, validation_manifest)
        require(validation.root_count > 0 and validation_manifest, "no validation roots/manifest")
    prompt_rows = read_prompt_records(config["validation_prompts"], "validation") if config["validation_prompts"] else []
    if prompt_rows:
        check_heldout(train_manifest, prompt_rows)
        if Path(args.capture).name == "index.json":
            check_heldout(split_manifest(captured_dataset(args.capture, "test"), "test"), prompt_rows)
    prompt_identity = file_identity(config["validation_prompts"]) if prompt_rows else None
    if resume:
        require(resume["validation_prompt_identity"] == prompt_identity, "resume validation prompts changed")
    prefixes = ["checkpoint"] if len(config["variants"]) == 1 else config["variants"]
    if resume:
        require(resume["variant"] in config["variants"], "resume arm is absent from saved variants")
        prefix = "checkpoint" if len(prefixes) == 1 else resume["variant"]
        require(Path(args.resume).resolve() == output / f"{prefix}-last.pt",
                "resume must name the current last checkpoint, not an older/best checkpoint")
    needs_init = config["init_head"] and (not resume or any(
        not (output / f"{prefix}-last.pt").exists() for prefix in prefixes))
    init = load_head_checkpoint(config["init_head"]) if needs_init else None
    init_identity = resume["init_head_identity"] if resume else None
    if init:
        current_identity = file_identity(config["init_head"])
        require(not resume or current_identity == init_identity, "common init-head changed before remaining arms")
        init_identity = current_identity
        init_manifest = checkpoint_manifest(init)
        require(init_manifest == train_manifest, "init-head must use the same training corpus")
        if validation is not None:
            check_heldout(init_manifest, validation_manifest)
        check_heldout(init_manifest, prompt_rows)
    target = GlimmerTarget(config["attention"])
    target.assert_frozen()
    width = target.config.hidden_size
    if init:
        require(init["rank"] == config["rank"], "init-head rank mismatch")
    probe_rows = tokenize_prompts(target, prompt_rows)
    training_data = FixedRootSample(dataset, config["overfit_roots"], config["seed"]) if config["overfit_roots"] else dataset
    max_depth = planned_depth(config)
    depths = [d for d in DEPTHS if d <= max_depth]
    if not resume:
        output.mkdir(parents=True)
        json_write(output / "train-manifest.json", train_manifest)
    else:
        require(checkpoint_manifest(resume) == train_manifest, "resume training manifest changed")
    metadata = command_metadata(args)
    for variant in config["variants"]:
        prefix = "checkpoint" if len(config["variants"]) == 1 else variant
        last_path, best_path = output / f"{prefix}-last.pt", output / f"{prefix}-best.pt"
        previous = None
        if resume and last_path.exists():
            previous = load_head_checkpoint(last_path)
            require(previous.get("training_format") == TRAINING_FORMAT and previous["training_config"] == config,
                    f"incompatible existing arm: {variant}")
            require(previous["capture_identity"] == identity and previous["variant"] == variant
                    and previous["output_dir"] == str(output) and previous["init_head_identity"] == init_identity
                    and previous["validation_prompt_identity"] == prompt_identity,
                    "existing arm belongs to a different run")
        if resume and variant == resume["variant"]:
            require(previous is not None and previous["update"] == resume["update"],
                    "resume must name the current last checkpoint, not an older/best checkpoint")
        if resume and config["variants"].index(variant) < config["variants"].index(resume["variant"]):
            require(previous is not None, "missing earlier comparison arm; cannot recover unambiguously")
        if previous:
            require(previous["update"] <= updates, "updates precede the saved cursor")
            if previous["update"] == updates:
                save_training_result(output, variant, previous)
                print(f"{variant}: already complete at update {updates}", flush=True)
                continue
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        head = make_head(width, config["rank"], variant != "fixed-ce", max_depth).to(target.device)
        if init:
            initialize_head(head, init, config["rank"], width)
        optimizer = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        sampler = training_data.new_sampler(config["seed"])
        completed, root_exposures, loss_exposures, elapsed_before = 0, 0, 0, 0.
        best_score, best_update, last_validation, last_probe = None, None, None, None
        if previous:
            require(resume_environment_matches(previous["environment"], target.environment),
                    "resume requires the same recorded runtime/environment (GPU UUID may differ)")
            head.load_state_dict(previous["head"], strict=True)
            optimizer.load_state_dict(previous["optimizer"])
            sampler.load_state_dict(previous["sampler"])
            completed = previous["update"]
            require(previous["scheduler"] == {"completed_updates": completed,
                    "schedule_updates": config["schedule_updates"], "warmup_updates": config["warmup_updates"]},
                    "scheduler/cursor mismatch")
            require(all(group["lr"] == learning_rate(config, completed) for group in optimizer.param_groups),
                    "optimizer LR does not match saved scheduler")
            root_exposures, loss_exposures = previous["root_exposures"], previous["loss_position_exposures"]
            elapsed_before = previous["training_s"]
            best_score, best_update = previous["best_score"], previous["best_update"]
            last_validation, last_probe = previous["validation"], previous["probe"]
            restore_rng(previous["rng"])
        target.reset_peak()
        target.sync()
        started = time.perf_counter()
        log_path = output / f"{variant}-training.jsonl"
        with log_path.open("a") as log:
            for update in range(completed + 1, updates + 1):
                depth = active_depth(config, update)
                lr = learning_rate(config, update)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                states, tokens = sampler.next_batch(config["batch_size"])
                states, tokens = states.to(target.device), tokens.to(target.device)
                require(len(states) == config["batch_size"], "sampler returned an incomplete training batch")
                optimizer.zero_grad(set_to_none=True)
                loss, per_depth = training_loss(head, states, tokens, target,
                    config["state_weight"] if variant in ("shared-state", "shared-state-norm") else 0.,
                    config["kl_weight"], config["ce_weight"], config["temperature"], depth,
                    config["teacher_argmax_diagnostic"],
                    state_norm_weight=config["state_weight"] if variant == "shared-state-norm" else 0.)
                require(torch.isfinite(loss).item(), "nonfinite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                root_exposures += len(states)
                loss_exposures += len(states) * depth
                improved = False
                event = {"variant": variant, "update": update, "planned_updates": updates, "depth": depth,
                         "lr": lr, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                         "per_depth": per_depth, "root_exposures": root_exposures,
                         "loss_position_exposures": loss_exposures}
                if validation is not None and (update % config["validation_every"] == 0 or update == updates):
                    last_validation = {"update": update, **offline_validation(target, head, validation, depths,
                        config["validation_roots"], config["validation_batch_size"], config["validation_seed"])}
                    event["validation"] = last_validation
                    if best_score is None or last_validation["selection_score"] < best_score:
                        best_score, best_update, improved = last_validation["selection_score"], update, True
                if config["probe_every"] and (update % config["probe_every"] == 0 or update == updates):
                    last_probe = {"update": update, **validation_probe(target, head, probe_rows,
                        config["probe_new_tokens"], config["record_divergence"])}
                    event["probe"] = last_probe
                if config["overfit_roots"] and (update % 100 == 0 or update == updates):
                    event["training_set_diagnostic_NOT_validation"] = offline_validation(
                        target, head, training_data, depths, config["overfit_roots"],
                        config["validation_batch_size"], config["validation_seed"])
                elapsed = time.perf_counter() - started
                event["eta_s"] = elapsed / (update - completed) * (updates - update)
                event["elapsed_s"] = elapsed_before + elapsed
                if update == 1 or update % 10 == 0 or update == updates or "validation" in event or "probe" in event:
                    line = json.dumps(event, allow_nan=False)
                    log.write(line + "\n")
                    log.flush()
                    print(line, flush=True)
                if improved or update % config["checkpoint_every"] == 0 or update == updates:
                    target.assert_frozen()
                    checkpoint = {**metadata, "training_format": TRAINING_FORMAT,
                        "environment": target.environment, "training_config": config, "planned_updates": updates,
                        "variant": variant, "state_transition": STATE_TRANSITION, "rank": config["rank"],
                        "hidden_size": width, "max_depth": max_depth, "depth_weights": WEIGHTS,
                        "state_weight": config["state_weight"] if variant in ("shared-state", "shared-state-norm") else 0.,
                        "state_norm_weight": config["state_weight"] if variant == "shared-state-norm" else 0.,
                        "ce_weight": config["ce_weight"], "kl_weight": config["kl_weight"],
                        "temperature": config["temperature"],
                        "objective": "teacher-argmax diagnostic" if config["teacher_argmax_diagnostic"] else
                            ("sequence CE + forward teacher KL" if config["kl_weight"] else "sequence CE"),
                        "token_policy": "teacher-forced; predicted hidden feedback after root",
                        "head_parameters": sum(p.numel() for p in head.parameters()),
                        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "rng": rng_state(), "sampler": sampler.state_dict(),
                        "scheduler": {"completed_updates": update, "schedule_updates": config["schedule_updates"],
                                      "warmup_updates": config["warmup_updates"]},
                        "update": update, "root_exposures": root_exposures, "loss_position_exposures": loss_exposures,
                        "dataset_unique_roots": dataset.root_count, "training_sample_unique_roots": training_data.root_count,
                        "dataset_sequence_tokens": dataset.token_count, "training_s": event["elapsed_s"],
                        "selection": SELECTION if validation is not None else "last update; no validation selection",
                        "best_score": best_score, "best_update": best_update,
                        "validation": last_validation, "probe": last_probe,
                        "train_manifest_path": str(output / "train-manifest.json"),
                        "capture_identity": identity, "validation_prompt_identity": prompt_identity,
                        "init_head_identity": init_identity, "output_dir": str(output), **target.memory()}
                    if improved:
                        save_checkpoint(best_path, checkpoint)
                    save_checkpoint(last_path, checkpoint)
                    if update == updates:
                        # Preserve historical head filenames; these now also carry resumable state.
                        save_training_result(output, variant, checkpoint)
        del optimizer, head


def validate_head_command(args):
    require(args.split == "validation", "validate-head never reads train/test roots")
    require(Path(args.capture).name == "index.json", "validation requires generated capture index.json")
    require(not Path(args.output).exists(), "validation output already exists")
    checkpoint = load_head_checkpoint(args.head)
    dataset = captured_dataset(args.capture, "validation")
    training = checkpoint_manifest(checkpoint)
    validation_manifest = split_manifest(dataset, "validation")
    require(validation_manifest, "no validation manifest")
    check_heldout(training, validation_manifest)
    rows = read_prompt_records(args.validation_prompts, "validation")
    check_heldout(training, rows)
    check_heldout(split_manifest(captured_dataset(args.capture, "test"), "test"), rows)
    probe_depth = getattr(args, "probe_depth", 1)
    require(probe_depth in DEPTHS, "invalid validation probe depth")
    require(checkpoint["variant"] != "fixed-ce" or probe_depth <= checkpoint["max_depth"],
            "cannot extend untrained fixed-depth blocks")
    target = GlimmerTarget(args.attention)
    head = head_from_checkpoint(checkpoint, target, evaluation_depth=probe_depth).to(target.device)
    evaluation_depths = [d for d in DEPTHS if d <= max(checkpoint["max_depth"], probe_depth)]
    report = {**command_metadata(args), "environment": target.environment, "head": args.head,
              "selection_definition": SELECTION, "split": "validation",
              "trained_depth": checkpoint["max_depth"], "evaluation_depths": evaluation_depths,
              "checkpoint_selection_depths": [d for d in DEPTHS if d <= checkpoint["max_depth"]]}
    report["offline"] = offline_validation(target, head, dataset,
        evaluation_depths, args.validation_roots, args.batch_size, args.seed)
    report["probe"] = validation_probe(target, head, tokenize_prompts(target, rows),
        args.max_new_tokens, args.record_divergence, depth=probe_depth)
    target.assert_frozen()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json_write(args.output, report)

def token_fidelity(reference, candidate):
    first = next((i for i, (a, b) in enumerate(zip(reference, candidate)) if a != b),
                 min(len(reference), len(candidate)) if len(reference) != len(candidate) else None)
    return {"exact_token_identity": reference == candidate,
            "first_divergence_position": first, "divergence_position_base": 0,
            "baseline_generated_tokens": len(reference), "candidate_generated_tokens": len(candidate),
            "generated_length_difference": len(candidate) - len(reference)}


def check_pair_fidelity(pair, record_divergence=False):
    pair.update(token_fidelity(pair["baseline"]["token_ids"], pair["candidate"]["token_ids"]))
    require(pair["exact_token_identity"] or record_divergence,
            "greedy identity failed; no retry or fallback permitted")


def fidelity_summary(pairs):
    divergent = [p for p in pairs if not p["exact_token_identity"]]
    return {"exact_token_identity": not divergent,
            "exact_match_rate": 1 - len(divergent) / len(pairs), "diverged_pairs": len(divergent),
            "median_first_divergence_position": statistics.median(p["first_divergence_position"] for p in divergent)
                if divergent else None,
            "max_absolute_generated_length_difference": max(abs(p["generated_length_difference"]) for p in pairs)}


def conditional_acceptance(pairs, depth):
    passes = [item for pair in pairs for item in pair["candidate"]["passes"]]
    rows = []
    for d in range(1, depth + 1):
        eligible = sum(len(p["drafts"]) >= d and p["accepted_drafts"] >= d - 1 for p in passes)
        accepted = sum(p["accepted_drafts"] >= d for p in passes)
        rows.append({"depth": d, "eligible_accepted_prefixes": eligible, "accepted_drafts": accepted,
                     "conditional_acceptance_rate": accepted / eligible if eligible else None})
    return rows


def diagnostic_prompt_ids(rows, count, seed):
    if count is None:
        return {r["id"] for r in rows}  # Historical replay-every-prompt default.
    require(0 <= count <= len(rows), "diagnostic-prompts must fit the actual heldout set")
    rng = random.Random(seed)
    groups = [[r["id"] for r in rows if r["category"] == c] for c in sorted(CATEGORIES)]
    for group in groups:
        rng.shuffle(group)
    ordered = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
    return set(ordered[:count])


def evaluate_command(args):
    import torch
    require(not Path(args.output).exists(), "evaluation output already exists")
    require(args.repeats > 0, "repeats must be positive")
    external = bool(args.eval_data)
    if external:
        evaluation = read_prompt_records(args.eval_data, "test")
        if Path(args.capture).suffix == ".pt":
            capture_rows = manifest(load_capture(args.capture)["records"])
        else:
            capture_rows = [r for split in ("train", "validation", "test")
                            for r in split_manifest(captured_dataset(args.capture, split), split)]
        check_heldout(capture_rows, evaluation)
    else:
        require(Path(args.capture).suffix == ".pt", "index.json evaluation requires external --eval-data")
        bundle = load_capture(args.capture)
        evaluation = [r for r in bundle["records"] if r["split"] == "eval"]
    require(all(sum(r.get("category") == c for r in evaluation) >= 2 for c in CATEGORIES),
            "pilot requires at least two heldout prompts in each of six categories")
    checkpoints = []
    for path in args.heads:
        ckpt = load_head_checkpoint(path)
        check_heldout(checkpoint_manifest(ckpt), evaluation)
        validation_file = ckpt.get("training_config", {}).get("validation_prompts")
        if validation_file:
            check_heldout(read_prompt_records(validation_file, "validation"), evaluation)
        checkpoints.append((path, ckpt))
    require(len({c["variant"] for _, c in checkpoints}) == len(checkpoints), "duplicate evaluation variant")
    target = GlimmerTarget(args.attention)
    if external:
        evaluation = tokenize_prompts(target, evaluation)
        check_heldout(capture_rows, evaluation)
    for row in evaluation:
        context_guard(len(row["token_ids"]), args.max_new_tokens, 8)
    seed = args.eval_seed if args.eval_seed is not None else 20261002
    diagnostics = diagnostic_prompt_ids(evaluation, args.diagnostic_prompts, seed)
    randomized = external or args.eval_seed is not None
    rng = random.Random(seed)
    report = {**command_metadata(args), "environment": target.environment, "status": "running",
              "fidelity_policy": "record_divergence_diagnostic" if args.record_divergence else "strict_identity",
              "baseline": "same HF target, no speculation, same BF16/backend/prompt/output/EOS policy",
              "intentional_differences": "shared/unshared blocks, state loss, draft depth; equal data/updates NOT parameters",
              "performance_scope": "standalone synchronized HF development pilot; not production serving or DFlash comparison",
              "order_seed": seed if randomized else None, "randomized_interleaved": randomized,
              "diagnostic_prompt_ids": sorted(diagnostics), "diagnostics_timed": False,
              "pairs": [], "heads": []}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            heads = []
            for path, ckpt in checkpoints:
                head = head_from_checkpoint(ckpt, target).eval()
                heads.append(head)
                report["heads"].append({"path": path, "variant": ckpt["variant"],
                    "parameters": sum(p.numel() for p in head.parameters()), "checkpoint_bytes": Path(path).stat().st_size,
                    "training_arguments": ckpt.get("training_config", ckpt.get("arguments")),
                    "selection": ckpt.get("selection"), "update": ckpt.get("update")})
            tasks = [(i, depth, repeat, index) for i, (_, ckpt) in enumerate(checkpoints)
                     for depth in DEPTHS if depth <= ckpt["max_depth"]
                     for repeat in range(args.repeats) for index in range(len(evaluation))]
            if randomized:
                rng.shuffle(tasks)
            warmed = set()
            for i, depth, repeat, index in tasks:
                head, (_, ckpt), row = heads[i], checkpoints[i], evaluation[index]
                if (i, depth) not in warmed:
                    # Every head stays on CPU except the active candidate; transfers are untimed.
                    decode(target, evaluation[0]["token_ids"], 10)
                    head.to(target.device)
                    decode(target, evaluation[0]["token_ids"], 10, depth, Drafter(head, target))
                    head.cpu()
                    warmed.add((i, depth))
                baseline_first = rng.randrange(2) == 0 if randomized else (repeat + index) % 2 == 0
                pair = {"id": row["id"], "category": row["category"], "repeat": repeat,
                        "variant": ckpt["variant"], "depth": depth, "prompt_token_ids": row["token_ids"],
                        "prompt_format": row.get("prompt_format", "raw"),
                        "order": ["baseline", "candidate"] if baseline_first else ["candidate", "baseline"]}
                report["pairs"].append(pair)
                for mode in pair["order"]:
                    head.to("cpu" if mode == "baseline" else target.device)
                    pair[mode] = decode(target, row["token_ids"], args.max_new_tokens,
                        0 if mode == "baseline" else depth, None if mode == "baseline" else Drafter(head, target))
                    pair[mode]["text"] = target.tokenizer.decode(pair[mode]["token_ids"])
                check_pair_fidelity(pair, args.record_divergence)
                pair["decode_speedup"] = pair["baseline"]["decode_s"] / pair["candidate"]["decode_s"]
                if repeat == 0 and row["id"] in diagnostics:
                    head.to(target.device)
                    diagnostic = decode(target, row["token_ids"], args.max_new_tokens,
                                        depth, Drafter(head, target), diagnostics=True)
                    require(diagnostic["token_ids"] == pair["candidate"]["token_ids"], "diagnostic replay identity failed")
                    pair["drift_separate_untimed_replay"] = summarize_drift(diagnostic["drift"])
                    pair["drift_raw"] = diagnostic["drift"]
                    roots = [p["cache_before"] - len(row["token_ids"])
                             for p in pair["candidate"]["passes"]] or [0]
                    chosen_drift = chosen_path_drift(target, head, row["token_ids"],
                        pair["candidate"]["token_ids"], depth, roots, distribution_metrics=external)
                    pair["chosen_path_drift_separate_diagnostic"] = summarize_drift(chosen_drift)
                    pair["chosen_path_drift_raw"] = chosen_drift
                head.cpu()
                json_write(args.output, report)
                identity = "OK" if pair["exact_token_identity"] else "DIVERGED"
                print(f"{ckpt['variant']} d={depth} {row['id']} identity={identity} accepted/pass="
                      f"{pair['candidate']['mean_accepted_drafts_per_pass']:.3f} "
                      f"decode_ratio={pair['decode_speedup']:.3f}", flush=True)
        report["summary"] = []
        for variant in sorted({p["variant"] for p in report["pairs"]}):
            for depth in DEPTHS:
                for category in (None, *sorted(CATEGORIES)):
                    group = [p for p in report["pairs"] if p["variant"] == variant and p["depth"] == depth
                             and (category is None or p["category"] == category)]
                    if not group:
                        continue
                    exact = [p for p in group if p["exact_token_identity"]]
                    accepted = sum(p["candidate"]["accepted_drafts"] for p in group)
                    passes = sum(p["candidate"]["verification_passes"] for p in group)
                    report["summary"].append({"variant": variant, "depth": depth, "category": category or "pooled",
                        "pairs": len(group), "exact_output_pairs": len(exact),
                        "median_paired_decode_speedup": statistics.median(p["decode_speedup"] for p in group),
                        "median_exact_output_decode_speedup": statistics.median(p["decode_speedup"] for p in exact) if exact else None,
                        "accepted_drafts_per_pass": accepted / passes if passes else 0.,
                        "conditional_acceptance_by_depth": conditional_acceptance(group, depth), **fidelity_summary(group)})
        target.assert_frozen()
        report["fidelity"] = fidelity_summary(report["pairs"])
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
    prepare = sub.add_parser("prepare-prompts", help="prepare pinned, source-disjoint teacher prompts")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=20261002)
    prepare.add_argument("--coding-fraction", type=float, default=.4)
    prepare.add_argument("--max-prompts", type=int, default=12000)
    generated = sub.add_parser("capture-generated", help="generate frozen-target responses and shard teacher states")
    generated.add_argument("--prompts", required=True)
    generated.add_argument("--output-dir", required=True)
    generated.add_argument("--train-token-budget", type=int, default=3000000)
    generated.add_argument("--validation-token-budget", type=int, default=150000)
    generated.add_argument("--test-token-budget", type=int, default=150000)
    generated.add_argument("--max-prompt-tokens", type=int, default=512)
    generated.add_argument("--max-new-tokens", type=int, default=512)
    generated.add_argument("--generation-batch-size", type=int, default=4)
    train = sub.add_parser("train", help="frozen-target training, validation and exact same-runtime resume")
    train.add_argument("--capture", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--resume", help="current last.pt; reconstruct options, optimizer, RNG and data cursor")
    # Suppressed defaults distinguish omitted options from incompatible explicit resume overrides.
    train.add_argument("--updates", type=int, default=argparse.SUPPRESS)
    train.add_argument("--variants", nargs="+", choices=VARIANTS, default=argparse.SUPPRESS)
    train.add_argument("--rank", type=int, choices=(64, 128, 256), default=argparse.SUPPRESS)
    train.add_argument("--train-depth", type=int, choices=DEPTHS, default=argparse.SUPPRESS)
    for key in ("batch_size", "seed", "schedule_updates", "warmup_updates", "checkpoint_every",
                "validation_every", "probe_every", "validation_roots", "validation_batch_size",
                "validation_seed", "probe_new_tokens", "overfit_roots"):
        train.add_argument("--" + key.replace("_", "-"), type=int, default=argparse.SUPPRESS)
    for key in ("lr", "state_weight", "ce_weight", "kl_weight", "temperature", "weight_decay", "grad_clip"):
        train.add_argument("--" + key.replace("_", "-"), type=float, default=argparse.SUPPRESS)
    for key in ("curriculum", "validation_prompts", "init_head"):
        train.add_argument("--" + key.replace("_", "-"), default=argparse.SUPPRESS)
    train.add_argument("--teacher-argmax-diagnostic", action="store_true", default=argparse.SUPPRESS,
                       help="Stage-0 only: hard teacher labels on <=64 fixed roots, NOT sequence CE")
    train.add_argument("--record-divergence", action="store_true", default=argparse.SUPPRESS)
    train.add_argument("--attention", choices=("sdpa", "eager"), default=argparse.SUPPRESS)
    validation = sub.add_parser("validate-head", help="fixed validation roots plus real cached depth-1 probe")
    validation.add_argument("--capture", required=True)
    validation.add_argument("--head", required=True)
    validation.add_argument("--probe-depth", type=int, choices=DEPTHS, default=1,
                            help="validation recursion depth; shared heads can be probed beyond trained depth")
    validation.add_argument("--split", choices=("validation",), default="validation")
    validation.add_argument("--output", required=True)
    validation.add_argument("--validation-prompts", default=str(Path(__file__).with_name("glimmer_mtp_validation.jsonl")))
    validation.add_argument("--validation-roots", type=int, default=1024)
    validation.add_argument("--batch-size", type=int, default=8)
    validation.add_argument("--seed", type=int, default=314159)
    validation.add_argument("--max-new-tokens", type=int, default=64)
    validation.add_argument("--record-divergence", action="store_true")
    evaluate = sub.add_parser("evaluate", help="real cached greedy A/B decode and separate drift replay")
    evaluate.add_argument("--capture", required=True)
    evaluate.add_argument("--heads", nargs="+", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--eval-data", help="external test-only JSONL, independent of capture/optimizer/validation")
    evaluate.add_argument("--diagnostic-prompts", type=int, help="stratified untimed replay subset; legacy default all")
    evaluate.add_argument("--eval-seed", type=int, help="randomize/interleave head/depth/prompt/repeat pairs")
    evaluate.add_argument("--max-new-tokens", type=int, choices=(64, 128), default=64)
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--record-divergence", action="store_true",
                          help="diagnostic only: record output differences instead of aborting; verifier unchanged")
    for command in (capture, generated, validation, evaluate):
        command.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    return p


def main():
    args = parser().parse_args()
    if args.command == "validate":
        records = read_records(args.data)
        print(json.dumps({"sequences": len(records), "splits": dict(Counter(r["split"] for r in records)),
                          "heldout_categories": dict(Counter(r["category"] for r in records if r["split"] == "eval"))}))
    else:
        {"capture": capture_command, "train": train_command, "evaluate": evaluate_command,
         "prepare-prompts": prepare_prompts_command, "capture-generated": capture_generated_command,
         "validate-head": validate_head_command}[args.command](args)


if __name__ == "__main__":
    main()
