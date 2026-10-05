#!/usr/bin/env python3
"""Opt-in disposable vLLM 0.27.1 hook. See qwen38_mtp_live_parity.py --help.

Run this installer INSIDE the pinned disposable container, after the existing
training overlay installer. No persistent launcher or existing patch is edited.
The hook observes the real runner -> proposer -> model -> sampler calls; it never
substitutes tensors. Copies/synchronizations invalidate all performance timings.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import functools
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import re
import shutil

REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
IMAGE = "vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
MAX_REQUESTS, MAX_ROUNDS, MAX_PROMPT, MAX_OUTPUT = 2, 16, 1024, 64
MAX_BYTES = 128 * 1024 * 1024
MARKER = "# QWEN38_MTP_PARITY_DISPOSABLE"
# Fresh official v0.27.1 source inspected 2026-10-05. AST method guards allow
# ONLY unrelated existing overlay/capture seams, not changed proposer semantics.
SOURCE_HASHES = {
    "v1/spec_decode/llm_base_proposer.py": "aab141f140be6ee7e3a64ce9d866fc139db9d86477c295970f85b311ee00c3f0",
    "v1/spec_decode/eagle.py": "b2b1f7d15117b43108be368756d9c6e9334d9415a070e46178e18bb7c6476378",
}
METHOD_HASHES = {
    ("v1/worker/gpu_model_runner.py", "GPUModelRunner", "propose_draft_token_ids"):
        "4b724b4fbd544f7738baa161925d9aabaf35c1c79918c09d6fde3054b9406a83",
    ("model_executor/models/qwen3_5_mtp.py", "Qwen3_5MultiTokenPredictor", "forward"):
        "3c7bd4d78b53ad37b3172ab55236fbe5d210e70729157c6b302ba9b87c039396",
    ("model_executor/models/qwen3_5_mtp.py", "Qwen3_5MTP", "forward"):
        "fc11fc949a8c552ce355d71c5d3e170657047927514863c6997223df7c829702",
}


def require(condition, message):
    if not condition:
        raise RuntimeError("MTP parity: " + message)


def method_hash(source, cls, method):
    nodes = [m for c in ast.parse(source).body if isinstance(c, ast.ClassDef) and c.name == cls
             for m in c.body if isinstance(m, ast.FunctionDef) and m.name == method]
    require(len(nodes) == 1, "missing/ambiguous source method " + cls + "." + method)
    return hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest()


def guard_sources(root, version):
    require(version == "0.27.1", "requires exactly vLLM 0.27.1")
    paths = set(SOURCE_HASHES) | {key[0] for key in METHOD_HASHES}
    sources = {p: (root / p).read_text() for p in paths}
    for path, digest in SOURCE_HASHES.items():
        require(hashlib.sha256(sources[path].encode()).hexdigest() == digest,
                "unsupported installed source: " + path)
    for (path, cls, method), digest in METHOD_HASHES.items():
        require(method_hash(sources[path], cls, method) == digest,
                "unsupported installed method: " + cls + "." + method)
    return sources


def validate_control(control, key):
    require(set(control) == {"request_id", "split", "prompt_id", "prompt_token_ids", "max_tokens"},
            "unexpected control fields (do not pass private configs)")
    native_id = re.fullmatch(r"b70-native-[0-9a-f]{32}", key)
    parity_id = re.fullmatch(r"parity-(train|dev)-[a-zA-Z0-9_-]{1,80}", key)
    require(control["request_id"] == key and (native_id or parity_id),
            "selected public train/dev request ID required")
    require(control["split"] in ("train", "dev") and
            (native_id or key.startswith("parity-" + control["split"] + "-")),
            "held-out/private requests are not allowed")
    require(isinstance(control["prompt_id"], str) and bool(control["prompt_id"]), "missing public prompt ID")
    ids = control["prompt_token_ids"]
    require(isinstance(ids, list) and 3 <= len(ids) <= MAX_PROMPT and
            all(type(t) is int and 0 <= t < 248320 for t in ids), "invalid/bounded prompt IDs")
    require(type(control["max_tokens"]) is int and 1 <= control["max_tokens"] <= MAX_OUTPUT,
            "output limit exceeded")
    return control


def private_write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


@contextmanager
def wrapped_methods(items):
    """Restore instance attributes even if a model call or collector fails."""
    originals = [(obj, name, name in obj.__dict__, obj.__dict__.get(name)) for obj, name, _ in items]
    try:
        for obj, name, wrapper in items:
            setattr(obj, name, wrapper)
        yield
    finally:
        for obj, name, existed, original in reversed(originals):
            if existed:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def bind(fn, args, kwargs):
    bound = inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def cpu(value):
    return None if value is None else value.detach().to(device="cpu", copy=True)


def guard_runner(runner):
    import torch

    spec, cfg = runner.speculative_config, runner.model_config
    require(spec is not None and spec.method == "mtp" and spec.num_speculative_tokens in (4, 8), "native D4/D8 required")
    require(not spec.disable_padded_drafter_batch, "only the pinned padded drafter path is supported")
    require(cfg.enforce_eager and cfg.dtype == torch.bfloat16 and cfg.quantization is None,
            "eager unquantized BF16 required")
    require(runner.scheduler_config.max_num_seqs == 1 and not runner.scheduler_config.async_scheduling,
            "C1 synchronous scheduling required")
    require(not runner.cache_config.enable_prefix_caching and runner.cache_config.cache_dtype == "auto",
            "cold prefix and auto/BF16 KV required")
    require(all(getattr(runner.parallel_config, name) == 1 for name in
                ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")) and
            not runner.parallel_config.use_ubatching, "single GPU without microbatching required")
    require(not runner.use_aux_hidden_state_outputs and
            not hasattr(runner.get_model(), "get_mtp_target_hidden_states"), "unsupported target hidden boundary")
    require(type(runner.get_model()).__name__ in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM") and
            cfg.hf_text_config.hidden_size == 5120 and cfg.hf_text_config.vocab_size == 248320,
            "wrong target model")
    require(runner.vllm_config.kv_transfer_config is None and runner.vllm_config.ec_transfer_config is None,
            "external cache transfer unsupported")
    drafter = runner.drafter
    require(type(drafter).__name__ == "EagleProposer" and type(drafter.model).__name__ == "Qwen3_5MTP",
            "unsupported proposer/model wrapper (disable graphs/compilation)")
    require(not drafter.needs_extra_input_slots and not drafter.parallel_drafting and
            not drafter.constant_draft_positions and not drafter.use_heterogeneous_vocab,
            "unsupported proposer layout")


class Collector:
    def __init__(self, root):
        import vllm

        self.root = Path(root)
        require(self.root.is_absolute() and self.root.is_dir() and not self.root.is_symlink(), "existing private absolute trace root required")
        require(self.root.stat().st_mode & 0o077 == 0, "trace root must be mode 0700")
        require(os.environ.get("QWEN38_MTP_PARITY_REVISION") == REVISION and
                os.environ.get("QWEN38_MTP_PARITY_IMAGE") == IMAGE, "explicit pinned model/image declaration required")
        sources = guard_sources(Path(vllm.__file__).parent, vllm.__version__)
        native = os.environ.get("B70_MTP_NATIVE_CAPTURE_DIR")
        require(native in (None, str(self.root / "native")), "native capture must use this cell's /native subdirectory")
        if native:
            path = Path(vllm.__file__).parent / "model_executor/models/b70_mtp_native_capture.py"
            sources[path.name] = path.read_text()
        for path, source in sources.items():
            private_write(self.root / (Path(path).name + ".source"), source.encode())
        private_write(self.root / "runtime.json", json.dumps({
            "vllm": vllm.__version__, "declared_image": IMAGE, "declared_model_revision": REVISION,
            "max_rounds_per_request": MAX_ROUNDS, "max_requests": MAX_REQUESTS, "max_bytes": MAX_BYTES,
            "native_capture_enabled": bool(native),
            "note": "image/model declarations must be checked against lead's actual container/snapshot; timings invalid",
        }, indent=2).encode())
        self.steps, self.controls, self.total_bytes = {}, {}, 0

    def run(self, runner, original, args, kwargs):
        import torch
        from vllm.forward_context import get_forward_context

        guard_runner(runner)
        bound = bind(original, args, kwargs)
        scheduler = bound["scheduler_output"]
        require(len(runner.input_batch.req_ids) == 1, "exactly one active request required")
        req_id = runner.input_batch.req_ids[0]
        match = re.fullmatch(r"chatcmpl-(b70-native-[0-9a-f]{32}|parity-(?:train|dev)-[a-zA-Z0-9_-]{1,80}?)(?:-[0-9a-f]{8})?", req_id)
        require(match is not None, "unselected request; use the diagnostic client")
        key = match[1]
        control = validate_control(json.loads((self.root / (key + ".control.json")).read_text()), key)
        require(key in self.steps or len(self.steps) < MAX_REQUESTS, "request bound exceeded")
        require(key not in self.controls or self.controls[key] == control, "control changed mid-request")
        self.controls[key] = control
        step = self.steps.get(key, 0)
        self.steps[key] = step + 1
        request = runner.requests[req_id]
        require(request.prompt_token_ids == control["prompt_token_ids"] and
                request.sampling_params.max_tokens == control["max_tokens"], "public request/control mismatch")
        require(not request.mm_features and request.prompt_embeds is None and request.lora_request is None,
                "only plain public text requests are supported")
        require(not scheduler.scheduled_cached_reqs.resumed_req_ids and not getattr(scheduler, "preempted_req_ids", None),
                "resumption/preemption loses prefix context")
        require(bound["sampling_metadata"].all_greedy, "greedy sampling required")
        if step >= MAX_ROUNDS:
            # Deliberately bounded prefix, not a silently complete-request trace.
            return original(*args, **kwargs)
        n = scheduler.total_num_scheduled_tokens
        require(scheduler.num_scheduled_tokens == {req_id: n} and 1 <= n <= MAX_PROMPT, "scheduled row bound/layout")
        sampled = bound["sampled_token_ids"]
        require(isinstance(sampled, torch.Tensor) and sampled.shape[0] == 1, "GPU sampler path required")
        trace = dict(request_id=key, step=step, depth=runner.speculative_config.num_speculative_tokens,
                     block_size=runner.drafter.block_size,
                     target_ids=cpu(runner.input_ids.gpu[:n]), target_positions=cpu(runner._get_positions(n)),
                     target_hidden=cpu(bound["hidden_states"][:n]), sampled=cpu(sampled),
                     scheduled_drafts=list(scheduler.scheduled_spec_decode_tokens.get(req_id, [])), calls=[], samples=[])
        drafter, model = runner.drafter, runner.drafter.model
        first, metadata, forward, sample = (drafter.set_inputs_first_pass, drafter.build_per_group_and_layer_attn_metadata,
                                            model.forward, drafter._sample_draft_tokens)
        state = {}

        def observe_first(*a, **kw):
            b = bind(first, a, kw)
            trace["first"] = {k: cpu(b[k]) for k in ("target_token_ids", "target_positions", "target_hidden_states",
                                                    "next_token_ids", "num_rejected_tokens_gpu")}
            result = first(*a, **kw)
            trace["first"]["selected"] = cpu(result[1])
            return result

        def observe_metadata(*a, **kw):
            state["cad"] = bind(metadata, a, kw)["common_attn_metadata"]
            return metadata(*a, **kw)

        def observe_forward(*a, **kw):
            b, cad = bind(forward, a, kw), state["cad"]
            count = b["hidden_states"].shape[0]
            require(1 <= count <= MAX_PROMPT and len(trace["calls"]) < trace["depth"], "forward bound")
            slots = get_forward_context().slot_mapping
            require(isinstance(slots, dict) and len(slots) == 1, "one MTP attention layer/slot map required")
            row = dict(input_ids=cpu(drafter.input_ids[:count] if b["input_ids"] is None else b["input_ids"]),
                       positions=cpu(b["positions"]), hidden=cpu(b["hidden_states"]),
                       inputs_embeds=cpu(b["inputs_embeds"]), slots=cpu(next(iter(slots.values()))),
                       query_start=cpu(cad.query_start_loc), seq_lens=cpu(cad.seq_lens),
                       block_table=cpu(cad.block_table_tensor))
            result = forward(*a, **kw)
            require(isinstance(result, torch.Tensor), "unexpected model output tuple")
            row["output"] = cpu(result)
            trace["calls"].append(row)
            return result

        def observe_sample(*a, **kw):
            hidden = cpu(bind(sample, a, kw)["hidden_states"])
            result = sample(*a, **kw)
            trace["samples"].append(dict(hidden=hidden, ids=cpu(result[0])))
            return result

        with wrapped_methods([(drafter, "set_inputs_first_pass", observe_first),
                              (drafter, "build_per_group_and_layer_attn_metadata", observe_metadata),
                              (model, "forward", observe_forward), (drafter, "_sample_draft_tokens", observe_sample)]):
            result = original(*args, **kwargs)
        trace["draft_ids"] = cpu(result)
        require(len(trace["calls"]) == trace["depth"] == len(trace["samples"]), "missing live model/sampler boundary")
        buffer = io.BytesIO()
        torch.save(trace, buffer)
        data = buffer.getvalue()
        require(self.total_bytes + len(data) <= MAX_BYTES, "trace byte budget exceeded; no truncated tensor written")
        private_write(self.root / f"{key}.{step:03d}.pt", data)
        self.total_bytes += len(data)
        return result


def install(runner_class):
    if not os.environ.get("QWEN38_MTP_PARITY_DIR"):
        return
    original = runner_class.propose_draft_token_ids

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        if not hasattr(self, "_qwen38_parity_collector"):
            self._qwen38_parity_collector = Collector(os.environ["QWEN38_MTP_PARITY_DIR"])
        return self._qwen38_parity_collector.run(self, original.__get__(self), args, kwargs)

    runner_class.propose_draft_token_ids = observed


def patch(root, version):
    guard_sources(root, version)
    target = root / "v1/worker/gpu_model_runner.py"
    source = target.read_text()
    require(MARKER not in source, "use a fresh disposable container, hook already installed")
    source += ("\n" + MARKER + "\nfrom vllm.v1.worker.qwen38_mtp_parity import install as _install_mtp_parity\n"
               "_install_mtp_parity(GPUModelRunner)\n")
    compile(source, str(target), "exec")
    shutil.copyfile(__file__, target.with_name("qwen38_mtp_parity.py"))
    target.write_text(source)
    print(MARKER + " installed; inert without QWEN38_MTP_PARITY_DIR", flush=True)


if __name__ == "__main__":
    import vllm
    patch(Path(vllm.__file__).parent, vllm.__version__)
