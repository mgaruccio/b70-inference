"""Read-only observations of ONE real DSpark proposal. No replacement arithmetic.

Imported by the temporary overlay. Unarmed/dummy/unrelated requests are ignored.
Callbacks always return None; method wrappers return the original object. Errors
invalidate the artifact, not the request. Synchronous CPU snapshots add latency;
this is not a throughput benchmark. No tensor is cast or written in the live path.
"""
import functools
import inspect
import json
import os
from pathlib import Path
import traceback
import weakref

from parity_common import K, TAPS, check_layout, first_request_matches, tensor_identity, weight_parts

_CURRENT = None
_DONE = False


def tensor(name, value):
    if _CURRENT is not None:
        _CURRENT.safe(_CURRENT.put, name, value)


def first(value):
    return value[0] if isinstance(value, tuple) else value


class Capture:
    def __init__(self, spec, batch, armed, out):
        self.spec, self.batch, self.armed, self.out = spec, batch, armed, out
        self.data, self.errors, self.handles, self.restores = {}, [], [], []
        self.n = batch.num_tokens
        self.bytes = 0
        self.meta = {"complete": False, "armed": armed, "request_ids": batch.req_ids,
                     "target_layer_ids_configured": TAPS, "past_key_values": None,
                     "num_computed_tokens": int(batch.num_computed_tokens_np[0]),
                     "draft_config": spec.model.config.to_dict(), "attention": []}

    def safe(self, callback, *args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception:
            self.errors.append(traceback.format_exc())
            return None

    def put(self, name, value):
        if name in self.data:
            raise ValueError(f"duplicate observation (not first proposal): {name}")
        size = value.numel() * value.element_size()
        if self.bytes + size > 256 << 20:
            raise ValueError("capture exceeds 256 MiB bound")
        self.data[name] = value.detach().to(device="cpu", copy=True)
        self.bytes += size

    def post(self, module, name, select=first):
        def callback(_module, _args, output):
            self.safe(lambda: self.put(name, select(output)))
            # Never return a tensor: PyTorch would replace the actual output.
        self.handles.append(module.register_forward_hook(callback))

    def pre(self, module, name):
        def callback(_module, args):
            self.safe(lambda: self.put(name, args[0]))
        self.handles.append(module.register_forward_pre_hook(callback))

    def observe_qk(self, module, prefix):
        def callback(_module, args):
            self.safe(lambda: self.put(prefix + "query_q_rope", args[0]))
            self.safe(lambda: self.put(prefix + "query_k_rope", args[1]))
        self.handles.append(module.register_forward_pre_hook(callback))

    def wrap(self, obj, name, before=None, after=None):
        original = getattr(obj, name)
        had_own = name in obj.__dict__
        own = obj.__dict__.get(name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if before is not None:
                self.safe(before, *args, **kwargs)
            output = original(*args, **kwargs)
            if after is not None:
                self.safe(after, output)
            return output

        setattr(obj, name, wrapped)
        self.restores.append((obj, name, had_own, own))

    def attach(self, args):
        s, m = self.spec, self.spec.model
        b = m.model
        if (s.draft_logits is not None or s.num_query_per_req != K or s._draft_topk is not None
                or s.enable_adaptive_verification or m.draft_id_to_target_id is not None):
            raise ValueError("capture requires corrected greedy full-vocab K7 fixed verification")
        target = s._parity_target()
        target_language_model = target.get_language_model() if hasattr(target, "get_language_model") else target
        target_backbone = target_language_model.model
        self.meta["target_aux_registration"] = list(target_backbone.aux_hidden_state_layers)
        aux = args["aux_hidden_states"]
        if len(aux) != len(TAPS):
            raise ValueError("expected exactly five ordered target aux tensors")
        for i, value in enumerate(aux):
            self.put(f"aux.{i}", value[:self.n])
        for name in ("input_ids", "positions"):
            self.put("target_" + name, getattr(self.batch, name)[:self.n])
        self.put("num_sampled", args["num_sampled"][:1])
        self.put("num_rejected", args["num_rejected"][:1])
        self.pre(b.fc, "fc_input")
        self.post(b.fc, "context_fc")
        self.post(b.embed_tokens, "shared_query_embedding_fp16")
        self.wrap(b, "embed_input_ids", after=lambda out: self.put("query_embedding", out))
        self.wrap(b, "precompute_and_store_context_kv", before=self.context)
        self.wrap(s, "_generate_draft", before=self.query)
        self.wrap(b, "forward", after=lambda out: self.put("final_hidden", out))
        for i, layer in enumerate(b.layers):
            p = f"layers.{i}."
            self.post(layer.input_layernorm, p + "query_input_norm")
            self.post(layer.self_attn.q_norm, p + "query_q_norm")
            self.post(layer.self_attn.k_norm, p + "query_k_norm")
            # get_rope caches one shared module across layers. Observe its actual
            # Q/K outputs at each distinct Attention input, not five hooks on RoPE.
            self.observe_qk(layer.self_attn.attn, p)
            self.post(layer.self_attn.attn, p + "query_attention")
            self.post(layer.self_attn.o_proj, p + "query_o_proj")
            self.post(layer.post_attention_layernorm, p + "query_post_norm")
            self.post(layer.mlp, p + "query_mlp")
            self.post(layer, p + "query_branch", lambda x: x[0])
            self.post(layer, p + "query_residual", lambda x: x[1])
        self.wrap(m, "compute_draft_logits", before=lambda hidden: self.put("head_input_bf16", hidden),
                  after=lambda out: self.put("base_logits", out))
        self.markov_step = 0
        self.wrap(m, "markov_embed", before=self.prev)
        self.wrap(m, "markov_bias", after=self.bias)
        self.wrap(s, "_sample_logits", before=self.corrected, after=self.sampled)

    def context(self, states, positions, slots=None):
        self.put("context_positions", positions)
        groups = slots if isinstance(slots, (list, tuple)) else [slots]
        self.data["context_slots"] = [v.detach().cpu().tolist() for v in groups]

    def query(self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings,
              num_tokens_across_dp=None, cudagraph_runtime_mode=None):
        s = self.spec
        if num_reqs != 1 or num_tokens_padded != K or str(cudagraph_runtime_mode) not in ("None", "NONE", "CUDAGraphMode.NONE"):
            raise ValueError("first-step diagnostic is eager C1 without padded queries")
        for name, value in (
            ("query_ids", s.input_buffers.input_ids[:K]),
            ("query_positions", s.input_buffers.positions[:K]),
            ("draft_seq_lens", s.input_buffers.seq_lens[:1]),
            ("sample_indices", s.sample_indices[:K]),
            ("sample_positions", s.sample_pos[:K]),
        ):
            self.put(name, value)
        # Record the exact logical-to-physical map consumed by every draft layer.
        self.data["query_slots"] = []
        for i, layer in enumerate(s.model.model.layers):
            attn = layer.self_attn.attn
            md = attn_metadata[attn.layer_name]
            causal = md.causal
            if hasattr(causal, "item"):
                causal = causal.item()
            info = {"layer": i, "name": attn.layer_name, "causal": bool(causal),
                    "sliding_window": layer.self_attn.sliding_window,
                    "seq_lens": md.seq_lens.detach().cpu().tolist(),
                    "query_start_loc": md.query_start_loc.detach().cpu().tolist(),
                    "kv_dtype": str(attn.kv_cache.dtype)}
            self.meta["attention"].append(info)
            if info["causal"] or info["sliding_window"] is not None or info["seq_lens"] != [self.n + K] or info["query_start_loc"] != [0, K]:
                raise ValueError(f"visibility differs from cache-free noncausal reference: {info}")
            self.put(f"block_table.{i}", md.block_table[:1])
            self.data["query_slots"].append(slot_mappings[attn.layer_name][:K].detach().cpu().tolist())
        # Upstream passes either one shared context map or one map per layer.
        if len(self.data["context_slots"]) == 1:
            self.data["context_slots"] *= len(s.model.model.layers)

    def prev(self, ids):
        self.put(f"markov_prev.{self.markov_step}", ids)

    def bias(self, out):
        self.put(f"markov_bias.{self.markov_step}", out)

    def corrected(self, logits, idx_map, sample_pos, step):
        if step != self.markov_step:
            raise ValueError("unexpected sequential sampling order")
        self.put(f"corrected_logits.{step}", logits)

    def sampled(self, out):
        self.put(f"sampled.{self.markov_step}", out)
        self.markov_step += 1

    def detach(self):
        for handle in self.handles:
            handle.remove()
        for obj, name, had_own, own in reversed(self.restores):
            if had_own:
                setattr(obj, name, own)
            else:
                delattr(obj, name)

    def finish(self, output):
        import torch
        self.put("proposed_ids", output)
        for required in ("final_hidden", "base_logits", "head_input_bf16", "context_fc", "context_norm"):
            if required not in self.data:
                raise ValueError(f"required native observation missing: {required}")
        if self.markov_step != K:
            raise ValueError("did not observe exactly seven real sampler calls")
        check_layout(self.data, self.n)
        b = self.spec.model.model
        config = self.meta["draft_config"]
        weights = {}
        for name, value in b.named_parameters():
            if name in ("embed_tokens.weight", "mask_embedding"):
                continue  # recorded query embeddings; unused separate-mask placeholder
            for key, start, end in weight_parts(name, value.shape, config):
                if key in weights:
                    raise ValueError(f"duplicate native-to-HF key: {key}")
                weights[key] = tensor_identity(value[start:end])
        if len(weights) != 62:
            raise ValueError(f"expected 62 native-to-HF weight mappings, got {len(weights)}")
        self.meta["weights"] = weights
        self.meta["shared_lm_head"] = tensor_identity(self.spec.model.lm_head.weight)
        self.meta["shared_embedding_replayed_from_capture"] = True
        self.meta["logit_scale"] = self.spec.model.logits_processor.scale
        if self.meta["logit_scale"] != 1.0:
            raise ValueError("unexpected logit scaling")
        self.meta["errors"] = self.errors
        self.meta["complete"] = not self.errors
        torch.save(self.data, self.out / "capture.pt")


def install(cls):
    """Wrap original public proposer; retain target only to observe aux registration."""
    load = cls.load_draft_model

    @functools.wraps(load)
    def load_wrapper(self, target_model, *args, **kwargs):
        result = load(self, target_model, *args, **kwargs)
        self._parity_target = weakref.ref(target_model)
        return result

    cls.load_draft_model = load_wrapper
    propose = cls.propose
    signature = inspect.signature(propose)

    @functools.wraps(propose)
    def wrapper(self, *args, **kwargs):
        global _CURRENT, _DONE
        directory = os.environ.get("DSPARK_PARITY_OUTPUT")
        if _DONE or not directory:
            return propose(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a["dummy_run"] or a["is_profile"]:
            return propose(self, *args, **kwargs)
        out = Path(directory)
        armed_path = out / "arm.json"
        if not armed_path.is_file():
            return propose(self, *args, **kwargs)
        try:
            armed = json.loads(armed_path.read_text())
            batch = a["input_batch"]
            matches = first_request_matches(batch, armed)
        except Exception:
            # A malformed arm does not affect the request or cause a retry.
            matches = False
        if not matches:
            return propose(self, *args, **kwargs)
        _DONE = True  # never collect a later cached proposal, even on failure
        try:
            capture = Capture(self, batch, armed, out)
        except Exception:
            traceback.print_exc()
            return propose(self, *args, **kwargs)
        _CURRENT = capture
        capture.safe(capture.attach, a)
        try:
            output = propose(self, *args, **kwargs)
            capture.safe(capture.finish, output)
            return output
        finally:
            _CURRENT = None
            capture.safe(capture.detach)
            capture.meta["errors"] = capture.errors
            capture.meta["complete"] = capture.meta["complete"] and not capture.errors
            try:
                (out / "capture.json").write_text(json.dumps(capture.meta, indent=2) + "\n")
            except Exception:
                traceback.print_exc()  # diagnostic I/O cannot replace a model result

    cls.propose = wrapper
