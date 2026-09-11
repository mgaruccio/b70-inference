#!/usr/bin/env python3
"""Replay captured inputs through the UNMODIFIED official HF DSpark model.

No full target, from_pretrained, auto_map execution, downloads, replacement
attention/RoPE/Markov implementation, or fallback backend. Eager HF is an
intentional kernel difference. --fetch-source fetches ONLY two pinned .py files.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
import urllib.request

from parity_common import CONFIG_SHA, K, OFFICIAL_SHA, SOURCE_URL, TAPS, check_layout, sha, tensor_identity


def official(source, fetch=False):
    package = source / "specforge/modeling/draft"
    if fetch:
        package.mkdir(parents=True, exist_ok=False)
        for name, digest in OFFICIAL_SHA.items():
            with urllib.request.urlopen(SOURCE_URL + name, timeout=30) as response:
                data = response.read()
            import hashlib
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"official source fingerprint differs: {name}")
            (package / name).write_bytes(data)
    for name, digest in OFFICIAL_SHA.items():
        if sha(package / name) != digest:
            raise RuntimeError(f"official source fingerprint differs: {name}")
    # Explicit packaging-only accommodation: official dspark.py imports this
    # absolute namespace. Empty namespace directories, NOT stubs or edited code.
    sys.path.insert(0, str(source.resolve()))
    dflash = importlib.import_module("specforge.modeling.draft.dflash")
    dspark = importlib.import_module("specforge.modeling.draft.dspark")
    for module in (dflash, dspark):
        if Path(module.__file__).resolve().parent != package.resolve():
            raise RuntimeError(f"shadowed official source: {module.__file__}")
    return dflash, dspark


def metrics(left, right, logits=False):
    import torch
    if list(left.shape) != list(right.shape):
        raise ValueError(f"shape mismatch: {list(left.shape)} != {list(right.shape)}")
    a, b = left.float().cpu(), right.float().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = {"shape": list(a.shape), "dtypes": [str(left.dtype), str(right.dtype)],
              "finite": finite, "exact": bool(torch.equal(left.cpu(), right.cpu()))}
    if not finite:
        result["nonfinite_counts"] = [int((~torch.isfinite(x)).sum()) for x in (a, b)]
        return result
    d = a - b
    ad, bd = a.double().flatten(), b.double().flatten()
    denom = ad.norm() * bd.norm()
    result.update(max_abs=float(d.abs().max()), rmse=float(d.square().mean().sqrt()),
                  cosine=float((ad @ bd) / denom) if denom else None)
    if logits:
        a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        ai, bi = a.argmax(-1), b.argmax(-1)
        result.update(
            top1_agreement=float((ai == bi).float().mean()), left_top1=ai.tolist(), right_top1=bi.tolist(),
            left_top1_rank_in_right=(1 + (b > b.gather(1, ai[:, None])).sum(-1)).tolist(),
            right_top1_rank_in_left=(1 + (a > a.gather(1, bi[:, None])).sum(-1)).tolist(),
            left_top1_margin=(a.topk(2).values[:, 0] - a.topk(2).values[:, 1]).tolist(),
            right_top1_margin=(b.topk(2).values[:, 0] - b.topk(2).values[:, 1]).tolist())
    return result


class Observations:
    def __init__(self, model, dflash, n):
        self.model, self.dflash, self.n = model, dflash, n
        self.data, self.handles = {}, []
        self.layer = None

    def put(self, name, value):
        if name in self.data:
            raise ValueError(f"duplicate reference stage: {name}")
        self.data[name] = value.detach().to(device="cpu", copy=True).squeeze(0)

    def hook(self, module, name, select=lambda out: out):
        def cb(_m, _args, output):
            self.put(name, select(output))
        self.handles.append(module.register_forward_hook(cb))

    def __enter__(self):
        self.hook(self.model.fc, "context_fc")
        self.hook(self.model.hidden_norm, "context_norm")
        self.hook(self.model, "final_hidden")
        for i, layer in enumerate(self.model.layers):
            p = f"layers.{i}."
            def active(_m, _args, _kwargs, index=i):
                self.layer = index
            self.handles.append(layer.register_forward_pre_hook(active, with_kwargs=True))
            self.hook(layer.input_layernorm, p + "query_input_norm")
            self.hook(layer.self_attn.q_norm, p + "query_q_norm")
            self.hook(layer.self_attn.k_norm, p + "context_k_norm", lambda x: x[:, :self.n])
            self.hook(layer.self_attn.k_norm, p + "query_k_norm", lambda x: x[:, self.n:])
            for part in ("k", "v"):
                def kv(_m, _args, output, prefix=p, part=part):
                    key = prefix + ("context_k_raw" if part == "k" else "context_v")
                    if key not in self.data:
                        self.put(key, output)
                    # Second call is query K/V, not another context observation.
                self.handles.append(getattr(layer.self_attn, part + "_proj").register_forward_hook(kv))
            def attention(_m, args, prefix=p):
                self.put(prefix + "query_attention", args[0])
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(attention))
            self.hook(layer.self_attn.o_proj, p + "query_o_proj")
            self.hook(layer.post_attention_layernorm, p + "query_post_norm")
            self.hook(layer.mlp, p + "query_mlp")
            self.hook(layer, p + "query_output")
        self.rotary = self.dflash.apply_rotary_pos_emb
        def rotary(*args, **kwargs):
            result = self.rotary(*args, **kwargs)
            q, k = result
            p = f"layers.{self.layer}."
            self.put(p + "query_q_rope", q.transpose(1, 2))
            self.put(p + "query_k_rope", k[:, :, self.n:].transpose(1, 2))
            self.put(p + "context_k_rope", k[:, :, :self.n].transpose(1, 2))
            return result
        self.dflash.apply_rotary_pos_emb = rotary
        return self

    def __exit__(self, *exc):
        self.dflash.apply_rotary_pos_emb = self.rotary
        for handle in self.handles:
            handle.remove()


def load_weights(model, draft, target, meta, device):
    import torch
    from safetensors import safe_open
    expected = set(model.state_dict())
    with safe_open(str(draft / "model.safetensors"), framework="pt", device="cpu") as f:
        if set(f.keys()) != expected or expected != set(meta["weights"]):
            raise ValueError({"checkpoint_missing": sorted(expected - set(f.keys())),
                              "checkpoint_unexpected": sorted(set(f.keys()) - expected),
                              "native_missing": sorted(expected - set(meta["weights"])),
                              "native_unexpected": sorted(set(meta["weights"]) - expected)})
        state = {}
        for key in sorted(expected):
            # The unused confidence projection is FP32 in native vLLM, BF16 on disk.
            dtype = torch.float32 if key.startswith("confidence_head.") else torch.bfloat16
            value = f.get_tensor(key).to(dtype)
            if tensor_identity(value) != meta["weights"][key]:
                raise ValueError(f"effective native/HF weight differs: {key}")
            state[key] = value.to(device)
    result = model.load_state_dict(state, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(str(result))
    index = json.loads((target / "model.safetensors.index.json").read_text())["weight_map"]
    head_key = "lm_head.weight"
    with safe_open(str(target / index[head_key]), framework="pt", device="cpu") as f:
        head = f.get_tensor(head_key).to(torch.float16)
    if tensor_identity(head) != meta["shared_lm_head"]:
        raise ValueError("shared target lm_head differs from the exact captured native head")
    return head.to(device)


def normalize_native(data):
    import torch
    result = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
    for i in range(5):
        p = f"layers.{i}."
        for name in ("context_k_raw", "context_k_norm", "context_k_rope", "context_v"):
            result[p + name] = data[name][i]
        # Native layer defers the residual add to the next fused norm. This
        # diagnostic CPU sum materializes the corresponding HF layer boundary;
        # both original addends are retained, and the norm outputs are compared.
        result[p + "query_output"] = data[p + "query_branch"] + data[p + "query_residual"]
    for i in range(K):
        result[f"corrected_logits.{i}"] = result[f"corrected_logits.{i}"].squeeze(0)
        result[f"markov_bias.{i}"] = result[f"markov_bias.{i}"].squeeze(0)
    result["proposed_ids"] = result["proposed_ids"].reshape(K)
    return result


def replay(model, dflash, inputs, data, head, n):
    import torch
    with Observations(model, dflash, n) as obs:
        # This is the official public forward, not a reimplementation. It
        # returns a tensor despite the source's CausalLMOutputWithPast annotation.
        hidden = model(position_ids=inputs["positions"].clone(), attention_mask=None,
                       noise_embedding=inputs["query"].clone(), target_hidden=inputs["target"].clone(),
                       past_key_values=None, use_cache=False, is_causal=False)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"unexpected official forward return: {type(hidden)}")
    base = torch.nn.functional.linear(hidden.to(head.dtype), head).squeeze(0)
    obs.data["base_logits"] = base.cpu()
    prev = data["query_ids"][:1].to(head.device).long()
    proposals = []
    for i in range(K):
        # Use the OFFICIAL Markov head and sample helper. No top-k/temperature.
        bias = model.markov_head.compute_step_bias(prev)
        corrected = base[i:i + 1] + bias
        proposed = dflash.sample(corrected, temperature=0.0)
        obs.data[f"markov_bias.{i}"] = bias[0].cpu()
        obs.data[f"corrected_logits.{i}"] = corrected[0].cpu()
        # Also hold the native prefix fixed to separate Markov arithmetic drift
        # from downstream divergence caused by an earlier different argmax.
        native_prev = data[f"markov_prev.{i}"].to(head.device).long()
        native_bias = model.markov_head.compute_step_bias(native_prev)
        obs.data[f"native_prefix_corrected.{i}"] = (base[i:i + 1] + native_bias)[0].cpu()
        proposals.append(proposed)
        prev = proposed
    obs.data["proposed_ids"] = torch.cat(proposals).cpu()
    return obs.data


def compare(native, ref):
    stages = ["context_fc", "context_norm"]
    for i in range(5):
        stages += [f"layers.{i}.{name}" for name in (
            "context_k_raw", "context_v", "context_k_norm", "context_k_rope", "query_input_norm",
            "query_q_norm", "query_k_norm", "query_q_rope", "query_k_rope", "query_attention",
            "query_o_proj", "query_post_norm", "query_mlp", "query_output")]
    stages += ["final_hidden", "base_logits"]
    stages += [key for i in range(K) for key in (f"markov_bias.{i}", f"corrected_logits.{i}")]
    report = {}
    for key in stages:
        a, b = native[key], ref[key]
        # Canonical token-major layout; native Q/K may flatten head dimensions.
        if a.numel() != b.numel():
            raise ValueError(f"stage size differs: {key}")
        report[key] = metrics(a.reshape(b.shape), b, logits="logits" in key)
    return {"stages": report,
            "first_nonexact_stage": next((key for key in stages if not report[key]["exact"]), None),
            "first_top1_difference": next((key for key in stages if report[key].get("top1_agreement", 1) < 1), None),
            "proposed_ids": {"left": native["proposed_ids"].tolist(), "right": ref["proposed_ids"].tolist(),
                             "equal": native["proposed_ids"].tolist() == ref["proposed_ids"].tolist()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--fetch-source", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    parser.add_argument("--approve-gpu-replay", action="store_true")
    args = parser.parse_args()
    # Offline is enforced independently of CLI usage or model caching state.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    dflash, dspark = official(args.official_source, args.fetch_source)
    import torch
    import transformers
    environment = {"torch": torch.__version__, "transformers": transformers.__version__,
                   "official_source_sha256": OFFICIAL_SHA, "device": args.device,
                   "attention_backend": "official HF eager", "argv": sys.argv,
                   "namespace_accommodation": "unmodified official files under specforge/modeling/draft"}
    if args.check_imports:
        print(json.dumps(environment, indent=2))
        return 0
    if not all((args.capture, args.draft, args.target, args.out)):
        parser.error("replay requires --capture --draft --target --out")
    if args.device == "xpu" and not args.approve_gpu_replay:
        parser.error("GPU replay requires lead approval: --approve-gpu-replay")
    args.out.mkdir(parents=False, exist_ok=False)
    try:
        meta = json.loads((args.capture / "capture.json").read_text())
        if not meta["complete"] or meta["errors"] or meta["num_computed_tokens"] != 0:
            raise ValueError("capture is not a complete first zero-cache proposal")
        data = torch.load(args.capture / "capture.pt", map_location="cpu", weights_only=True)
        n = len(data["target_input_ids"])
        check_layout(data, n)
        if sha(args.draft / "config.json") != CONFIG_SHA:
            raise ValueError("draft config differs from pinned official checkpoint")
        config = dspark.DSparkConfig(**json.loads((args.draft / "config.json").read_text()))
        config._attn_implementation = "eager"
        if config.dflash_config["target_layer_ids"] != TAPS:
            raise ValueError("official aux order differs")
        # Construct on CPU under BF16; no full target or random GPU model exists.
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = dspark.DSparkDraftModel(config)
        finally:
            torch.set_default_dtype(old_dtype)
        model.eval()
        head = load_weights(model, args.draft, args.target, meta, args.device)
        model.to(args.device)
        with torch.inference_mode():
            target = torch.cat([data[f"aux.{i}"] for i in range(5)], dim=-1).to(torch.bfloat16)
            if not torch.equal(target, data["fc_input"]):
                raise ValueError("captured fc input is not ordered aux concat cast once to BF16")
            if not torch.equal(data["shared_query_embedding_fp16"].to(torch.bfloat16), data["query_embedding"]):
                raise ValueError("captured query embeddings differ from shared target boundary")
            inputs = {"target": target.unsqueeze(0).to(args.device),
                      "query": data["query_embedding"].unsqueeze(0).to(args.device),
                      "positions": torch.cat((data["context_positions"], data["query_positions"])).unsqueeze(0).to(args.device)}
            one = replay(model, dflash, inputs, data, head, n)
            two = replay(model, dflash, inputs, data, head, n)  # real second forward, fresh cache=None
        native = normalize_native(data)
        torch.save(one, args.out / "official-first.pt")
        torch.save(two, args.out / "official-repeat.pt")
        report = {"status": "measured-not-threshold-certified", "tier": "development", "environment": environment,
                  "weight_mapping": "all 62 keys + exact effective shared head checked; no missing/unexpected",
                  "native_vs_official": compare(native, one), "official_self_repeatability": compare(one, two),
                  "native_prefix_corrected": {str(i): metrics(native[f"corrected_logits.{i}"], one[f"native_prefix_corrected.{i}"], True) for i in range(K)},
                  "target_aux_registration": meta["target_aux_registration"],
                  "native_query_output_boundary": "CPU BF16 branch + residual diagnostic sum; raw addends retained",
                  "tolerance": "none imposed; report raw error, exact first difference, repeatability and decision margins"}
        (args.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "first_nonexact_stage": report["native_vs_official"]["first_nonexact_stage"],
                          "proposed_ids": report["native_vs_official"]["proposed_ids"]}, indent=2))
    except Exception:
        (args.out / "failure.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
