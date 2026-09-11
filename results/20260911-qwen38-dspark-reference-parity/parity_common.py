"""Shared, stdlib-importable helpers for this one-shot experiment only."""
import hashlib
from pathlib import Path

REVISION = "b9a5dbdf03bc999c6c73c426b19c2d9041cea393"
SOURCE_URL = f"https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/raw/{REVISION}/"
OFFICIAL_SHA = {
    "dflash.py": "9825996703de73bd436a6aaf57ae203ef92d599249d9cf49078576d52f4e56a4",
    "dspark.py": "75bba7c469166bd2d6a6877b9964d035b0cec9a83190ab9f3de8c6175883a114",
}
CONFIG_SHA = "dd65fb1b01c2adea69512ff2990a79d58eb7fe2c7ea97375aa66f657a29a5bfd"
OVERLAY_SHA = "3cd21f859c2977978cc8f3e9f15f65a47b80d7e206e17bf1e099aee56eaab737"
TAPS = [5, 19, 33, 47, 61]
K = 7
MAX_CONTEXT = 512
CODE = "Write a detailed tutorial on implementing a bounded LRU cache in Python using collections.OrderedDict. Include a complete class, explain get and put behavior, discuss edge cases, and include tests. Continue with a worked example and complexity analysis. Be precise and use meaningful prose and code."


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_identity(value):
    """Exact effective weight bytes, streaming rows; never copy a whole head to CPU."""
    import torch
    h = hashlib.sha256()
    value = value.detach()
    rows = max(1, (8 << 20) // (value[0].numel() * value.element_size()))
    for start in range(0, value.shape[0], rows):
        chunk = value[start:start + rows].contiguous().cpu().view(torch.uint8)
        h.update(chunk.numpy().tobytes())
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": h.hexdigest()}


def weight_parts(name, shape, config):
    """Native TP=1 fused weights -> exact official checkpoint keys/row spans."""
    if ".qkv_proj." in name:
        q = config["num_attention_heads"] * config["head_dim"]
        kv = config["num_key_value_heads"] * config["head_dim"]
        if shape[0] != q + 2 * kv:
            raise ValueError(f"unexpected fused QKV shape: {name} {shape}")
        return [(name.replace(".qkv_proj.", f".{part}_proj."), a, b)
                for part, a, b in (("q", 0, q), ("k", q, q + kv), ("v", q + kv, q + 2 * kv))]
    if ".gate_up_proj." in name:
        size = config["intermediate_size"]
        if shape[0] != 2 * size:
            raise ValueError(f"unexpected fused MLP shape: {name} {shape}")
        return [(name.replace(".gate_up_proj.", ".gate_proj."), 0, size),
                (name.replace(".gate_up_proj.", ".up_proj."), size, 2 * size)]
    return [(name, 0, shape[0])]


def first_request_matches(batch, armed, *, dummy=False, profile=False):
    """Check cheap CPU fields before reading any device data. No chunk/cache replay."""
    ids = armed.get("prompt_token_ids", [])
    if dummy or profile or batch.num_reqs != 1 or not 0 < len(ids) <= MAX_CONTEXT:
        return False
    if (batch.num_tokens != len(ids) or batch.num_draft_tokens != 0
            or int(batch.num_computed_tokens_np[0]) != 0
            or int(batch.num_computed_prefill_tokens_np[0]) != 0
            or int(batch.prefill_len_np[0]) != len(ids)):
        return False
    return batch.input_ids[:batch.num_tokens].detach().cpu().tolist() == ids


def check_layout(tensors, n):
    """A logically empty cache: every visible context/query position is written now."""
    if tensors["context_positions"].tolist() != list(range(n)):
        raise ValueError("context is not the complete, zero-cache prefill")
    if tensors["query_positions"].tolist() != list(range(n, n + K)):
        raise ValueError("query positions are not the absolute anchor + six noise slots")
    if tensors["sample_positions"].tolist() != list(range(n + 1, n + K + 1)):
        raise ValueError("sample positions are not query positions + 1")
    if tensors["sample_indices"].tolist() != list(range(K)):
        raise ValueError("not the sample-from-anchor layout")
    if tensors["query_ids"].tolist()[1:] != [248070] * (K - 1):
        raise ValueError("unexpected noise token IDs")
    if tensors["draft_seq_lens"].tolist() != [n + K]:
        raise ValueError("attention exposes a stale/missing KV prefix")
    for group in tensors["context_slots"]:
        if len(group) != n or min(group) < 0 or len(set(group)) != n:
            raise ValueError("context slots are missing, evicted or aliased")
    for group in tensors["query_slots"]:
        if len(group) != K or min(group) < 0 or len(set(group)) != K:
            raise ValueError("query slots are missing or aliased")
    if len(tensors["context_slots"]) != len(tensors["query_slots"]):
        raise ValueError("cache group count differs")
    for context, query in zip(tensors["context_slots"], tensors["query_slots"]):
        if set(context) & set(query):
            raise ValueError("context/query cache slots overlap")
