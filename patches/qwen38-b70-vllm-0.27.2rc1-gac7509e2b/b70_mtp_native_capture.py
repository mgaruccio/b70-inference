"""Opt-in, synchronous CPU feature copies from the *native* quantized verifier.

Pinned seam/source (research supplied by the lead, checked against the image):
https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/worker/gpu_model_runner.py
https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/step3p5.py

After rejection sampling, BEFORE the proposer reuses input/position buffers,
verify inputs at L..L+n with m valid sampled outputs contribute hidden rows
L..L+m-1. Outputs occupy L+1..L+m. Even on full acceptance the bonus token's
hidden is NOT available until the next forward. Prefill contributes every prompt
row, including chunks whose sampler output is discarded. No replay/teacher model.

Private control + per-step candidate files are the only capture protocol. The
client finalizes after the nonstreaming API returns authoritative token IDs;
runner request state/async CPU token placeholders are never final authority.
Capture timings are NOT performance measurements. Supported: text, C1, native
MTP4, TP=PP=DP=1, cold complete prefixes (unique API cache_salt). Prefix caching
and async scheduling may stay enabled; hits, gaps and preemption fail closed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat

import torch

HIDDEN_SIZE = 5120
VOCAB_SIZE = 248320
MAX_SEQUENCE_TOKENS = 32768
MAX_REQUESTS = 4096
MAX_TOTAL_TOKENS = 2097152
KEY_PATTERN = r"b70-native-[0-9a-f]{32}"


def private_directory(path, *, create=False):
    """No symlinks, Git artifacts, shared directories, or implicit chmod repairs."""
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink() or (ancestor / ".git").exists():
            raise ValueError("private capture paths must be nonsymlink paths outside Git")
    if create:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("private capture directory requires mode 0700")
    if info.st_uid != os.getuid() and os.geteuid() != 0:
        raise ValueError("private capture directory must belong to this user")
    return path


def _private_open(path, mode):
    parent = private_directory(Path(path).parent)
    flags = os.O_NOFOLLOW | (os.O_WRONLY | os.O_CREAT | os.O_EXCL if mode == "wb" else os.O_RDONLY)
    fd = os.open(path, flags, 0o600)
    try:
        # Docker's root worker writes into the host user's private bind mount.
        # Preserve that owner so the host client can read mode-0600 candidates.
        owner = parent.stat()
        if mode == "wb" and os.geteuid() == 0:
            os.fchown(fd, owner.st_uid, owner.st_gid)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != owner.st_uid or info.st_nlink != 1):
            raise ValueError("private capture file requires an owned, unlinked mode-0600 regular file")
        return os.fdopen(fd, mode)
    except BaseException:
        os.close(fd)
        raise


def save_json(path, data):
    with _private_open(path, "wb") as handle:
        handle.write((json.dumps(data, ensure_ascii=True, indent=2) + "\n").encode())


def load_json(path, max_bytes=8 * 1024 * 1024):
    with _private_open(path, "rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("capture JSON size bound exceeded")
    return json.loads(data)


def save_tensor(path, data):
    # Only publish a fully serialized file; link fails rather than overwriting.
    partial = Path(str(path) + ".partial")
    with _private_open(partial, "wb") as handle:
        torch.save(data, handle)
    os.link(partial, path, follow_symlinks=False)
    partial.unlink()


def token_ids(ids, *, empty=False):
    if (not isinstance(ids, list) or (not empty and not ids)
            or any(type(t) is not int or not 0 <= t < VOCAB_SIZE for t in ids)):
        raise ValueError("expected exact pinned-vocabulary token IDs")
    return ids


def validate_control(data):
    if set(data) != {"request_id", "prompt_token_ids", "max_tokens"}:
        raise ValueError("invalid native capture control fields")
    if not isinstance(data["request_id"], str) or not re.fullmatch(KEY_PATTERN, data["request_id"]):
        raise ValueError("invalid native capture request ID")
    prompt = token_ids(data["prompt_token_ids"])
    if (type(data["max_tokens"]) is not int or data["max_tokens"] < 1
            or not 3 <= len(prompt) + data["max_tokens"] <= MAX_SEQUENCE_TOKENS):
        raise ValueError("native capture per-request token bound exceeded")
    return data


def _validate_runner(runner):
    if (runner.speculative_config is None or runner.speculative_config.method != "mtp"
            or runner.speculative_config.num_speculative_tokens != 4):
        raise ValueError("native capture requires native MTP4")
    if runner.scheduler_config.max_num_seqs != 1:
        raise ValueError("native capture requires C1")
    parallel = runner.parallel_config
    if any(getattr(parallel, k) != 1 for k in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"
    )) or parallel.use_ubatching:
        raise ValueError("native capture requires TP=PP=DP=1 without microbatching")
    if (runner.cache_config.kv_sharing_fast_prefill
            or runner.vllm_config.kv_transfer_config is not None
            or runner.vllm_config.ec_transfer_config is not None):
        raise ValueError("native capture requires ordinary complete text prefill")
    model = runner.get_model()
    if (runner.use_aux_hidden_state_outputs or runner.is_pooling_model
            or runner.model_config.hf_text_config.hidden_size != HIDDEN_SIZE
            or type(model).__name__ not in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
            or hasattr(model, "get_mtp_target_hidden_states")):
        raise ValueError("native capture requires the pinned Qwen post-final-norm hidden source")


def _cpu(tensor):
    return tensor.detach().to(device="cpu", copy=True)


def _sample_ids(sampled):
    if (not isinstance(sampled, torch.Tensor) or sampled.ndim != 2
            or sampled.shape[0] != 1 or not 1 <= sampled.shape[1] <= 5
            or sampled.dtype not in (torch.int32, torch.int64)):
        raise ValueError("invalid current GPU sampler output")
    values = _cpu(sampled[0]).tolist()
    # Match rejection_sampler.parse_output's vocabulary/placeholder filter,
    # but additionally require a valid contiguous prefix, never compress a gap.
    valid = [v != -1 and 0 <= v < VOCAB_SIZE for v in values]
    count = sum(valid)
    if valid != [True] * count + [False] * (len(valid) - count):
        raise ValueError("non-contiguous sampled output")
    return values[:count]


class NativeCapture:
    def __init__(self, root):
        self.root = private_directory(root)
        self.max_total = int(os.environ.get("B70_MTP_NATIVE_MAX_TOKENS", "131072"))
        self.max_requests = int(os.environ.get("B70_MTP_NATIVE_MAX_REQUESTS", "1024"))
        if not 1 <= self.max_total <= MAX_TOTAL_TOKENS or not 1 <= self.max_requests <= MAX_REQUESTS:
            raise ValueError("invalid native capture run bounds")
        self.reserved = 0
        self.seen = set()
        self.active = None

    def step(self, runner, scheduler, spec_metadata, hidden_states, sampler_output):
        _validate_runner(runner)
        batch = runner.input_batch
        if len(batch.req_ids) != 1 or batch.num_reqs != 1:
            raise ValueError("native capture requires exactly one active request")
        req_id = batch.req_ids[0]
        match = re.fullmatch(r"chatcmpl-(" + KEY_PATTERN + r")(?:-[0-9a-f]{8})?", req_id)
        if not match:
            raise ValueError("native capture requires a client-controlled request ID")
        key = match[1]
        directory = private_directory(self.root / key)
        control = validate_control(load_json(directory / "control.json"))
        if control["request_id"] != key:
            raise ValueError("native capture control/request mismatch")
        if (scheduler.scheduled_cached_reqs.resumed_req_ids
                or getattr(scheduler, "preempted_req_ids", None)):
            raise ValueError("native capture rejects resumed/preempted requests")
        request = runner.requests[req_id]
        if (request.mm_features or request.prompt_embeds is not None or request.lora_request is not None
                or (request.prompt_is_token_ids is not None and not all(request.prompt_is_token_ids))):
            raise ValueError("native capture rejects multimodal, LoRA, or embedding input")
        if (request.prompt_token_ids != control["prompt_token_ids"] or request.sampling_params is None
                or request.sampling_params.max_tokens != control["max_tokens"]):
            raise ValueError("native capture requires the exact server-rendered prompt and length bound")
        if self.active is None or self.active["req_id"] != req_id:
            reserve = len(control["prompt_token_ids"]) + control["max_tokens"] + 4
            if key in self.seen or len(self.seen) >= self.max_requests or self.reserved + reserve > self.max_total:
                raise ValueError("native capture run budget or repeated-request violation")
            self.seen.add(key)
            self.reserved += reserve
            self.active = dict(req_id=req_id, control=control, rows=0, steps=0, dtype=None)
        active = self.active
        if active["control"] != control:
            raise ValueError("native capture control changed mid-request")
        n = scheduler.num_scheduled_tokens.get(req_id, 0)
        if n < 1 or scheduler.num_scheduled_tokens != {req_id: n} or scheduler.total_num_scheduled_tokens != n:
            raise ValueError("invalid native capture scheduled rows")
        # Read ACTUAL GPU positions, not possibly deferred async CPU metadata.
        positions = _cpu(runner._get_positions(n))
        if positions.ndim == 2 and positions.shape[0] == 3:
            if not torch.equal(positions, positions[0:1].expand_as(positions)):
                raise ValueError("native capture rejects non-text rotary positions")
            positions = positions[0].clone()
        start = active["rows"]
        if (positions.dtype != torch.int64 or positions.ndim != 1
                or not torch.equal(positions, torch.arange(start, start + n))):
            raise ValueError("native capture prefix-cache hit, gap, or reordered positions")
        limit = len(control["prompt_token_ids"]) + control["max_tokens"] + 4
        if start + n > limit or active["steps"] >= limit:
            raise ValueError("native capture per-request row bound exceeded")
        drafts = scheduler.scheduled_spec_decode_tokens.get(req_id, [])
        d = len(drafts)
        if d > 4 or (spec_metadata is None and d) or (
            spec_metadata is not None and list(spec_metadata.num_draft_tokens) != [d]
        ):
            raise ValueError("native capture speculative metadata mismatch")
        outputs = _sample_ids(sampler_output.sampled_token_ids)
        prompt_len = len(control["prompt_token_ids"])
        prefill = start < prompt_len
        if prefill:
            if d or start + n > prompt_len:
                raise ValueError("native capture mixed prompt/draft rows unsupported")
            keep = n
            if start + n < prompt_len:
                outputs = []  # Chunked-prefill samples are discarded by vLLM.
            elif len(outputs) != 1:
                raise ValueError("native capture final prefill requires one sampled token")
            output_start = start + n
        else:
            if n != d + 1 or not 1 <= len(outputs) <= n:
                raise ValueError("native capture invalid accepted prefix")
            keep = len(outputs)
            output_start = start + 1
        if (not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2
                or hidden_states.shape[0] < n or hidden_states.shape[1] != HIDDEN_SIZE
                or hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
            raise ValueError("native capture invalid final hidden tensor")
        ids = _cpu(runner.input_ids.gpu[:keep])
        if ids.ndim != 1 or len(ids) != keep or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("native capture invalid runtime IDs")
        token_ids(ids.tolist())
        if prefill and ids.tolist() != control["prompt_token_ids"][start:start + keep]:
            raise ValueError("native capture runtime prompt mismatch")
        if not prefill and ids[1:].tolist() != outputs[:-1]:
            raise ValueError("native capture accepted draft input/output mismatch")
        hidden = _cpu(hidden_states[:keep])  # Independent storage before proposer/graph reuse.
        if not bool(torch.isfinite(hidden).all()) or active["dtype"] not in (None, hidden.dtype):
            raise ValueError("native capture nonfinite or changing hidden dtype")
        payload = dict(
            request_id=key, step=active["steps"], prefill=prefill, num_drafts=d,
            input_ids=ids.to(torch.int64), positions=positions[:keep].clone(),
            target_last_hidden_states=hidden,
            output_ids=torch.tensor(outputs, dtype=torch.int64),
            output_positions=torch.arange(output_start, output_start + len(outputs)),
        )
        save_tensor(directory / f"step-{active['steps']:06d}.pt", payload)
        active.update(rows=start + keep, steps=active["steps"] + 1, dtype=hidden.dtype)


def finalize_request(directory, prompt_ids, generated_ids):
    """Validate candidates against API IDs; return a trainer record, never pad hidden.

    T IDs/mask rows, H observed hidden/position rows: T-2 <= H <= T.
    MTP continuation consumes h[t] with x[t+1] and predicts x[t+2].
    """
    directory = private_directory(directory)
    control = validate_control(load_json(directory / "control.json"))
    token_ids(prompt_ids)
    token_ids(generated_ids)
    if prompt_ids != control["prompt_token_ids"] or len(generated_ids) > control["max_tokens"]:
        raise ValueError("authoritative API prompt/length mismatch")
    ids = prompt_ids + generated_ids
    if not 3 <= len(ids) <= MAX_SEQUENCE_TOKENS:
        raise ValueError("final trajectory length out of bounds")
    chunks = sorted(directory.glob("step-*.pt"))
    limit = len(prompt_ids) + control["max_tokens"] + 4
    if not chunks or len(chunks) > limit:
        raise ValueError("missing or excessive native candidates")
    inputs, positions, hidden, outputs = [], [], [], []
    rows, out_rows = 0, len(prompt_ids)
    acceptance = {"zero": 0, "partial": 0, "full": 0}
    dtype = None
    for i, path in enumerate(chunks):
        if path.name != f"step-{i:06d}.pt":
            raise ValueError("missing/reordered native capture step")
        with _private_open(path, "rb") as handle:
            if os.fstat(handle.fileno()).st_size > limit * (HIDDEN_SIZE * 4 + 64) + 65536:
                raise ValueError("native candidate file size bound exceeded")
            raw = torch.load(handle, weights_only=True, map_location="cpu")
        if raw["request_id"] != control["request_id"] or raw["step"] != i:
            raise ValueError("native candidate identity mismatch")
        x, p, h = raw["input_ids"], raw["positions"], raw["target_last_hidden_states"]
        y, q = raw["output_ids"], raw["output_positions"]
        if (x.dtype != torch.int64 or x.ndim != 1 or p.dtype != torch.int64
                or not torch.equal(p, torch.arange(rows, rows + len(x)))
                or len(x) < 1 or rows + len(x) > limit
                or h.shape != (len(x), HIDDEN_SIZE)
                or h.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or dtype not in (None, h.dtype) or not bool(torch.isfinite(h).all())
                or y.dtype != torch.int64 or y.ndim != 1 or q.dtype != torch.int64
                or not torch.equal(q, torch.arange(out_rows, out_rows + len(y)))):
            raise ValueError("native candidate shape, position, dtype, or coverage mismatch")
        token_ids(x.tolist())
        token_ids(y.tolist(), empty=True)
        d = raw["num_drafts"]
        if type(d) is not int or not 0 <= d <= 4:
            raise ValueError("invalid candidate draft count")
        if raw["prefill"] is True:
            if d or rows >= len(prompt_ids) or rows + len(x) > len(prompt_ids):
                raise ValueError("invalid candidate prefill")
            if len(y) != int(rows + len(x) == len(prompt_ids)):
                raise ValueError("invalid candidate prefill sampling")
        elif raw["prefill"] is False:
            if rows < len(prompt_ids) or not 1 <= len(x) == len(y) <= d + 1 or out_rows != rows + 1:
                raise ValueError("invalid candidate verification mapping")
            if d:
                acceptance["zero" if len(y) == 1 else "full" if len(y) == d + 1 else "partial"] += 1
        else:
            raise ValueError("invalid candidate phase")
        inputs.append(x)
        positions.append(p)
        hidden.append(h)
        outputs.append(y)
        rows += len(x)
        out_rows += len(y)
        dtype = h.dtype
        # The API can finish while async scheduling has a later, discarded
        # forward in flight. Do not wait for/read its incomplete tail files.
        # This prefix must already prove every authoritative output and all
        # training hidden rows; a missing committed step still fails below.
        if rows >= max(len(prompt_ids), len(ids) - 2) and out_rows >= len(ids):
            break
    observed = min(rows, len(ids))
    if not max(len(prompt_ids), len(ids) - 2) <= observed <= len(ids):
        raise ValueError("insufficient native final hidden coverage")
    if (torch.cat(inputs)[:observed].tolist() != ids[:observed]
            or torch.cat(outputs)[:len(generated_ids)].tolist() != generated_ids):
        raise ValueError("native features/sampler IDs do not match authoritative API trajectory")
    return dict(
        input_ids=torch.tensor(ids, dtype=torch.int64),
        positions=torch.cat(positions)[:observed].clone(),
        target_last_hidden_states=torch.cat(hidden)[:observed].clone(),
        loss_mask=torch.tensor([False] * len(prompt_ids) + [True] * len(generated_ids)),
        metadata=dict(source="native-quantized-verifier", request_id=control["request_id"],
                      prompt_length=len(prompt_ids), trajectory_length=len(ids),
                      observed_hidden_length=observed, raw_hidden_length=rows,
                      raw_output_length=out_rows - len(prompt_ids), acceptance_blocks=acceptance),
    )


def capture_native_step(runner, scheduler_output, spec_decode_metadata, hidden_states, sampler_output):
    root = os.environ.get("B70_MTP_NATIVE_CAPTURE_DIR")
    if root is None:
        return
    from vllm.forward_context import is_forward_context_available

    if torch.compiler.is_compiling() or is_forward_context_available():
        raise RuntimeError("native capture must run outside forward/compilation context")
    if os.environ.get("B70_MTP_CAPTURE_DIR"):
        raise ValueError("native capture and teacher replay capture are mutually exclusive")
    collector = getattr(runner, "_b70_mtp_native_capture", None)
    if collector is None:
        collector = NativeCapture(root)
        runner._b70_mtp_native_capture = collector
    if collector.root != Path(root).absolute():
        raise ValueError("native capture directory changed during run")
    collector.step(runner, scheduler_output, spec_decode_metadata, hidden_states, sampler_output)
