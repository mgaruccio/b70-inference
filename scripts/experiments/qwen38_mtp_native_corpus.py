#!/usr/bin/env python3
"""Private native-MTP corpus client; no generated tool is ever executed.

Run with the pinned ML runtime, not the interactive Pi Python environment.
`prepare` writes (never starts) a temporary reference launcher. Explicitly choose
--capture or --no-capture; both keep native MTP4, prefix caching and thinking.
`generate` calls /tokenize then /v1/chat/completions (nonstreaming) with unique
cache_salt and return_token_ids=true. All reasoning/tool token IDs count as
response, irrespective of the parsed message. No local rendering/retokenization.

Input JSONL: OpenAI messages, optional tools/tool_choice=auto, sampling options,
chat_template_kwargs (including enable_thinking), and split=train|heldout. Only
text content is supported. Each record is one continuation, not a tool agent.
All output directories must be NEW, outside Git, mode 0700; files are 0600.

Lead E2E: prepare matched on/off launchers on the idle pinned B70; start one at a
time using the existing host procedure; run generate --synthetic against the
loopback HTTP API, compare authoritative token IDs, inspect coverage/acceptance
counts (zero/partial/full and EOS), then train and run capture-OFF stock/tuned
ABBA. Keep these private outputs and exact launch configs; stop the disposable
server between cells, verify the persistent launcher hash, remove only explicitly
selected private run directories after use. CPU protocol tests are NOT that E2E.
No throughput claim may use capture-on timings.

Pinned API source verified in the image:
https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/entrypoints/openai/chat_completion/serving.py
https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/entrypoints/serve/tokenize/protocol.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import urllib.parse
import urllib.request
import uuid

PATCHES = Path(__file__).resolve().parents[2] / "patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b"
SAMPLING = {"temperature", "seed", "top_p", "top_k", "min_p", "presence_penalty",
            "frequency_penalty", "repetition_penalty", "stop", "stop_token_ids", "ignore_eos"}
CHAT = {"messages", "tools", "chat_template_kwargs", "add_generation_prompt",
        "continue_final_message", "add_special_tokens"}


def runtime():
    spec = importlib.util.spec_from_file_location("b70_mtp_native_capture", PATCHES / "b70_mtp_native_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("temporary native launcher anchor mismatch")
    return text.replace(old, new, 1)


def launcher_text(original, out, capture, max_tokens, max_requests):
    import qwen38_mtp_reference as reference

    text = reference.launcher_text(original, out)  # Includes the persistent SHA guard.
    text = replace_once(text, "--no-enable-prefix-caching", "--enable-prefix-caching")
    text = replace_once(text, r'enable_thinking\":false', r'enable_thinking\":true')
    mounts = " -v " + shlex.quote(str(out / "native-patches") + ":/mtp-native:ro")
    if capture:
        mounts += (" -e B70_MTP_NATIVE_CAPTURE_DIR=/profile/features"
                   f" -e B70_MTP_NATIVE_MAX_TOKENS={max_tokens}"
                   f" -e B70_MTP_NATIVE_MAX_REQUESTS={max_requests}")
    marker = "exec docker run --rm --name qwen38 --ipc=host"
    text = replace_once(text, marker, marker + mounts)
    text = replace_once(text, "exec vllm serve /model ",
                        "python /mtp-native/patch_mtp_native_capture.py; exec vllm serve /model ")
    # A temporary launcher and any shell redirections remain private.
    return text.replace("\n", "\numask 077\n", 1)


def prepare(args, rt):
    import qwen38_mtp_reference as reference

    original = args.launcher.read_bytes()
    guard = args.guard.read_bytes()
    if hashlib.sha256(guard).hexdigest() != reference.dflash.probe.GUARD_SHA:
        raise ValueError("prefill guard mismatch")
    out = rt.private_directory(args.output, create=True)
    text = launcher_text(original, out, args.capture, args.max_total_tokens, args.max_requests)
    source = rt.private_directory(out / "reference-source", create=True)
    source = rt.private_directory(source / "patches", create=True)
    native = rt.private_directory(out / "native-patches", create=True)
    hashes = {}
    for name in reference.PATCHES:
        data = (args.reference_patches / name).read_bytes()
        with rt._private_open(source / name, "wb") as handle:
            handle.write(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    for name in ("patch_mtp_native_capture.py", "b70_mtp_native_capture.py"):
        data = (PATCHES / name).read_bytes()
        with rt._private_open(native / name, "wb") as handle:
            handle.write(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    for name, data in (("launcher.sh", text.encode()), ("persistent-launcher.sh", original),
                       ("patch_uniform_decode_prefill.py", guard)):
        with rt._private_open(out / name, "wb") as handle:
            handle.write(data)
    if args.capture:
        rt.private_directory(out / "features", create=True)
    rt.save_json(out / "launch-config.json", dict(
        tier="development", image=reference.IMAGE, capture=args.capture,
        speculative_tokens=4, prefix_caching=True, thinking=True,
        context=212992, draft_quantization="existing S+M1 RTN INT4",
        persistent_launcher_sha256=hashlib.sha256(original).hexdigest(),
        temporary_launcher_sha256=hashlib.sha256(text.encode()).hexdigest(),
        prefill_guard_sha256=hashlib.sha256(guard).hexdigest(), patch_sha256=hashes,
        max_total_tokens=args.max_total_tokens, max_requests=args.max_requests,
        command=["bash", str(out / "launcher.sh")],
        intentional_differences=["reference balanced mode/graph sizes/prefill guard/loopback binding",
                                 "restore production prefix cache and default thinking",
                                 "native capture enabled" if args.capture else "native capture disabled"],
    ))
    if args.launcher.read_bytes() != original:
        raise ValueError("persistent launcher changed during preparation")


def synthetic_records(max_tokens):
    return [
        {"messages": [{"role": "user", "content": "Implement a Python function that merges two sorted integer lists. Explain its complexity."}],
         "chat_template_kwargs": {"enable_thinking": True}, "max_tokens": max_tokens, "split": "train"},
        {"messages": [{"role": "user", "content": "Inspect the build status using the supplied tool. Do not run a build."}],
         "tools": [{"type": "function", "function": {"name": "build_status", "description": "Read build status",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}],
         "tool_choice": "auto", "chat_template_kwargs": {"enable_thinking": False},
         "max_tokens": max_tokens, "split": "heldout"},
        {"messages": [{"role": "user", "content": "Reply with just OK and stop."}],
         "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": max_tokens, "split": "heldout"},
        {"messages": [{"role": "user", "content": "Write a small unit test for integer addition."}],
         "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 1, "split": "train"},
    ]


def validate_record(record, default_max_tokens):
    if not isinstance(record, dict) or set(record) - CHAT - SAMPLING - {"tool_choice", "max_tokens", "split"}:
        raise ValueError("unsupported corpus record fields")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("corpus record requires OpenAI messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError("invalid OpenAI message")
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(c, dict) or set(c) != {"type", "text"}
                   or c["type"] != "text" or not isinstance(c["text"], str) for c in content):
                raise ValueError("multimodal corpus input is unsupported")
        elif content is not None and not isinstance(content, str):
            raise ValueError("invalid text content")
        if set(message) - {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning", "reasoning_content"}:
            raise ValueError("unsupported message fields")
    if record.get("tool_choice", "auto") != "auto":
        raise ValueError("only automatic tool selection preserves the shared tokenize/chat contract")
    split = record.get("split", "train")
    maximum = record.get("max_tokens", default_max_tokens)
    if split not in ("train", "heldout") or type(maximum) is not int or not 1 <= maximum <= 32768:
        raise ValueError("invalid split or generation bound")
    body = {k: v for k, v in record.items() if k in CHAT | SAMPLING | {"tool_choice"}}
    body.setdefault("temperature", 0)
    body.setdefault("seed", 42)
    body.setdefault("chat_template_kwargs", {"enable_thinking": True})
    body["max_tokens"] = maximum
    return split, body


def input_records(args, rt):
    if args.synthetic:
        yield from synthetic_records(args.max_tokens)
        return
    # Deliberate explicit path only. Never discover or read private sessions.
    with rt._private_open(args.records, "rb") as handle:
        for _ in range(args.max_requests):
            line = handle.readline(8 * 1024 * 1024 + 1)
            if not line:
                return
            if len(line) > 8 * 1024 * 1024:
                raise ValueError("private record size bound exceeded")
            yield json.loads(line)
        if handle.read(1):
            raise ValueError("private corpus request bound exceeded")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("corpus API redirects are forbidden")


def post(base_url, path, body, timeout):
    parsed = urllib.parse.urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment):
        raise ValueError("private corpus requires a loopback HTTP API (use an SSH tunnel if remote)")
    request = urllib.request.Request(base_url.rstrip("/") + path,
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        data = response.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024:
        raise ValueError("corpus API response bound exceeded")
    return json.loads(data)


def generate(args, rt):
    config = rt.load_json(args.server_config)
    if (config.get("capture") is not bool(args.capture_dir) or config.get("speculative_tokens") != 4
            or config.get("prefix_caching") is not True):
        raise ValueError("server launch config does not match native corpus mode")
    out = rt.private_directory(args.output, create=True)
    if args.capture_dir:
        capture = rt.private_directory(args.capture_dir)
    rt.save_json(out / "config.json", dict(
        tier="development", server=config, base_url=args.base_url, model=args.model,
        input_kind="synthetic" if args.synthetic else "explicit-private-jsonl",
        input_path=str(args.records) if args.records else None, max_requests=args.max_requests,
        max_total_tokens=args.max_total_tokens, default_max_tokens=args.max_tokens,
        timeout=args.timeout, capture_directory=str(args.capture_dir) if args.capture_dir else None,
        argv=sys.argv, semantics="native on-policy; API IDs; all generated reasoning/tool tokens are response",
    ))
    for split in ("train", "heldout"):
        rt.private_directory(out / split, create=True)
    counts = dict(requests=0, prompt_tokens=0, generated_tokens=0, observed_hidden_rows=0,
                  useful_positions=0, acceptance_blocks=dict(zero=0, partial=0, full=0), finish_reasons={})
    reserved = 0
    try:
        for index, record in enumerate(input_records(args, rt)):
            if index >= min(args.max_requests, config["max_requests"]):
                raise ValueError("native corpus request bound exceeded")
            split, body = validate_record(record, args.max_tokens)
            key = "b70-native-" + uuid.uuid4().hex
            body.update(model=args.model, stream=False, n=1, request_id=key,
                        cache_salt=uuid.uuid4().hex, return_token_ids=True)
            request_dir = rt.private_directory(out / f"request-{index:06d}", create=True)
            rt.save_json(request_dir / "request.json", body)
            rendered = post(args.base_url, "/tokenize", {k: v for k, v in body.items() if k in CHAT | {"model"}}, args.timeout)
            rt.save_json(request_dir / "tokenize.json", rendered)
            prompt = rt.token_ids(rendered["tokens"])
            control = rt.validate_control(dict(request_id=key, prompt_token_ids=prompt, max_tokens=body["max_tokens"]))
            reserved += len(prompt) + body["max_tokens"] + 4
            if reserved > min(args.max_total_tokens, config["max_total_tokens"]):
                raise ValueError("native corpus token budget exceeded")
            if args.capture_dir:
                directory = rt.private_directory(capture / key, create=True)
                rt.save_json(directory / "control.json", control)
            response = post(args.base_url, "/v1/chat/completions", body, args.timeout)
            rt.save_json(request_dir / "response.json", response)
            if (response.get("id") != "chatcmpl-" + key or len(response["choices"]) != 1
                    or response.get("prompt_token_ids") != prompt):
                raise ValueError("API response identity or exact rendered prompt mismatch")
            choice = response["choices"][0]
            generated = rt.token_ids(choice["token_ids"])
            if (choice.get("finish_reason") not in ("length", "stop", "tool_calls")
                    or len(generated) > body["max_tokens"]
                    or response["usage"]["prompt_tokens"] != len(prompt)
                    or response["usage"]["completion_tokens"] != len(generated)):
                raise ValueError("API generation completion/usage mismatch")
            if args.capture_dir:
                payload = rt.finalize_request(directory, prompt, generated)
                payload["prompt_id"] = f"native-{index:06d}"
                rt.save_tensor(out / split / f"native-{index:06d}.pt", payload)
                metadata = payload["metadata"]
                counts["observed_hidden_rows"] += metadata["observed_hidden_length"]
                counts["useful_positions"] += int(payload["loss_mask"][2:].sum())
                for name, value in metadata["acceptance_blocks"].items():
                    counts["acceptance_blocks"][name] += value
            counts["requests"] += 1
            counts["prompt_tokens"] += len(prompt)
            counts["generated_tokens"] += len(generated)
            reason = choice["finish_reason"]
            counts["finish_reasons"][reason] = counts["finish_reasons"].get(reason, 0) + 1
        if not counts["requests"]:
            raise ValueError("empty native corpus")
    except Exception as exc:
        rt.save_json(out / "failure.json", dict(error_type=type(exc).__name__, counts=counts))
        raise
    rt.save_json(out / "summary.json", dict(status="completed", capture=bool(args.capture_dir), **counts))
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="write a temporary launcher only; never start a server")
    prep.add_argument("--launcher", type=Path, required=True)
    prep.add_argument("--guard", type=Path, required=True)
    prep.add_argument("--reference-patches", type=Path, required=True)
    capture = prep.add_mutually_exclusive_group(required=True)
    capture.add_argument("--capture", dest="capture", action="store_true")
    capture.add_argument("--no-capture", dest="capture", action="store_false")
    gen = sub.add_parser("generate", help="call the existing loopback API; retain private IDs/features")
    gen.add_argument("--server-config", type=Path, required=True)
    gen.add_argument("--base-url", default="http://127.0.0.1:8000")
    gen.add_argument("--model", default="qwen38")
    gen.add_argument("--timeout", type=float, default=1800)
    gen.add_argument("--max-tokens", type=int, default=128)
    inputs = gen.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--synthetic", action="store_true")
    inputs.add_argument("--records", type=Path)
    mode = gen.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-dir", type=Path)
    mode.add_argument("--capture-off", action="store_true")
    for command in (prep, gen):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--max-total-tokens", type=int, default=131072)
        command.add_argument("--max-requests", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        rt = runtime()
        if not 1 <= args.max_total_tokens <= rt.MAX_TOTAL_TOKENS or not 1 <= args.max_requests <= rt.MAX_REQUESTS:
            raise ValueError("native corpus budget out of bounds")
        if args.command == "prepare":
            prepare(args, rt)
            print("Temporary native launcher prepared; no server started.")
        else:
            if not 1 <= args.max_tokens <= rt.MAX_SEQUENCE_TOKENS or not 0 < args.timeout <= 86400:
                raise ValueError("invalid generation bound")
            counts = generate(args, rt)
            print(json.dumps(counts, sort_keys=True))  # Aggregate counts only, never record contents.
    except Exception as exc:
        print("Native corpus failed (" + type(exc).__name__ + "); no private content printed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
