#!/usr/bin/env python3
"""Serial public-API B70 research cell. Leaves Glimmer down; never edits defaults."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

IMAGE = "vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
LAUNCHER_SHA = "63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4"
GUARD_SHA = "baa4647398874c19175ea74fe6f5d8dd6c2d83fc4bd0e5f2a68558afd983f5ad"
BASE = "http://127.0.0.1:8000"
CODE = "Write a detailed tutorial on implementing a bounded LRU cache in Python using collections.OrderedDict. Include a complete class, explain get and put behavior, discuss edge cases, and include tests. Continue with a worked example and complexity analysis. Be precise and use meaningful prose and code."
PROSE = "Explain how a relational database executes a SQL query, from parsing and planning through indexing, joins, transactions, and returning rows. Write a detailed technical tutorial with concrete examples and tradeoffs, including common performance pitfalls. Use complete sentences and avoid repeating yourself."
CALIBRATION_TOPICS = (
    "Python context managers and resource cleanup", "Rust ownership and borrowing",
    "JavaScript promises and cancellation", "Go channels and worker pools",
    "SQL window functions with employee salary examples", "CSS grid and accessible forms",
    "HTTP cache validation and conditional requests", "POSIX file permissions",
    "binary search invariants", "graph traversal with breadth first search",
    "heap based event scheduling", "UTF-8 parsing and malformed input",
    "unit testing dates and timezones", "floating point summation error",
    "database transactions and write skew", "rate limiting using token buckets",
    "command line argument parsing", "idempotent message processing",
    "JSON serialization and schema validation", "log rotation and retention",
    "image convolution with a small numerical example", "probability and Bayes rule",
    "network retries with exponential backoff", "immutable application state",
    "configuration precedence and environment variables", "stream processing with generators",
    "differential testing of sorting algorithms", "security boundaries in multi-tenant services",
    "regular expressions and finite automata", "numeric integration with Simpson's rule",
    "documenting a public API", "testing a compiler lexer",
)
HELDOUT_TOPICS = (
    "Implement a disjoint-set union structure and explain path compression.",
    "Explain DNS resolution including negative caching and DNSSEC limitations.",
    "Review a Python data race in a lazy initialization function and propose a fix.",
    "Give a worked example of exact rational arithmetic with fractions.",
    "Design a TOML configuration for a backup utility and explain each option.",
    "Explain B-tree page splits with a concrete insertion sequence.",
    "Write and explain a TypeScript discriminated union for an editor's events.",
    "Compare event sourcing with snapshots and ordinary relational updates.",
)
# Held-out functional checks, separate from all activation-calibration prompts.
CODE_TASKS = (
    ("gcd", "gcd(a, b), the nonnegative greatest common divisor of two integers, including zero and negatives",
     "assert gcd(48,18)==6; assert gcd(-15,10)==5; assert gcd(0,0)==0"),
    ("merge", "merge_intervals(items), returning sorted merged [start,end] lists; touching closed intervals merge; empty input returns []",
     "assert merge_intervals([[5,7],[1,3],[3,4]])==[[1,4],[5,7]]; assert merge_intervals([])==[]; assert merge_intervals([[1,9],[2,3]])==[[1,9]]"),
    ("brackets", "balanced(s), True if (), [] and {} are correctly nested, ignoring other characters",
     "assert balanced('a{b[()]}'); assert not balanced('([)]'); assert balanced(''); assert not balanced(']')"),
    ("lowerbound", "lower_bound(a, x), the first index with value >= x in a sorted list, or len(a), without importing bisect",
     "assert lower_bound([1,2,2,4],2)==1; assert lower_bound([],1)==0; assert lower_bound([1,3],4)==2; assert lower_bound([1,3],0)==0"),
    ("chunks", "chunks(xs, n), returning a list of consecutive sublists of size at most n; reject n <= 0 with ValueError",
     "assert chunks([1,2,3,4,5],2)==[[1,2],[3,4],[5]]; assert chunks([],3)==[]\ntry:\n chunks([1],0)\nexcept ValueError:\n pass\nelse:\n raise AssertionError('n=0 accepted')"),
    ("dedup", "dedup(xs), returning the unique hashable elements in first-occurrence order",
     "assert dedup([3,1,3,2,1])==[3,1,2]; assert dedup([])==[]; assert dedup(['b','a','b'])==['b','a']"),
    ("transpose", "transpose(rows), returning the transposed rectangular list of lists, [] for [], and raising ValueError for ragged input",
     "assert transpose([[1,2],[3,4]])==[[1,3],[2,4]]; assert transpose([])==[]\ntry:\n transpose([[1,2],[3]])\nexcept ValueError:\n pass\nelse:\n raise AssertionError('ragged accepted')"),
    ("rle", "rle(s), returning consecutive run-length pairs as (character, count) tuples",
     "assert rle('aaabbcaa')==[('a',3),('b',2),('c',1),('a',2)]; assert rle('')==[]; assert rle('x')==[('x',1)]"),
)


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def command(*args, timeout=60, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout, **kwargs).stdout


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return response.read().decode()


def post(path, payload, timeout=180):
    request = urllib.request.Request(BASE + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def counters():
    return {
        key: float(value)
        for line in get("/metrics").splitlines()
        if line.startswith("vllm:") and re.search(r"(spec_decode_num_(?:drafts|draft_tokens|accepted_tokens)_total|prefix_cache_(?:hits|queries)_total)", line)
        for key, value in [line.rsplit(" ", 1)]
    }


def sandbox(source, checks):
    """Execute model code only in an isolated no-network, read-only CPU container."""
    program = source + "\n" + checks + "\nprint('FUNCTIONAL_PASS')\n"
    try:
        output = command(
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "32", "--memory", "512m", "--cpus", "1",
            "--user", "65534:65534", "--entrypoint", "python", "-i", IMAGE,
            "-I", "-c", "import signal; signal.alarm(8); exec(compile(__import__('sys').stdin.read(), '<answer>', 'exec'))",
            input=program, timeout=20,
        )
        return {"pass": "FUNCTIONAL_PASS" in output, "stdout": output}
    except (subprocess.SubprocessError, OSError) as error:
        return {"pass": False, "error": str(error), "stderr": getattr(error, "stderr", None)}


class Cell:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.proc = None
        self.log = None
        self.rows = []
        self.summary = {"status": "starting", "head": args.head, "suite": args.suite,
                        "cascade_enabled": bool(args.cascade_patch), "cascade_probability_ratio": args.alpha}
        self.launcher = Path.home() / "inference/launchers/start-qwen38.sh"
        self.original = self.launcher.read_bytes()
        self.power = Path("/sys/class/drm/card0/device/hwmon/hwmon2/power1_cap")

    def start(self):
        if hashlib.sha256(self.original).hexdigest() != LAUNCHER_SHA:
            raise RuntimeError("persistent launcher changed")
        if self.power.read_text().strip() != "275000000":
            raise RuntimeError("power cap is not 275 W")
        running = command("docker", "ps", "--format", "{{.Names}}").splitlines()
        if running:
            raise RuntimeError(f"GPU cell requires no running containers; found {running}")
        guard = self.args.guard.resolve()
        if hashlib.sha256(guard.read_bytes()).hexdigest() != GUARD_SHA:
            raise RuntimeError("prefill guard mismatch")
        (self.out / "runner.py").write_bytes(Path(__file__).read_bytes())
        (self.out / "patch_uniform_decode_prefill.py").write_bytes(guard.read_bytes())
        text = self.original.decode()
        marker = "exec docker run --rm --name qwen38 --ipc=host"
        mounts = f' -v "{self.out}:/profile" -v "{guard}:/patch_guard.py:ro"'
        patches = "python /patch_guard.py; "
        if self.args.head != "dense":
            patch = self.args.head_patch.resolve()
            (self.out / "patch_lossy_lmhead.py").write_bytes(patch.read_bytes())
            mounts += f' -v "{patch}:/patch_head.py:ro" -e B70_TARGET_HEAD_MODE={self.args.head}'
            patches += "python /patch_head.py; "
            if self.args.head == "capture":
                mounts += " -e B70_HEAD_CAPTURE_ROOT=/profile"
            else:
                mounts += f' -v "{self.args.packed_head.resolve()}:/calibrated-head.pt:ro" -e B70_TARGET_HEAD_FILE=/calibrated-head.pt'
        if self.args.cascade_patch:
            patch = self.args.cascade_patch.resolve()
            (self.out / "patch_cascade.py").write_bytes(patch.read_bytes())
            mounts += f' -v "{patch}:/patch_cascade.py:ro" -e B70_CASCADE_ALPHA={self.args.alpha}'
            patches += "python /patch_cascade.py; "
        if text.count(marker) != 1 or text.count("exec vllm serve /model ") != 1:
            raise RuntimeError("launcher anchors changed")
        text = text.replace(marker, marker + mounts)
        flags = '--performance-mode balanced --compilation-config "{\\"cudagraph_capture_sizes\\":[1,2,4,8]}" --cudagraph-metrics '
        text = text.replace("exec vllm serve /model ", patches + "exec vllm serve /model " + flags)
        (self.out / "launcher.sh").write_text(text)
        command("bash", "-n", str(self.out / "launcher.sh"))
        self.log = (self.out / "server.log").open("w")
        self.proc = subprocess.Popen(["bash", str(self.out / "launcher.sh")], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 720
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Qwen exited at startup: {self.proc.returncode}")
            try:
                get("/health")
                break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError("Qwen startup exceeded 720s")
        models = json.loads(get("/v1/models"))
        save(self.out / "models.json", models)
        if not any(m["id"] == "qwen38" and m["max_model_len"] == 212992 for m in models["data"]):
            raise RuntimeError("wrong served model or context")
        self.summary["models"] = models
        self.summary["status"] = "running"
        print("CELL_READY=" + str(self.out), flush=True)

    def chat(self, label, prompt, count=512, forced=True):
        payload = {
            "model": "qwen38", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": 42, "max_tokens": count, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "cache_salt": label, "return_token_ids": True,
        }
        if forced:
            payload["ignore_eos"] = True
        save(self.out / f"{label}-request.json", payload)
        before = counters()
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        start = time.monotonic()
        first = None
        text, tokens, prompt_ids, usage, finish, done = "", [], [], {}, None, False
        with urllib.request.urlopen(req, timeout=300) as response, (self.out / f"{label}-sse.jsonl").open("w") as events:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                now = time.monotonic()
                events.write(json.dumps({"elapsed": now - start, "data": data}) + "\n")
                if data == "[DONE]":
                    done = True
                    break
                obj = json.loads(data)
                if obj.get("error"):
                    raise RuntimeError(str(obj["error"]))
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("prompt_token_ids") is not None:
                    prompt_ids = obj["prompt_token_ids"]
                for choice in obj.get("choices", []):
                    piece = choice.get("delta", {}).get("content") or ""
                    if piece and first is None:
                        first = now
                    text += piece
                    tokens.extend(choice.get("token_ids") or [])
                    finish = choice.get("finish_reason") or finish
        end = time.monotonic()
        after = counters()
        n = usage.get("completion_tokens", 0)
        ok = bool(done and text.strip() and n > 0 and finish and len(tokens) == n and len(prompt_ids) == usage.get("prompt_tokens"))
        if forced:
            ok = bool(ok and n == count and finish == "length")
        row = {"label": label, "content": text, "token_ids": tokens, "prompt_token_ids": prompt_ids,
               "usage": usage, "finish_reason": finish, "transport_pass": ok,
               "output_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               "ttft_s": None if first is None else first - start,
               "decode_tps": None if first is None or n < 2 else (n - 1) / (end - first),
               "elapsed_s": end - start, "metric_deltas": {k: after[k] - before.get(k, 0) for k in after}}
        save(self.out / f"{label}-result.json", row)
        self.rows.append(row)
        print(json.dumps({k: row[k] for k in ("label", "usage", "decode_tps", "transport_pass")}), flush=True)
        if not ok:
            raise RuntimeError("transport failure: " + label)
        return row

    def gates(self):
        a = self.chat("canary-arithmetic", "What is 19 + 23? Reply with only the integer.", 32, False)
        j = self.chat("canary-json", 'Reply with only this exact JSON object and no markdown: {"answer":42}', 64, False)
        c = self.chat("canary-code", "Output only Python source, without markdown fences: define a function add(a, b) that returns a + b.", 96, False)
        self.summary["canaries"] = [a["content"].strip() == "42", json.loads(j["content"]) == {"answer": 42},
                                    ast.dump(ast.parse(c["content"])) == ast.dump(ast.parse("def add(a,b): return a+b"))]
        if not all(self.summary["canaries"]):
            raise RuntimeError("canary failed")
        lengths = list(range(1, 129)) + [133, 197, 261]
        with (self.out / "boundaries.jsonl").open("w") as log:
            for n in lengths:
                payload = {"model": "qwen38", "prompt": [42] * n, "max_tokens": 1, "temperature": 0,
                           "seed": 42, "ignore_eos": True, "logprobs": 1, "cache_salt": f"boundary-{n}"}
                obj = post("/v1/completions", payload)
                choice = obj.get("choices", [{}])[0]
                probs = (choice.get("logprobs") or {}).get("token_logprobs", [])
                ok = (obj.get("usage", {}).get("prompt_tokens") == n and len(probs) == 1 and
                      all(isinstance(p, (int, float)) and math.isfinite(p) for p in probs) and choice.get("finish_reason") == "length")
                log.write(json.dumps({"request": payload, "response": obj, "pass": ok}) + "\n")
                log.flush()
                if not ok:
                    raise RuntimeError(f"nonfinite/bad boundary {n}")
        self.summary["finite_boundaries"] = len(lengths)

    def collect(self):
        for split, prompts, count in (("calibration", CALIBRATION_TOPICS, 384), ("heldout", HELDOUT_TOPICS, 256)):
            for i, prompt in enumerate(prompts):
                label = f"{split}-{i:02d}"
                save(self.out / "capture-request.json", {"split": split, "name": label})
                try:
                    if split == "calibration":
                        prompt = f"Write a detailed technical explanation of {prompt}. Include concrete examples and code where appropriate."
                    self.chat(label, prompt, count)
                finally:
                    (self.out / "capture-request.json").unlink(missing_ok=True)
        self.summary["capture_files"] = {split: len(list((self.out / split).glob("*.pt"))) for split in ("calibration", "heldout")}

    def quality(self):
        results = []
        for label, description, checks in CODE_TASKS:
            row = self.chat("functional-" + label, "Output only Python source without markdown fences. Implement " + description + ".", 512, False)
            source = row["content"].strip()
            if source.startswith("```") and source.endswith("```"):
                source = "\n".join(source.splitlines()[1:-1])
            result = {"label": label, **sandbox(source, checks)}
            results.append(result)
            save(self.out / "functional.json", results)
        self.summary["functional"] = results
        if not all(r["pass"] for r in results):
            raise RuntimeError("held-out functional test failed; do not rank candidate speed")

    def speed(self):
        for label, prompt in (("code", CODE), ("prose", PROSE)):
            self.chat(label + "-warmup", prompt)
            rows = [self.chat(f"{label}-{i}", prompt) for i in range(5)]
            self.summary[label] = {"median_tps": statistics.median(r["decode_tps"] for r in rows),
                                   "range_tps": [min(r["decode_tps"] for r in rows), max(r["decode_tps"] for r in rows)],
                                   "unique_outputs": len({r["output_sha256"] for r in rows})}

    def long_context(self):
        def tokenize(text):
            return post("/tokenize", {"model": "qwen38", "prompt": text, "add_special_tokens": False})["tokens"]
        template_request = {"model": "qwen38", "messages": [{"role": "user", "content": ""}],
                            "add_generation_prompt": True, "add_special_tokens": False,
                            "chat_template_kwargs": {"enable_thinking": False}}
        template = post("/tokenize", template_request)
        save(self.out / "long-template-request.json", template_request)
        save(self.out / "long-template-response.json", template)
        # Insert content before the final user im_end, preserving the server's
        # actual non-thinking assistant prefix. This remains Qwen-specific.
        end_user = max(i for i, token in enumerate(template["tokens"]) if token == 248046)
        header, footer = template["tokens"][:end_user], template["tokens"][end_user:]
        filler = tokenize("Background engineering note: services validate inputs, bound queue sizes, retry idempotent operations, record errors, and test rollback behavior.\n")
        start = tokenize("Read the following deployment notes and answer the question at the end.\n")
        needle = tokenize("\nThe deployment recovery code for project Cedar is 684219.\n")
        tail = tokenize("\nQuestion: What is the deployment recovery code for project Cedar? Reply with only the six digits.")

        def retrieve(label, prompt, timeout):
            payload = {"model": "qwen38", "prompt": prompt, "max_tokens": 32, "temperature": 0, "seed": 42,
                       "cache_salt": "heldout-" + label + "-cedar-v2", "logprobs": 1}
            save(self.out / f"{label}-request.json", payload)
            started = time.monotonic()
            obj = post("/v1/completions", payload, timeout=timeout)
            save(self.out / f"{label}-response.json", obj)
            choice = obj.get("choices", [{}])[0]
            probs = (choice.get("logprobs") or {}).get("token_logprobs", [])
            finite = bool(probs) and all(p is not None and math.isfinite(p) for p in probs)
            count = obj.get("usage", {}).get("prompt_tokens")
            ok = (choice.get("text", "").strip() == "684219" and finite and count == len(prompt)
                  and choice.get("finish_reason") == "stop")
            result = {"pass": ok, "protocol": "nonthinking-chat-v2", "elapsed_s": time.monotonic() - started,
                      "prompt_tokens": count, "answer": choice.get("text"), "finite_logprobs": finite,
                      "finish_reason": choice.get("finish_reason")}
            print(label.upper() + "=" + json.dumps(result), flush=True)
            return result

        smoke = retrieve("long-format-smoke", header + start + needle + tail + footer, 180)
        self.summary["long_format_smoke"] = smoke
        if not smoke["pass"]:
            raise RuntimeError("non-thinking retrieval format smoke failed; skipping costly long prefill")
        needed = 200000 - len(header) - len(start) - len(needle) - len(tail) - len(footer)
        background = (filler * (needed // len(filler) + 1))[:needed]
        prompt = header + start + background[:needed // 2] + needle + background[needed // 2:] + tail + footer
        assert len(prompt) == 200000
        result = retrieve("long", prompt, 1200)
        self.summary["long_context"] = result
        if not result["pass"]:
            raise RuntimeError("200k context retrieval failed")

    def close(self):
        (self.out / "capture-request.json").unlink(missing_ok=True)
        if self.proc is not None:
            names = command("docker", "ps", "--format", "{{.Names}}").splitlines()
            if "qwen38" in names:
                command("docker", "stop", "--time", "30", "qwen38")
            self.proc.wait(timeout=60)
        if self.log is not None:
            self.log.close()
        self.summary["launcher_unchanged"] = self.launcher.read_bytes() == self.original
        self.summary["power_unchanged"] = self.power.read_text().strip() == "275000000"
        save(self.out / "summary.json", self.summary)
        print("GLIMMER_LEFT_DOWN=1", flush=True)
        print("SUMMARY=" + json.dumps(self.summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--guard", type=Path, default=Path("/tmp/qwen-b70-patch-uniform-decode-prefill.py"))
    parser.add_argument("--head", choices=("dense", "capture", "gptq"), default="dense")
    parser.add_argument("--head-patch", type=Path)
    parser.add_argument("--packed-head", type=Path)
    parser.add_argument("--cascade-patch", type=Path)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--suite", choices=("capture", "compare", "long", "compare-long"), default="compare")
    args = parser.parse_args()
    if args.head != "dense" and not args.head_patch:
        parser.error("--head-patch required for capture/gptq")
    if args.head == "gptq" and not args.packed_head:
        parser.error("--packed-head required for gptq")
    if args.suite == "capture" and (args.head != "capture" or args.cascade_patch):
        parser.error("calibration requires dense capture without cascade")
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("alpha must be finite and in [0,1]")
    if bool(args.cascade_patch) != (args.alpha > 0):
        parser.error("--cascade-patch and a positive --alpha must be used together")
    cell = Cell(args)
    print("ARTIFACT_DIR=" + str(cell.out), flush=True)
    try:
        cell.start()
        cell.gates()
        if args.suite == "capture":
            cell.collect()
        elif args.suite == "long":
            cell.long_context()
        else:
            cell.quality()
            cell.speed()
            if args.suite == "compare-long":
                cell.long_context()
        cell.summary["status"] = "completed"
    except Exception as error:
        cell.summary.update(status="failed", error=repr(error))
        raise
    finally:
        cell.close()


if __name__ == "__main__":
    main()
