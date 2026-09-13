#!/usr/bin/env python3
"""Stdlib-only contract checks; never imports an evaluator or runs generated code."""
from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
QUALITY_PATH = ROOT / "quality.py"
DOCKERFILE = ROOT / "Dockerfile"


def load_quality():
    spec = importlib.util.spec_from_file_location("grouped_quality_fixture", QUALITY_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load quality.py fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_no_optional_top_level_imports(tree: ast.Module) -> None:
    forbidden = {
        "torch",
        "vllm",
        "transformers",
        "datasets",
        "evalplus",
        "lm_eval",
        "numpy",
        "requests",
        "openai",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert not forbidden.intersection(names), f"optional/ML import at module scope: {names}"


def request_and_scoring_fixtures(q):
    row = {
        "id": "fixture/1",
        "prompt": "Say fixture",
        "messages": [{"role": "user", "content": "Say fixture"}],
        "stop_sequences": [],
    }
    payload = q._build_request(row, "qwen38")
    assert payload == {
        "model": "qwen38",
        "messages": row["messages"],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 42,
        "max_tokens": 4096,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": "C1",
        "return_token_ids": True,
    }
    assert "ignore_eos" not in payload
    bad_fields, bad_errors = q._extract_response_fields(
        {"choices": [{"message": {"content": "fixture"}, "finish_reason": "stop"}]},
        payload,
    )
    assert bad_fields["output_token_ids"] is None
    assert {error["kind"] for error in bad_errors} >= {
        "missing_output_token_ids", "missing_prompt_token_ids"
    }
    assert q._first_divergence([1, 2], [1, 2]) is None
    assert q._first_divergence([1, 2], [1, 9]) == 2
    assert q._first_divergence([1], [1, 2]) == 2
    assert q._gsm_extract_strict("work\n#### 1,200") == "1,200"
    assert q._gsm_extract_strict("answer: 1200") is None
    assert q._gsm_extract_flexible("answer: $1,200") == "$1,200"
    assert q._gsm_normalize("Reasoning #### $1,200.") == "1200"


def http_generation_fixture(q):
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
            payload = json.loads(body)
            seen.append(payload)
            response = {
                "id": "fixture-response",
                "object": "chat.completion",
                "model": "qwen38",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fixture"},
                        "token_ids": [11, 12],
                        "finish_reason": "stop",
                    }
                ],
                "prompt_token_ids": [101],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
            encoded = json.dumps(response, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            data = directory / "data.jsonl"
            row = {
                "id": "fixture/1",
                "task": "external",
                "prompt": "Say fixture",
                "messages": [{"role": "user", "content": "Say fixture"}],
                "prompt_sha256": q._sha256_json("Say fixture"),
                "messages_sha256": q._sha256_json(
                    [{"role": "user", "content": "Say fixture"}]
                ),
            }
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = directory / "target.jsonl"
            args = SimpleNamespace(
                data=str(data),
                out=str(output),
                arm="target",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="qwen38",
                timeout=5.0,
                api_key=None,
                limit=1,
                force=False,
            )
            assert q.generate(args) == 0
            records = q._read_jsonl(output)
            assert len(records) == 1 and records[0]["ok"]
            assert records[0]["output_token_ids"] == [11, 12]
            assert records[0]["prompt_token_ids"] == [101]
            assert records[0]["finish_reason"] == "stop"
            assert records[0]["truncated"] is False
            assert seen and seen[0]["stream"] is False
            assert seen[0]["return_token_ids"] is True
            assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
            assert "ignore_eos" not in seen[0]
            assert records[0]["request_sha256"] == hashlib.sha256(
                records[0]["request_body"].encode()
            ).hexdigest()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def docker_fixture():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "@sha256:" in text and "FROM python:3.11.11-slim-bookworm@" in text
    assert '"lm-eval==0.4.13"' in text
    assert '"evalplus==0.3.1"' in text
    assert "QUALITY_EVAL_SANDBOX=1" in text
    assert "USER 65532:65532" in text
    assert "vllm" not in text.lower()


def main() -> int:
    tree = ast.parse(QUALITY_PATH.read_text(encoding="utf-8"), filename=str(QUALITY_PATH))
    assert_no_optional_top_level_imports(tree)
    q = load_quality()
    assert q.EXPECTED_PRIMARY_COUNTS == {
        "ifeval": 541,
        "gsm8k": 1319,
        "humanevalplus": 164,
        "mbppplus": 378,
    }
    assert q.EXPECTED_REQUEST_TOTAL == 2466
    assert q.IFEVAL_REVISION and q.GSM_REVISION
    request_and_scoring_fixtures(q)
    http_generation_fixture(q)
    docker_fixture()
    print("PASS: stdlib AST, pinned constants, request contract, GSM extraction, HTTP capture, Docker sandbox contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
