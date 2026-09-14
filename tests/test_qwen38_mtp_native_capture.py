"""CPU-only native seam/trajectory/HTTP protocol checks, not GPU E2E evidence.

B70_MTP_TEST_VLLM_ROOT includes read-only checks of the actual pinned source.
Use the pinned disposable image, no devices, --network none. Synthetic HTTP
fixtures bind loopback only, never run vLLM, and never read private records.
"""
from __future__ import annotations

import ast
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).parents[1]
PATCHES = ROOT / "patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b"
CLI = ROOT / "scripts/experiments/qwen38_mtp_native_corpus.py"
KEY = "b70-native-" + "a" * 32


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rt(monkeypatch):
    pytest.importorskip("torch")
    module = load(PATCHES / "b70_mtp_native_capture.py")
    monkeypatch.setattr(module, "HIDDEN_SIZE", 4)
    for name in ("B70_MTP_NATIVE_CAPTURE_DIR", "B70_MTP_NATIVE_MAX_TOKENS", "B70_MTP_NATIVE_MAX_REQUESTS"):
        monkeypatch.delenv(name, raising=False)
    return module


@pytest.fixture
def client():
    return load(CLI)


def request_fixture(rt, root, prompt=None, maximum=32, key=KEY):
    prompt = prompt or [11, 12, 13]
    directory = rt.private_directory(root / key, create=True)
    rt.save_json(directory / "control.json", dict(request_id=key, prompt_token_ids=prompt, max_tokens=maximum))
    req_id = "chatcmpl-" + key + "-1234abcd"
    request = NS(prompt_token_ids=prompt, sampling_params=NS(max_tokens=maximum),
                 mm_features=[], prompt_embeds=None, lora_request=None, prompt_is_token_ids=None,
                 output_token_ids=[-1] * 100)
    runner = NS(
        speculative_config=NS(method="mtp", num_speculative_tokens=4),
        cache_config=NS(enable_prefix_caching=True, kv_sharing_fast_prefill=False),
        scheduler_config=NS(max_num_seqs=1, async_scheduling=True),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, use_ubatching=False),
        vllm_config=NS(kv_transfer_config=None, ec_transfer_config=None),
        model_config=NS(hf_text_config=NS(hidden_size=rt.HIDDEN_SIZE)),
        use_aux_hidden_state_outputs=False, is_pooling_model=False,
        get_model=lambda: type("Qwen3_5ForCausalLM", (), {})(),
        requests={req_id: request},
        input_batch=NS(req_ids=[req_id], num_reqs=1, num_computed_tokens_cpu=[999999],
                       token_ids_cpu=[[-1] * 100], sampled_token_ids_cpu=[[-1] * 5]),
    )
    return runner, directory


def block(rt, runner, start, ids, outputs, drafts=0, *, spec=True):
    torch = rt.torch
    req_id = runner.input_batch.req_ids[0]
    scheduler = NS(num_scheduled_tokens={req_id: len(ids)}, total_num_scheduled_tokens=len(ids),
                   scheduled_spec_decode_tokens={req_id: ids[1:]} if drafts else {},
                   scheduled_cached_reqs=NS(resumed_req_ids=set()), preempted_req_ids=set())
    assert not drafts or len(ids) == drafts + 1
    # Padding and async CPU token metadata intentionally disagree with GPU input.
    runner.input_ids = NS(gpu=torch.tensor(ids + [999, 999], dtype=torch.int32))
    runner.positions = torch.tensor(list(range(start, start + len(ids))) + [999, 999])
    runner._get_positions = lambda n: runner.positions[:n]
    hidden = (torch.arange(start, start + len(ids) + 2, dtype=torch.float32) + 100)[:, None].repeat(1, rt.HIDDEN_SIZE)
    sampled = NS(sampled_token_ids=torch.tensor([outputs + [-1] * (5 - len(outputs))], dtype=torch.int32))
    return scheduler, NS(num_draft_tokens=[drafts]) if spec else None, hidden, sampled


def step(rt, capture, runner, start, ids, outputs, drafts=0, **kwargs):
    values = block(rt, runner, start, ids, outputs, drafts, **kwargs)
    capture.step(runner, *values)
    return values


def read_step(rt, directory, i):
    return rt.torch.load(directory / f"step-{i:06d}.pt", weights_only=True)


@pytest.mark.parametrize("m", [1, 3, 5])
def test_exact_zero_partial_full_and_missing_terminal_hidden(rt, tmp_path, m):
    runner, directory = request_fixture(rt, tmp_path)
    capture = rt.NativeCapture(tmp_path)
    step(rt, capture, runner, 0, [11, 12, 13], [20], spec=False)
    outputs = [21, 22, 23, 24][:m - 1] + [31]
    values = step(rt, capture, runner, 3, [20, 21, 22, 23, 24], outputs, drafts=4)
    # Mutating the graph/input buffers after the seam cannot alter saved features.
    runner.input_ids.gpu.fill_(-1)
    runner.positions.fill_(999)
    values[2].fill_(-99)
    raw = read_step(rt, directory, 1)
    assert raw["positions"].tolist() == list(range(3, 3 + m))
    assert raw["input_ids"].tolist() == [20, 21, 22, 23, 24][:m]
    assert raw["output_positions"].tolist() == list(range(4, 4 + m))
    assert raw["output_ids"].tolist() == outputs
    assert raw["target_last_hidden_states"][:, 0].tolist() == list(range(103, 103 + m))
    result = rt.finalize_request(directory, [11, 12, 13], [20] + outputs)
    total = len(result["input_ids"])
    assert len(result["positions"]) == total - 1
    assert len(result["loss_mask"]) == total
    assert result["positions"].tolist() == list(range(total - 1))
    assert result["metadata"]["observed_hidden_length"] == total - 1
    assert result["metadata"]["acceptance_blocks"][{1: "zero", 3: "partial", 5: "full"}[m]] == 1
    # Native MTP alignment h[t], x[t+1] -> x[t+2], NOT h[t+1].
    assert len(result["target_last_hidden_states"][:total - 2]) == len(result["input_ids"][2:])
    assert result["input_ids"][1:-1].tolist() == ([11, 12, 13, 20] + outputs)[1:-1]


def test_full_accept_bonus_hidden_first_appears_next_round(rt, tmp_path):
    runner, directory = request_fixture(rt, tmp_path)
    capture = rt.NativeCapture(tmp_path)
    step(rt, capture, runner, 0, [11, 12, 13], [20])
    step(rt, capture, runner, 3, [20, 21, 22, 23, 24], [21, 22, 23, 24, 31], drafts=4)
    assert read_step(rt, directory, 1)["positions"].tolist() == [3, 4, 5, 6, 7]
    step(rt, capture, runner, 8, [31], [32], spec=False)
    assert read_step(rt, directory, 2)["input_ids"].tolist() == [31]
    assert read_step(rt, directory, 2)["positions"].tolist() == [8]
    result = rt.finalize_request(directory, [11, 12, 13], [20, 21, 22, 23, 24, 31, 32])
    assert result["target_last_hidden_states"][:, 0].tolist() == list(range(100, 109))


def test_chunked_prefill_includes_every_prompt_row_discards_samples(rt, tmp_path):
    prompt = [11, 12, 13, 14, 15, 16]
    runner, directory = request_fixture(rt, tmp_path, prompt)
    capture = rt.NativeCapture(tmp_path)
    for start in (0, 2, 4):
        step(rt, capture, runner, start, prompt[start:start + 2], [20 if start == 4 else 999], spec=False)
    assert read_step(rt, directory, 0)["output_ids"].numel() == 0
    result = rt.finalize_request(directory, prompt, [20])
    assert result["positions"].tolist() == list(range(6))
    assert result["input_ids"].tolist() == prompt + [20]


@pytest.mark.parametrize("generated", [[20], [20, 21], [20, 21, 22, 23, 24, 31]])
def test_eos_length_truncation_drops_only_terminal_candidates(rt, tmp_path, generated):
    runner, directory = request_fixture(rt, tmp_path)
    capture = rt.NativeCapture(tmp_path)
    step(rt, capture, runner, 0, [11, 12, 13], [20])
    step(rt, capture, runner, 3, [20, 21, 22, 23, 24], [21, 22, 23, 24, 31], drafts=4)
    result = rt.finalize_request(directory, [11, 12, 13], generated)
    total, observed = len(result["input_ids"]), len(result["positions"])
    assert total - 2 <= observed <= total
    assert result["input_ids"].tolist() == [11, 12, 13] + generated
    hidden = result["target_last_hidden_states"]
    assert hidden.untyped_storage().nbytes() == hidden.numel() * hidden.element_size()


def test_async_terminal_tail_need_not_finish_but_committed_output_must_exist(rt, tmp_path):
    runner, directory = request_fixture(rt, tmp_path)
    capture = rt.NativeCapture(tmp_path)
    step(rt, capture, runner, 0, [11, 12, 13], [20])
    with rt._private_open(directory / "step-000001.pt.partial", "wb") as handle:
        handle.write(b"a future async forward is still saving")
    result = rt.finalize_request(directory, [11, 12, 13], [20])
    assert result["positions"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        rt.finalize_request(directory, [11, 12, 13], [20, 21])


@pytest.mark.parametrize("bad", ["gap", "cache-hit", "reorder", "resumed", "preempted", "mm", "lora", "bounds", "sample-gap", "draft-mismatch", "nan"])
def test_capture_fail_closed(rt, tmp_path, bad):
    runner, directory = request_fixture(rt, tmp_path, maximum=2 if bad == "bounds" else 32)
    capture = rt.NativeCapture(tmp_path)
    if bad != "cache-hit":
        step(rt, capture, runner, 0, [11, 12, 13], [20])
    vals = block(rt, runner, 3, [20, 21, 22, 23, 24], [21, 22, 23, 24, 31], 4)
    if bad in ("gap", "reorder"):
        runner.positions[0] += 1 if bad == "gap" else -1
    elif bad == "resumed":
        vals[0].scheduled_cached_reqs.resumed_req_ids.add(runner.input_batch.req_ids[0])
    elif bad == "preempted":
        vals[0].preempted_req_ids.add(runner.input_batch.req_ids[0])
    elif bad in ("mm", "lora"):
        setattr(runner.requests[runner.input_batch.req_ids[0]], "mm_features" if bad == "mm" else "lora_request", [1])
    elif bad == "bounds":
        # A second full-accept block cannot fit the bounded candidate trajectory.
        capture.step(runner, *vals)
        vals = block(rt, runner, 8, [31, 32, 33, 34, 35], [32], 4)
    elif bad == "sample-gap":
        vals[3].sampled_token_ids[0, 1] = -1
    elif bad == "draft-mismatch":
        vals[3].sampled_token_ids[0, 0] = 888
    elif bad == "nan":
        vals[2][0, 0] = float("nan")
    with pytest.raises(ValueError):
        capture.step(runner, *vals)


@pytest.mark.parametrize("bad", ["prompt", "output", "unobserved-output", "missing-prefix", "missing-step", "position", "hidden"])
def test_finalize_rejects_mismatch_and_missing_features(rt, tmp_path, bad):
    runner, directory = request_fixture(rt, tmp_path)
    capture = rt.NativeCapture(tmp_path)
    step(rt, capture, runner, 0, [11, 12, 13], [20])
    step(rt, capture, runner, 3, [20], [21])
    prompt, output = [11, 12, 13], [20, 21]
    if bad == "prompt":
        prompt[0] = 99
    elif bad == "output":
        output[0] = 99
    elif bad == "unobserved-output":
        output += [22, 23, 24]
    elif bad in ("missing-prefix", "missing-step"):
        (directory / ("step-000000.pt" if bad == "missing-prefix" else "step-000001.pt")).unlink()
    else:
        path = directory / "step-000001.pt"
        raw = read_step(rt, directory, 1)
        if bad == "position":
            raw["positions"][0] += 1
        else:
            raw["target_last_hidden_states"] = raw["target_last_hidden_states"][:0]
        path.unlink()
        rt.save_tensor(path, raw)
    with pytest.raises(ValueError):
        rt.finalize_request(directory, prompt, output)


def test_bounds_disabled_hook_and_private_no_overwrite(rt, tmp_path, monkeypatch):
    rt.capture_native_step(None, None, None, None, None)
    runner, directory = request_fixture(rt, tmp_path)
    monkeypatch.setenv("B70_MTP_NATIVE_MAX_TOKENS", "3")
    with pytest.raises(ValueError, match="budget"):
        step(rt, rt.NativeCapture(tmp_path), runner, 0, [11, 12, 13], [20])
    with pytest.raises(FileExistsError):
        rt.save_json(directory / "control.json", {})
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError):
        rt.private_directory(public)
    repo = tmp_path / "repo"
    repo.mkdir(mode=0o700)
    (repo / ".git").write_text("gitdir: somewhere")
    with pytest.raises(ValueError):
        rt.private_directory(repo / "out", create=True)
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError):
        rt.private_directory(link)
    assert (directory / "control.json").stat().st_mode & 0o777 == 0o600


def test_root_worker_preserves_host_bind_mount_owner(rt, tmp_path):
    if os.geteuid() != 0:
        pytest.skip("root-only Docker bind-mount ownership test")
    directory = rt.private_directory(tmp_path / "host-user", create=True)
    os.chown(directory, 12345, 12345)
    rt.save_json(directory / "control.json", {"synthetic": True})
    assert (directory / "control.json").stat().st_uid == 12345
    assert rt.load_json(directory / "control.json") == {"synthetic": True}


def test_source_patch_once_idempotent_and_seam_order():
    patch = load(PATCHES / "patch_mtp_native_capture.py")
    source = "class Runner:\n    def sample_tokens(self):\n" + patch.OLD + "        self.propose_draft_token_ids()\n"
    changed = patch.patched_source(source)
    assert patch.patched_source(changed) == changed
    tree = ast.parse(changed)
    calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
             for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert calls == ["_update_states_after_model_execute", "capture_native_step", "propose_draft_token_ids"]
    for invalid in ("", source + source, changed.replace("hidden_states, sampler_output", "wrong, sampler_output")):
        with pytest.raises(RuntimeError):
            patch.patched_source(invalid)


def test_actual_pinned_source_seam_and_api_contract():
    root = os.environ.get("B70_MTP_TEST_VLLM_ROOT")
    if not root:
        pytest.skip("set B70_MTP_TEST_VLLM_ROOT to inspect the pinned image's source")
    root = Path(root)
    patch = load(PATCHES / "patch_mtp_native_capture.py")
    source = (root / "v1/worker/gpu_model_runner.py").read_text()
    changed = patch.patched_source(source)
    assert patch.patched_source(changed) == changed
    assert changed.index(patch.NEW) < changed.index("        def propose_draft_token_ids(sampled_token_ids):")
    api = (root / "entrypoints/openai/chat_completion/serving.py").read_text()
    assert "final_res.prompt_token_ids if request.return_token_ids else None" in api
    assert "as_list(output.token_ids)" in api
    assert 'f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"' in api
    processor = (root / "v1/engine/input_processor.py").read_text()
    assert 'f"{request.external_req_id}-{random_uuid():.8}"' in processor
    target = (root / "model_executor/models/qwen3_next.py").read_text()
    assert "hidden_states, _ = self.norm(hidden_states, residual)" in target


@pytest.mark.parametrize("capture", [False, True])
def test_temporary_launcher_restores_production_flags_only(client, tmp_path, monkeypatch, capture):
    monkeypatch.syspath_prepend(str(CLI.parent))
    import qwen38_mtp_reference as reference
    original = b'''#!/bin/bash
COOKBOOK="$ROOT/src/intel-arc-pro-b70-inference-cookbook"
exec docker run --rm --name qwen38 --ipc=host -p "0.0.0.0:${PORT:-8000}:8000" image bash -c '
exec vllm serve /model --enable-prefix-caching --default-chat-template-kwargs "{\\"enable_thinking\\":true}" --speculative-config "{\\"method\\":\\"mtp\\",\\"num_speculative_tokens\\":4}"
'
'''
    monkeypatch.setattr(reference.dflash.probe, "LAUNCHER_SHA", hashlib.sha256(original).hexdigest())
    result = client.launcher_text(original, tmp_path, capture, 1234, 12)
    assert "--no-enable-prefix-caching" not in result
    assert "--enable-prefix-caching" in result
    assert r'enable_thinking\":true' in result
    assert "--no-async-scheduling" not in result
    assert "patch_mtp_native_capture.py" in result
    assert ("B70_MTP_NATIVE_CAPTURE_DIR" in result) is capture
    subprocess.run(["bash", "-n"], input=result, text=True, check=True)
    with pytest.raises(RuntimeError, match="persistent"):
        client.launcher_text(original + b"# altered", tmp_path, capture, 1234, 12)


@pytest.mark.parametrize("capture", [False, True])
def test_cli_synthetic_http_protocol(rt, tmp_path, capture):
    # Use the REAL CLI subprocess and HTTP transport; only the inference service
    # is synthetic. The on-mode fixture invokes the same NativeCapture.step.
    rt.HIDDEN_SIZE = 5120  # CLI child loads the actual pinned dimensions.
    capture_dir = rt.private_directory(tmp_path / "features", create=True)
    collector = rt.NativeCapture(capture_dir)
    requests, errors = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                if self.path == "/tokenize":
                    result = {"tokens": [11, 12, 13], "count": 3}
                else:
                    assert self.path == "/v1/chat/completions"
                    assert body["return_token_ids"] is True and body["stream"] is False
                    assert body["n"] == 1 and len(body["cache_salt"]) == 32
                    result = {"id": "chatcmpl-" + body["request_id"], "prompt_token_ids": [11, 12, 13],
                              "choices": [{"index": 0, "token_ids": [20], "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": None,
                                                       "reasoning": "synthetic reasoning not retokenized",
                                                       "tool_calls": [{"function": {"name": "NEVER_EXECUTE", "arguments": "{}"}}]}}],
                              "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
                    if capture:
                        # CLI already created the request control; construct only
                        # the fake runner (not a replacement collection path).
                        key = body["request_id"]
                        temporary = rt.private_directory(tmp_path / (key + "-fixture"), create=True)
                        runner, _ = request_fixture(rt, temporary, maximum=body["max_tokens"], key=key)
                        step(rt, collector, runner, 0, [11, 12, 13], [20])
                data = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                errors.append(exc)
                self.send_error(500)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "launch-config.json"
    rt.save_json(config, dict(capture=capture, speculative_tokens=4, prefix_caching=True,
                              max_requests=16, max_total_tokens=131072))
    command = [sys.executable, str(CLI), "generate", "--output", str(tmp_path / "out"),
               "--server-config", str(config), "--base-url", f"http://127.0.0.1:{server.server_port}",
               "--synthetic", "--max-tokens", "8", "--timeout", "10"]
    command += ["--capture-dir", str(capture_dir)] if capture else ["--capture-off"]
    try:
        process = subprocess.run(command, text=True, capture_output=True, timeout=40)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert not errors
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["requests"] == 4 and summary["generated_tokens"] == 4
    assert summary["observed_hidden_rows"] == (12 if capture else 0)
    assert "NEVER_EXECUTE" not in process.stdout + process.stderr
    chat = [b for p, b in requests if p == "/v1/chat/completions"]
    assert len(set(b["cache_salt"] for b in chat)) == 4
    assert any("tools" in b for b in chat)
    assert {b["chat_template_kwargs"]["enable_thinking"] for b in chat} == {True, False}
    for path in (tmp_path / "out").rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    assert len(list((tmp_path / "out").glob("*/*.pt"))) == (4 if capture else 0)
    # Existing output is an error before sending another request, never clobber.
    process = subprocess.run(command, text=True, capture_output=True, timeout=20)
    assert process.returncode == 1 and "FileExistsError" in process.stderr


def test_private_input_protocol_validation_and_sanitized_failure(rt, client, tmp_path, capsys, monkeypatch):
    record = {"messages": [{"role": "assistant", "content": None, "tool_calls": []},
                           {"role": "tool", "tool_call_id": "abc", "content": "synthetic result"}],
              "tools": [], "chat_template_kwargs": {"enable_thinking": True}, "split": "heldout"}
    split, body = client.validate_record(record, 12)
    assert split == "heldout" and body["messages"] == record["messages"]
    with pytest.raises(ValueError):
        client.validate_record({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}, 12)
    with pytest.raises(ValueError):
        client.validate_record({**record, "tool_choice": "none"}, 12)
    monkeypatch.setattr(client, "runtime", lambda: rt)
    monkeypatch.setattr(client, "generate", lambda *a: (_ for _ in ()).throw(ValueError("SECRET-CONTENT")))
    assert client.main(["generate", "--output", str(tmp_path / "out"), "--server-config", "unused",
                        "--synthetic", "--capture-off"]) == 1
    assert "SECRET-CONTENT" not in capsys.readouterr().err
