"""CPU/unit coverage for the bounded Glimmer teacher-data slice.

The official-model/GPU journey remains a lead-owned isolated-runtime check;
these tests exercise the data contract with a deliberately tiny fake target.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/experiments/glimmer_mtp_data.py"
spec = importlib.util.spec_from_file_location("glimmer_mtp_data", MODULE_PATH)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                             return_dict, return_tensors=None):
        assert tokenize and add_generation_prompt and return_dict
        text = messages[0]["content"]
        # Different prompt widths force true left-padding/position handling.
        tokens = [1, 2, 3, 4, 5] if "long" in text else [1, 2, 3, 4]
        payload = {"input_ids": torch.tensor([tokens], dtype=torch.long),
                   "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long)}
        if return_tensors is None:
            payload = {key: value.tolist() for key, value in payload.items()}
        return payload


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        batch = input_ids.shape[0]
        # Return padded input + fixed-length rows.  EOS and trailing pads test
        # that capture keeps EOS but never turns pads into future tokens.
        responses = []
        for row in range(batch):
            if row % 2:
                response = [20, 21, 22, 23, 24, 25, 26, 27, 99, 0]
            else:
                response = [10, 11, 12, 13, 14, 15, 16, 17, 99, 0]
            responses.append(response)
        return torch.cat((input_ids, torch.tensor(responses, dtype=torch.long,
                                                  device=input_ids.device)), dim=1)

    def parameters(self):
        return iter(())


class FakeTarget:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.device = torch.device("cpu")
        self.eos = {99}
        self.config = SimpleNamespace(eos_token_id=99, pad_token_id=0)
        self.environment = {"model": "fake-for-unit-test"}
        self.forward_calls = []

    def assert_frozen(self):
        return None

    def forward(self, token_ids, cache, logits_to_keep=1):
        assert cache is None
        assert logits_to_keep == 1
        self.forward_calls.append(list(token_ids))
        values = torch.arange(len(token_ids) * data.WIDTH, dtype=torch.float32)
        states = values.reshape(len(token_ids), data.WIDTH).to(torch.bfloat16)
        logits = torch.empty((len(token_ids), 1), dtype=torch.bfloat16)
        return states, logits, None


def _prompt_rows():
    return [
        {"id": "train-1", "family": "family-train-1", "split": "train", "text": "short train"},
        {"id": "train-2", "family": "family-train-2", "split": "train", "text": "long train"},
        {"id": "validation-1", "family": "family-validation", "split": "validation", "text": "validation"},
        {"id": "test-1", "family": "family-test", "split": "test", "text": "test"},
    ]


def _capture(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps(row) for row in _prompt_rows()) + "\n")
    args = SimpleNamespace(
        prompts=prompts,
        output_dir=tmp_path / "capture",
        train_token_budget=20,
        validation_token_budget=1,
        test_token_budget=1,
        max_prompt_tokens=32,
        max_new_tokens=10,
        generation_batch_size=2,
        shard_token_budget=9,
        attention="eager",
    )
    target = FakeTarget()
    index = data.capture_generated(args, target)
    return args, target, index


def test_prepare_prompts_streams_deduplicates_and_splits(monkeypatch, tmp_path):
    coding_sha = "a" * 40
    general_sha = "b" * 40
    coding_rows = [
        {"id": "c0", "input": "Write a parser."},
        {"id": "c1", "input": "Shared prompt"},
        {"id": "c2", "input": "Write a scheduler."},
    ]
    general_rows = [
        {"prompt_id": "g0", "messages": [{"role": "user", "content": " shared   prompt "}]},
        {"prompt_id": "g1", "messages": [{"role": "user", "content": "Explain caching."}]},
        {"prompt_id": "g2", "messages": [{"role": "system", "content": "skip"}, {"role": "user", "content": "Design a test."}]},
        {"prompt_id": "g3", "messages": [{"role": "user", "content": "Summarize a protocol."}]},
        {"prompt_id": "g4", "messages": [{"role": "user", "content": "Give an example."}]},
    ]

    datasets_module = ModuleType("datasets")
    calls = []

    def load_dataset(repo, *, split, revision, streaming):
        assert streaming is True
        calls.append((repo, split, revision))
        return iter(coding_rows if repo == "nvidia/OpenCodeInstruct" else general_rows)

    datasets_module.load_dataset = load_dataset
    hub_module = ModuleType("huggingface_hub")

    class FakeApi:
        def dataset_info(self, repo, revision):
            assert revision == "main"
            return SimpleNamespace(sha=coding_sha if repo.startswith("nvidia/") else general_sha)

    hub_module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    output = tmp_path / "prompts.jsonl"
    args = SimpleNamespace(output=output, seed=7, coding_fraction=0.4, max_prompts=5)
    summary = data.prepare_prompts(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    companion = json.loads((tmp_path / "prompts.jsonl.meta.json").read_text())

    assert summary == companion
    assert len(rows) == 5
    assert sum(row["source"] == "nvidia/OpenCodeInstruct" for row in rows) == 2
    assert sum(row["source"] == "HuggingFaceH4/ultrachat_200k" for row in rows) == 3
    assert not any(row["text"].casefold() == "shared prompt" and row["source"].startswith("HuggingFace") for row in rows)
    assert all(row["revision"] in (coding_sha, general_sha) for row in rows)
    families = {}
    for row in rows:
        assert row["split"] in data.SPLITS
        assert families.setdefault(row["family"], row["split"]) == row["split"]
    assert {row["split"] for row in rows} == set(data.SPLITS)
    assert calls == [
        ("nvidia/OpenCodeInstruct", "train", coding_sha),
        ("HuggingFaceH4/ultrachat_200k", "train_sft", general_sha),
    ]


def test_capture_shards_alignment_eos_and_frozen_target(tmp_path):
    args, target, index = _capture(tmp_path)
    assert index["status"] == "complete"
    assert index["state_kind"] == data.STATE_KIND
    assert index["width"] == data.WIDTH
    assert index["dtype"] == "bfloat16"
    assert index["root_count"] == 4
    assert all(shard["state_token_count"] <= 14 for shard in index["shards"])
    assert sum(shard["state_token_count"] for shard in index["shards"]) == index["state_token_count"]
    assert all(len(call) > 0 for call in target.forward_calls)
    # The model saw real left-padding/position IDs for the first batch.
    generation_call = target.model.calls[0]
    assert generation_call["attention_mask"].tolist() == [[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    assert generation_call["position_ids"].tolist() == [[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]]

    dataset = data.CapturedDataset(args.output_dir / "index.json", split="train")
    assert dataset.root_count == 2
    assert dataset.token_count == 27
    assert dataset.manifest() == [
        {key: row[key] for key in ("id", "family", "split", "text")}
        for row in _prompt_rows()[:2]
    ]
    states, tokens = dataset.iter_batches(8, limit_roots=99, seed=3).__next__()
    assert states.shape == (2, 9, data.WIDTH)
    assert states.dtype == torch.bfloat16
    assert tokens.shape == (2, 10)
    assert tokens.dtype == torch.long
    assert set(tokens[:, 0].tolist()) == {4, 5}
    assert all(99 in row for row in tokens.tolist())
    assert all(0 not in row for row in tokens.tolist())


def test_sampler_resume_is_exact_and_validation_is_independent(tmp_path):
    args, _, _ = _capture(tmp_path)
    dataset = data.CapturedDataset(args.output_dir / "index.json", split="train")
    first = dataset.new_sampler(123)
    first.next_batch(1)
    state = first.state_dict()
    state_path = tmp_path / "sampler-state.pt"
    torch.save(state, state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    expected_states, expected_tokens = first.next_batch(3)

    resumed = dataset.new_sampler(123)
    resumed.load_state_dict(state)
    actual_states, actual_tokens = resumed.next_batch(3)
    assert torch.equal(expected_states, actual_states)
    assert torch.equal(expected_tokens, actual_tokens)

    # Validation iteration has its own fresh deterministic sampler and is
    # capped at the split's root count, regardless of requested limit.
    validation = data.CapturedDataset(args.output_dir / "index.json", split="validation")
    batches = list(validation.iter_batches(4, limit_roots=100, seed=123))
    assert sum(batch[0].shape[0] for batch in batches) == validation.root_count == 1


def test_capture_index_refuses_partial_and_path_traversal(tmp_path):
    args, _, index = _capture(tmp_path)
    index_path = args.output_dir / "index.json"
    original = json.loads(index_path.read_text())

    partial = dict(original)
    partial["status"] = "incomplete_budget"
    partial["complete"] = False
    index_path.write_text(json.dumps(partial))
    with pytest.raises(data.CaptureFormatError, match="incomplete"):
        data.CapturedDataset(index_path)

    unsafe = dict(original)
    unsafe["shards"] = [dict(original["shards"][0], path="../outside.pt")] + original["shards"][1:]
    index_path.write_text(json.dumps(unsafe))
    with pytest.raises(data.CaptureFormatError, match="traversal|escapes"):
        data.CapturedDataset(index_path)


def test_legacy_pt_path_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    path.write_bytes(b"not a capture")
    with pytest.raises(data.CaptureFormatError, match="index.json"):
        data.CapturedDataset(path)


@pytest.mark.parametrize("output", [[[10, 11]], [[0, 2, 3, 10]]])
def test_generation_refuses_missing_or_corrupted_input_prefix(output):
    target = FakeTarget()
    target.model.generate = lambda **kwargs: torch.tensor(output)
    with pytest.raises(ValueError, match="padded input prefix"):
        data._generate_batch(target, [[1, 2, 3]], 10, 0, {99}, torch)


def test_generation_type_error_is_not_retried_with_changed_options():
    target = FakeTarget()
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise TypeError("generation failed")

    target.model.generate = fail
    with pytest.raises(TypeError, match="generation failed"):
        data._generate_batch(target, [[1, 2, 3]], 10, 0, {99}, torch)
    assert len(calls) == 1
    assert calls[0]["use_cache"] is True
    assert calls[0]["num_beams"] == 1
