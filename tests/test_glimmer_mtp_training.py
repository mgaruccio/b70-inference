"""CPU contract tests, NOT substitutes for the lead's real Glimmer CLI/GPU gate."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/glimmer_recursive_mtp.py"
spec = importlib.util.spec_from_file_location("glimmer_training_under_test", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
VALIDATION = SCRIPT.with_name("glimmer_mtp_validation.jsonl")
TEST = SCRIPT.with_name("glimmer_mtp_test.jsonl")


class TinyTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([0] if add_special_tokens else []) + [ord(c) % 29 + 1 for c in text[:12]]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [0, 2] + self.encode(messages[0]["content"], False) + [3]

    def decode(self, tokens):
        return " ".join(map(str, tokens))


class TinyTarget(torch.nn.Module):
    """Frozen CPU test double. The production target and decoder are not replaced."""
    def __init__(self, attention="sdpa"):
        super().__init__()
        torch.manual_seed(700)
        self.embedding = torch.nn.Embedding(31, 12)
        self.norm = torch.nn.RMSNorm(12, eps=1e-5)
        self.lm_head = torch.nn.Linear(12, 31, bias=False)
        self.config = SimpleNamespace(hidden_size=12, output_multiplier=.7, final_logit_softcapping=20.)
        self.device, self.torch, self.eos = torch.device("cpu"), torch, set()
        self.tokenizer = TinyTokenizer()
        self.environment = {"runtime": "CPU unit double, not Glimmer", "attention": attention}
        self.eval().requires_grad_(False)

    def sync(self):
        pass

    def reset_peak(self):
        pass

    def memory(self):
        return {}

    def assert_frozen(self):
        assert all(not p.requires_grad and p.grad is None for p in self.parameters())

    def forward(self, tokens, cache, logits_to_keep=0):
        cache = [] if cache is None else cache
        state = cache[-1] if cache else torch.zeros(12)
        states = []
        for token in tokens:
            state = self.norm(state + self.embedding(torch.tensor(token)))
            states.append(state)
            cache.append(state)
        states = torch.stack(states)
        logits = pilot.project(self.lm_head, states[-logits_to_keep:] if logits_to_keep else states, self.config)
        return states, logits, cache

    def prefill(self, tokens):
        states, logits, cache = self.forward(tokens, None, 1)
        return states[-1].clone(), int(logits[-1].argmax()), cache

    def verify(self, tokens, cache):
        states, logits, cache = self.forward(tokens, cache)
        return states, logits.argmax(-1).tolist(), cache

    def crop(self, cache, keep):
        del cache[keep:]

    @staticmethod
    def cache_length(cache):
        return len(cache)


class ContractDataset:
    """The agreed helper API with tiny tensors; no shard implementation in this slice."""
    def __init__(self, path=None, split="train"):
        generator = torch.Generator().manual_seed({"train": 51, "validation": 52, "test": 53}[split])
        self.states = torch.randn(17, 9, 12, generator=generator)
        self.tokens = torch.randint(0, 31, (17, 10), generator=generator)
        self.root_count, self.token_count, self.split = 17, 170, split
        self.metadata = {"model": pilot.MODEL, "revision": pilot.REVISION, "state_kind": "target_final_norm"}
        self.seen = []
        self.iterated = []

    def manifest(self):
        return [{"id": self.split + "-source", "family": self.split + "-source-family",
                 "split": self.split, "text": self.split + " source document not an external fixture"}]

    def new_sampler(self, seed):
        dataset = self

        class Sampler:
            def __init__(self):
                self.rng, self.order, self.cursor = random.Random(seed), [], 0

            def next_batch(self, size):
                indices = []
                for _ in range(size):
                    if self.cursor == len(self.order):
                        self.order = list(range(dataset.root_count))
                        self.rng.shuffle(self.order)
                        self.cursor = 0
                    indices.append(self.order[self.cursor])
                    self.cursor += 1
                dataset.seen.extend(indices)
                return dataset.states[indices], dataset.tokens[indices]

            def state_dict(self):
                return {"rng": self.rng.getstate(), "order": self.order, "cursor": self.cursor}

            def load_state_dict(self, state):
                self.rng.setstate(state["rng"])
                self.order, self.cursor = state["order"], state["cursor"]

        return Sampler()

    def iter_batches(self, batch_size, limit_roots, seed):
        self.iterated.append((self.split, batch_size, limit_roots, seed))
        ids = random.Random(seed).sample(range(self.root_count), min(limit_roots, self.root_count))
        for start in range(0, len(ids), batch_size):
            selected = ids[start:start + batch_size]
            yield self.states[selected], self.tokens[selected]


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    datasets = {split: ContractDataset(split=split) for split in ("train", "validation", "test")}
    monkeypatch.setattr(pilot, "data_helper", lambda: SimpleNamespace(CapturedDataset=lambda path, split: datasets[split]))
    monkeypatch.setattr(pilot, "GlimmerTarget", TinyTarget)
    index = tmp_path / "index.json"
    index.write_text("{}")
    return index, datasets


def train_args(index, output, *flags):
    return pilot.parser().parse_args(["train", "--capture", str(index), "--output-dir", str(output), *map(str, flags)])


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def checkpoint(head, variant="shared-ce", depth=1, rank=64):
    return {"model": pilot.MODEL, "revision": pilot.REVISION, "state_transition": pilot.STATE_TRANSITION,
            "variant": variant, "max_depth": depth, "rank": rank, "hidden_size": 12,
            "head": head.state_dict(), "train_manifest": ContractDataset().manifest()}


def test_legacy_defaults_and_new_flags_are_dependency_free():
    config, updates = pilot.training_config(train_args("old.pt", "heads"))
    assert (updates, config["rank"], config["train_depth"], config["ce_weight"], config["kl_weight"]) == (100, 64, 8, 1., 0.)
    assert config["lr"] == 3e-4 and config["weight_decay"] == .01 and config["grad_clip"] == 1.
    assert config["validation_every"] == config["probe_every"] == config["warmup_updates"] == 0
    for command in ("prepare-prompts", "capture-generated", "train", "validate-head", "evaluate"):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), command, "--help"], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-S", "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; assert 'glimmer_mtp_data' not in sys.modules", str(SCRIPT)],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_delegation_uses_agreed_helper_api_lazily(monkeypatch):
    calls = []
    marker = object()
    monkeypatch.setattr(pilot, "GlimmerTarget", lambda attention: marker)
    monkeypatch.setattr(pilot, "data_helper", lambda: SimpleNamespace(
        prepare_prompts=lambda args: calls.append(("prepare", args)),
        capture_generated=lambda args, target: calls.append(("capture", args, target)),
        CapturedDataset=lambda path, split: (path, split)))
    prepare = pilot.parser().parse_args(["prepare-prompts", "--output", "prompts.jsonl"])
    generated = pilot.parser().parse_args(["capture-generated", "--prompts", "prompts.jsonl", "--output-dir", "capture"])
    pilot.prepare_prompts_command(prepare)
    pilot.capture_generated_command(generated)
    assert calls == [("prepare", prepare), ("capture", generated, marker)]
    assert generated.generation_batch_size == 4 and generated.train_token_budget == 3000000
    assert pilot.captured_dataset("capture/index.json", "validation") == (Path("capture/index.json"), "validation")
    with pytest.raises(ValueError, match="unsupported capture layout"):
        pilot.captured_dataset("capture/unknown.json")


def test_fixed_prompt_splits_categories_and_independence():
    validation = pilot.read_prompt_records(VALIDATION, "validation")
    test = pilot.read_prompt_records(TEST, "test")
    legacy = pilot.read_records(pilot.FIXTURES)
    assert len(validation) == 12 and len(test) == 60
    for category in pilot.CATEGORIES:
        assert sum(r["category"] == category for r in validation) == 2
        assert sum(r["category"] == category for r in test) == 10
    pilot.check_heldout(legacy, validation + test)
    pilot.check_heldout(validation, test)
    assert len({r["family"] for r in validation + test}) == 72
    selected = pilot.diagnostic_prompt_ids(test, 6, 45)
    assert {r["category"] for r in test if r["id"] in selected} == pilot.CATEGORIES
    assert selected == pilot.diagnostic_prompt_ids(test, 6, 45)
    assert pilot.diagnostic_prompt_ids(test, 0, 45) == set()
    assert len(pilot.diagnostic_prompt_ids(test, None, 45)) == 60


def test_chat_tokenizes_template_once_without_duplicate_bos():
    calls = []
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)) or [1, 2, 3],
        encode=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chat must not be re-encoded")))
    rows = [{"text": "user instruction", "prompt_format": "chat"}]
    assert pilot.tokenize_prompts(SimpleNamespace(tokenizer=tokenizer), rows)[0]["token_ids"] == [1, 2, 3]
    assert calls == [([{"role": "user", "content": "user instruction"}], {"tokenize": True, "add_generation_prompt": True})]


@pytest.mark.parametrize("update,depth", [(1, 2), (10000, 2), (10001, 4), (20000, 4), (20001, 8), (60000, 8)])
def test_curriculum_edges(update, depth):
    config = {**pilot.TRAIN_DEFAULTS, "curriculum": "2:10000,4:10000,8:40000"}
    assert pilot.active_depth(config, update) == depth
    with pytest.raises(ValueError, match="exceeds curriculum"):
        pilot.active_depth(config, 60001)


@pytest.mark.parametrize("value", ["2", "3:10", "2:0", "4:10,2:10", "2:10,2:10", "2:10,", "x:8"])
def test_invalid_curriculum_refused(value):
    with pytest.raises(ValueError, match="curriculum"):
        pilot.parse_curriculum(value)


def test_warmup_cosine_and_first_stage_fixed_horizon():
    config, updates = pilot.training_config(train_args("index.json", "head", "--updates", 20000,
        "--schedule-updates", 50000, "--warmup-updates", 1000, "--rank", 128, "--train-depth", 1))
    assert updates == 20000
    assert pilot.learning_rate(config, 1) == pytest.approx(3e-7)
    assert pilot.learning_rate(config, 1000) == 3e-4
    assert 0 < pilot.learning_rate(config, 20000) < 3e-4
    assert pilot.learning_rate(config, 50000) == 0
    ckpt = {"training_format": pilot.TRAINING_FORMAT, "training_config": config, "planned_updates": updates}
    resumed, updates = pilot.training_config(train_args("index.json", "head", "--updates", 50000), ckpt)
    assert updates == 50000 and resumed == config
    with pytest.raises(ValueError, match="fixed LR horizon"):
        pilot.training_config(train_args("index.json", "head", "--updates", 50001), ckpt)
    with pytest.raises(ValueError, match="incompatible resume override.*rank"):
        pilot.training_config(train_args("index.json", "head", "--rank", 64), ckpt)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_active_depth_loss_renormalizes_ce_kl_and_detaches_teacher(depth):
    target = TinyTarget()
    head = pilot.make_head(12, 64, True)
    data = ContractDataset()
    states = data.states[:2].clone().requires_grad_()
    loss, metrics = pilot.training_loss(head, states, data.tokens[:2], target,
        state_weight=.2, kl_weight=1., ce_weight=.25, temperature=2., depth=depth)
    expected = sum(w * (.25 * m["ce"] + m["teacher_kl"] + .2 * (m["normalized_mse"] + m["cosine_distance"]))
                   for w, m in zip(pilot.WEIGHTS[:depth], metrics)) / sum(pilot.WEIGHTS[:depth])
    assert float(loss.detach()) == pytest.approx(expected)
    state = head.step(states[:, 0].detach(), target.embedding(data.tokens[:2, 1]), 1, target.norm)
    predicted = pilot.project(target.lm_head, state, target.config).float()
    teacher = pilot.project(target.lm_head, states[:2, 1].detach(), target.config).float()
    assert metrics[0]["ce"] == pytest.approx(float(F.cross_entropy(predicted, data.tokens[:2, 2]).detach()))
    assert metrics[0]["teacher_kl"] == pytest.approx(float(F.kl_div(
        F.log_softmax(predicted / 2., -1), F.softmax(teacher / 2., -1), reduction="batchmean").detach() * 4))
    loss.backward()
    assert states.grad is None
    target.assert_frozen()
    assert len(metrics) == depth


def test_offline_validation_is_unique_independent_and_rng_neutral():
    target = TinyTarget()
    head = pilot.make_head(12, 64, True)
    data = ContractDataset(split="validation")
    before = pilot.rng_state()
    first = pilot.offline_validation(target, head, data, [1, 2, 4, 8], 1024, 5, 19)
    assert_nested_equal(before, pilot.rng_state())
    torch.rand(50)
    random.random()
    second = pilot.offline_validation(target, head, data, [1, 2, 4, 8], 1024, 5, 19)
    assert first["per_depth"] == second["per_depth"]
    assert first["root_count"] == first["available_unique_roots"] == 17
    assert all(entry[2] == 17 for entry in data.iterated)
    assert head.training
    for row in first["per_depth"]:
        assert 0 <= row["teacher_argmax_agreement"] <= row["teacher_top5_recall"] <= 1
        assert row["sequence_ce"] > 0 and row["teacher_kl"] > 0
        assert row["predicted_rms"] > 0 and row["teacher_rms"] > 0
    target.assert_frozen()


@pytest.mark.parametrize("curriculum", [None, "2:3,4:3,8:2"])
def test_public_train_resume_matches_uninterrupted_cpu_exact(runtime, tmp_path, curriculum):
    index, datasets = runtime
    common = ["--variants", "shared-state", "--rank", "128", "--batch-size", "3", "--schedule-updates", "8",
              "--warmup-updates", "3", "--ce-weight", ".25", "--kl-weight", "1", "--checkpoint-every", "1",
              "--validation-every", "2", "--validation-roots", "9", "--validation-batch-size", "4"]
    common += ["--curriculum", curriculum] if curriculum else ["--train-depth", "1"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    pilot.train_command(train_args(index, full, *common, "--updates", 8))
    expected_order = datasets["train"].seen[:]
    datasets["train"].seen.clear()
    pilot.train_command(train_args(index, resumed, *common, "--updates", 2))
    saved = load(resumed / "checkpoint-last.pt")
    assert saved["training_config"]["schedule_updates"] == 8
    assert "schedule" not in saved and "train_manifest" not in saved
    assert saved["root_exposures"] == 6
    pilot.train_command(train_args(index, resumed, "--resume", resumed / "checkpoint-last.pt", "--updates", 8))
    actual, expected = load(resumed / "checkpoint-last.pt"), load(full / "checkpoint-last.pt")
    for key in ("head", "optimizer", "scheduler", "rng", "sampler", "training_config", "best_score", "best_update"):
        assert_nested_equal(actual[key], expected[key])
    assert datasets["train"].seen == expected_order
    assert actual["root_exposures"] == 24
    assert actual["loss_position_exposures"] == (3 * (3 * 2 + 3 * 4 + 2 * 8) if curriculum else 24)
    assert datasets["test"].iterated == []
    assert actual["objective"] == "sequence CE + forward teacher KL"
    summary = json.loads((resumed / "shared-state.json").read_text())
    assert summary["updates_completed"] == 8 and summary["checkpoint_bytes"] > 0
    assert "optimizer" not in summary and "head" not in summary and "sampler" not in summary


def test_legacy_sampler_matches_historical_random_choice_and_resume():
    data = ContractDataset()
    legacy = object.__new__(pilot.LegacyCapture)
    legacy.records = [{"states": data.states[0], "token_ids": list(range(10))}]
    # batch reads nine states and ten tokens from root zero.
    legacy.roots, legacy.root_count = [(0, 0)], 1
    sampler = legacy.new_sampler(31)
    states, tokens = sampler.next_batch(3)
    restored = legacy.new_sampler(99)
    restored.load_state_dict(sampler.state_dict())
    assert_nested_equal(sampler.next_batch(2), restored.next_batch(2))
    assert states.shape == (3, 9, 12) and tokens.shape == (3, 10)


def test_resume_refuses_override_capture_and_old_checkpoint(runtime, tmp_path):
    index, _ = runtime
    output = tmp_path / "run"
    pilot.train_command(train_args(index, output, "--variants", "shared-ce", "--rank", 128,
        "--train-depth", 1, "--updates", 1, "--schedule-updates", 3))
    last = output / "checkpoint-last.pt"
    for flags in (("--rank", 64), ("--kl-weight", 1), ("--seed", 4), ("--schedule-updates", 5)):
        with pytest.raises(ValueError, match="incompatible resume override"):
            pilot.train_command(train_args(index, output, "--resume", last, "--updates", 2, *flags))
    index.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="resume capture changed"):
        pilot.train_command(train_args(index, output, "--resume", last, "--updates", 2))
    with pytest.raises(ValueError, match="not resumable"):
        pilot.training_config(train_args(index, output), checkpoint(pilot.make_head(12, 64, True)))


@pytest.mark.parametrize("variant", pilot.VARIANTS)
def test_warm_start_copies_one_block_without_sharing_independent_storage(variant):
    source = pilot.make_head(12, 128, True, 1)
    destination = pilot.make_head(12, 128, variant != "fixed-ce", 8)
    ckpt = checkpoint(source, rank=128)
    pilot.initialize_head(destination, ckpt, 128, 12)
    assert len(destination.blocks) == (8 if variant == "fixed-ce" else 1)
    for block in destination.blocks:
        assert_nested_equal(source.blocks[0].state_dict(), block.state_dict())
        assert block.up.weight.data_ptr() != source.blocks[0].up.weight.data_ptr()
    assert len({b.up.weight.data_ptr() for b in destination.blocks}) == len(destination.blocks)
    with pytest.raises(ValueError, match="rank mismatch"):
        pilot.initialize_head(destination, ckpt, 256, 12)
    with pytest.raises(ValueError, match="representation width"):
        pilot.initialize_head(destination, ckpt, 128, 24)
    with pytest.raises(ValueError, match="state-transition"):
        pilot.initialize_head(destination, {**ckpt, "state_transition": "unknown"}, 128, 12)
    with pytest.raises(ValueError, match="one-step"):
        pilot.initialize_head(destination, {**ckpt, "max_depth": 8}, 128, 12)


def test_best_checkpoint_is_validation_only_and_last_is_separate(runtime, tmp_path, monkeypatch):
    index, datasets = runtime
    scores = iter([3., 1., 2.])
    calls = []

    def validate(target, head, dataset, depths, *args):
        assert dataset.split == "validation"
        calls.append(depths)
        return {"selection_score": next(scores), "per_depth": [], "root_count": 17}

    monkeypatch.setattr(pilot, "offline_validation", validate)
    output = tmp_path / "selection"
    pilot.train_command(train_args(index, output, "--variants", "shared-ce", "--train-depth", 1,
        "--updates", 3, "--validation-every", 1))
    assert load(output / "checkpoint-best.pt")["update"] == 2
    assert load(output / "checkpoint-last.pt")["update"] == 3
    assert load(output / "checkpoint-last.pt")["best_score"] == 1
    assert calls == [[1], [1], [1]] and not datasets["test"].iterated


def test_sequential_three_arm_recovery_preserves_completed_and_missing_arms(runtime, tmp_path, monkeypatch):
    index, _ = runtime
    common = ["--rank", "64", "--curriculum", "2:1,4:1", "--batch-size", "2", "--updates", "2",
              "--schedule-updates", "2", "--warmup-updates", "1", "--checkpoint-every", "1"]
    full, resumed = tmp_path / "three-full", tmp_path / "three-resume"
    pilot.train_command(train_args(index, full, *common))
    save = pilot.save_checkpoint

    def interrupt(path, value):
        save(path, value)
        if Path(path).name == "shared-ce-last.pt" and value["update"] == 1:
            raise RuntimeError("injected after durable arm checkpoint")

    monkeypatch.setattr(pilot, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="injected"):
        pilot.train_command(train_args(index, resumed, *common))
    before = (resumed / "fixed-ce-last.pt").read_bytes()
    monkeypatch.setattr(pilot, "save_checkpoint", save)
    pilot.train_command(train_args(index, resumed, "--resume", resumed / "shared-ce-last.pt", "--updates", 2))
    assert before == (resumed / "fixed-ce-last.pt").read_bytes()
    for variant in pilot.VARIANTS:
        actual, expected = load(resumed / f"{variant}-last.pt"), load(full / f"{variant}-last.pt")
        for key in ("head", "optimizer", "sampler", "rng"):
            assert_nested_equal(actual[key], expected[key])


def test_validation_probe_uses_actual_decode_and_counts_only_drafts(monkeypatch):
    target = TinyTarget()
    head = pilot.make_head(12, 64, True, 1)
    rows = pilot.tokenize_prompts(target, pilot.read_prompt_records(VALIDATION, "validation"))
    before = pilot.rng_state()
    result = pilot.validation_probe(target, head, rows, 8)
    assert_nested_equal(before, pilot.rng_state())
    assert len(result["pairs"]) == 12 and len(result["categories"]) == 6
    assert result["fidelity"]["exact_token_identity"]
    assert result["accepted_drafts"] == sum(p["candidate"]["accepted_drafts"] for p in result["pairs"])
    for pair in result["pairs"]:
        run = pair["candidate"]
        assert run["accepted_drafts"] == sum(p["accepted_drafts"] for p in run["passes"])
        assert run["accepted_drafts"] < run["generated_tokens"]
        assert run["verification_passes"] > 0
    assert all(c["prompts"] == 2 and c["proposed_drafts"] > 0 for c in result["categories"])


def test_validate_head_public_dispatch_is_validation_only(runtime, tmp_path, monkeypatch):
    index, datasets = runtime
    head_path = tmp_path / "head.pt"
    torch.save(checkpoint(pilot.make_head(12, 64, True, 1)), head_path)
    output = tmp_path / "validation.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "validate-head", "--capture", str(index), "--head", str(head_path),
        "--output", str(output), "--validation-roots", "8", "--max-new-tokens", "8"])
    pilot.main()
    result = json.loads(output.read_text())
    assert result["offline"]["root_count"] == 8 and len(result["probe"]["pairs"]) == 12
    assert datasets["validation"].iterated and not datasets["train"].iterated and not datasets["test"].iterated
    with pytest.raises(SystemExit):
        pilot.parser().parse_args(["validate-head", "--capture", str(index), "--head", str(head_path),
                                  "--output", str(output), "--split", "test"])


def test_overfit_diagnostic_is_explicit_bounded_and_not_validation(runtime, tmp_path):
    index, _ = runtime
    with pytest.raises(ValueError, match="teacher-argmax diagnostic"):
        pilot.training_config(train_args(index, tmp_path / "bad", "--teacher-argmax-diagnostic"))
    output = tmp_path / "overfit"
    pilot.train_command(train_args(index, output, "--variants", "shared-ce", "--train-depth", 1,
        "--overfit-roots", 8, "--teacher-argmax-diagnostic", "--updates", 1))
    saved = load(output / "checkpoint-last.pt")
    assert saved["objective"] == "teacher-argmax diagnostic" and saved["training_sample_unique_roots"] == 8
    assert saved["validation"] is None and saved["best_score"] is None
    assert not (output / "checkpoint-best.pt").exists()
    events = [json.loads(line) for line in (output / "shared-ce-training.jsonl").read_text().splitlines()]
    assert events[-1]["training_set_diagnostic_NOT_validation"]["root_count"] == 8


def test_external_evaluation_matrix_randomized_with_only_six_untimed_replays(runtime, tmp_path, monkeypatch):
    index, _ = runtime
    heads = []
    for variant in pilot.VARIANTS:
        path = tmp_path / f"{variant}.pt"
        torch.save(checkpoint(pilot.make_head(12, 64, variant != "fixed-ce"), variant, 8), path)
        heads.append(str(path))
    calls, replays, reports = [], [], []

    def decode(target, prompt, count, depth=0, drafter=None, diagnostics=False):
        calls.append((depth, diagnostics))
        return {"token_ids": [7, 8], "decode_s": 1., "accepted_drafts": int(depth > 0),
                "verification_passes": 1, "mean_accepted_drafts_per_pass": float(depth > 0),
                "passes": [{"cache_before": len(prompt), "drafts": [8] * depth,
                            "accepted_drafts": int(depth > 0)}], "drift": []}

    monkeypatch.setattr(pilot, "decode", decode)
    monkeypatch.setattr(pilot, "chosen_path_drift", lambda *args, **kwargs: replays.append(args[-2]) or [])
    monkeypatch.setattr(pilot, "json_write", lambda path, value: reports.append(copy.deepcopy(value)) if value["status"] != "running" else None)
    args = pilot.parser().parse_args(["evaluate", "--capture", str(index), "--heads", *heads,
        "--output", str(tmp_path / "evaluate.json"), "--eval-data", str(TEST), "--diagnostic-prompts", "6",
        "--max-new-tokens", "128", "--repeats", "2", "--eval-seed", "90"])
    pilot.evaluate_command(args)
    report = reports[-1]
    assert report["status"] == "complete" and len(report["pairs"]) == 3 * 4 * 60 * 2
    assert len(replays) == sum(diagnostic for _, diagnostic in calls) == 3 * 4 * 6
    assert report["randomized_interleaved"] and report["order_seed"] == 90
    assert len(report["diagnostic_prompt_ids"]) == 6
    assert len({p["variant"] for p in report["pairs"][:20]}) > 1
    assert {tuple(p["order"]) for p in report["pairs"]} == {("baseline", "candidate"), ("candidate", "baseline")}
    assert len(report["summary"]) == 3 * 4 * 7
    assert all(s["exact_output_pairs"] == s["pairs"] for s in report["summary"])
    first_order = [(p["variant"], p["depth"], p["repeat"], p["id"], p["order"]) for p in report["pairs"]]
    pilot.evaluate_command(args)
    assert first_order == [(p["variant"], p["depth"], p["repeat"], p["id"], p["order"]) for p in reports[-1]["pairs"]]


def test_external_evaluation_rejects_optimizer_leakage_before_loading_target(runtime, tmp_path, monkeypatch):
    index, _ = runtime
    source = checkpoint(pilot.make_head(12, 64, True, 1))
    source["train_manifest"] = pilot.read_prompt_records(TEST, "test")[:1]
    path = tmp_path / "leaky.pt"
    torch.save(source, path)
    monkeypatch.setattr(pilot, "GlimmerTarget", lambda *a: (_ for _ in ()).throw(AssertionError("should refuse before model load")))
    args = pilot.parser().parse_args(["evaluate", "--capture", str(index), "--heads", str(path),
        "--output", str(tmp_path / "eval.json"), "--eval-data", str(TEST)])
    with pytest.raises(ValueError, match="checkpoint training"):
        pilot.evaluate_command(args)


def test_common_warm_start_then_resume_needs_no_source_checkpoint(runtime, tmp_path):
    index, _ = runtime
    source = pilot.make_head(12, 128, True, 1)
    init_path = tmp_path / "step1-best.pt"
    torch.save(checkpoint(source, rank=128), init_path)
    output = tmp_path / "warm"
    pilot.train_command(train_args(index, output, "--variants", "fixed-ce", "--rank", 128,
        "--init-head", init_path, "--updates", 1, "--schedule-updates", 4, "--curriculum", "2:1,4:1,8:2"))
    saved = load(output / "checkpoint-last.pt")
    assert saved["rank"] == 128 and saved["head_parameters"] == 8 * (3 * 12 * 128 + 12)
    assert saved["init_head_identity"]["path"] == str(init_path)
    init_path.unlink()  # Same-arm recovery uses the saved trained head, never reinitializes.
    pilot.train_command(train_args(index, output, "--resume", output / "checkpoint-last.pt", "--updates", 4))
    assert load(output / "checkpoint-last.pt")["update"] == 4


def test_warm_start_rejects_different_corpus_and_rank(runtime, tmp_path):
    index, _ = runtime
    source = checkpoint(pilot.make_head(12, 64, True, 1))
    source["train_manifest"] = ContractDataset(split="test").manifest()
    init_path = tmp_path / "different.pt"
    torch.save(source, init_path)
    with pytest.raises(ValueError, match="same training corpus"):
        pilot.train_command(train_args(index, tmp_path / "bad", "--init-head", init_path, "--updates", 1))
    source["train_manifest"] = ContractDataset().manifest()
    torch.save(source, init_path)
    with pytest.raises(ValueError, match="rank mismatch"):
        pilot.train_command(train_args(index, tmp_path / "bad-rank", "--init-head", init_path, "--rank", 128, "--updates", 1))


def test_last_checkpoint_not_best_or_alias_is_required_for_resume(runtime, tmp_path):
    index, _ = runtime
    output = tmp_path / "run"
    pilot.train_command(train_args(index, output, "--variants", "shared-ce", "--train-depth", 1,
        "--updates", 1, "--validation-every", 1))
    for filename in ("checkpoint-best.pt", "shared-ce.pt"):
        with pytest.raises(ValueError, match="current last checkpoint"):
            pilot.train_command(train_args(index, output, "--resume", output / filename, "--updates", 2))


def test_rank_256_capacity_and_checkpoint_roundtrip(runtime, tmp_path):
    index, _ = runtime
    output = tmp_path / "capacity"
    pilot.train_command(train_args(index, output, "--variants", "shared-ce", "--rank", 256,
        "--train-depth", 1, "--updates", 1))
    saved = pilot.load_head_checkpoint(output / "checkpoint-last.pt")
    assert saved["head_parameters"] == 3 * 12 * 256 + 12
    restored = pilot.head_from_checkpoint(saved, TinyTarget())
    assert_nested_equal(saved["head"], restored.state_dict())
    with pytest.raises(RuntimeError, match="size mismatch"):
        pilot.head_from_checkpoint({**saved, "rank": 128}, TinyTarget())


def test_chosen_path_projection_metrics_are_untimed_same_prefix():
    target, head = TinyTarget(), pilot.make_head(12, 64, True)
    prompt = [4, 2, 7]
    with torch.inference_mode():
        run = pilot.decode(target, prompt, 12, 4, pilot.Drafter(head, target))
        rows = pilot.chosen_path_drift(target, head, prompt, run["token_ids"], 4, [0, 3], distribution_metrics=True)
    assert len(rows) == 8
    assert all(0 <= r["teacher_argmax_agreement"] <= r["teacher_top5_recall"] <= 1 for r in rows)
    assert all(r["teacher_kl"] >= -1e-6 and r["teacher_rms"] > 0 for r in rows)
    summary = pilot.summarize_drift(rows)
    assert len(summary["chosen_token_prefix"]) == 4
    assert all("teacher_kl" in row and "state_mse" in row for row in summary["chosen_token_prefix"])


def test_conditional_acceptance_uses_live_accepted_prefix_not_all_proposals():
    pairs = [{"candidate": {"passes": [
        {"drafts": [1, 2, 3, 4], "accepted_drafts": 0},
        {"drafts": [1, 2, 3, 4], "accepted_drafts": 2},
        {"drafts": [1, 2, 3, 4], "accepted_drafts": 4},
    ]}}]
    rows = pilot.conditional_acceptance(pairs, 4)
    assert [r["eligible_accepted_prefixes"] for r in rows] == [3, 2, 2, 1]
    assert [r["accepted_drafts"] for r in rows] == [2, 2, 1, 1]
    assert [r["conditional_acceptance_rate"] for r in rows] == [2 / 3, 1., .5, 1.]


def test_legacy_evaluate_defaults_keep_all_replays_and_alternating_order(tmp_path, monkeypatch):
    rows = pilot.read_records(pilot.FIXTURES)
    tokenizer = TinyTokenizer()
    rows = [{**row, "token_ids": tokenizer.encode(row["text"])} for row in rows]
    # Preserve real fixture IDs/text/families; avoid fake-tokenizer collisions in the split check.
    for i, row in enumerate(rows):
        row["token_ids"] = [i % 30 + 1, i // 30 + 1]
    monkeypatch.setattr(pilot, "load_capture", lambda path: {"records": rows})
    monkeypatch.setattr(pilot, "GlimmerTarget", TinyTarget)
    source = checkpoint(pilot.make_head(12, 64, True, 1))
    source["train_manifest"] = [r for r in rows if r["split"] == "train"]
    path = tmp_path / "legacy-head.pt"
    torch.save(source, path)
    output = tmp_path / "legacy-eval.json"
    args = pilot.parser().parse_args(["evaluate", "--capture", "old.pt", "--heads", str(path), "--output", str(output)])
    pilot.evaluate_command(args)  # Actual decoder on the unit target; no mock timed path.
    result = json.loads(output.read_text())
    assert not result["randomized_interleaved"] and len(result["pairs"]) == 12
    assert result["fidelity"]["exact_token_identity"]
    assert len(result["diagnostic_prompt_ids"]) == 12
    for i, pair in enumerate(result["pairs"]):
        assert pair["order"] == (["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"])
        assert "chosen_path_drift_raw" in pair


def test_capture_wide_metadata_is_filtered_to_actual_training_and_validation_splits(runtime, tmp_path, monkeypatch):
    index, datasets = runtime
    catalog = [r for data in datasets.values() for r in data.manifest()]
    for data in datasets.values():
        monkeypatch.setattr(data, "manifest", lambda: catalog)
    output = tmp_path / "all-metadata"
    pilot.train_command(train_args(index, output, "--variants", "shared-ce", "--train-depth", 1,
        "--updates", 1, "--validation-every", 1))
    saved = load(output / "checkpoint-last.pt")
    manifest = pilot.checkpoint_manifest(saved)
    assert len(manifest) == 1 and manifest[0]["split"] == "train"
    assert saved["validation"]["root_count"] == datasets["validation"].root_count
    assert not datasets["test"].iterated


@pytest.mark.parametrize("record", [False, True])
def test_evaluate_retains_divergence_without_relaxing_verification(runtime, tmp_path, monkeypatch, record):
    index, _ = runtime
    path = tmp_path / "head.pt"
    torch.save(checkpoint(pilot.make_head(12, 64, True, 1)), path)
    reports = []

    def decode(target, prompt, count, depth=0, drafter=None, diagnostics=False):
        return {"token_ids": [7, 9] if depth else [7, 8], "decode_s": 1., "accepted_drafts": 0,
                "verification_passes": 1, "mean_accepted_drafts_per_pass": 0.,
                "passes": [{"drafts": [9] if depth else [], "accepted_drafts": 0}], "drift": []}

    monkeypatch.setattr(pilot, "decode", decode)
    monkeypatch.setattr(pilot, "json_write", lambda path, value: reports.append(copy.deepcopy(value)) if value["status"] != "running" else None)
    args = pilot.parser().parse_args(["evaluate", "--capture", str(index), "--heads", str(path),
        "--output", str(tmp_path / "out.json"), "--eval-data", str(TEST), "--diagnostic-prompts", "0",
        *(["--record-divergence"] if record else [])])
    if record:
        pilot.evaluate_command(args)
        assert reports[-1]["status"] == "complete" and len(reports[-1]["pairs"]) == 60
        assert reports[-1]["fidelity"]["exact_match_rate"] == 0.
        assert all(row["median_exact_output_decode_speedup"] is None for row in reports[-1]["summary"])
    else:
        with pytest.raises(ValueError, match="greedy identity failed"):
            pilot.evaluate_command(args)
        assert reports[-1]["status"] == "failed" and len(reports[-1]["pairs"]) == 1
    assert all(not pair["exact_token_identity"] for pair in reports[-1]["pairs"])
    assert all(pair["candidate"]["accepted_drafts"] == 0 for pair in reports[-1]["pairs"])
