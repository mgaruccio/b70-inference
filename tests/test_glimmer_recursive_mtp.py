"""CPU contract tests, NOT a substitute for the official 30B GPU journey."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, PretrainedConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/glimmer_recursive_mtp.py"
spec = importlib.util.spec_from_file_location("glimmer_recursive_mtp", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_dependency_free_cli_and_fixture_split():
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "validate"],
                            capture_output=True, text=True, check=True)
    assert '"train": 24' in result.stdout
    assert '"eval": 12' in result.stdout
    assert len(pilot.read_records(pilot.FIXTURES)) == 36


@pytest.mark.parametrize("field,value,match", [
    ("family", "train-family", "family"),
    ("text", " training   text ", "sequence text"),
    ("token_ids", [1, 2], "tokenized sequence"),
    ("id", "train", "sequence id"),
])
def test_overlap_rejected(field, value, match):
    rows = [dict(id="train", family="train-family", split="train", text="training text", token_ids=[1, 2]),
            dict(id="eval", family="eval-family", split="eval", text="held out", token_ids=[3, 4], category="code")]
    rows[1][field] = value
    with pytest.raises(ValueError, match=match):
        pilot.validate_records(rows)


def test_checkpoint_overlap_and_context_refusals():
    training = [dict(family="x", text="hello world", token_ids=[1, 2])]
    with pytest.raises(ValueError, match="training family"):
        pilot.check_heldout(training, [dict(family="x", text="different", token_ids=[3])])
    with pytest.raises(ValueError, match="training text"):
        pilot.check_heldout(training, [dict(family="y", text="hello  world", token_ids=[3])])
    with pytest.raises(ValueError, match="training tokens"):
        pilot.check_heldout(training, [dict(family="y", text="different", token_ids=[1, 2])])
    pilot.context_guard(1600, 128, 8)
    with pytest.raises(ValueError, match="SWA"):
        pilot.context_guard(1700, 128, 8)
    with pytest.raises(ValueError, match="unsupported depth"):
        pilot.context_guard(10, 10, 3)


def cache_config(window=2048):
    return PretrainedConfig(num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"],
                            sliding_window=window)


def update_cache(cache, tokens):
    values = torch.tensor(tokens, dtype=torch.float32).reshape(1, 1, -1, 1)
    for i in range(2):
        cache.update(values.clone(), values.clone(), i)


def test_real_hybrid_dynamic_cache_partial_prefix_rollback():
    config = cache_config()
    cache = DynamicCache(config=config)
    update_cache(cache, [11, 12, 13, 14, 15])
    update_cache(cache, [16, 17, 99, 98])
    pilot.crop_cache(cache, 7, config.layer_types)
    for layer in cache.layers:
        assert layer.keys.flatten().tolist() == [11, 12, 13, 14, 15, 16, 17]
    update_cache(cache, [18, 19])
    pilot.check_cache(cache, 9, config.layer_types)
    for layer in cache.layers:
        assert layer.keys.flatten().tolist() == list(range(11, 20))
    with pytest.raises(ValueError, match="length mismatch"):
        pilot.check_cache(cache, 8, config.layer_types)
    with pytest.raises(ValueError, match="ordinary"):
        pilot.check_cache(object(), 9, config.layer_types)


def test_saturated_or_unknown_cache_refused():
    config = cache_config(window=8)
    cache = DynamicCache(config=config)
    update_cache(cache, list(range(8)))
    with pytest.raises(ValueError, match="saturated"):
        pilot.crop_cache(cache, 5, config.layer_types, window=8)
    config = cache_config()
    cache = DynamicCache(config=config)
    update_cache(cache, [1, 2])
    with pytest.raises(ValueError, match="layer type"):
        pilot.check_cache(cache, 2, ["full_attention", "full_attention"])


# A prefix-dependent CPU oracle exercises the production decode control flow with
# real HF hybrid KV rollback. It is explicitly not a model/performance benchmark.
def advance(state, token):
    checksum, length, _ = state.tolist()
    return torch.tensor([(int(checksum) * 13 + token) % 97, length + 1, 1.])


def next_token(state):
    return (int(state[0]) * 7 + int(state[1]) * 11) % 97 + 1


def prefix_state(tokens):
    state = torch.tensor([0., 0., 1.])
    for token in tokens:
        state = advance(state, token)
    return state


def reference(prompt, budget, eos=()):
    state = prefix_state(prompt)
    result = []
    for _ in range(budget):
        token = next_token(state)
        result.append(token)
        if token in eos:
            break
        state = advance(state, token)
    return result


class CacheOracle:
    def __init__(self, eos=()):
        self.eos = set(eos)
        self.config = cache_config()
        self.prefills = 0
        self.blocks = []
        self.rollbacks = []

    def sync(self):
        pass

    def reset_peak(self):
        pass

    def memory(self):
        return {"peak_allocated_bytes": 0}

    def prefill(self, prompt):
        self.prefills += 1
        self.cache = DynamicCache(config=self.config)
        update_cache(self.cache, prompt)
        state = prefix_state(prompt)
        return state, next_token(state), self.cache

    def verify(self, tokens, cache):
        self.blocks.append(tokens)
        prefix = [int(t) for t in cache.layers[0].keys.flatten().tolist()]
        state = prefix_state(prefix)
        states, logits_argmax = [], []
        for token in tokens:
            state = advance(state, token)
            states.append(state)
            logits_argmax.append(next_token(state))
        update_cache(cache, tokens)
        return torch.stack(states), logits_argmax, cache

    def crop(self, cache, keep):
        self.rollbacks.append((cache.get_seq_length(), keep))
        pilot.crop_cache(cache, keep, self.config.layer_types)

    @staticmethod
    def cache_length(cache):
        return cache.get_seq_length()


class OracleDrafter:
    def __init__(self, fault=None):
        self.fault = fault
        self.roots = []

    def draft(self, root, anchor, depth, eos):
        self.roots.append(root.clone())
        token, state = anchor, root
        drafts, states = [], []
        for i in range(depth):
            state = advance(state, token)
            states.append(state)
            token = next_token(state)
            if i == self.fault:
                token = token % 97 + 1
            drafts.append(token)
            if token in eos:
                break
        return drafts, states


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
@pytest.mark.parametrize("fault", [None, 0, 1, 3])
@pytest.mark.parametrize("budget", [1, 2, 5, 17])
def test_greedy_identity_zero_full_first_middle_rejection_and_budget(depth, fault, budget):
    prompt = [9, 14, 27, 6]
    target, drafter = CacheOracle(), OracleDrafter(fault)
    run = pilot.decode(target, prompt, budget, depth, drafter)
    baseline = pilot.decode(CacheOracle(), prompt, budget)
    assert run["token_ids"] == baseline["token_ids"] == reference(prompt, budget)
    assert run["generated_tokens"] == budget
    assert target.prefills == 1
    assert all(1 <= len(block) <= depth + 1 for block in target.blocks)
    assert run["target_calls_total"] == 1 + len(target.blocks)
    assert run["proposed_drafts"] == sum(run["proposed_by_depth"])
    assert run["accepted_drafts"] == sum(run["accepted_by_depth"])
    committed = list(prompt)
    for index, record in enumerate(run["passes"]):
        assert torch.equal(drafter.roots[index], prefix_state(committed))
        assert record["cache_before"] == len(committed)
        committed += [record["seed"], *record["drafts"][:record["accepted_drafts"]]]
        assert record["cache_after"] == len(committed)
    if fault == 0:
        assert run["accepted_drafts"] == 0
        assert all(after == before - len(block) + 1
                   for (before, after), block in zip(target.rollbacks, target.blocks))
    if fault is None:
        assert run["accepted_drafts"] == run["proposed_drafts"]


@pytest.mark.parametrize("position,fault", [(0, None), (1, None), (3, None), (3, 2), (5, 0)])
def test_eos_anchor_accepted_draft_or_corrected_mismatch(position, fault):
    prompt = [7, 11, 19]
    eos = {reference(prompt, 12)[position]}
    target = CacheOracle(eos)
    run = pilot.decode(target, prompt, 12, 8, OracleDrafter(fault))
    assert run["token_ids"] == reference(prompt, 12, eos)
    assert run["token_ids"][-1] in eos
    assert len(run["token_ids"]) <= position + 1


def test_alignment_bonus_and_diagnostic_prefix_separation():
    prompt = [1, 7, 3]
    run = pilot.decode(CacheOracle(), prompt, 10, 4, OracleDrafter(1), diagnostics=True)
    first = run["passes"][0]
    assert first["accepted_drafts"] == 1
    assert first["next_correct_token"] == reference(prompt, 3)[2]
    assert [r["chosen_prefix"] for r in run["drift"][:4]] == [True, True, False, False]
    assert all(r["normalized_mse"] == 0 for r in run["drift"])
    summary = pilot.summarize_drift(run["drift"])
    assert summary["chosen_token_prefix"][0]["depth"] == 1
    assert summary["conditional_after_rejection"][0]["depth"] == 3
    with pytest.raises(ValueError, match="anchor"):
        pilot.accepted_prefix([1, 2], [1, 2])


def small_frozen_target():
    torch.manual_seed(7)
    target = torch.nn.Module()
    target.embedding = torch.nn.Embedding(31, 12)
    target.norm = torch.nn.RMSNorm(12, eps=1e-5)
    target.lm_head = torch.nn.Linear(12, 31, bias=False)
    target.config = SimpleNamespace(output_multiplier=.19611613513818404, final_logit_softcapping=20.)
    return target.eval().requires_grad_(False)


@pytest.mark.parametrize("shared,state_weight", [(False, 0.), (True, 0.), (True, .2)])
def test_full_depth_losses_backpropagate_only_into_head(shared, state_weight):
    target = small_frozen_target()
    head = pilot.make_head(12, 4, shared)
    states = target.norm(torch.randn(2, 9, 12)).detach()
    tokens = torch.randint(0, 31, (2, 10))
    before = {k: v.clone() for k, v in target.state_dict().items()}
    loss, metrics = pilot.training_loss(head, states, tokens, target, state_weight, kl_weight=.01)
    loss.backward()
    assert [m["depth"] for m in metrics] == list(range(1, 9))
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert all(p.grad is None and not p.requires_grad for p in target.parameters())
    torch.optim.AdamW(head.parameters(), lr=.001).step()
    assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())
    assert len(head.blocks) == (1 if shared else 8)
    assert all("embedding" not in k and "lm_head" not in k and "norm" not in k for k in head.state_dict())
    f = io.BytesIO()
    torch.save({"head": head.state_dict(), "variant": "shared-ce"}, f)
    f.seek(0)
    assert set(torch.load(f, weights_only=True)["head"]) == set(head.state_dict())


def test_token_shift_hidden_feedback_and_explicit_depth_weighting():
    target = small_frozen_target()
    head = pilot.make_head(12, 4, True)
    states = target.norm(torch.randn(1, 9, 12)).detach()
    tokens = torch.arange(10).reshape(1, 10)
    seen_tokens, head_inputs, head_outputs = [], [], []
    target.embedding.register_forward_pre_hook(lambda module, args: seen_tokens.append(args[0].clone()))
    head.blocks[0].register_forward_pre_hook(lambda module, args: head_inputs.append(args[0].clone()))
    head.blocks[0].register_forward_hook(lambda module, args, output: head_outputs.append(output.clone()))
    loss, metrics = pilot.training_loss(head, states, tokens, target)
    assert [t.item() for t in seen_tokens] == list(range(1, 9))
    assert torch.equal(head_inputs[0], states[:, 0])
    assert all(torch.equal(head_inputs[d], head_outputs[d - 1]) for d in range(1, 8))
    for d in range(1, 9):
        logits = pilot.project(target.lm_head, head_outputs[d - 1], target.config)
        expected = torch.nn.functional.cross_entropy(logits, tokens[:, d + 1])
        assert metrics[d - 1]["ce"] == pytest.approx(float(expected.detach()))
    assert float(loss.detach()) == pytest.approx(sum(w * m["ce"] for w, m in zip(pilot.WEIGHTS, metrics)) / sum(pilot.WEIGHTS))
    changed = states.clone()
    changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 100
    other, other_metrics = pilot.training_loss(head, changed, tokens, target)
    assert torch.equal(loss, other)  # Teacher states after root affect no CE-only prediction.
    assert [m["ce"] for m in metrics] == [m["ce"] for m in other_metrics]


def test_parameter_budget_and_scaled_softcap_projection():
    shared = pilot.make_head(12, 4, True)
    fixed = pilot.make_head(12, 4, False)
    count = lambda h: sum(p.numel() for p in h.parameters())
    assert count(shared) == 3 * 12 * 4 + 12
    assert count(fixed) == 8 * count(shared)
    target = small_frozen_target()
    states = torch.randn(2, 12)
    raw = target.lm_head(states)
    assert torch.equal(pilot.project(target.lm_head, states, target.config),
                       torch.tanh(raw * target.config.output_multiplier / 20.) * 20.)


def test_real_drafter_consumes_selected_token_and_previous_predicted_state():
    target = small_frozen_target()
    target.torch, target.device = torch, torch.device("cpu")
    head = pilot.make_head(12, 4, True)
    seen_tokens, inputs, outputs = [], [], []
    target.embedding.register_forward_pre_hook(lambda module, args: seen_tokens.append(int(args[0])))
    head.blocks[0].register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
    head.blocks[0].register_forward_hook(lambda module, args, output: outputs.append(output.clone()))
    root = target.norm(torch.randn(12))
    with torch.no_grad():
        drafts, states = pilot.Drafter(head, target).draft(root, 17, 8, set())
    assert seen_tokens == [17, *drafts[:-1]]
    assert torch.equal(inputs[0], root)
    assert all(torch.equal(inputs[i], states[i - 1]) for i in range(1, 8))


def test_chosen_path_drift_uses_genuine_same_prefix_states_at_every_depth():
    prompt = [8, 2, 9]
    generated = reference(prompt, 12)
    calls, consumed = [], []

    def forward(tokens, cache, logits_to_keep):
        calls.append(tokens)
        assert cache is None and logits_to_keep == 1
        return torch.stack([prefix_state(tokens[:i + 1]) for i in range(len(tokens))]), None, None

    class Head:
        def step(self, state, embedding, depth, norm):
            consumed.append(int(embedding))
            return advance(state, int(embedding))

    target = SimpleNamespace(forward=forward, torch=torch, device="cpu", embedding=lambda token: token, norm=None)
    rows = pilot.chosen_path_drift(target, Head(), prompt, generated, 8, [0, 3])
    assert calls == [[*prompt, *generated]]
    assert consumed == generated[:8] + generated[3:11]
    assert [r["depth"] for r in rows] == list(range(1, 9)) * 2
    assert all(r["normalized_mse"] == 0 and r["chosen_prefix"] for r in rows)


def test_rejected_eos_draft_does_not_end_generation():
    class EosDrafter:
        def draft(self, root, anchor, depth, eos):
            return [100], [advance(root, anchor)]

    prompt = [3, 5, 7]
    run = pilot.decode(CacheOracle(eos={100}), prompt, 17, 8, EosDrafter())
    assert run["token_ids"] == reference(prompt, 17)
    assert run["accepted_drafts"] == 0
    assert run["proposed_drafts"] == 16


@pytest.mark.parametrize("reference,candidate,first", [
    ([1, 2], [1, 2], None), ([1, 2], [1, 3], 1), ([1, 2], [1], 1),
    ([1], [1, 2], 1), ([], [1], 0), ([1], [], 0), ([], [], None),
])
def test_token_fidelity_includes_length_only_divergence(reference, candidate, first):
    result = pilot.token_fidelity(reference, candidate)
    assert result["exact_token_identity"] == (reference == candidate)
    assert result["first_divergence_position"] == first
    assert result["divergence_position_base"] == 0
    assert result["generated_length_difference"] == len(candidate) - len(reference)


def test_divergence_recording_is_explicit_and_summary_never_claims_identity():
    pair = {"baseline": {"token_ids": [1, 2]}, "candidate": {"token_ids": [1, 3, 4]}}
    with pytest.raises(ValueError, match="greedy identity failed"):
        pilot.check_pair_fidelity(pair)
    pilot.check_pair_fidelity(pair, record_divergence=True)
    assert pair["exact_token_identity"] is False
    same = {"baseline": {"token_ids": [1, 2]}, "candidate": {"token_ids": [1, 2]}}
    pilot.check_pair_fidelity(same)
    summary = pilot.fidelity_summary([pair, same])
    assert summary == {"exact_token_identity": False, "exact_match_rate": .5,
                       "diverged_pairs": 1, "median_first_divergence_position": 1,
                       "max_absolute_generated_length_difference": 1}


def test_evaluate_parser_keeps_strict_default_and_opt_in_recording():
    argv = ["evaluate", "--capture", "capture.pt", "--heads", "head.pt", "--output", "out.json"]
    assert pilot.parser().parse_args(argv).record_divergence is False
    assert pilot.parser().parse_args([*argv, "--record-divergence"]).record_divergence is True


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_update_preserves_post_normalized_state_with_signed_norm(shared, dtype):
    head = pilot.make_head(12, 4, shared)
    signed_scale = torch.tensor([1., -1., 2., -2.] * 3)
    raw = torch.randn(2, 12)
    state = (raw * (raw.square().mean(-1, keepdim=True) + 1e-5).rsqrt() * signed_scale).to(dtype)
    embedding = torch.randn_like(state)
    with torch.no_grad():
        for block in head.blocks:
            block.up.weight.zero_()
        def forbidden_norm(value):
            raise AssertionError("the trunk final norm must not be reapplied")
        for depth in range(1, 9):
            state_after = head.step(state, embedding, depth, forbidden_norm)
            assert torch.equal(state_after, state)
            assert state_after.dtype == dtype
            state = state_after


@pytest.mark.parametrize("tag", [None, "repeated-target-norm", "unknown"])
def test_old_or_unknown_state_transition_checkpoint_is_refused(tag):
    with pytest.raises(ValueError, match="state-transition mismatch"):
        pilot.check_head_transition({"state_transition": tag})
    pilot.check_head_transition({"state_transition": pilot.STATE_TRANSITION})
