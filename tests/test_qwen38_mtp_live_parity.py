"""CPU-only diagnostic logic tests. Synthetic fixtures are NOT live parity evidence.

Run in the existing external ML runtime:
  python -m unittest discover -s tests -p test_qwen38_mtp_live_parity.py -v
No GPU allocation, network requests, model downloads or vLLM server are used.
"""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = load("scripts/experiments/qwen38_mtp_live_parity.py", "parity_test")
hook = load("patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/qwen38_mtp_parity.py", "hook_test")
try:
    import torch
except ImportError:
    torch = None


class Guards(unittest.TestCase):
    def test_control_requires_selected_public_bounded_request(self):
        key = "parity-train-mbpp11"
        control = dict(request_id=key, prompt_id="mbpp11", split="train", prompt_token_ids=[1, 2, 3], max_tokens=64)
        self.assertEqual(hook.validate_control(control, key), control)
        for bad in (dict(control, split="test"), dict(control, max_tokens=65),
                    dict(control, prompt_token_ids=[True, 2, 3]), dict(control, signed_url="secret"),
                    dict(control, prompt_token_ids=list(range(1025)))):
            with self.assertRaises(RuntimeError):
                hook.validate_control(bad, key)

    def test_source_and_version_guards_fail_closed(self):
        source = "class Runner:\n    def propose(self, x):\n        return x + 1\n"
        expected = hook.method_hash(source, "Runner", "propose")
        self.assertNotEqual(expected, hook.method_hash(source.replace("+ 1", "+ 2"), "Runner", "propose"))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root / "runner.py").write_text(source)
            with patch.object(hook, "SOURCE_HASHES", {}), patch.object(hook, "METHOD_HASHES", {("runner.py", "Runner", "propose"): expected}):
                hook.guard_sources(root, "0.27.1")
                with self.assertRaisesRegex(RuntimeError, "exactly"):
                    hook.guard_sources(root, "0.27.2rc1")
                (root / "runner.py").write_text(source.replace("+ 1", "+ 2"))
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    hook.guard_sources(root, "0.27.1")

    def test_wrappers_restore_and_do_not_replace_return_values(self):
        class Model:
            def forward(self, x):
                return x
        model = Model()
        value = object()
        original = model.forward
        with self.assertRaisesRegex(ValueError, "test failure"):
            with hook.wrapped_methods([(model, "forward", lambda x: original(x))]):
                self.assertIs(model.forward(value), value)
                raise ValueError("test failure")
        self.assertNotIn("forward", model.__dict__)
        self.assertIs(model.forward(value), value)

    def test_writes_never_overwrite_or_follow_symlink(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "trace.pt"
            hook.private_write(path, b"first")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                hook.private_write(path, b"second")
            link = Path(directory) / "link.pt"
            link.symlink_to(path)
            with self.assertRaises(FileExistsError):
                hook.private_write(link, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_hook_inert_without_opt_in(self):
        class Runner:
            def propose_draft_token_ids(self):
                return 5
        original = Runner.propose_draft_token_ids
        with patch.dict(hook.os.environ, {}, clear=True):
            hook.install(Runner)
        self.assertIs(Runner.propose_draft_token_ids, original)

    def test_client_preserves_api_tokens_and_original_capture_control(self):
        import json
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            source = root / "train-requests.jsonl"
            source.write_text(json.dumps(dict(prompt_id="mbpp11", messages=[dict(role="user", content="public test")])) + "\n")
            args = SimpleNamespace(root=root, requests=source, split="train", prompt_id="mbpp11", base_url="http://127.0.0.1:8000")
            response = dict(prompt_token_ids=[1, 2, 3], choices=[dict(token_ids=[4] * 64)])
            with patch.object(parity, "post", side_effect=[dict(tokens=[1, 2, 3]), response]) as post:
                parity.request_command(args, hook)
            self.assertEqual([c.args[1] for c in post.call_args_list], ["/tokenize", "/v1/chat/completions"])
            body = post.call_args_list[1].args[2]
            key = body["request_id"]
            self.assertTrue(body["return_token_ids"])
            self.assertEqual(len(key.removeprefix("b70-native-")), 32)
            control = json.loads((root / "native" / key / "control.json").read_text())
            self.assertEqual(control, dict(request_id=key, prompt_token_ids=[1, 2, 3], max_tokens=64))


@unittest.skipIf(torch is None, "requires external CPU torch runtime")
class TensorLogic(unittest.TestCase):
    def test_fixed_tolerance_reports_amplitude_not_just_cosine(self):
        x = torch.tensor([[1., 2., 3.]])
        self.assertTrue(parity.errors(x, x)["pass"])
        result = parity.errors(x * 1.1, x)
        self.assertFalse(result["pass"])
        self.assertGreater(result["cosine"][0], 0.999)
        self.assertGreater(result["norm_error"][0], 0)
        self.assertGreater(result["max_abs"][0], 0.29)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            parity.errors(x * float("nan"), x)

    def test_sampler_holes_and_nontext_rotary_rejected(self):
        self.assertEqual(parity.valid_samples(torch.tensor([[2, 3, -1]])), [2, 3])
        for values in ([[2, -1, 3]], [[-1]], [[2, 248320]], [[2, -2]]):
            with self.assertRaises(ValueError):
                parity.valid_samples(torch.tensor(values))
        self.assertEqual(parity.positions(torch.tensor([[1, 2]] * 3)), [1, 2])
        with self.assertRaises(ValueError):
            parity.positions(torch.tensor([[1, 2], [2, 3], [1, 2]]))

    def test_slot_prefix_and_rejected_suffix_reuse(self):
        def call(start, count):
            pos = list(range(start, start + count))
            return dict(positions=torch.tensor(pos), query_start=torch.tensor([0, count]),
                        seq_lens=torch.tensor([start + count]), block_table=torch.tensor([[7, 8, 9]]),
                        slots=torch.tensor([14 + p for p in pos]))
        slots = {}
        parity.check_slots(call(0, 3), 3, 2, slots)
        parity.check_slots(call(3, 3), 1, 2, slots)  # rejected suffix not retained
        self.assertEqual(slots, {0: 14, 1: 15, 2: 16, 3: 17})
        parity.check_slots(call(4, 1), 1, 2, slots)
        broken = call(5, 1)
        broken["block_table"][0, 0] = 10
        with self.assertRaisesRegex(ValueError, "prefix KV"):
            parity.check_slots(broken, 1, 2, slots)
        with self.assertRaisesRegex(ValueError, "prefix KV"):
            parity.check_slots(call(1, 1), 1, 2, {})


@unittest.skipIf(torch is None, "requires external CPU torch/transformers runtime")
class HFReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.trainer = load("scripts/experiments/qwen38_train_mtp.py", "unchanged_trainer")
        try:
            cls.trainer.runtime()
        except cls.trainer.TrainingError as exc:
            raise unittest.SkipTest(str(exc))
        raw = dict(model_type="qwen3_5", text_config=dict(
            model_type="qwen3_5_text", hidden_size=128, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=32,
            max_position_embeddings=128, mtp_num_hidden_layers=1, hidden_act="silu",
            rms_norm_eps=1e-6, rope_parameters=dict(rope_type="default", rope_theta=10000000.,
            partial_rotary_factor=0.5, mrope_section=[1, 1, 0], mrope_interleaved=True)))
        config = cls.trainer.native_config(raw)
        weights = {k: (torch.zeros(shape) if len(shape) == 1 else torch.randn(shape) * .02).bfloat16()
                   for k, shape in cls.trainer.expected_mtp_shapes(config).items()}
        cls.model = cls.trainer.build_native_mtp(config, weights).to(torch.bfloat16).eval()
        cls.embedding = torch.nn.Embedding(32, 128).to(torch.bfloat16).requires_grad_(False)
        cls.head = torch.nn.Linear(128, 32, bias=False).to(torch.bfloat16).requires_grad_(False)

    def fixture(self, depth=4, acceptance=(0, 2, 4)):
        """Artificial test trace built with HF itself; never called by live CLI."""
        t = self.trainer
        prompt, stream, traces, drafts, cursor = [1, 2, 3], [1, 2, 3, 4], [], [], 0
        cache = t.runtime().Cache()
        with torch.inference_mode():
            for step, accepted in enumerate((None,) + acceptance):
                ids = prompt if step == 0 else [stream[cursor]] + drafts
                outputs = [4] if step == 0 else drafts[:accepted] + [27]
                keep = len(prompt) if step == 0 else len(outputs)
                pos = torch.arange(cursor, cursor + len(ids))
                hidden = torch.randn(len(ids), 128).bfloat16()
                shifted = ids[1:] + [0]
                shifted[keep - 1] = outputs[-1]
                trace = dict(request_id="parity-train-unit", step=step, depth=depth, block_size=2,
                             target_ids=torch.tensor(ids), target_positions=pos, target_hidden=hidden,
                             sampled=torch.tensor([outputs + [-1] * (depth + 1 - len(outputs))]),
                             scheduled_drafts=drafts.copy(), calls=[], samples=[], first=dict(
                                 target_token_ids=torch.tensor(ids), target_positions=pos,
                                 target_hidden_states=hidden, next_token_ids=torch.tensor([outputs[-1]]),
                                 selected=torch.tensor([keep - 1]),
                                 num_rejected_tokens_gpu=None if step == 0 else torch.tensor([len(ids) - keep])))
                for d in range(depth):
                    start = cursor if d == 0 else cursor + keep - 1 + d
                    cache = t.prefix_cache(cache, start) if start else t.runtime().Cache()
                    input_ids = torch.tensor(shifted) if d == 0 else prediction
                    positions = torch.arange(start, start + len(input_ids))
                    feedback = hidden if d == 0 else output[selected:selected + 1]
                    with t.autocast("cpu", dtype=torch.bfloat16):
                        output = self.model(input_ids, feedback, positions, self.embedding, past_key_values=cache)
                        selected = keep - 1 if d == 0 else 0
                        prediction = self.head(output[selected:selected + 1]).argmax(-1)
                    trace["calls"].append(dict(input_ids=input_ids, positions=positions, hidden=feedback,
                                               inputs_embeds=None, output=output, slots=positions + 14,
                                               query_start=torch.tensor([0, len(input_ids)]),
                                               seq_lens=torch.tensor([start + len(input_ids)]),
                                               block_table=torch.arange(7, 80).unsqueeze(0)))
                    trace["samples"].append(dict(hidden=output[selected:selected + 1], ids=prediction))
                drafts = [sample["ids"].item() for sample in trace["samples"]]
                trace["draft_ids"] = torch.tensor([drafts])
                traces.append(trace)
                if step:
                    stream.extend(outputs)
                cursor += keep
        return traces, dict(request_id="parity-train-unit", prompt_token_ids=prompt), stream[len(prompt):]

    def test_original_capture_rows_must_match_live_selected_prefix(self):
        traces, _, _ = self.fixture()
        trace = traces[2]  # partial rejection, not only prefill
        keep = trace["first"]["selected"].item() + 1
        native = dict(request_id=trace["request_id"], step=trace["step"],
                      input_ids=trace["target_ids"][:keep], positions=trace["target_positions"][:keep],
                      target_last_hidden_states=trace["target_hidden"][:keep],
                      output_ids=torch.tensor(parity.valid_samples(trace["sampled"])))
        self.assertEqual(parity.compare_native_step(trace, native), keep)
        native["positions"] = native["positions"] + 1
        with self.assertRaisesRegex(ValueError, "row selection"):
            parity.compare_native_step(trace, native)

    def run_replay(self, fixture):
        with torch.inference_mode():
            return parity.replay(*fixture, self.trainer, self.model, self.embedding, self.head, "cpu")

    def test_hf_prefix_replay_all_rejection_classes_and_d8(self):
        for depth, accepts in ((4, (0, 2, 4)), (8, (0, 3, 8))):
            result = self.run_replay(self.fixture(depth, accepts))
            self.assertEqual(result["untested_classes"], [])
            self.assertTrue(result["observed_numeric_pass"])
            self.assertGreater(result["trainer_aligned_rows"], 3)

    def test_prefill_does_not_claim_unseen_rejections(self):
        result = self.run_replay(self.fixture(acceptance=()))
        self.assertEqual(result["untested_classes"], ["full_acceptance", "partial_rejection", "zero_acceptance"])

    def test_wrong_index_position_feedback_and_missing_prompt_fail(self):
        fixture = self.fixture()
        for key in ("selected", "position", "feedback", "missing"):
            bad = copy.deepcopy(fixture)
            if key == "selected":
                bad[0][1]["first"]["selected"] += 1
            elif key == "position":
                bad[0][1]["calls"][1]["positions"] += 1
            elif key == "feedback":
                bad[0][1]["calls"][1]["hidden"] = bad[0][1]["calls"][1]["hidden"] + 1
            else:
                bad[0].pop(0)
            with self.assertRaises(ValueError, msg=key):
                self.run_replay(bad)

    def test_argmax_mismatch_is_failure_even_when_norms_match(self):
        fixture = self.fixture(acceptance=())
        wrong = (fixture[0][0]["samples"][-1]["ids"].item() + 1) % 32
        fixture[0][0]["samples"][-1]["ids"] = torch.tensor([wrong])
        fixture[0][0]["draft_ids"] = fixture[0][0]["draft_ids"].clone()
        fixture[0][0]["draft_ids"][0, -1] = wrong
        result = self.run_replay(fixture)
        self.assertFalse(result["observed_numeric_pass"])
        self.assertFalse(result["rows"][-1]["argmax_equal"])


if __name__ == "__main__":
    unittest.main()
