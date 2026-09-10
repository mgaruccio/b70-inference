"""CPU contracts for the fixed verification-prefix overlay, not native proof.

The fixture is the COMPLETE, unmodified Apache-2.0 async_scheduler.py from
vLLM 73029d42441321b631779db3475031f5ec26dd6c (URL in the patch). No network,
vLLM, torch, GPU or XPU dependency is needed. We execute the actual original
and patched AsyncScheduler class plus the existing BF16 config guards with
isolated imports. The parent Scheduler/request/cache stubs below model only
counter handoffs; they do not replace the lead's native GDN/API oracle.

Run: python -m unittest discover -s tests -p 'test_qwen38_dflash2_verification.py' -v
The GPU source-pin regression additionally uses B70_DFLASH2_VERIFY_GPU_SOURCE
(default: the local pinned /tmp source export); only that test skips if absent.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/qwen38_dflash2_verification/async_scheduler.py"
ORIGINAL = FIXTURE.read_text(encoding="utf-8")


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = load_script("verification_patch", "patch-vllm-qwen38-dflash2-verification.py")
bf16 = load_script("bf16_patch", "patch-vllm-qwen38-dflash2-bf16.py")
PATCHED = patch.patch_text(ORIGINAL)
STATUS = NS(RUNNING="running", PREEMPTED="preempted", FINISHED="finished")
EOS = 2


def config(int4=False):
    target = NS(dtype="float16", quantization="gptq", head_dtype=None,
                enforce_eager=True, get_vocab_size=lambda: 248320,
                get_hidden_size=lambda: 5120)
    hf = NS(architectures=["DFlash2DraftModel"], dtype="bfloat16",
            num_hidden_layers=5, hidden_size=5120, vocab_size=248320,
            draft_vocab_size=None, tie_word_embeddings=False,
            dflash_config={"block_size": 8, "selector_rank": 256,
                           "selector_top_k": 16, "conv_kernel_size": 2,
                           "conv_group_size": 16, "target_layer_ids": [5, 19, 33, 47, 61]})
    if int4:
        hf.quantization_config = {
            "quant_method": "gptq", "bits": 4, "group_size": 128,
            "desc_act": False, "sym": True, "lm_head": False,
            "checkpoint_format": "gptq",
            "modules_in_block_to_quantize": ["self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"],
        }
    return NS(
        scheduler_config=NS(async_scheduling=True, max_num_seqs=1),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           data_parallel_size=1, decode_context_parallel_size=1,
                           prefill_context_parallel_size=1),
        speculative_config=NS(
            method="dflash", num_speculative_tokens=7,
            # SpeculativeConfig normalizes DFlash to parallel drafting at init.
            num_speculative_tokens_per_batch_size=None, parallel_drafting=True,
            draft_sample_method="greedy", rejection_sample_method="standard",
            quantization="gptq" if int4 else None, kv_cache_dtype="auto",
            use_heterogeneous_vocab=False, use_local_argmax_reduction=False,
            enforce_eager=True, target_model_config=target,
            draft_model_config=NS(hf_config=hf, dtype="bfloat16",
                                  quantization="gptq" if int4 else None)),
        use_v2_model_runner=False, num_speculative_tokens=7,
        num_sampled_tokens_per_step=1, dynamic_sd_lookup=None,
        lora_config=None, kv_transfer_config=None, ec_transfer_config=None,
    )


class Request:
    def __init__(self, request_id="r", prompt=8, max_output=100):
        self.request_id = request_id
        self.prompt = prompt
        self.max_output = max_output
        self.output_token_ids = []
        self.spec_token_ids = []
        self.num_computed_tokens = 0
        self.num_output_placeholders = 0
        self.num_in_flight_tokens = 0
        self.num_stale_output_tokens = 0
        self.drop_stale_output = False
        self.is_prefill_chunk = True
        self.use_structured_output = False
        self.status = STATUS.RUNNING

    @property
    def num_tokens(self):
        return self.prompt + len(self.output_token_ids)


class SchedulerStub:
    """Only the parent handoffs needed to execute the real async overrides."""

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = self.num_spec_tokens
        self.num_sampled_tokens_per_step = vllm_config.num_sampled_tokens_per_step
        self.dynamic_sd_lookup = vllm_config.dynamic_sd_lookup
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        self.current_step = 0
        self.requests = {}
        self.cached = []
        self.kv_cache_manager = NS(cache_blocks=lambda r, n: self.cached.append((r.request_id, n)))

    def _update_after_schedule(self, output):
        # Pinned parent's counter/prefill handoff, excluding unrelated managers.
        for rid, n in output.num_scheduled_tokens.items():
            r = self.requests[rid]
            r.num_computed_tokens += n
            r.num_in_flight_tokens += n
            r.is_prefill_chunk = r.num_computed_tokens < r.num_tokens + r.num_output_placeholders
            output.has_structured_output_requests |= r.use_structured_output and not r.is_prefill_chunk

    def _update_request_with_output(self, r, ids):
        stopped = False
        for n, token in enumerate(ids, 1):
            r.output_token_ids.append(token)
            if token == EOS or len(r.output_token_ids) == r.max_output:
                del ids[n:]
                r.status = STATUS.FINISHED
                stopped = True
                break
        return ids, stopped


def module(name, **attrs):
    result = ModuleType(name)
    result.__dict__.update(attrs)
    return result


@contextmanager
def runtime(cap=None, cfg=None, *, source=PATCHED, env=None, int4=False, xpu=True):
    environment = {"B70_DFLASH2_BF16": "1", "VLLM_USE_V2_MODEL_RUNNER": "0"}
    if cap is not None:
        environment["B70_DFLASH2_VERIFY_CAP"] = cap
    if int4:
        environment["B70_DFLASH2_INT4"] = "1"
    environment.update(env or {})
    modules = {name: module(name) for name in (
        "vllm", "vllm.config", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched")}
    modules.update({
        "vllm.logger": module("vllm.logger", init_logger=lambda name: NS()),
        "vllm.v1.core.sched.output": module("output", SchedulerOutput=NS),
        "vllm.v1.core.sched.scheduler": module("scheduler", Scheduler=SchedulerStub),
        "vllm.v1.request": module("request", Request=Request, RequestStatus=STATUS),
        "vllm.platforms": module("platforms", current_platform=NS(is_xpu=lambda: xpu)),
        "torch": module("torch", float16="float16", bfloat16="bfloat16"),
    })
    with mock.patch.dict(os.environ, environment, clear=True), mock.patch.dict(sys.modules, modules):
        guards = module("vllm.config.speculative")
        exec(bf16.CONFIG_HELPERS, guards.__dict__)
        with mock.patch.dict(sys.modules, {"vllm.config.speculative": guards}):
            actual = module("async_scheduler_under_test")
            exec(compile(source, str(FIXTURE), "exec"), actual.__dict__)
            yield actual.AsyncScheduler(cfg if cfg is not None else config(int4))


def schedule(s, r=None, *, drafts=None, n=None, generation=7):
    """Feed a chosen real schedule to the override; not a scheduler simulator."""
    if r is None:
        output = NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={})
    else:
        s.requests[r.request_id] = r
        if drafts is None:
            drafts = r.spec_token_ids
        output = NS(num_scheduled_tokens={r.request_id: n if n is not None else len(drafts) + 1},
                    scheduled_spec_decode_tokens={r.request_id: drafts} if drafts else {})
        # Parent schedule rebinds, never clears the shared placeholder list.
        r.spec_token_ids = []
    output.num_spec_tokens_to_schedule = generation
    output.pending_structured_output_tokens = False
    output.has_structured_output_requests = False
    s.current_step += 1
    s._update_after_schedule(output)
    return output


def deliver(s, r, output, ids):
    """Model the parent's rejection/stale drains, then call the REAL override.

    No claim about update_from_output, allocator, worker, or device execution.
    In-flight counts are scheduled input counts, not accepted output counts.
    """
    n = output.num_scheduled_tokens[r.request_id]
    r.num_in_flight_tokens -= n
    stale = r.num_stale_output_tokens > 0
    if stale:
        r.num_stale_output_tokens -= n
        assert r.num_stale_output_tokens >= 0
    if r.status == STATUS.FINISHED or (stale and r.drop_stale_output):
        return [], r.status == STATUS.FINISHED
    drafts = len(output.scheduled_spec_decode_tokens.get(r.request_id, ()))
    if drafts and ids and not stale:
        rejected = drafts - max(len(ids) - s.num_sampled_tokens_per_step, 0)
        if r.num_computed_tokens > 0:
            r.num_computed_tokens -= rejected
        if r.num_output_placeholders > 0:
            r.num_output_placeholders -= rejected
    return s._update_request_with_output(r, list(ids), is_stale=stale)


def preempt(r, *, drop=False):
    # Pinned parent zeroes counters/rebinds IDs before late async delivery.
    r.status = STATUS.PREEMPTED
    r.num_computed_tokens = 0
    r.spec_token_ids = []
    r.num_stale_output_tokens = r.num_in_flight_tokens
    r.num_output_placeholders = 0
    r.drop_stale_output = drop


def warm(s, r):
    output = schedule(s, r, drafts=[], n=r.prompt)
    deliver(s, r, output, [101])


class PrefixTests(unittest.TestCase):
    def test_off_and_fixed_caps_change_only_next_placeholders(self):
        for cap in (None, "1", "3", "7"):
            with self.subTest(cap=cap), runtime(cap) as s:
                r = Request()
                original_initial_list = s._spec_token_placeholders
                self.assertEqual(original_initial_list, [-1] * 7)
                warm(s, r)
                self.assertEqual(r.spec_token_ids, [-1] * (int(cap) if cap else 7))
                self.assertIs(r.spec_token_ids, s._spec_token_placeholders)
                for actual_length in (7, 1, 3, 0, 2):
                    # Previous full/short histories must not be charged the NEW cap.
                    ids = list(range(20, 20 + actual_length))
                    output = schedule(s, r, drafts=ids)
                    self.assertEqual(output.num_spec_tokens_to_schedule, 7)
                    self.assertEqual(output.num_scheduled_tokens, {"r": actual_length + 1})
                    self.assertEqual(r.num_output_placeholders, actual_length + 1)
                    self.assertEqual(ids, list(range(20, 20 + actual_length)))
                    deliver(s, r, output, [30] * (actual_length + 1))
                    self.assertEqual(r.num_output_placeholders, 0)
                    self.assertEqual(r.num_in_flight_tokens, 0)
                self.assertEqual(s.num_spec_tokens, 7)
                self.assertEqual(s.num_lookahead_tokens, 7)
                self.assertEqual(s.vllm_config.speculative_config.num_speculative_tokens, 7)
                self.assertEqual(original_initial_list, [-1] * 7)

    def test_aliases_survive_slicing_rebinding_and_preemption(self):
        with runtime("3") as s:
            r = Request()
            warm(s, r)
            held = r.spec_token_ids
            first = schedule(s, r)
            self.assertIs(first.scheduled_spec_decode_tokens["r"], held)
            self.assertIsNot(r.spec_token_ids, held)
            second_held = r.spec_token_ids
            second = schedule(s, r, drafts=second_held[:1])
            self.assertIsNot(second.scheduled_spec_decode_tokens["r"], second_held)
            preempt(r)
            self.assertEqual(held, [-1] * 3)
            self.assertEqual(second_held, [-1] * 3)
            self.assertEqual(first.scheduled_spec_decode_tokens["r"], [-1] * 3)
            self.assertEqual(second.scheduled_spec_decode_tokens["r"], [-1])

    def test_prefill_and_empty_steps(self):
        for cap in (None, "1", "3", "7"):
            with self.subTest(cap=cap), runtime(cap) as s:
                r = Request(prompt=20)
                first = schedule(s, r, drafts=[], n=10)
                self.assertTrue(r.is_prefill_chunk)
                self.assertEqual(r.spec_token_ids, [])
                self.assertEqual(r.num_output_placeholders, 0)
                deliver(s, r, first, [])
                last = schedule(s, r, drafts=[], n=10)
                self.assertFalse(r.is_prefill_chunk)
                self.assertEqual(r.num_output_placeholders, 1)
                deliver(s, r, last, [10])
                self.assertEqual(r.num_in_flight_tokens, 0)
                held = r.spec_token_ids
                empty = schedule(s)
                self.assertEqual(empty.num_spec_tokens_to_schedule, 7)
                self.assertIs(r.spec_token_ids, held)
                self.assertEqual(r.num_output_placeholders, 0)

    def test_overlapping_accept_reject_and_final_partial_drains(self):
        for cap in ("1", "3", "7"):
            with self.subTest(cap=cap), runtime(cap) as s:
                r = Request()
                warm(s, r)
                k = int(cap)
                for accepted in (0, k, k // 2):
                    first = schedule(s, r)
                    second = schedule(s, r)
                    self.assertEqual(r.num_output_placeholders, 2 * (k + 1))
                    deliver(s, r, first, [30] * (accepted + 1))
                    self.assertEqual(r.num_output_placeholders, k + 1)
                    self.assertEqual(r.num_in_flight_tokens, k + 1)
                    deliver(s, r, second, [40] * (k + 1))
                    self.assertEqual(r.num_output_placeholders, 0)
                    self.assertEqual(r.num_in_flight_tokens, 0)
                    self.assertEqual(r.num_computed_tokens, r.num_tokens - 1)
                r.max_output = len(r.output_token_ids) + 2
                final = schedule(s, r, drafts=r.spec_token_ids[:1])
                ids, stopped = deliver(s, r, final, [50, 51])
                self.assertEqual(ids, [50, 51])
                self.assertTrue(stopped)
                self.assertEqual(r.num_output_placeholders, 0)
                self.assertEqual(r.num_in_flight_tokens, 0)

    def test_eos_trimming_preserves_upstream_unused_placeholder_behavior(self):
        for cap in ("1", "3", "7"):
            with self.subTest(cap=cap), runtime(cap) as s:
                r = Request()
                warm(s, r)
                k = int(cap)
                output = schedule(s, r)
                ids, stopped = deliver(s, r, output, [EOS] + [50] * k)
                self.assertEqual(ids, [EOS])
                self.assertTrue(stopped)
                self.assertEqual(r.num_in_flight_tokens, 0)
                # The original async method subtracts only tokens before EOS.
                # Unused placeholders remain on the FINISHED request, which the
                # real scheduler retires; inventing a drain here would hide drift.
                self.assertEqual(r.num_output_placeholders, k)
                self.assertEqual(r.status, STATUS.FINISHED)

    def test_preemption_deliver_and_drop_stale_do_not_underflow(self):
        for cap in ("1", "3", "7"):
            for drop in (False, True):
                with self.subTest(cap=cap, drop=drop), runtime(cap) as s:
                    r = Request()
                    warm(s, r)
                    first, second = schedule(s, r), schedule(s, r)
                    cache_before = list(s.cached)
                    preempt(r, drop=drop)
                    deliver(s, r, first, [30])
                    self.assertEqual(r.num_stale_output_tokens, int(cap) + 1)
                    deliver(s, r, second, [40] * (int(cap) + 1))
                    self.assertEqual(r.num_in_flight_tokens, 0)
                    self.assertEqual(r.num_stale_output_tokens, 0)
                    self.assertEqual(r.num_output_placeholders, 0)
                    self.assertEqual(r.num_computed_tokens, 0)
                    self.assertEqual(s.cached, cache_before)
                    r.status = STATUS.RUNNING
                    resumed = schedule(s, r, drafts=[], n=r.num_tokens)
                    deliver(s, r, resumed, [60])
                    self.assertEqual(r.num_output_placeholders, 0)
                    self.assertEqual(r.num_in_flight_tokens, 0)
                    self.assertEqual(len(r.spec_token_ids), int(cap))

    def test_cap_seven_and_disabled_match_unpatched_trace_exactly(self):
        def trace(source, cap):
            with runtime(cap, source=source) as s:
                r = Request()
                warm(s, r)
                records = []
                for k, accepted in ((7, 0), (3, 3), (1, 0), (7, 7), (0, 0)):
                    r.use_structured_output = True
                    first = schedule(s, r, drafts=r.spec_token_ids[:k])
                    second = schedule(s, r, drafts=r.spec_token_ids[:k])
                    deliver(s, r, first, [30] * (accepted + 1))
                    deliver(s, r, second, [40] * (k + 1))
                    records.append(deepcopy((vars(first), vars(second), vars(r), s.cached,
                                             s._spec_token_placeholders)))
                eos = schedule(s, r)
                deliver(s, r, eos, [EOS] + [50] * 7)
                records.append(deepcopy((vars(eos), vars(r), s.cached)))
                return records
        expected = trace(ORIGINAL, None)
        self.assertEqual(trace(PATCHED, None), expected)
        self.assertEqual(trace(PATCHED, "7"), expected)

    def test_off_does_not_restrict_other_original_modes_or_dynamic_widths(self):
        cfg = config()
        cfg.scheduler_config.max_num_seqs = 4
        cfg.use_v2_model_runner = True
        cfg.speculative_config = None
        cfg.num_speculative_tokens = 3
        cfg.dynamic_sd_lookup = [3]
        with runtime(cfg=cfg, env={"B70_DFLASH2_BF16": "0"}) as s:
            r = Request(prompt=1)
            out = schedule(s, r, n=1, generation=2)
            self.assertEqual(r.spec_token_ids, [-1, -1])
            self.assertEqual(out.num_spec_tokens_to_schedule, 2)
            self.assertEqual(r.next_decode_eligible_step, s.current_step + 1)

    def test_cap_is_read_once_at_init(self):
        with runtime("3") as s:
            os.environ["B70_DFLASH2_VERIFY_CAP"] = "1"
            r = Request()
            warm(s, r)
            self.assertEqual(r.spec_token_ids, [-1] * 3)


class GuardTests(unittest.TestCase):
    def test_rejects_noncanonical_environment(self):
        for value in ("", "0", "2", "4", "8", "-1", "01", "+1", " 1", "3 ", "7\n", "1.0", "auto", "off", "１"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "exactly 1, 3, or 7"):
                with runtime(value):
                    pass

    def test_rejects_unsupported_scheduler_and_spec_modes(self):
        changes = {
            "scheduler_config.async_scheduling": [False, None, "true"],
            "scheduler_config.max_num_seqs": [0, 2, True, 1.0],
            "use_v2_model_runner": [True],
            "num_speculative_tokens": [1, 3, 7.0],
            "num_sampled_tokens_per_step": [0],
            "dynamic_sd_lookup": [[], [7]],
            "lora_config": [NS()], "kv_transfer_config": [NS()], "ec_transfer_config": [NS()],
            "speculative_config": [None],
            "speculative_config.method": ["eagle", "ngram"],
            "speculative_config.num_speculative_tokens": [3, 7.0],
            "speculative_config.num_speculative_tokens_per_batch_size": [[], [(1, 1, 7)]],
            "speculative_config.parallel_drafting": [False, None, 1],
            "speculative_config.draft_sample_method": ["probabilistic"],
            "speculative_config.rejection_sample_method": ["synthetic", "typical"],
            "speculative_config.use_heterogeneous_vocab": [True],
            "speculative_config.use_local_argmax_reduction": [True],
            "speculative_config.kv_cache_dtype": ["fp8"],
        }
        for field in ("tensor", "pipeline", "data", "decode_context", "prefill_context"):
            changes[f"parallel_config.{field}_parallel_size"] = [2]
        for path, values in changes.items():
            for value in values:
                with self.subTest(path=path, value=value):
                    cfg = config()
                    obj = cfg
                    *parents, name = path.split(".")
                    for part in parents:
                        obj = getattr(obj, part)
                    setattr(obj, name, value)
                    with self.assertRaises(ValueError), runtime("3", cfg):
                        pass

    def test_reuses_real_checkpoint_and_bf16_int4_guards(self):
        for int4 in (False, True):
            with self.subTest(valid_int4=int4), runtime("3", int4=int4) as s:
                self.assertEqual(s._b70_dflash2_verify_cap, 3)
        changes = {
            "architectures": ["DFlashDraftModel"], "dtype": "float16",
            "num_hidden_layers": 4, "hidden_size": 4096, "vocab_size": 10,
            "draft_vocab_size": 10, "tie_word_embeddings": True,
        }
        for field, value in changes.items():
            cfg = config()
            setattr(cfg.speculative_config.draft_model_config.hf_config, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError), runtime("1", cfg):
                pass
        for field in config().speculative_config.draft_model_config.hf_config.dflash_config:
            cfg = config()
            cfg.speculative_config.draft_model_config.hf_config.dflash_config[field] = 0
            with self.subTest(field=field), self.assertRaises(ValueError), runtime("7", cfg):
                pass
        cfg = config(int4=True)
        cfg.speculative_config.draft_model_config.hf_config.quantization_config["bits"] = 8
        with self.assertRaises(ValueError), runtime("3", cfg, int4=True):
            pass
        for env in ({"B70_DFLASH2_BF16": "0"}, {"VLLM_USE_V2_MODEL_RUNNER": "1"}):
            with self.subTest(env=env), self.assertRaises(ValueError), runtime("3", env=env):
                pass
        with self.assertRaises(ValueError), runtime("3", xpu=False):
            pass

    def test_missing_or_malformed_config_fails_closed(self):
        for value in (None, [], "bad"):
            cfg = config()
            cfg.speculative_config.draft_model_config.hf_config.dflash_config = value
            with self.subTest(value=value), self.assertRaises(ValueError), runtime("3", cfg):
                pass
        cfg = config()
        del cfg.speculative_config.num_speculative_tokens_per_batch_size
        with self.assertRaisesRegex(ValueError, "malformed"), runtime("3", cfg):
            pass

    def test_generation_or_dynamic_drift_refused_before_counter_mutation(self):
        for generation in (0, 1, 3, 8, 7.0, "7", None):
            with self.subTest(generation=generation), runtime("3") as s:
                r = Request()
                before = deepcopy(vars(r))
                with self.assertRaisesRegex(RuntimeError, "generation K7"):
                    schedule(s, r, generation=generation)
                self.assertEqual(vars(r), before)
        with runtime("3") as s:
            s.dynamic_sd_lookup = [7]
            with self.assertRaisesRegex(RuntimeError, "dynamic SD"):
                schedule(s)


class SourceTests(unittest.TestCase):
    def test_fixture_pin_and_exact_idempotence(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), patch.ORIGINAL_SHA256)
        self.assertNotEqual(PATCHED, ORIGINAL)
        self.assertEqual(patch.patch_text(PATCHED), PATCHED)
        self.assertEqual(PATCHED.count(patch.MARKER), 1)
        # Output updates, including stale/EOS behavior, are untouched verbatim.
        tail = "    def _update_request_with_output(\n"
        self.assertEqual(ORIGINAL[ORIGINAL.index(tail):], PATCHED[PATCHED.index(tail):])

    def test_prefill_guarded_gpu_accepted_but_unguarded_rejected(self):
        source_path = Path(os.environ.get(
            "B70_DFLASH2_VERIFY_GPU_SOURCE",
            "/tmp/qwen38-boundary-source-73029d424/v1/worker/gpu_model_runner.py",
        ))
        if not source_path.is_file():
            self.skipTest(f"pinned unguarded GPU source export unavailable: {source_path}")
        original = source_path.read_text(encoding="utf-8")
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),
                         "d620deb484fee968aeefefcf8cc901cf2665118e1a2415ee1bb906fd697b3054")
        # Exact edits from deployed patch_uniform_decode_prefill.py, SHA256
        # baa4647398874c19175ea74fe6f5d8dd6c2d83fc4bd0e5f2a68558afd983f5ad.
        signature = (
            "    def _is_uniform_decode(\n"
            "        max_num_scheduled_tokens: int,\n"
            "        uniform_decode_query_len: int,\n"
            "        num_tokens: int,\n"
            "        num_reqs: int,\n"
        )
        classifier = "                (max_num_scheduled_tokens == uniform_decode_query_len)\n"
        call = (
            "            num_tokens=num_tokens,\n"
            "            num_reqs=num_reqs,\n"
        )
        edits = (
            (signature, signature + "        has_prefill: bool = False,\n"),
            (classifier, "                not has_prefill  # B70_FIX_UNIFORM_DECODE_PREFILL\n"
                         "                and (max_num_scheduled_tokens == uniform_decode_query_len)\n"),
            (call + "            force_uniform_decode=force_uniform_decode,\n",
             call + "            has_prefill=bool(\n"
                    "                (self.input_batch.num_computed_tokens_cpu[:num_reqs]\n"
                    "                 < self.input_batch.num_prompt_tokens[:num_reqs]).any()\n"
                    "            ),\n"
                    "            force_uniform_decode=force_uniform_decode,\n"),
        )
        guarded = original
        for old, new in edits:
            self.assertEqual(guarded.count(old), 1)
            guarded = guarded.replace(old, new, 1)
        expected = "00f22cb5fe8bc2f05cc93b0500faaa55d8b3a77754fde3e23767648f621f4dd4"
        relative = "v1/worker/gpu_model_runner.py"
        self.assertEqual(hashlib.sha256(guarded.encode()).hexdigest(), expected)
        self.assertEqual(patch.DEPENDENCY_SHA256[relative], expected)
        with self.package_root() as (root, target):
            with mock.patch.dict(patch.DEPENDENCY_SHA256, {relative: expected}):
                gpu = root / relative
                gpu.write_text(original, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "handoff dependency"):
                    patch.apply(root)
                self.assertEqual(target.read_text(), ORIGINAL)
                gpu.write_text(guarded, encoding="utf-8")
                self.assertTrue(patch.apply(root))
                self.assertFalse(patch.apply(root))
                gpu.write_text(original, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "handoff dependency"):
                    patch.apply(root)
                self.assertEqual(target.read_text(), PATCHED)

    def test_source_drift_partial_moved_and_tampered_overlay_rejected(self):
        candidates = [
            ORIGINAL + "\n", ORIGINAL.replace("num_spec_tokens", "num_spec_tokens_changed", 1),
            ORIGINAL.replace("logger =", "# changed\nlogger =", 1),
            ORIGINAL.replace("\n", "\r\n"),
            PATCHED.replace("int(cap)", "int(cap) + 1"),
            PATCHED + f"\n# {patch.MARKER}\n",
            PATCHED.replace(patch.MARKER, "REMOVED"),
            PATCHED.replace(patch.INIT, "") + patch.INIT,
            PATCHED.replace(patch.STEP_GUARD, "") + patch.STEP_GUARD,
        ]
        for old, new in patch.transformations():
            candidates.append(ORIGINAL.replace(old, new, 1))
            candidates.append(PATCHED.replace(new, old, 1))
        for index, source in enumerate(candidates):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                patch.patch_text(source)

    @contextmanager
    def package_root(self):
        # All test writes stay beneath this worker checkout and are cleaned up.
        with tempfile.TemporaryDirectory(prefix=".verification-test-", dir=ROOT) as tmp:
            root = Path(tmp)
            dependencies = {}
            for relative in patch.DEPENDENCY_SHA256:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                source = f"# dependency stub for {relative}\n".encode()
                path.write_bytes(source)
                dependencies[relative] = hashlib.sha256(source).hexdigest()
            path = root / patch.RELATIVE_PATH
            path.write_text(ORIGINAL, encoding="utf-8")
            # Stub bytes test the file protocol; fresh full-source pins are also
            # checked against actual upstream+BF16 sources in the worker smoke.
            with mock.patch.object(patch, "DEPENDENCY_SHA256", dependencies):
                yield root, path

    def test_apply_writes_only_async_and_replay_does_not_write(self):
        with self.package_root() as (root, target):
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.py")}
            self.assertTrue(patch.apply(root))
            self.assertEqual(target.read_text(), PATCHED)
            with mock.patch.object(Path, "write_bytes", side_effect=AssertionError("unexpected write")):
                self.assertFalse(patch.apply(root))
            for relative, content in before.items():
                if relative.as_posix() != patch.RELATIVE_PATH:
                    self.assertEqual((root / relative).read_bytes(), content)

    def test_every_handoff_dependency_is_required_even_on_replay(self):
        for relative in patch.DEPENDENCY_SHA256:
            for already in (False, True):
                with self.subTest(relative=relative, already=already), self.package_root() as (root, target):
                    if already:
                        patch.apply(root)
                    before = target.read_bytes()
                    (root / relative).write_text("# drift\n")
                    with self.assertRaisesRegex(RuntimeError, "handoff dependency"):
                        patch.apply(root)
                    self.assertEqual(target.read_bytes(), before)
                    (root / relative).unlink()
                    with self.assertRaises(FileNotFoundError):
                        patch.apply(root)
                    self.assertEqual(target.read_bytes(), before)

    def test_compile_failure_cannot_write(self):
        with self.package_root() as (root, target):
            bad = (("logger = init_logger(__name__)", "logger = ("),)
            with mock.patch.object(patch, "transformations", return_value=bad):
                with self.assertRaises(SyntaxError):
                    patch.apply(root)
            self.assertEqual(target.read_text(), ORIGINAL)


if __name__ == "__main__":
    unittest.main()
