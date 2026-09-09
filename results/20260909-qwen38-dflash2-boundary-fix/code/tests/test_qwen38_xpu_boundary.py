"""CPU-only patch/routing contracts; NOT a substitute for native continuation tests."""
import ast
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch as mock_patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/patch-vllm-qwen38-xpu-boundary.py"
spec = importlib.util.spec_from_file_location("xpu_boundary", PATH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

# Exact stopped-container function at vLLM 73029d424. Keep independent of the
# patcher's insertion text; its full-function pin must validate this fixture.
PINNED_IMPL = '''def _gdn_attention_core_xpu_impl(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op wrapping the XPU SYCL GDN kernel for torch.compile."""
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata_raw = forward_context.attn_metadata

    if attn_metadata_raw is None:
        return

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata = attn_metadata_raw[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)

    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    num_prefills = attn_metadata.num_prefills
    num_decodes = attn_metadata.num_decodes
    num_spec_decodes = attn_metadata.num_spec_decodes

    has_initial_state = attn_metadata.has_initial_state

    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
    non_spec_state_indices_tensor = (
        non_spec_state_indices_tensor.contiguous()
        if non_spec_state_indices_tensor is not None
        else None
    )

    spec_query_start_loc = attn_metadata.spec_query_start_loc
    spec_token_indx = attn_metadata.spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501

    spec_sequence_masks = attn_metadata.spec_sequence_masks
    if spec_sequence_masks is not None:
        if non_spec_token_indx is not None:
            non_spec_token_indx = non_spec_token_indx.to(torch.int32)
        if spec_token_indx is not None:
            spec_token_indx = spec_token_indx.to(torch.int32)

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        ssm_state=self.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_prefills=num_prefills,  # type: ignore[attr-defined]
        num_decodes=num_decodes,  # type: ignore[attr-defined]
        num_spec_decodes=num_spec_decodes,  # type: ignore[attr-defined]
        has_initial_state=has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_token_indx=non_spec_token_indx,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        spec_query_start_loc=spec_query_start_loc,  # type: ignore[attr-defined]
        spec_token_indx=spec_token_indx,  # type: ignore[attr-defined]
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,  # type: ignore[attr-defined]
        num_actual_tokens=num_actual_tokens,  # type: ignore[attr-defined]
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )
'''
SOURCE = ("import torch\n\n" + PINNED_IMPL + patch.END
          + "):\n    pass\n\ndef register():\n    direct_register_custom_op(\n"
          + patch.REGISTRATION + "    )\n")


class Tensor:
    """Small row-view spy, with no device-value reads in the wrapper allowed."""
    def __init__(self, shape, fill=0, *, dtype="bf16", device="xpu:0",
                 storage=None, offset=0):
        self.shape = tuple(shape)
        self.dtype, self.device = dtype, device
        self.storage = [fill] * math.prod(shape) if storage is None else storage
        self.offset = offset

    @property
    def ndim(self):
        return len(self.shape)

    def size(self, dim):
        return self.shape[dim]

    def numel(self):
        return math.prod(self.shape)

    def values(self):
        return self.storage[self.offset:self.offset + self.numel()]

    def __getitem__(self, key):
        start, stop, step = key.indices(self.size(0))
        assert step == 1
        return Tensor((max(0, stop - start), *self.shape[1:]), dtype=self.dtype,
                      device=self.device, storage=self.storage,
                      offset=self.offset + start * math.prod(self.shape[1:]))

    def copy_(self, other):
        assert self.shape == other.shape
        self.storage[self.offset:self.offset + self.numel()] = other.values()
        return self

    def new_zeros(self, shape):
        return Tensor(shape, dtype=self.dtype, device=self.device)

    def new_empty(self, shape):
        return Tensor(shape, fill=-999, dtype=self.dtype, device=self.device)

    def new_tensor(self, values):
        return Tensor((len(values),), dtype=self.dtype, device=self.device,
                      storage=list(values))

    def contiguous(self):
        return self

    def to(self, dtype):
        if dtype == self.dtype:
            return self
        return Tensor(self.shape, dtype=dtype, device=self.device, storage=self.values())

    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return Tensor(shape, dtype=self.dtype, device=self.device,
                      storage=self.storage, offset=self.offset)


def ints(values):
    return Tensor((len(values),), dtype="int32", storage=list(values))


class Metadata(SimpleNamespace):
    pass


def make_case(length=7, physical=None, actual=None, accepted=8):
    physical = length if physical is None else physical
    metadata = Metadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=1, num_spec_decode_tokens=length,
        num_actual_tokens=length if actual is None else actual,
        has_initial_state=None, non_spec_query_start_loc=None,
        non_spec_token_indx=ints([]), non_spec_state_indices_tensor=None,
        spec_query_start_loc=ints([0, length]), spec_token_indx=ints(range(length)),
        spec_state_indices_tensor=Tensor((1, 8), dtype="int32", storage=list(range(2, 10))),
        num_accepted_tokens=ints([accepted]), spec_sequence_masks=ints([1]),
    )
    layer = SimpleNamespace(
        prefix="layer", num_k_heads=2, num_v_heads=2, head_k_dim=2, head_v_dim=3,
        tp_size=1, gqa_interleaved_layout=False, activation="silu",
        conv1d=SimpleNamespace(weight=Tensor((10, 1, 4)), bias=None),
        kv_cache=(Tensor((16, 10, 10)), Tensor((16, 2, 3, 2))),
        A_log=Tensor((2,), dtype="fp32"), dt_bias=Tensor((2,)),
    )
    outputs = [Tensor((physical, 2, 3), fill=-17), Tensor((physical, 2, 3), fill=-23)]
    inputs = [Tensor((physical, 10), storage=list(range(physical * 10))),
              Tensor((physical, 4), storage=list(range(physical * 4)))]
    return SimpleNamespace(metadata=metadata, layer=layer, outputs=outputs, inputs=inputs)


class NativeSpy:
    def __init__(self, strict=False):
        self.calls = []
        self.strict = strict

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.strict:
            width = kwargs["spec_state_indices_tensor"].size(1)
            assert kwargs["spec_token_indx"].numel() == kwargs["num_spec_decodes"] * width
        for tensor, base in zip(args[:2], (1000, 2000)):
            n = min(kwargs["num_actual_tokens"], tensor.size(0))
            prefix = tensor[:n]
            prefix.copy_(Tensor(prefix.shape, storage=[base + i for i in range(prefix.numel())]))


def execute(case, *, patched=True, strict=False, no_metadata=False):
    tree = ast.parse(patch.patch_text(SOURCE) if patched else SOURCE)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_gdn_attention_core_xpu_impl")
    native = NativeSpy(strict)
    torch = SimpleNamespace(Tensor=Tensor, int32="int32",
                            arange=lambda n, dtype, device: Tensor(
                                (n,), dtype=dtype, device=device, storage=list(range(n))),
                            ops=SimpleNamespace(_xpu_C=SimpleNamespace(gdn_attention=native)))
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "patched-impl", "exec"), namespace)
    context = SimpleNamespace(no_compile_layers={"layer": case.layer},
                              attn_metadata=None if no_metadata else {"layer": case.metadata})
    modules = {
        "vllm.forward_context": SimpleNamespace(get_forward_context=lambda: context),
        "vllm.v1.attention.backends.gdn_attn": SimpleNamespace(GDNAttentionMetadata=Metadata),
    }
    with mock_patch.dict(sys.modules, modules):
        namespace["_gdn_attention_core_xpu_impl"](*case.outputs, *case.inputs, "layer")
    return native.calls


class PatchTests(unittest.TestCase):
    def test_transform_replay_syntax_and_scope(self):
        result = patch.patch_text(SOURCE)
        self.assertEqual(patch.patch_text(result), result)
        self.assertEqual(result.replace(patch.BEFORE_CALL, "").replace(patch.AFTER_CALL, ""), SOURCE)
        self.assertEqual(result.count(patch.CALL), 1)
        self.assertEqual(result.count(patch.MARKER), 1)
        compile(result, "patched-source", "exec")

    def test_drift_fails_closed(self):
        result = patch.patch_text(SOURCE)
        bad_sources = (
            SOURCE.replace("tp_size=self.tp_size", "tp_size=1"),
            SOURCE.replace("eager_break_during_capture(_gdn_attention_core_xpu_impl)", "other"),
            SOURCE + PINNED_IMPL, SOURCE + "# " + patch.MARKER,
            result.replace(patch.BEFORE_CALL, ""),
            result.replace(patch.AFTER_CALL, ""),
            result.replace(patch.BEFORE_CALL, "").replace(
                patch.AFTER_CALL, patch.BEFORE_CALL + patch.AFTER_CALL),
            result.replace("block_tokens = spec_state_indices_tensor.size(1)", "block_tokens = 8"),
            result.replace("tp_size=self.tp_size", "tp_size=1"),
            result + "# " + patch.MARKER,
        )
        for source in bad_sources:
            with self.subTest(source=source[-100:]), self.assertRaises(RuntimeError):
                patch.patch_text(source)

    def test_compile_checked_on_first_apply_and_replay(self):
        for source in (SOURCE, patch.patch_text(SOURCE)):
            with self.assertRaises(SyntaxError):
                patch.patch_text(source + "\nif broken\n")

    def test_cli_offline_copy_replay_and_no_write_on_drift(self):
        # All temporary writes stay inside this checkout, never the source copy.
        with tempfile.TemporaryDirectory(prefix=".boundary-test-", dir=ROOT) as directory:
            path = Path(directory) / "_xpu_ops.py"
            path.write_text(SOURCE)
            command = [sys.executable, str(PATH), "--root", directory]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(path.read_text(), patch.patch_text(SOURCE))
            mtime = path.stat().st_mtime_ns
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
            changed = path.read_text().replace("tp_size=self.tp_size", "tp_size=1")
            path.write_text(changed)
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("implementation changed", failed.stderr)
            self.assertEqual(path.read_text(), changed)

    @unittest.skipUnless(os.environ.get("QWEN38_BOUNDARY_SOURCE_ROOT"), "optional stopped-image source")
    def test_real_pinned_source_read_only(self):
        source = (Path(os.environ["QWEN38_BOUNDARY_SOURCE_ROOT"]) / "_xpu_ops.py").read_text()
        start, end = source.index(patch.START), source.index(patch.END)
        self.assertEqual(source[start:end], PINNED_IMPL)
        result = patch.patch_text(source)
        self.assertEqual(patch.patch_text(result), result)
        self.assertEqual(result.replace(patch.BEFORE_CALL, "").replace(patch.AFTER_CALL, ""), source)


class RoutingTests(unittest.TestCase):
    def test_reproduces_fixed_width_contract_before_fix(self):
        with self.assertRaises(AssertionError):
            execute(make_case(), patched=False, strict=True)
        self.assertEqual(len(execute(make_case(), strict=True)), 1)

    def test_partial_prefix_all_lengths_previous_acceptance_and_padding(self):
        for length in range(1, 8):
            for accepted in range(1, 9):
                for physical, actual in ((length, length), (8, length), (8, 8), (12, 12)):
                    with self.subTest(length=length, accepted=accepted, physical=physical, actual=actual):
                        case = make_case(length, physical, actual, accepted)
                        metadata_before = vars(case.metadata).copy()
                        original_inputs = [t.values() for t in case.inputs]
                        (args, kwargs), = execute(case, strict=True)
                        for i, tensor in enumerate(args[:4]):
                            self.assertEqual(tensor.size(0), 8)
                            original = (case.outputs + case.inputs)[i]
                            self.assertIsNot(tensor, original)
                            self.assertEqual(tensor.dtype, original.dtype)
                            self.assertEqual(tensor.device, original.device)
                            self.assertEqual(tensor.shape[1:], original.shape[1:])
                        for temp, original, before in zip(args[2:4], case.inputs, original_inputs):
                            self.assertEqual(temp[:length].values(), original[:length].values())
                            self.assertEqual(temp[length:].values(), [0] * temp[length:].numel())
                            self.assertEqual(original.values(), before)
                        for i, original in enumerate(case.outputs):
                            self.assertEqual(original[:length].values(), args[i][:length].values())
                            self.assertEqual(original[length:].values(), [-17 if i == 0 else -23]
                                             * original[length:].numel())
                        self.assertEqual(kwargs["num_actual_tokens"], 8)
                        self.assertEqual(kwargs["spec_query_start_loc"].values(), [0, 8])
                        self.assertEqual(kwargs["spec_token_indx"].values(), list(range(8)))
                        for name in ("spec_state_indices_tensor", "num_accepted_tokens"):
                            self.assertIs(kwargs[name], getattr(case.metadata, name))
                        self.assertEqual(kwargs["num_accepted_tokens"].values(), [accepted])
                        self.assertEqual(case.metadata.spec_query_start_loc.values(), [0, length])
                        self.assertEqual(case.metadata.spec_token_indx.values(), list(range(length)))
                        self.assertEqual(case.metadata.spec_state_indices_tensor.values(), list(range(2, 10)))
                        self.assertIs(kwargs["conv_state"], case.layer.kv_cache[0])
                        self.assertIs(kwargs["ssm_state"], case.layer.kv_cache[1])
                        self.assertEqual(vars(case.metadata), metadata_before)

    def test_full_width_fast_path_and_metadata_free_warmup_unchanged(self):
        for physical in (8, 12):
            case = make_case(8, physical)
            (args, kwargs), = execute(case, strict=True)
            for got, original in zip(args[:4], case.outputs + case.inputs):
                self.assertIs(got, original)
            for name in ("spec_query_start_loc", "spec_token_indx", "spec_state_indices_tensor",
                         "num_accepted_tokens"):
                self.assertIs(kwargs[name], getattr(case.metadata, name))
            self.assertEqual(kwargs["num_actual_tokens"], 8)
        case = make_case()
        before = [t.values() for t in case.outputs]
        self.assertEqual(execute(case, no_metadata=True), [])
        self.assertEqual([t.values() for t in case.outputs], before)

    def test_non_spec_mixed_and_multi_request_paths_not_reclassified(self):
        for changes in ({"num_spec_decodes": 0, "num_decodes": 1},
                        {"num_prefills": 1, "num_prefill_tokens": 1},
                        {"num_spec_decodes": 2}, {"num_spec_decode_tokens": 0}):
            case = make_case()
            vars(case.metadata).update(changes)
            (args, kwargs), = execute(case)
            for got, original in zip(args[:4], case.outputs + case.inputs):
                self.assertIs(got, original)
            self.assertEqual(kwargs["num_actual_tokens"], case.metadata.num_actual_tokens)
            self.assertEqual(kwargs["num_prefills"], case.metadata.num_prefills)
            self.assertEqual(kwargs["num_spec_decodes"], case.metadata.num_spec_decodes)

    def test_inconsistent_partial_c1_metadata_fails_before_native_call(self):
        changes = (
            {"num_prefill_tokens": 1}, {"num_decode_tokens": 1},
            {"spec_state_indices_tensor": Tensor((2, 8), dtype="int32")},
            {"spec_query_start_loc": ints([0, 7, 7])}, {"num_accepted_tokens": ints([8, 1])},
            {"spec_token_indx": ints(range(8))}, {"num_actual_tokens": 6},
            {"non_spec_token_indx": ints([0])}, {"non_spec_state_indices_tensor": ints([2])},
            {"num_accepted_tokens": None}, {"spec_sequence_masks": None},
        )
        for change in changes:
            with self.subTest(change=change):
                case = make_case()
                vars(case.metadata).update(change)
                with self.assertRaisesRegex(RuntimeError, "unsupported C1 partial"):
                    execute(case)
        with self.assertRaisesRegex(RuntimeError, "unsupported C1 partial"):
            execute(make_case(7, physical=6))


if __name__ == "__main__":
    unittest.main()
