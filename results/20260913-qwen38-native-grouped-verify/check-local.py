"""Stdlib-only fixtures/AST/shell checks. NOT compiler, GPU or numerical qualification."""
import argparse
import ast
import itertools
import math
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parent


def strides(shape):
    return tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))


class Tensor:
    """Lazy symbolic values: no Torch import, device, compiler, or ML runtime."""
    device = SimpleNamespace(type="xpu")

    def __init__(self, shape, dtype="fp16", pitch=None, value=lambda i: i):
        self.shape, self.dtype = tuple(shape), dtype
        self.pitch = tuple(pitch) if pitch is not None else strides(self.shape)
        self.value = value

    def stride(self, i=None):
        return self.pitch if i is None else self.pitch[i]

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.pitch == strides(self.shape)

    def reshape(self, *shape):
        assert math.prod(shape) == self.numel()
        def value(i):
            linear = sum(x * s for x, s in zip(i, strides(shape)))
            old = tuple((linear // s) % n for n, s in zip(self.shape, strides(self.shape)))
            return self.value(old)
        return Tensor(shape, self.dtype, value=value)

    def view(self, *shape):
        assert self.is_contiguous()
        result = self.reshape(*shape)
        result.view_parent = self
        return result

    def permute(self, *order):
        inverse = tuple(order.index(i) for i in range(len(order)))
        return Tensor(tuple(self.shape[i] for i in order), self.dtype,
                      tuple(self.pitch[i] for i in order),
                      lambda i: self.value(tuple(i[x] for x in inverse)))

    def contiguous(self):
        return Tensor(self.shape, self.dtype, value=self.value)

    def clamp_min(self, n):
        return Tensor(self.shape, self.dtype, value=lambda i: max(n, self.value(i)))

    def copy_(self, other):
        assert self.shape == other.shape
        if hasattr(self, "view_parent"):
            parent = self.view_parent
            def value(i):
                linear = sum(x * s for x, s in zip(i, strides(parent.shape)))
                index = tuple((linear // s) % n for n, s in zip(self.shape, strides(self.shape)))
                return other.value(index)
            parent.value = value
        self.value = other.value
        return self


def python_seam_fixture():
    tree = ast.parse((ROOT / "grouped_verify.py").read_text())
    assert not any(isinstance(n, ast.Attribute) and n.attr in
                   ("item", "cpu", "numpy", "tolist") for n in ast.walk(tree))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Import) and
                 any(a.name == "torch" for a in n.names))]
    calls = []
    live = [0]

    def operator(q, k, v, used, table, cu, ks, vs, maximum, scale):
        assert q.shape == (1, 120, 256) and q.is_contiguous()
        assert [cu.value((i,)) for i in range(2)] == [0, 1]
        assert used.value((0,)) == max(live[0], 1)
        assert table.shape == (1, 128)
        for t, h, d in itertools.product(range(5), range(24), (0, 127, 255)):
            assert q.value((0, (h // 6) * 30 + t * 6 + h % 6, d)) == (t, h, d)
        calls.append(live[0])
        return q  # An identity payload tests inverse unpack, not attention math.

    torch = SimpleNamespace(
        Tensor=Tensor, float16="fp16", float8_e4m3fn="fp8", float32="fp32", int32="i32",
        arange=lambda n, **kw: Tensor((n,), kw["dtype"], value=lambda i: i[0]),
        empty=lambda shape, **kw: Tensor(shape, kw["dtype"], value=lambda i: "unwritten"),
        ops=SimpleNamespace(b70_grouped_verify=SimpleNamespace(forward=operator)))
    namespace = {"torch": torch}
    exec(compile(tree, str(ROOT / "grouped_verify.py"), "exec"), namespace)
    q = Tensor((5, 24, 256), pitch=(12288, 512, 1))
    kv = Tensor((176, 1664, 4, 256), "fp8", (1664 * 4 * 256, 256, 1664 * 256, 1))
    used = Tensor((1,), "i32", value=lambda i: live[0])
    descale = Tensor((1, 4), "fp32", (0, 0))
    args = [q, kv, kv, Tensor(q.shape, value=lambda i: "unwritten"), Tensor((2,), "i32"), used,
            Tensor((1, 128), "i32"), 212992, descale, descale, .0625, None, (-1, -1), 0.0]
    fallback = []
    def native(*a, **kw):
        fallback.append(True)
        return "fallback"
    for length in (0, 1, 4, 1665, 65541):
        live[0] = length
        assert namespace["unsupported_reason"](*args) is None
        y = namespace["dispatch"](native, *args)
        assert y is args[3]
        for t, h, d in itertools.product(range(5), range(24), (0, 127, 255)):
            assert y.value((t, h, d)) == (t, h, d)
    args[3] = None
    allocated = namespace["dispatch"](native, *args)
    assert allocated.shape == (5, 24, 256)
    assert allocated.value((4, 23, 255)) == (4, 23, 255)
    for index, bad in ((0, Tensor((4, 24, 256))), (6, Tensor((5, 128), "i32")),
                       (8, Tensor((1, 4), "fp32")), (3, q), (7, 212993)):
        bad_args = args.copy()
        bad_args[index] = bad
        assert namespace["dispatch"](native, *bad_args) == "fallback"
    def broken(*a):
        raise RuntimeError("eligible failure")
    namespace["packed_forward"] = broken
    try:
        namespace["dispatch"](native, *args)
    except RuntimeError as exc:
        assert str(exc) == "eligible failure"
    else:
        raise AssertionError("eligible exception swallowed")
    assert len(fallback) == 5 and len(calls) == 6


def mask_fixture():
    # Include q8 and q16 global row tiles, <5 lengths, and key boundaries.
    # No device execution; real graph mutation qualification remains required.
    for used in (0, 1, 2, 3, 4, 5, 6, 63, 64, 65, 1663, 1664, 1665, 1985, 2049, 65541):
        for width in (8, 16):
            for tile, local in itertools.product(range(32 // width), range(width)):
                row = tile * width + local
                limit = max(max(used, 1) - 4 + row // 6, 1)
                for key in (0, 1, 63, 64, 1663, 1664, max(used - 1, 0), used):
                    candidate = row < 30 and key < limit
                    expected = row < 30 and key < max(used - 4 + row // 6, 1)
                    assert candidate == expected
    assert max(1665 - 4 + 16 // 6, 1) != max(1665 - 4 + 0 // 6, 1)
    # Lowest finite max + zero sum is native's empty-row/split representation.
    assert math.exp(-math.inf - (-3.4028235e38)) == 0


def upstream_fixture():
    patcher = runpy.run_path(str(ROOT / "patch-native.py"))
    for name in ("NATIVE_SHA", "TLA_SHA"):
        assert len(patcher[name]) == 40 and int(patcher[name], 16)
    url = "https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/" + patcher["NATIVE_SHA"] + "/"
    files = [("collective/chunk_prefill_mainloop.hpp", "patched_mainloop"),
             ("paged_decode.hpp", "patched_config")]
    for relative, function in files:
        with urllib.request.urlopen(url + patcher["BASE"] + relative, timeout=25) as response:
            old = response.read().decode()
        new = patcher[function](old)
        assert new != old
        if function == "patched_mainloop":
            assert "bool PackedVerify_ = false" in new
            assert "cute::max(seq_len - 4 + group_row / 6, 1)" in new
            assert "group_row >= 30 || key_pos >= causal_limit" in new
            assert old.split("struct DecodeFwdMainloop")[0].split("template <\n    class DispatchPolicy_,\n    bool PagedKV_,")[0] == new.split("template <\n    class DispatchPolicy_,\n    bool PagedKV_,")[0]
            assert old[old.rfind("  // Single step of blocked softmax."):] == new[new.rfind("  // Single step of blocked softmax."):]
        try:
            patcher[function](new)
        except ValueError:
            pass
        else:
            raise AssertionError("patch accepted an already-patched header")


def serving_overlay_fixture():
    """Exercise the import hook with fake modules, never Torch or an XPU."""
    import contextlib
    import hashlib
    import io
    import os
    import sys
    import tempfile
    import types

    names = ("torch", "vllm_xpu_kernels", "vllm_xpu_kernels.flash_attn_interface", "grouped_verify")
    saved_modules = {name: sys.modules.get(name) for name in names}
    saved_env = {name: os.environ.get(name) for name in (
        "B70_GROUPED_SERVING", "B70_GROUPED_SERVING_LIBRARY"
    )}

    class FakeTensor:
        def __init__(self, shape, pitch, dtype="fp16"):
            self.shape = tuple(shape)
            self._pitch = tuple(pitch)
            self.dtype = dtype
            self.device = "xpu"

        def stride(self):
            return self._pitch

    loaded = []
    dispatch_natives = []
    native_calls = []
    eligible_failure = []
    torch = types.ModuleType("torch")
    torch.ops = SimpleNamespace(load_library=lambda path: loaded.append(path))
    torch.xpu = SimpleNamespace(is_current_stream_capturing=lambda: True)

    fa = types.ModuleType("vllm_xpu_kernels.flash_attn_interface")
    original_vllm_fa2 = object()

    def native(*args, **kwargs):
        native_calls.append((args, kwargs))
        return "native"

    fa._spec_decode_varlen_fwd = native
    fa._vllm_fa2_C = original_vllm_fa2
    kernels = types.ModuleType("vllm_xpu_kernels")
    kernels.flash_attn_interface = fa

    def unsupported_reason(*args, **kwargs):
        key = args[1] if len(args) > 1 else kwargs.get("k")
        return None if getattr(key, "shape", ()) == (176, 1664, 4, 256) else "wrong KV geometry"

    def dispatch(original, *args, **kwargs):
        dispatch_natives.append(original)
        if eligible_failure:
            raise RuntimeError("eligible failure")
        if unsupported_reason(*args, **kwargs) is not None:
            return original(*args, **kwargs)
        return "candidate"

    seam = types.ModuleType("grouped_verify")
    seam.dispatch = dispatch
    seam.unsupported_reason = unsupported_reason
    sys.modules.update({
        "torch": torch,
        "vllm_xpu_kernels": kernels,
        "vllm_xpu_kernels.flash_attn_interface": fa,
        "grouped_verify": seam,
    })

    try:
        with tempfile.NamedTemporaryFile(dir=ROOT, delete=True) as library:
            library.write(b"CPU fixture, not a shared object")
            library.flush()
            namespace = runpy.run_path(str(ROOT / "serving-overlay.py"))
            assert namespace["install"]() is False
            assert namespace["install_worker_profile"](type("Worker", (), {})) is False
            assert not loaded

            os.environ["B70_GROUPED_SERVING"] = "1"
            os.environ["B70_GROUPED_SERVING_LIBRARY"] = library.name
            worker = type("Worker", (), {})
            try:
                namespace["install_worker_profile"](worker)
            except RuntimeError as exc:
                assert "hash mismatch" in str(exc)
            else:
                raise AssertionError("library hash mismatch was not rejected")
            assert not loaded
            library.seek(0)
            overlay_globals = namespace["install_worker_profile"].__globals__
            overlay_globals["EXPECTED_LIBRARY_SHA256"] = hashlib.sha256(
                library.read()
            ).hexdigest()
            q = FakeTensor((5, 24, 256), (12288, 512, 1))
            good_k = FakeTensor(
                (176, 1664, 4, 256),
                (176 * 1664 * 4 * 256, 256, 1664 * 256, 1),
                "fp8",
            )
            bad_k = FakeTensor((136, 1664, 4, 256), (136 * 1664 * 4 * 256, 256, 1664 * 256, 1), "fp8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                assert namespace["install_worker_profile"](worker) is True
                wrapped = fa._spec_decode_varlen_fwd
                assert wrapped(q, bad_k, bad_k) == "native"
                assert wrapped(q, good_k, good_k) == "candidate"
                assert wrapped(q, good_k, good_k) == "candidate"
                assert wrapped(q, bad_k, bad_k) == "native"
            text = output.getvalue()
            assert loaded == [library.name]
            assert all(original is native for original in dispatch_natives)
            assert fa._vllm_fa2_C is original_vllm_fa2
            assert text.count('"event":"unsupported-q5"') == 1
            assert text.count('"event":"eligible-dispatch"') == 1
            assert '"capture_state":"capturing"' in text
            assert '"q_shape":[5,24,256]' in text
            assert '"k_shape":[176,1664,4,256]' in text
            assert '"reason":"wrong KV geometry"' in text
            eligible_failure.append(True)
            try:
                wrapped(q, good_k, good_k)
            except RuntimeError as exc:
                assert str(exc) == "eligible failure"
            else:
                raise AssertionError("eligible exception swallowed")
            assert len(native_calls) == 2
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name, previous in saved_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def launch_contract_fixture():
    """Check the disposable wrapper's pure launch/evidence contract."""
    import tempfile
    namespace = runpy.run_path(str(ROOT / "run-serving.py"))
    assert namespace["EXPECTED_LIBRARY_SHA256"] == (
        "4630ef2db027c3443ff63b16a611699c0db250c2cc53ed67aaad8a1a318f4490"
    )
    mounts = namespace["_candidate_mounts"](
        ROOT / "serving-overlay.py",
        ROOT / "qwen38_step_timing_patch.py",
        ROOT / "grouped_verify.py",
        ROOT / "libb70_grouped_verify.so",
    )
    assert [mount["container"] for mount in mounts] == [
        "/experiment/qwen38_step_timing_overlay.py",
        "/experiment/qwen38_step_timing_patch.py",
        "/experiment/grouped_verify.py",
        "/candidate/libb70_grouped_verify.so",
    ]
    assert all(mount["mode"] == "ro" for mount in mounts)
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        out = Path(directory)
        (out / "server.log").write_text(
            "Capturing CUDA graphs (decode, FULL)\n"
            "[B70_GROUPED_SERVING] {\"event\":\"eligible-dispatch\"}\n"
            "| FULL         |\n",
            encoding="utf-8",
        )
        evidence = namespace["_candidate_execution_evidence"](out)
        assert evidence["eligible_dispatch_log_count"] == 1
        assert evidence["full_graph_capture_seen"]
        assert evidence["full_graph_run_seen"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", action="store_true", help="also check pinned headers over HTTPS")
    options = parser.parse_args()
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    for path in ROOT.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    python_seam_fixture()
    mask_fixture()
    serving_overlay_fixture()
    launch_contract_fixture()
    if options.upstream:
        upstream_fixture()
    print("PASS: stdlib pack/unpack, device-metadata seam, fail-closed dispatch, two-tile mask, serving hook/launch contract, Python AST, shell syntax"
          + (", pinned upstream patch/drift checks" if options.upstream else ""))
