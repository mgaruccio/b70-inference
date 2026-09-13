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
    if options.upstream:
        upstream_fixture()
    print("PASS: stdlib pack/unpack, device-metadata seam, fail-closed dispatch, two-tile mask, Python AST, shell syntax"
          + (", pinned upstream patch/drift checks" if options.upstream else ""))
