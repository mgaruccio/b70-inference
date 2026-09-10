"""CPU-only checks for the pinned Qwen 3.8 DFlash2 cache-group overlay.

The tests execute the actual pinned vLLM grouping function with small local
spec stubs; they do not import vLLM or launch an XPU workload.  The lead owns
native allocator and public API validation.

The source export can be overridden with B70_DFLASH2_CACHE_GROUP_SOURCE.  The
pinned export is intentionally read-only and source-dependent tests skip when
it is unavailable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "scripts/patch-vllm-qwen38-dflash2-cache-groups.py"
SOURCE_PATH = Path(
    os.environ.get(
        "B70_DFLASH2_CACHE_GROUP_SOURCE",
        str(ROOT / "results/20260910-qwen38-dflash2-cache-efficiency/reference-source/kv_cache_utils.py"),
    )
)

spec = importlib.util.spec_from_file_location("dflash2_cache_groups", PATCH_PATH)
assert spec is not None and spec.loader is not None
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, tuple[object, ...]]] = []

    def warning(self, message: str, *args: object) -> None:
        self.warnings.append((message, args))


class _Spec:
    def __init__(self, page_size_bytes: int) -> None:
        self.page_size_bytes = page_size_bytes

    def __hash__(self) -> int:
        return hash((type(self), self.page_size_bytes))

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is type(self)
            and other.page_size_bytes == self.page_size_bytes  # type: ignore[union-attr]
        )

    @classmethod
    def merge(cls, specs: list["_Spec"]) -> "_Spec":
        if not all(type(spec) is cls for spec in specs):
            raise ValueError("mixed cache-spec classes")
        if len({spec.page_size_bytes for spec in specs}) != 1:
            raise ValueError("mixed page sizes")
        return specs[0]


class MambaSpec(_Spec):
    pass


class FullAttentionSpec(_Spec):
    pass


class SlidingWindowSpec(_Spec):
    pass


class OtherSpec(_Spec):
    pass


@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: _Spec


def _create_kv_cache_group_specs(
    kv_cache_spec: dict[str, _Spec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    groups = []
    for layer_names in grouped_layer_names:
        specs = [kv_cache_spec[layer_name] for layer_name in layer_names]
        groups.append(
            KVCacheGroupSpec(layer_names, type(specs[0]).merge(specs))
        )
    return groups


def _load_pinned_source() -> str:
    if not SOURCE_PATH.is_file():
        raise unittest.SkipTest(f"pinned source export unavailable: {SOURCE_PATH}")
    return SOURCE_PATH.read_text(encoding="utf-8")


def _load_group_function(source: str):
    start = source.index(patch.START)
    end = source.index(patch.END, start)
    function_source = source[start:end]
    scope = {
        "defaultdict": defaultdict,
        "KVCacheSpec": _Spec,
        "KVCacheGroupSpec": KVCacheGroupSpec,
        "MambaSpec": MambaSpec,
        "FullAttentionSpec": FullAttentionSpec,
        "SlidingWindowSpec": SlidingWindowSpec,
        "cdiv": lambda value, divisor: (value + divisor - 1) // divisor,
        "create_kv_cache_group_specs": _create_kv_cache_group_specs,
        "logger": _Logger(),
    }
    exec(
        compile(
            "from __future__ import annotations\n" + function_source,
            "pinned_kv_cache_utils.py",
            "exec",
        ),
        scope,
    )
    return scope["_get_kv_cache_groups_uniform_page_size"]


def _layout_specs(page_size: int = 4096) -> dict[str, _Spec]:
    result: dict[str, _Spec] = {}
    for index in range(48):
        result[f"mamba.{index}"] = MambaSpec(page_size)
    for index in range(16):
        result[f"full.{index}"] = FullAttentionSpec(page_size)
    for index in range(5):
        result[f"sliding.{index}"] = SlidingWindowSpec(page_size)
    return result


class CacheGroupOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _load_pinned_source()
        digest = hashlib.sha256(
            cls.source[
                cls.source.index(patch.START) : cls.source.index(patch.END)
            ].encode()
        ).hexdigest()
        if digest != patch.ORIGINAL_FUNCTION_SHA256:
            raise AssertionError(
                f"pinned function hash changed: {digest}"
            )
    def test_default_source_vs_patched_geometry(self) -> None:
        pristine = self.source
        patched = patch.patch_text(pristine)
        self.assertNotIn(patch.MARKER, pristine)
        self.assertIn(patch.MARKER, patched)

        default_groups = _load_group_function(pristine)(_layout_specs())
        patched_groups = _load_group_function(patched)(_layout_specs())
        self.assertEqual(
            [len(group.layer_names) for group in default_groups],
            [5] * 8 + [4] * 2 + [4] * 4 + [5],
        )
        self.assertEqual(
            [len(group.layer_names) for group in patched_groups],
            [8] * 8 + [5],
        )

    def test_patched_geometry_has_coverage_homogeneity_and_three_padding_slots(self) -> None:
        specs = _layout_specs()
        groups = _load_group_function(patch.patch_text(self.source))(specs)

        names = [name for group in groups for name in group.layer_names]
        self.assertEqual(len(names), len(specs))
        self.assertEqual(len(set(names)), len(specs))
        self.assertEqual(set(names), set(specs))
        expected_names = (
            [[f"mamba.{index}" for index in range(offset, 48, 6)] for offset in range(6)]
            + [[f"full.{index}" for index in range(offset, 16, 2)] for offset in range(2)]
            + [[f"sliding.{index}" for index in range(5)]]
        )
        self.assertEqual([group.layer_names for group in groups], expected_names)
        self.assertEqual(
            [type(group.kv_cache_spec) for group in groups],
            [MambaSpec] * 6 + [FullAttentionSpec] * 2 + [SlidingWindowSpec],
        )
        self.assertTrue(
            all(
                len({type(specs[name]) for name in group.layer_names}) == 1
                for group in groups
            )
        )
        self.assertEqual(
            {group.kv_cache_spec.page_size_bytes for group in groups}, {4096}
        )
        self.assertEqual(sum(8 - len(group.layer_names) for group in groups), 3)

    def test_geometry_guard_rejects_unsupported_count_type_and_page_layout(self) -> None:
        function = _load_group_function(patch.patch_text(self.source))

        missing = _layout_specs()
        del missing["mamba.47"]
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            function(missing)

        wrong_type = _layout_specs()
        wrong_type["full.0"] = OtherSpec(4096)
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            function(wrong_type)

        non_uniform = _layout_specs()
        non_uniform["mamba.0"] = MambaSpec(8192)
        with self.assertRaisesRegex(RuntimeError, "uniform page"):
            function(non_uniform)

    def test_idempotence_drift_and_partial_marker_fail_closed(self) -> None:
        patched = patch.patch_text(self.source)
        self.assertEqual(patch.patch_text(patched), patched)

        drifted = self.source.replace(
            "group_size = min_num_layers", "group_size = min_num_layers + 1", 1
        )
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            patch.patch_text(drifted)

        partial = patched.replace("group_size = 8", "group_size = 7", 1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            patch.patch_text(partial)

        moved_marker = patched.replace(patch.MARKER, patch.MARKER + "_MOVED", 1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            patch.patch_text(moved_marker)

        marker_outside_function = self.source + f"\n# {patch.MARKER}\n"
        with self.assertRaisesRegex(RuntimeError, "partial/changed"):
            patch.patch_text(marker_outside_function)

    def test_root_cli_applies_and_replays_without_vllm_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "v1/core/kv_cache_utils.py"
            target.parent.mkdir(parents=True)
            target.write_text(self.source, encoding="utf-8")
            command = [sys.executable, str(PATCH_PATH), "--root", str(root)]

            first = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, check=False
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("patched", first.stdout)
            expected = patch.patch_text(self.source)
            self.assertEqual(target.read_text(encoding="utf-8"), expected)

            second = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, check=False
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already patched", second.stdout)
            self.assertEqual(target.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
