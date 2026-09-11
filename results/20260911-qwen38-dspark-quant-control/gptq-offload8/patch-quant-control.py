#!/usr/bin/env python3
"""Compose the supplied canonical DSpark overlay with this diagnostic-only guard.

No GPTQ spoofing, kernel selection override, production edit or numeric change.
The canonical prepare/apply enforces all-pristine or exact full replay atomically.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import runpy

EXTRA_PINS = {
    "model_executor/layers/quantization/fp8.py": "f71856c684febd1a57414ecf9a37f770790e66214f94ed3d237912247a18eb08",
    "model_executor/kernels/linear/__init__.py": "d0710fdbe617209ef99e22e053a0ffc8a0ece8c83d04eea5083f6aaa8c19d2d8",
    "model_executor/kernels/linear/scaled_mm/xpu.py": "ba123182e4f9505f553227e06c8adeb68a0307800dea69f6e6614e25cd7541d1",
    "model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py": "26807d4a0a78dbbdcb5724c15905f3a39007924f858494cce9e88953e4cbbc76",
    "model_executor/layers/quantization/input_quant_fp8.py": "b16ee19f3f75affbaad6feb2099583932f8e4d5c51616b84d8aa5316a72b6ef2",
    "model_executor/layers/quantization/utils/fp8_utils.py": "f64a5408cd6d0fced5cd3dcc19b83883e5e0485c29bf6b34ff28217e6b387cd3",
    "model_executor/layers/quantization/auto_gptq.py": "65ee92761c623185a38d6ea96f6d6e6bea1fbff09b005d4dd2249512362936bf",
    "model_executor/kernels/linear/mixed_precision/xpu.py": "de94f0fc2813c5f86e44369490809a7c1aab44cac8e20e5a64a5a1896ba7d9ba",
    "model_executor/offloader/uva.py": "f0b16c969227b34a99032008419dc1ba8b8c5126f800d97b50a366adc957cd75",
    "model_executor/offloader/base.py": "5157a59232715e7247761588efb88fc44e14970b580722a1f0b2e8b3a23a10fe",
    "model_executor/utils.py": "f208310647a012797e0b0e9631d3498135c7f9aef831e35e42e96eceb7623f89",
}
QUANT_GUARD = '''    qc = getattr(target.hf_config, "quantization_config", {})
    if (not isinstance(qc, dict) or qc.get("quant_method") != "gptq"
            or qc.get("bits") != 4 or qc.get("group_size") != 128
            or qc.get("sym") is not True or qc.get("desc_act") is not False
            or qc.get("lm_head") is not False):
        raise ValueError("B70 DSpark requires target GPTQ Int4 symmetric G128, no act-order/head quantization")
'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("canonical overlay guard anchor missing/ambiguous; do not bypass")
    return source.replace(old, new, 1)


def build_overlay(canonical, expected_sha):
    data = Path(canonical).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise RuntimeError("canonical overlay SHA256 mismatch")
    loaded = runpy.run_path(str(canonical))
    ns = loaded["prepare"].__globals__
    helpers = ns["CONFIG_HELPERS"]
    helpers = replace_once(helpers, 'target.quantization not in ("gptq", "auto_gptq")',
                           'target.quantization not in ("gptq", "auto_gptq", "fp8")')
    # Preserve the original GPTQ guard verbatim inside the else branch. Every
    # dtype/architecture/vocab/cache/rejection/draft guard outside it is unchanged.
    helpers = replace_once(helpers, QUANT_GUARD,
        '    if target.quantization == "fp8":\n'
        '        from vllm._b70_quant_observe import validate_fp8_target\n'
        '        validate_fp8_target(target)\n'
        '    else:\n' + "".join("    " + line + "\n" for line in QUANT_GUARD.splitlines()))
    ns["CONFIG_HELPERS"] = helpers
    for name, pin in EXTRA_PINS.items():
        if name in ns["PINNED_SHA256"] and ns["PINNED_SHA256"][name] != pin:
            raise RuntimeError("canonical/native source pin conflict: " + name)
    ns["PINNED_SHA256"] = {**ns["PINNED_SHA256"], **EXTRA_PINS}
    canonical_transformations = ns["transformations"]

    def transformations():
        edits = canonical_transformations()
        hooks = {
            "model_executor/offloader/uva.py": [
                ('            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()\n',
                 '            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()\n'
                 '            from vllm._b70_quant_observe import record_offload\n'
                 '            record_offload(self, p)\n')],
            "model_executor/model_loader/utils.py": [
                ('                p._vllm_is_uva_offloaded = True\n',
                 '                p._vllm_is_uva_offloaded = True\n'
                 '                from vllm._b70_quant_observe import record_reoffload\n'
                 '                record_reoffload(p)\n')],
            "model_executor/model_loader/base_loader.py": [
                ('        return model.eval()\n',
                 '        from vllm._b70_quant_observe import inspect_target\n'
                 '        inspect_target(model, model_config)\n'
                 '        return model.eval()\n')],
            "v1/worker/gpu/model_runner.py": [
                ('        time_after_load = time.perf_counter()\n',
                 '        time_after_load = time.perf_counter()\n'
                 '        from vllm._b70_quant_observe import inspect_loaded_runner\n'
                 '        inspect_loaded_runner(self)\n')],
        }
        for name, pairs in hooks.items():
            edits.setdefault(name, []).extend(pairs)
        return edits

    ns["transformations"] = transformations
    return ns


# The existing driver's source_check uses runpy and the canonical prepare API.
if os.environ.get("B70_QUANT_CANONICAL_SHA"):
    _overlay = build_overlay(os.environ.get("B70_QUANT_CANONICAL", "/quant/canonical.py"),
                             os.environ["B70_QUANT_CANONICAL_SHA"])
    PINNED_SHA256 = _overlay["PINNED_SHA256"]
    prepare = _overlay["prepare"]
    apply = _overlay["apply"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="exported pristine package root, CPU transform tests only")
    args = parser.parse_args()
    if os.environ.get("B70_QUANT_CONTROL_ARM") not in ("gptq", "fp8"):
        parser.error("explicit B70_QUANT_CONTROL_ARM=gptq|fp8 is required")
    if not os.environ.get("B70_QUANT_CANONICAL_SHA"):
        parser.error("B70_QUANT_CANONICAL_SHA is required")
    root = args.root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        root = Path(next(iter(spec.submodule_search_locations)))
    apply(root)


if __name__ == "__main__":
    main()
