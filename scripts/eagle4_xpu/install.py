#!/usr/bin/env python3
"""Install the opt-in EAGLE4 chain adapter in the pinned XPU vLLM container.

Run before starting vLLM (all engine/registry subprocesses must see the overlay):
    python /overlay/eagle4_xpu/install.py
Use --check for a read-only compatibility check. Re-running is a no-op. This does
not launch a server, change a checkpoint, or install the official tree harness.

Image: vllm/vllm-openai-xpu@sha256:
f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
Speculative config: method=eagle3, model=<released checkpoint>,
num_speculative_tokens=<chain length>, quantization=null, kv_cache_dtype=auto.
Use --dtype float16 --enforce-eager --max-num-seqs 1, TP1 and unquantized KV.
The real HTTP/XPU acceptance test belongs to the invoking experiment, not this
installer's source checks or the CPU unit tests.
"""

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path


PINNED_VERSION = "0.27.2rc1.dev77+gac7509e2b"
BASE_PATH = "model_executor/models/llama_eagle3.py"
BASE_SHA256 = "2fc44507dc81f5793f41e5c9686d9d963f2c5fe63e66d3e1cdfa49a95e185fee"
MODEL_PATH = "model_executor/models/eagle4_xpu.py"

# Small source overlays, not new proposer/verifier hooks. Other model families
# keep their existing behavior, including ordinary Eagle3 architecture names.
PATCHES = {
    "model_executor/models/registry.py": (
        "_SPECULATIVE_DECODING_MODELS = {\n",
        "_SPECULATIVE_DECODING_MODELS = {\n"
        '    "LlamaForCausalLMEagle4": ("eagle4_xpu", "LlamaForCausalLMEagle4"),\n',
    ),
    "transformers_utils/configs/eagle.py": (
        '                if arch.startswith("Eagle3") or arch.endswith("Eagle3")\n',
        '                if (arch.startswith("Eagle3") or arch.endswith("Eagle3")\n'
        '                    or arch == "LlamaForCausalLMEagle4")\n',
    ),
    "config/speculative.py": (
        "                # Automatically detect the method\n",
        '                # EAGLE4_XPU_CHAIN: the released weights use Eagle3 verification.\n'
        '                if "LlamaForCausalLMEagle4" in self.draft_model_config.architectures:\n'
        '                    if self.method not in (None, "draft_model", "eagle3"):\n'
        '                        raise ValueError("EAGLE4 requires method=eagle3 (chain mode)")\n'
        '                    self.method = "eagle3"\n\n'
        "                # Automatically detect the method\n",
    ),
}


def patch_source(source: str, before: str, after: str, path: str) -> str:
    if source.count(after) == 1:
        # An insertion contains its own before-anchor; do not accept another copy.
        remainder = source.replace(after, "", 1)
        if before in remainder:
            raise ValueError(f"Ambiguous EAGLE4 overlay in {path}")
        return source
    if source.count(before) != 1 or "EAGLE4_XPU_CHAIN" in source or (
        "LlamaForCausalLMEagle4" in source
    ):
        raise ValueError(f"Unsupported or conflicting EAGLE4 overlay source: {path}")
    return source.replace(before, after, 1)


def prepare_overlay(root: Path) -> dict[Path, str]:
    """Validate every dependency/transform before the first write."""
    version_tree = ast.parse((root / "_version.py").read_text())
    versions = [
        ast.literal_eval(node.value)
        for node in version_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    if versions != [PINNED_VERSION]:
        raise ValueError(f"EAGLE4 overlay requires vLLM {PINNED_VERSION}; found {versions}")
    base = (root / BASE_PATH).read_bytes()
    if hashlib.sha256(base).hexdigest() != BASE_SHA256:
        raise ValueError("EAGLE4 overlay requires the pinned, unmodified EAGLE3 model base")
    updates = {}
    for relative, (before, after) in PATCHES.items():
        path = root / relative
        source = path.read_text()
        patched = patch_source(source, before, after, relative)
        compile(patched, str(path), "exec")
        if patched != source:
            updates[path] = patched
    model_path = root / MODEL_PATH
    model_source = Path(__file__).with_name("model.py").read_text()
    compile(model_source, str(model_path), "exec")
    if model_path.exists():
        if model_path.read_text() != model_source:
            raise ValueError(f"Refusing to overwrite a different adapter at {model_path}")
    else:
        updates[model_path] = model_source
    return updates


def install(root: Path, *, check: bool = False) -> list[Path]:
    updates = prepare_overlay(root)
    if not check:
        for path, source in updates.items():
            path.write_text(source)
    return list(updates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="vllm package directory")
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args()
    root = args.root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            parser.error("vLLM not found; supply --root")
        root = Path(spec.origin).parent
    paths = install(root.resolve(), check=args.check)
    print(f"EAGLE4 chain overlay: {'checked' if args.check else 'installed'}; "
          f"{len(paths)} file(s) {'would change' if args.check else 'changed'}")


if __name__ == "__main__":
    main()
