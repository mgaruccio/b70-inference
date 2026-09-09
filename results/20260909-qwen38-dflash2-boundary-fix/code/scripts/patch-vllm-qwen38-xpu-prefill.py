#!/usr/bin/env python3
"""Pinned 73029d424 XPU GDN one-token-prefill fix, for disposable research cells.

The runner's uniform-decode guard alone is insufficient without speculation:
GDN also calls split_decodes_and_prefills with short extends treated as decodes.
Use that existing helper's is_prefilling-aware mode on XPU. Synthetic metadata
without an is_prefilling tensor retains upstream behavior for warmup/capture.
Sources: vllm/vllm@73029d424 v1/attention/backends/{gdn_attn,utils}.py;
related prefill-classification issue/fix: https://github.com/vllm-project/vllm/pull/53059.
"""
import argparse
import importlib.util
from pathlib import Path

MARKER = "B70_XPU_GDN_SHORT_PREFILL"
IMPORT = "from vllm.config import VllmConfig\n"
OLD = "                split_decodes_and_prefills(m, decode_threshold=1)\n"
NEW = """                split_decodes_and_prefills(
                    m, decode_threshold=1,
                    # B70_XPU_GDN_SHORT_PREFILL: initialize fresh one-token state.
                    treat_short_extends_as_decodes=(
                        not current_platform.is_xpu() or m.is_prefilling is None
                    ),
                )
"""


def patch_text(source):
    if MARKER in source:
        if source.count(NEW) != 1 or source.count("from vllm.platforms import current_platform\n") != 1:
            raise RuntimeError("partial/changed XPU GDN prefill patch")
        return source
    if source.count(OLD) != 1 or source.count(IMPORT) != 1:
        raise RuntimeError("pinned GDN short-prefill anchors changed")
    result = source.replace(IMPORT, IMPORT + "from vllm.platforms import current_platform\n")
    result = result.replace(OLD, NEW)
    compile(result, "gdn_attn.py", "exec")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="vllm package root; omit inside serving container")
    args = parser.parse_args()
    root = args.root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("vllm package not found")
        root = Path(spec.origin).parent
    path = root / "v1/attention/backends/gdn_attn.py"
    before = path.read_text()
    after = patch_text(before)
    if after != before:
        path.write_text(after)
    print(f"{MARKER}: {path}", flush=True)


if __name__ == "__main__":
    main()
