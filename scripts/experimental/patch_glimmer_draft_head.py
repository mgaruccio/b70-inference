#!/usr/bin/env python3
"""Opt-in hooks for the pinned vLLM V2 DFlash worker; never patch target logits.

Run inside a disposable container with glimmer_draft_head.py on PYTHONPATH.
Both source anchors are validated before either file is written. The normal
repository launcher does not invoke this experimental patcher.
"""
from __future__ import annotations

import argparse
from pathlib import Path

LOAD_PATH = "v1/worker/gpu/spec_decode/dflash/utils.py"
SAMPLE_PATH = "v1/worker/gpu/spec_decode/speculator.py"
LOAD_OLD = "    return dflash_model\n"
LOAD_NEW = (
    "    from glimmer_draft_head import attach_draft_head\n"
    "    attach_draft_head(dflash_model, vllm_config)\n"
    "    return dflash_model\n"
)
SAMPLE_OLD = (
    "    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
    "        if self.use_local_argmax_reduction:\n"
)
SAMPLE_NEW = (
    "    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
    "        if hasattr(self.model, 'glimmer_draft_head'):\n"
    "            return self.model.glimmer_draft_head(hidden_states)\n"
    "        if self.use_local_argmax_reduction:\n"
)


def patch_sources(root: Path) -> list[str]:
    pending: list[tuple[Path, str]] = []
    for relative, old, new in (
        (LOAD_PATH, LOAD_OLD, LOAD_NEW),
        (SAMPLE_PATH, SAMPLE_OLD, SAMPLE_NEW),
    ):
        path = root / relative
        source = path.read_text()
        if source.count(new) == 1:
            continue
        if source.count(old) != 1 or "glimmer_draft_head" in source:
            raise ValueError(f"Unsupported or partially patched source: {relative}")
        candidate = source.replace(old, new, 1)
        compile(candidate, str(path), "exec")
        pending.append((path, candidate))
    for path, candidate in pending:
        path.write_text(candidate)
    return [str(path.relative_to(root)) for path, _ in pending]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Pinned installed vllm package directory")
    args = parser.parse_args()
    changed = patch_sources(args.root)
    print("Draft-only V2 hooks: " + (", ".join(changed) or "already patched"))


if __name__ == "__main__":
    main()
