#!/usr/bin/env python3
"""Compose the campaign import with the canonical disposable timing patch.

The canonical patch remains the owner of the XPU graph replay hook and worker
profile lifecycle.  This small wrapper reuses its idempotent source transform,
then adds the campaign-only draft attribution import.  It is run inside the
throwaway serving container and never edits a host checkout.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import stat
import sys


MARKER = "B70_MTP4_DRAFT_ATTRIBUTION_INSTALLED"
DRAFT_IMPORT_BLOCK = '''

# B70_MTP4_DRAFT_ATTRIBUTION_INSTALLED: bounded post-profile scopes only.
import os as _B70_DRAFT_ATTRIBUTION_OS
if _B70_DRAFT_ATTRIBUTION_OS.getenv("B70_DRAFT_ATTRIBUTION") == "1":
    import draft_annotations as _B70_DRAFT_ANNOTATIONS
    _B70_DRAFT_ANNOTATIONS.install_worker_profile(XPUWorker)
'''


def _canonical_module():
    """Load qwen38_step_timing_patch from the mounted canonical timing dir."""
    try:
        import qwen38_step_timing_patch as module  # type: ignore[import-not-found]
        return module
    except ImportError:
        candidates = [
            Path(__file__).resolve().parent.parent / "timing",
            Path(__file__).resolve().parents[2] / "scripts" / "experiments",
        ]
        path = next(
            (candidate / "qwen38_step_timing_patch.py" for candidate in candidates
             if (candidate / "qwen38_step_timing_patch.py").is_file()),
            None,
        )
        if path is None:
            checked = ", ".join(str(candidate) for candidate in candidates)
            raise RuntimeError(f"canonical timing patch is unavailable; checked: {checked}")
        spec = importlib.util.spec_from_file_location("qwen38_step_timing_patch", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load canonical timing patch: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def patch_text(source: str) -> str:
    """Return a compiled, idempotent canonical-plus-campaign worker source."""
    canonical = _canonical_module()
    base = canonical.patch_text(source)
    if MARKER in base:
        if base.count(MARKER) != 1 or base.count(DRAFT_IMPORT_BLOCK.strip()) != 1:
            raise RuntimeError("partial/ambiguous draft attribution worker patch")
        compile(base, "xpu_worker.py", "exec")
        return base
    patched = base.rstrip() + DRAFT_IMPORT_BLOCK + "\n"
    compile(patched, "xpu_worker.py", "exec")
    return patched


def _find_root(canonical, explicit: Path | None) -> Path:
    return canonical._find_root(explicit)  # type: ignore[attr-defined]


def patch_file(path: Path) -> bool:
    before = path.read_text(encoding="utf-8")
    after = patch_text(before)
    if after == before:
        return False
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(after, encoding="utf-8")
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="installed vLLM package root")
    args = parser.parse_args(argv)
    canonical = _canonical_module()
    root = _find_root(canonical, args.root)
    path = root / "v1" / "worker" / "xpu_worker.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    changed = patch_file(path)
    print(f"{MARKER}: {path} ({'patched' if changed else 'already patched'})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
