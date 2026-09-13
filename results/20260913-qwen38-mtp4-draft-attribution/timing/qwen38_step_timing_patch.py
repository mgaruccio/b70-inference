#!/usr/bin/env python3
"""Apply the disposable step-timing import to an installed vLLM XPU worker.

The patch changes only the serving container's writable copy of
``vllm/v1/worker/xpu_worker.py``.  It is intentionally opt-in and refuses to
silently rewrite a source tree whose import anchor is ambiguous.

Container startup (with both files mounted read-only) is:

    -v "$HERE/qwen38_step_timing_overlay.py:/experiment/qwen38_step_timing_overlay.py:ro" \
    -v "$HERE/qwen38_step_timing_patch.py:/experiment/qwen38_step_timing_patch.py:ro" \
    -e B70_STEP_TIMING=1 -e B70_STEP_TIMING_DIR=/output/step-timing \
    PYTHONPATH=/experiment /opt/venv/bin/python -P \
        /experiment/qwen38_step_timing_patch.py \
        --root /opt/venv/lib/python3.12/site-packages/vllm
    PYTHONPATH=/experiment exec vllm serve ...

Required environment for a native ``/start_profile`` -> ``/stop_profile``
window:

    B70_STEP_TIMING=1
    B70_STEP_TIMING_DIR=/output/step-timing
    B70_STEP_TIMING_MAX_SAMPLES=32

The existing vLLM profiler flags may be enabled independently.  To use the
overlay without the torch kernel profiler, add ``B70_STEP_TIMING_AUTO=1``;
the first non-capture graph replay starts a bounded window and writes at the
sample bound.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import stat


MARKER = "B70_STEP_TIMING_OVERLAY_INSTALLED"
IMPORT_ANCHOR = "import torch\n"
IMPORT_BLOCK = '''import torch

# B70_STEP_TIMING_OVERLAY_INSTALLED: disposable, opt-in graph timing only.
_B70_STEP_TIMING_OVERLAY = None
if os.getenv("B70_STEP_TIMING") == "1":
    import qwen38_step_timing_overlay as _B70_STEP_TIMING_OVERLAY
    _B70_STEP_TIMING_OVERLAY.install()
'''
END_BLOCK = '''

if _B70_STEP_TIMING_OVERLAY is not None:
    _B70_STEP_TIMING_OVERLAY.install_worker_profile(XPUWorker)
'''


def patch_text(source: str) -> str:
    """Return a compiled, idempotently patched worker source."""
    if MARKER in source:
        if source.count(MARKER) != 1 or source.count(END_BLOCK.strip()) != 1:
            raise RuntimeError("partial/ambiguous step-timing worker patch")
        compile(source, "xpu_worker.py", "exec")
        return source
    if source.count(IMPORT_ANCHOR) != 1:
        raise RuntimeError("xpu_worker.py torch import anchor changed or is ambiguous")
    if "class XPUWorker" not in source:
        raise RuntimeError("xpu_worker.py has no XPUWorker class")
    patched = source.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)
    patched = patched.rstrip() + END_BLOCK + "\n"
    compile(patched, "xpu_worker.py", "exec")
    return patched


def _find_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm package not found; pass --root inside the serving image")
    return Path(spec.origin).parent


def patch_file(path: Path) -> bool:
    before = path.read_text()
    after = patch_text(before)
    if after == before:
        return False
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(after)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="installed vllm package root")
    args = parser.parse_args()
    root = _find_root(args.root)
    path = root / "v1" / "worker" / "xpu_worker.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    changed = patch_file(path)
    print(f"{MARKER}: {path} ({'patched' if changed else 'already patched'})", flush=True)


if __name__ == "__main__":
    main()
