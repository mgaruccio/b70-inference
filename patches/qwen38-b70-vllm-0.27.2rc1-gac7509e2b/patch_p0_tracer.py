#!/usr/bin/env python3
"""Install or remove the Qwen B70 P0 tracer bootstrap overlay.

This writes a small bootstrap file (``sitecustomize.py`` by default) that adds
this patch directory to ``sys.path`` and calls ``p0_tracer.install_hooks()``.
It does not rewrite pinned vLLM source files.
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path

OVERLAY_VERSION = "qwen38-b70-p0-tracer-v1"
IMAGE_DIGEST = (
    "sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
)
SOURCE_COMMIT = "ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9"
PATCH_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/b70_p0_tracer.pth"
)

OVERLAY_MARKER = "B70_P0_TRACER_V1"
REFUSED_TARGETS = frozenset(
    {
        "gpu_model_runner.py",
        "gdn_attn.py",
        "rejection_sampler.py",
    }
)


def _bootstrap_text() -> str:
    patch_dir = str(PATCH_DIR)
    # A single "import ..." line is valid both as sitecustomize.py and as a
    # site .pth hook. Debian images often already own stdlib sitecustomize.py.
    return (
        "import sys; "
        f"sys.path.insert(0, {patch_dir!r}); "
        "from p0_tracer import install_import_hook; "
        f"install_import_hook()  # {OVERLAY_MARKER}\n"
    )


def _validate_target(path: Path) -> None:
    if path.name in REFUSED_TARGETS:
        raise ValueError(
            f"refusing to patch vLLM source file {path.name}; "
            "use the bootstrap target instead"
        )


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read()


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def patch_bytes(original: bytes) -> bytes:
    """Return bootstrap bytes with the tracer installed."""
    marker = OVERLAY_MARKER.encode("utf-8")
    bootstrap = _bootstrap_text().encode("utf-8")
    if marker in original:
        if bootstrap == original:
            return original
        raise ValueError("target already contains an incompatible P0 tracer bootstrap")
    if original.strip():
        raise ValueError(
            "target bootstrap already exists without the P0 tracer marker; "
            "refusing to overwrite foreign bootstrap content"
        )
    return bootstrap


def unpatch_bytes(original: bytes) -> bytes:
    """Remove the tracer bootstrap, restoring an empty bootstrap file."""
    marker = OVERLAY_MARKER.encode("utf-8")
    if marker not in original:
        return original
    bootstrap = _bootstrap_text().encode("utf-8")
    if original != bootstrap:
        raise ValueError("target contains an incompatible P0 tracer bootstrap")
    return b""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"bootstrap file to write (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="remove this overlay and restore the previous bootstrap bytes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report without writing the target",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = args.path
    try:
        _validate_target(path)
        original = _read_bytes(path) if path.exists() else b""
        transformed = unpatch_bytes(original) if args.reverse else patch_bytes(original)
        compile(transformed.decode("utf-8"), str(path), "exec")
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"{OVERLAY_VERSION}: refusing {path}: {exc}", file=sys.stderr)
        return 2

    if transformed == original:
        state = "already at original bootstrap" if args.reverse else "already patched"
        print(f"{OVERLAY_VERSION}: {state}: {path}")
        return 0

    if not args.dry_run:
        try:
            if transformed or original:
                _write_bytes_atomic(path, transformed)
            elif path.exists():
                path.unlink()
        except OSError as exc:
            print(f"{OVERLAY_VERSION}: failed to write {path}: {exc}", file=sys.stderr)
            return 2
        prefix = "reversed" if args.reverse else "patched"
    else:
        prefix = "would reverse" if args.reverse else "would patch"

    if args.reverse:
        detail = "removed P0 tracer bootstrap"
    else:
        detail = "installed import-time P0 tracer bootstrap"
    print(f"{OVERLAY_VERSION}: {prefix} {path}: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
