#!/usr/bin/env python3
"""Apply the Qwen B70 v1 GDN metadata-buffer overlay.

This is intentionally a text overlay for the exact vLLM XPU image/source pin
recorded next to this file.  It must run after the existing Qwen MTP and
partial-final-group boundary patches.  The transform is deliberately limited
to static metadata buffers in ``gdn_attn.py``; it does not touch the model
runner, kernels, or inference configuration.
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path

OVERLAY_VERSION = "qwen38-b70-gdn-metadata-v1"
IMAGE_DIGEST = (
    "sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
)
SOURCE_COMMIT = "ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9"
DEFAULT_TARGET = Path("/workspace/vllm/vllm/v1/attention/backends/gdn_attn.py")

BOUNDARY_MARKER = "B70_MTP_PARTIAL_FINAL_GROUP"
OVERLAY_MARKER = "B70_GDN_METADATA_V1"

# The boundary patch is a prerequisite.  These anchors intentionally include
# its behavior, not just its marker, so a similarly named but incompatible
# target cannot pass the precondition.
BOUNDARY_PURE_SOURCE = """            if num_prefills == 0 and num_decodes == 0:
                expected_spec_token_size = num_spec_decodes * (self.num_spec + 1)
                actual_spec_token_size = query_start_loc_cpu[-1].item()
                if actual_spec_token_size < expected_spec_token_size:
                    # B70_MTP_PARTIAL_FINAL_GROUP: The max-sequence boundary can
                    # truncate the final speculative group. The XPU GDN kernel
                    # requires complete groups, so process this final partial
                    # group through the existing stateful non-spec prefill path.
                    spec_sequence_masks = None
                    spec_sequence_masks_cpu = None
                    num_prefills = num_spec_decodes
                    num_prefill_tokens = actual_spec_token_size
                    num_spec_decodes = 0
                    num_spec_decode_tokens = 0
                    spec_token_indx = None
                    non_spec_token_indx = None
                    spec_state_indices_tensor = None
                    non_spec_state_indices_tensor = block_table_tensor[:, 0]
                    spec_query_start_loc = None
                    non_spec_query_start_loc = query_start_loc
                    non_spec_query_start_loc_cpu = query_start_loc_cpu
                    num_accepted_tokens = None
                else:
                    spec_token_indx = torch.arange(
                        expected_spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                    # Filter by spec_sequence_masks to exclude padded sequences
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                    non_spec_state_indices_tensor = None
                    # Padded sequences are always at the back, so the first
                    # num_spec_decodes + 1 entries of query_start_loc already
                    # contain the correct cumulative token counts.
                    spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                    non_spec_query_start_loc = None
                    non_spec_query_start_loc_cpu = None
            else:
"""


BOUNDARY_PARTIAL_PATCHED = """                if actual_spec_token_size < expected_spec_token_size:
                    # B70_MTP_PARTIAL_FINAL_GROUP: The max-sequence boundary can
                    # truncate the final speculative group. The XPU GDN kernel
                    # requires complete groups, so process this final partial
                    # group through the existing stateful non-spec prefill path.
                    spec_sequence_masks = None
                    spec_sequence_masks_cpu = None
                    num_prefills = num_spec_decodes
                    num_prefill_tokens = actual_spec_token_size
                    num_spec_decodes = 0
                    num_spec_decode_tokens = 0
                    spec_token_indx = None
                    non_spec_token_indx = None
                    spec_state_indices_tensor = None
                    non_spec_state_indices_tensor = block_table_tensor[:, 0]
                    spec_query_start_loc = None
                    non_spec_query_start_loc = query_start_loc
                    non_spec_query_start_loc_cpu = query_start_loc_cpu
                    num_accepted_tokens = None
"""

BOUNDARY_FINALIZE_PATCHED = """            if spec_sequence_masks_cpu is not None:
                assert num_accepted_tokens is not None
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
"""

INIT_SOURCE = """        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
"""

INIT_PATCHED = """        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        # B70_GDN_METADATA_V1: reuse the static speculative token arange.
        self.spec_token_arange: torch.Tensor = torch.arange(
            self.decode_cudagraph_max_bs * (self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
"""

PURE_SOURCE = """                    spec_token_indx = torch.arange(
                        expected_spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                    # Filter by spec_sequence_masks to exclude padded sequences
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
"""

PURE_PATCHED = """                    if expected_spec_token_size > self.spec_token_arange.numel():
                        raise RuntimeError(
                            "speculative metadata buffer is too small"
                        )
                    spec_token_indx = self.spec_token_arange[
                        :expected_spec_token_size
                    ]
                    non_spec_token_indx = self.non_spec_token_indx[:0]
                    # Spec decode rows are compacted to the front; padded rows are
                    # at the back. Slice instead of CPU-mask indexing to avoid
                    # launching a tiny index kernel for every metadata build.
                    spec_state_indices_tensor = block_table_tensor[
                        :num_spec_decodes, : self.num_spec + 1
                    ]
"""

COPY_SOURCE = """            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]
"""

COPY_PATCHED = """            assert non_spec_token_indx is not None and spec_token_indx is not None
            if non_spec_token_indx.numel() > 0:
                self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                    non_spec_token_indx, non_blocking=True
                )
                non_spec_token_indx = self.non_spec_token_indx[
                    : non_spec_token_indx.size(0)
                ]

            if spec_token_indx.data_ptr() == self.spec_token_arange.data_ptr():
                spec_token_indx = self.spec_token_arange[: spec_token_indx.size(0)]
            else:
                self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                    spec_token_indx, non_blocking=True
                )
                spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]
"""


def _count(text: str, needle: str) -> int:
    return text.count(needle)


def _require_count(text: str, needle: str, expected: int, label: str) -> None:
    actual = _count(text, needle)
    if actual != expected:
        raise ValueError(
            f"{label}: expected {expected} occurrence(s), found {actual}; "
            "refusing to modify the target"
        )


def _validate_boundary_source(text: str) -> None:
    _require_count(text, BOUNDARY_MARKER, 1, "boundary marker")
    _require_count(text, BOUNDARY_PURE_SOURCE, 1, "post-boundary pure-spec block")
    _require_count(text, BOUNDARY_FINALIZE_PATCHED, 1, "post-boundary finalize block")


def _validate_boundary_after_overlay(text: str) -> None:
    _require_count(text, BOUNDARY_MARKER, 1, "boundary marker")
    _require_count(text, BOUNDARY_PARTIAL_PATCHED, 1, "boundary partial-group block")
    _require_count(text, BOUNDARY_FINALIZE_PATCHED, 1, "boundary finalize block")


def _validate_overlay(text: str) -> None:
    _require_count(text, OVERLAY_MARKER, 1, "overlay marker")
    _require_count(text, INIT_PATCHED, 1, "preallocated arange block")
    _require_count(text, PURE_PATCHED, 1, "pure-spec reuse block")
    _require_count(text, COPY_PATCHED, 1, "metadata-copy block")
    _validate_boundary_after_overlay(text)
    for needle, label in (
        (INIT_SOURCE, "unpatched init block"),
        (PURE_SOURCE, "unpatched pure-spec block"),
        (COPY_SOURCE, "unpatched metadata-copy block"),
    ):
        _require_count(text, needle, 0, label)


def patch_text(text: str) -> str:
    """Return text with the overlay applied, or unchanged when already valid."""
    if OVERLAY_MARKER in text:
        _validate_overlay(text)
        return text

    _validate_boundary_source(text)
    for needle, label in (
        (INIT_SOURCE, "init block"),
        (PURE_SOURCE, "pure-spec block"),
        (COPY_SOURCE, "metadata-copy block"),
    ):
        _require_count(text, needle, 1, label)
    for needle, label in (
        (INIT_PATCHED, "partial preallocated init block"),
        (PURE_PATCHED, "partial pure-spec block"),
        (COPY_PATCHED, "partial metadata-copy block"),
    ):
        _require_count(text, needle, 0, label)

    patched = text.replace(INIT_SOURCE, INIT_PATCHED, 1)
    patched = patched.replace(PURE_SOURCE, PURE_PATCHED, 1)
    patched = patched.replace(COPY_SOURCE, COPY_PATCHED, 1)
    _validate_overlay(patched)
    return patched


def unpatch_text(text: str) -> str:
    """Remove this overlay while preserving the preceding boundary patch."""
    if OVERLAY_MARKER not in text:
        _validate_boundary_source(text)
        return text

    _validate_overlay(text)
    restored = text.replace(INIT_PATCHED, INIT_SOURCE, 1)
    restored = restored.replace(PURE_PATCHED, PURE_SOURCE, 1)
    restored = restored.replace(COPY_PATCHED, COPY_SOURCE, 1)
    _validate_boundary_source(restored)
    return restored


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def _write_text_atomic(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"target gdn_attn.py (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="remove this overlay and restore the post-boundary source",
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
        original = _read_text(path)
        transformed = unpatch_text(original) if args.reverse else patch_text(original)
        compile(transformed, str(path), "exec")
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"{OVERLAY_VERSION}: refusing {path}: {exc}", file=sys.stderr)
        return 2

    if transformed == original:
        state = "already at post-boundary source" if args.reverse else "already patched"
        print(f"{OVERLAY_VERSION}: {state}: {path}")
        return 0

    if not args.dry_run:
        try:
            _write_text_atomic(path, transformed)
        except OSError as exc:
            print(f"{OVERLAY_VERSION}: failed to write {path}: {exc}", file=sys.stderr)
            return 2
        prefix = "reversed" if args.reverse else "patched"
    else:
        prefix = "would reverse" if args.reverse else "would patch"
    print(
        f"{OVERLAY_VERSION}: {prefix} {path}: "
        "static spec-token arange, zero-length non-spec copy avoidance, "
        "and static-buffer copy reuse"
        if not args.reverse
        else f"{OVERLAY_VERSION}: {prefix} {path}: restored post-boundary metadata blocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
