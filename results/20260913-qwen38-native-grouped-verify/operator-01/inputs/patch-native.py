#!/usr/bin/env python3
"""Emit a two-header patch against pinned native source; never edit the source."""
import difflib
from pathlib import Path
import sys

NATIVE_SHA = "1796aa8bc8db4ac68d9cd19636cef88f3af81d2b"
TLA_SHA = "cd763790ad2f74d7294435ecf77682bac0062c3a"
BASE = "csrc/xpu/attn/xe_2/"


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned source mismatch: expected one {old!r}")
    return text.replace(old, new, 1)


def patched_mainloop(source):
    # Scope edits to DecodeFwdMainloop; prefill is byte-for-byte unchanged.
    marker = "template <\n    class DispatchPolicy_,\n    bool PagedKV_,"
    if source.count(marker) != 1:
        raise ValueError("Pinned DecodeFwdMainloop declaration not found")
    prefill, decode = source.split(marker)
    decode = once(decode, "    bool LocalMask_ = false>",
                  "    bool LocalMask_ = false,\n    bool PackedVerify_ = false>")
    decode = once(decode, "    bool LocalMask_>\nstruct DecodeFwdMainloop<",
                  "    bool LocalMask_,\n    bool PackedVerify_>\nstruct DecodeFwdMainloop<")
    decode = once(decode, "    LocalMask_> {", "    LocalMask_,\n    PackedVerify_> {")
    decode = once(decode, "  static constexpr bool LocalMask = LocalMask_;", """  static constexpr bool LocalMask = LocalMask_;
  static constexpr bool PackedVerify = PackedVerify_;
  static_assert(!PackedVerify || (PagedKV && !LocalMask && !CausalMask),
                "Packed verification requires paged, non-local decode");
  static_assert(!PackedVerify || (get<0>(TileShapeQK{}) == 16 &&
                                  get<1>(TileShapeQK{}) == 64),
                "Packed verification is qualified only for q16/p64");""")
    decode = once(decode, "      /* Local/sliding window masking */", """      // Experimental C1/MTP4 verification: Q rows are t*6 + group, not
      // positions in a single-token decode. seq_len comes from the DEVICE
      // lengths tensor on every launch/replay (Python clamps it to >= 1).
      // partition_C of the GLOBAL identity tile includes blk_qv[0]*16;
      // using a tile-local row would incorrectly remask rows 16..29.
      if constexpr (PackedVerify) {
        Tensor cVerify = make_identity_tensor(make_shape(30, seq_len));
        Tensor gVerify = local_tile(
            cVerify, take<0, 2>(TileShapeQK{}),
            make_coord(get<0>(blk_qv), K));
        auto cVerify_thread = thr_mma_qk.partition_C(gVerify);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int group_row = get<0>(cVerify_thread(i));
          int key_pos = get<1>(cVerify_thread(i));
          int causal_limit = cute::max(seq_len - 4 + group_row / 6, 1);
          if (group_row >= 30 || key_pos >= causal_limit) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* Local/sliding window masking */""")
    return prefill + marker + decode


def patched_config(source):
    source = once(source, "    typename GmemTiledCopyO = void>\nstruct PagedDecodeConfig",
                  "    typename GmemTiledCopyO = void,\n    bool PackedVerify = false>\nstruct PagedDecodeConfig")
    return once(source, "        GmemTiledCopyV,\n        Local>;",
                "        GmemTiledCopyV,\n        Local,\n        PackedVerify>;")


def patch(source_dir):
    edits = [(BASE + "collective/chunk_prefill_mainloop.hpp", patched_mainloop),
             (BASE + "paged_decode.hpp", patched_config)]
    # Validate both files before emitting anything, so drift cannot half-apply.
    pairs = [(name, (Path(source_dir) / name).read_text(), edit) for name, edit in edits]
    changes = [(name, old, edit(old)) for name, old, edit in pairs]
    return "".join("".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile="a/" + name,
        tofile="b/" + name)) for name, old, new in changes)


if __name__ == "__main__":
    print(patch(sys.argv[1]), end="")
