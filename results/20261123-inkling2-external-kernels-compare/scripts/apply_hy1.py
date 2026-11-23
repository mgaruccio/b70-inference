"""Hybrid experts on an hc-ck1 copy: CUTLASS grouped GEMM for multi-token shapes, our decode GEMV for
<= INK2_HYBRID_DEC_MAX tokens on NVFP4 layers (checkpoint nibble layout, no second weight copy).
INK2_CUTLASS_MXFP8=0 keeps the MXFP8 layers (20-23) entirely on our Triton kernels."""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / 'vllm/models/inkling2/nvidia'

# 1. moe_xpu_dec.py: SPLIT_HALF constexpr (True = existing split-half layout, False = checkpoint layout:
#    byte j of a 32-wide chunk holds k = 32c + 2j (low nibble) and 32c + 2j + 1 (high nibble)).
dec = root / 'ops/moe_xpu_dec.py'
s = dec.read_text()
old = """def _row_dot(xrow, W, S, Amax, Denominator, expert, rows,
             K: tl.constexpr, NVFP4: tl.constexpr, CH: tl.constexpr):"""
new = """def _row_dot(xrow, W, S, Amax, Denominator, expert, rows,
             K: tl.constexpr, NVFP4: tl.constexpr, CH: tl.constexpr, SPLIT_HALF: tl.constexpr = True):"""
assert s.count(old) == 1; s = s.replace(old, new)
old = """            b = tl.load(wrow + (cc * 16)[None, :, None] + j16[None, None, :]).to(tl.int32)  # [BN, CH, 16]
            xa = tl.load(xrow + (cc * 32)[:, None] + j16[None, :]).to(tl.float32)  # [CH, 16]
            xb = tl.load(xrow + (cc * 32 + 16)[:, None] + j16[None, :]).to(tl.float32)
            pa = tl.sum(_f16_e2m1(b & 15) * xa[None, :, :], 2)  # [BN, CH]
            pb = tl.sum(_f16_e2m1(b >> 4) * xb[None, :, :], 2)
"""
new = """            b = tl.load(wrow + (cc * 16)[None, :, None] + j16[None, None, :]).to(tl.int32)  # [BN, CH, 16]
            if SPLIT_HALF:
                xa = tl.load(xrow + (cc * 32)[:, None] + j16[None, :]).to(tl.float32)  # [CH, 16]
                xb = tl.load(xrow + (cc * 32 + 16)[:, None] + j16[None, :]).to(tl.float32)
                pa = tl.sum(_f16_e2m1(b & 15) * xa[None, :, :], 2)  # [BN, CH]
                pb = tl.sum(_f16_e2m1(b >> 4) * xb[None, :, :], 2)
            else:
                xe = tl.load(xrow + (cc * 32)[:, None] + 2 * j16[None, :]).to(tl.float32)  # [CH, 16]
                xo = tl.load(xrow + (cc * 32)[:, None] + 2 * j16[None, :] + 1).to(tl.float32)
                p = _f16_e2m1(b & 15) * xe[None, :, :] + _f16_e2m1(b >> 4) * xo[None, :, :]
                lo = (j16 < 8)[None, None, :]  # bytes 0-7: k 32c..32c+15 (scale 2c); 8-15: scale 2c+1
                pa = tl.sum(tl.where(lo, p, 0.0), 2)
                pb = tl.sum(tl.where(lo, 0.0, p), 2)
"""
assert s.count(old) == 1; s = s.replace(old, new)
old = """def _dec_gate_up(X, Ids, W, S, Amax, Denominator, Out, num_routes,
                 N: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr, NVFP4: tl.constexpr,
                 GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr, BN: tl.constexpr, CH: tl.constexpr):"""
new = old[:-2] + ",\n                 SPLIT_HALF: tl.constexpr = True):"
assert s.count(old) == 1; s = s.replace(old, new)
old = "        acc = _row_dot(X + token.to(tl.int64) * K, W, S, Amax, Denominator, expert, rows, K, NVFP4, CH)"
new = "        acc = _row_dot(X + token.to(tl.int64) * K, W, S, Amax, Denominator, expert, rows, K, NVFP4, CH, SPLIT_HALF)"
assert s.count(old) == 1; s = s.replace(old, new)
old = """def _dec_down_combine(Act, Ids, Weights, W, S, Amax, Denominator, Out,
                      N: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr, NVFP4: tl.constexpr,
                      BN: tl.constexpr, CH: tl.constexpr):"""
new = old[:-2] + ", SPLIT_HALF: tl.constexpr = True):"
assert s.count(old) == 1; s = s.replace(old, new)
old = """                         K, NVFP4, CH)"""
new = """                         K, NVFP4, CH, SPLIT_HALF)"""
assert s.count(old) == 1; s = s.replace(old, new)
old = "def packed_experts_dec(x, ids, weights, is_padding, w13, w2, gate_cap, up_cap, cfg=None):"
new = "def packed_experts_dec(x, ids, weights, is_padding, w13, w2, gate_cap, up_cap, cfg=None, split_half=True):"
assert s.count(old) == 1; s = s.replace(old, new)
old = "        n13, hidden, topk, nvfp4, gate_cap, up_cap, bn1, ch1, num_warps=nw1,"
new = "        n13, hidden, topk, nvfp4, gate_cap, up_cap, bn1, ch1, split_half, num_warps=nw1,"
assert s.count(old) == 1; s = s.replace(old, new)
old = "        hidden, inter, topk, nvfp4, bn2, ch2, num_warps=nw2,"
new = "        hidden, inter, topk, nvfp4, bn2, ch2, split_half, num_warps=nw2,"
assert s.count(old) == 1; s = s.replace(old, new)
dec.write_text(s)

# 2. moe.py: per-layer CUTLASS opt-out for MXFP8 and the small-batch NVFP4 dispatch.
moe = root / 'moe.py'
s = moe.read_text()
old = '            self._use_cutlass = os.getenv("INK2_CUTLASS_MOE", "0") == "1"\n'
new = ('            self._use_cutlass = os.getenv("INK2_CUTLASS_MOE", "0") == "1" and (\n'
       '                self.nvfp4 or os.getenv("INK2_CUTLASS_MXFP8", "1") != "0")\n'
       '            self._dec_max = int(os.getenv("INK2_HYBRID_DEC_MAX", "0"))\n')
assert s.count(old) == 1; s = s.replace(old, new)
old = """            if getattr(self, "_use_cutlass", False):
                from .ops.moe_cutlass import cutlass_experts
"""
new = """            if getattr(self, "_use_cutlass", False):
                if self.nvfp4 and x.shape[0] <= self._dec_max:
                    from .ops.moe_xpu_dec import packed_experts_dec

                    return packed_experts_dec(
                        x, ids, weights, is_padding,
                        (self._w13_w, self._w13_s, self.w13_weight_amax, self.w13_weight_outer_scale_denominator),
                        (self._w2_w, self._w2_s, self.w2_weight_amax, self.w2_weight_outer_scale_denominator),
                        self.gate_cap, self.up_cap, split_half=False,
                    )
                from .ops.moe_cutlass import cutlass_experts
"""
assert s.count(old) == 1; s = s.replace(old, new)
moe.write_text(s)
print('patched', dec, moe)
