"""hc-hy2 = hc-hy1 with a reshape/split checkpoint-layout NVFP4 GEMV (contiguous x loads, no masks)."""
import pathlib, sys
dec = pathlib.Path(sys.argv[1]) / 'vllm/models/inkling2/nvidia/ops/moe_xpu_dec.py'
s = dec.read_text()
old = """                xe = tl.load(xrow + (cc * 32)[:, None] + 2 * j16[None, :]).to(tl.float32)  # [CH, 16]
                xo = tl.load(xrow + (cc * 32)[:, None] + 2 * j16[None, :] + 1).to(tl.float32)
                p = _f16_e2m1(b & 15) * xe[None, :, :] + _f16_e2m1(b >> 4) * xo[None, :, :]
                lo = (j16 < 8)[None, None, :]  # bytes 0-7: k 32c..32c+15 (scale 2c); 8-15: scale 2c+1
                pa = tl.sum(tl.where(lo, p, 0.0), 2)
                pb = tl.sum(tl.where(lo, 0.0, p), 2)
"""
new = """                # byte j holds k = 32c + 2j (low) | 32c + 2j + 1 (high); bytes 0-7 use scale 2c, 8-15 2c+1.
                xv = tl.load(xrow + (cc * 32)[:, None] + tl.arange(0, 32)[None, :]).to(tl.float32)  # [CH, 32]
                xe, xo = tl.split(tl.reshape(xv, (CH, 16, 2)))  # [CH, 16] each
                p = _f16_e2m1(b & 15) * xe[None, :, :] + _f16_e2m1(b >> 4) * xo[None, :, :]  # [BN, CH, 16]
                ph = tl.sum(tl.reshape(p, (rows.shape[0], CH, 2, 8)), 3)  # [BN, CH, 2]
                pa, pb = tl.split(ph)
"""
assert s.count(old) == 1
dec.write_text(s.replace(old, new)); print('patched', dec)
