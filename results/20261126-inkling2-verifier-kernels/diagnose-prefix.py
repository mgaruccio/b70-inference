"""Identify exact FP8 storage codes whose fused conversion differs from stock on B70."""
import json
import torch
import vllm_inkling2_xpu.attention as a

with torch.inference_mode():
    codes = torch.arange(256, dtype=torch.uint8).repeat(1, 8, 64, 1)
    kv = codes.to('xpu').view(torch.float8_e4m3fn)
    pages = torch.zeros((1,), dtype=torch.int32, device='xpu')
    out = torch.zeros((2, 8, 64, 128), dtype=torch.bfloat16, device='xpu')
    ref = torch.empty_like(out)
    a.gather_prefix_kv_torch(kv, pages, ref[0], ref[1])
    a.gather_prefix_kv_xpu(kv, pages, out)
    stock, cand = [t.view(torch.int16).cpu().permute(1, 2, 0, 3).reshape(-1, 256) for t in (ref, out)]
    rows = []
    for code in range(256):
        diff = stock[:, code] != cand[:, code]
        if diff.any():
            pairs = torch.stack((stock[diff, code], cand[diff, code]), dim=1).unique(dim=0).tolist()
            rows.append(dict(code=hex(code), differing=int(diff.sum()), stock_candidate_hex=[[hex(x & 65535), hex(y & 65535)] for x,y in pairs]))
    print(json.dumps(dict(probe='all_byte_mappings', mismatches=rows)))
