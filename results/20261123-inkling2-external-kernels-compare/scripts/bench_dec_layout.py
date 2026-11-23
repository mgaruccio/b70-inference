"""Microbench: decode expert GEMV, 1 token, real Inkling dims (128 experts, top-8), split-half layout vs
checkpoint layout variants. Variant selected by env INK2_DEC_CK_VARIANT in the patched kernel."""
import json
import sys

import torch

sys.path.insert(0, sys.argv[1])
from vllm.models.inkling2.nvidia.ops.moe_xpu_dec import packed_experts_dec  # noqa: E402
from vllm.models.inkling2.nvidia.ops.moe_xpu_v3 import repack_nvfp4_split_half  # noqa: E402

H, I, E, topk, dev = 2304, 1152, 128, 8, 'xpu'
g = torch.Generator().manual_seed(1)


def nvfp4(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), generator=g, dtype=torch.uint8)
    s = (torch.randint(4, 11, (E, n, k // 16), generator=g) << 3 | torch.randint(0, 8, (E, n, k // 16), generator=g))
    return w.to(dev), s.to(torch.uint8).view(torch.float8_e4m3fn).to(dev)


w13, s13 = nvfp4(2 * I, H)
w2, s2 = nvfp4(H, I)
a13, a2 = w13.clone(), w2.clone()
repack_nvfp4_split_half(a13); repack_nvfp4_split_half(a2)
amax = torch.rand(E, generator=g).to(dev) + 0.5
den = torch.tensor([1.0], device=dev)
res = {}
for tokens in (1, 4):
    x = (torch.randn(tokens, H, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(tokens)]).to(torch.int32).to(dev)
    wts = torch.softmax(torch.randn(tokens, topk, generator=g), -1).to(dev)
    outs = {}
    for name, sh, W13, W2 in (('split_half', True, a13, a2), ('checkpoint', False, w13, w2)):
        f = lambda: packed_experts_dec(x, ids, wts, None, (W13, s13, amax, den), (W2, s2, amax, den), 0.0, 0.0,
                                       split_half=sh)
        for _ in range(20):
            o = f()
        torch.xpu.synchronize()
        st, en = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        st.record()
        for _ in range(200):
            o = f()
        en.record(); torch.xpu.synchronize()
        outs[name] = o
        res[f'{name}_t{tokens}_us'] = round(st.elapsed_time(en) / 200 * 1000, 1)
    d = (outs['split_half'].float() - outs['checkpoint'].float()).abs().max().item()
    res[f'maxdiff_t{tokens}'] = d
print(json.dumps(res))
