"""Checkpoint-layout NVFP4 decode GEMV (split_half=False) vs the existing split-half GEMV on the same
weights (repacked copy). Real Inkling 2 Mini dims, random valid NVFP4 codes and E4M3 scales."""
import json
import sys

import torch

sys.path.insert(0, sys.argv[1])
from vllm.models.inkling2.nvidia.ops.moe_xpu_dec import packed_experts_dec  # noqa: E402
from vllm.models.inkling2.nvidia.ops.moe_xpu_v3 import repack_nvfp4_split_half  # noqa: E402

cfg = json.load(open(sys.argv[2]))
tc = cfg.get('text_config', cfg)
H, I = tc['hidden_size'], tc['intermediate_size']
E, topk, dev = 16, tc['num_experts_per_tok'], 'xpu'
g = torch.Generator().manual_seed(0)


def nvfp4(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), generator=g, dtype=torch.uint8)
    # E4M3 scales: positive normal codes only (exponent 4..10), no NaN
    s = (torch.randint(4, 11, (E, n, k // 16), generator=g) << 3 | torch.randint(0, 8, (E, n, k // 16), generator=g))
    return w.to(dev), s.to(torch.uint8).view(torch.float8_e4m3fn).to(dev)


w13, s13 = nvfp4(2 * I, H)
w2, s2 = nvfp4(H, I)
amax13 = torch.rand(E, generator=g).to(dev) + 0.5
amax2 = torch.rand(E, generator=g).to(dev) + 0.5
den = torch.tensor([2688.0], device=dev)
worst = 0.0
for tokens in (1, 2, 4):
    x = (torch.randn(tokens, H, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(tokens)]).to(torch.int32).to(dev)
    wts = torch.softmax(torch.randn(tokens, topk, generator=g), -1).to(dev)
    ck = packed_experts_dec(x, ids, wts, None, (w13, s13, amax13, den), (w2, s2, amax2, den), 0.0, 0.0,
                            split_half=False)
    a13, a2 = w13.clone(), w2.clone()
    repack_nvfp4_split_half(a13); repack_nvfp4_split_half(a2)
    sh = packed_experts_dec(x, ids, wts, None, (a13, s13, amax13, den), (a2, s2, amax2, den), 0.0, 0.0)
    torch.xpu.synchronize()
    d = (ck.float() - sh.float()).abs()
    rel = (d.max() / sh.float().abs().max()).item()
    worst = max(worst, rel)
    print(json.dumps(dict(tokens=tokens, max_abs=d.max().item(), ref_max=sh.float().abs().max().item(),
                          rel=rel, exact_frac=(d == 0).float().mean().item())))
print('PASS' if worst < 2e-2 else 'FAIL', worst)
