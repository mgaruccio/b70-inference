"""Router GEMM candidates: determinism, batch invariance, accuracy vs fp64."""
import json, pathlib, sys
import torch, triton, triton.language as tl
import torch.nn.functional as F
root = pathlib.Path(sys.argv[1]); dev = 'xpu'
torch.manual_seed(0)
K, N = 2048, 256
w = (torch.randn(N, K) * 0.02).to(torch.bfloat16).to(dev)
X = torch.randn(512, K).to(torch.bfloat16).to(dev)
ref = (X.double().cpu() @ w.double().cpu().t())

@triton.jit
def router_kernel(x_ptr, w_ptr, o_ptr, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(x_ptr + rm[:, None] * K + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(w_ptr + rn[None, :] * K + (k0 + rk)[:, None])
        acc += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc, mask=rm[:, None] < M)

def triton_router(x, w, BM=16, BN=64, BK=64):
    M = x.shape[0]; o = torch.empty(M, N, device=x.device, dtype=torch.float32)
    router_kernel[(triton.cdiv(M, BM), N // BN)](x, w, o, M, N, K, BM, BN, BK)
    return o

def cand():
    yield 'fp32_linear', lambda x: F.linear(x.float(), w.float())
    yield 'mm_out_dtype', lambda x: torch.mm(x, w.t(), out_dtype=torch.float32)
    yield 'bf16_linear', lambda x: F.linear(x, w).float()
    yield 'triton_bf16_fp32acc', lambda x: triton_router(x, w)
res = {'fp32_matmul_precision': torch.get_float32_matmul_precision()}
for name, f in cand():
    try:
        out = {}
        for M in (1, 7, 8, 47, 95, 512):
            x = X[:M].contiguous()
            ys = [f(x) for _ in range(20)]
            torch.xpu.synchronize()
            det = all(torch.equal(ys[0], y) for y in ys[1:])
            err = ((ys[0].double().cpu() - ref[:M]).abs().max() / ref[:M].abs().max()).item()
            out[M] = {'deterministic': det, 'maxrel_vs_fp64': err, 'row0': ys[0][0].cpu()}
        base = out[1]['row0']
        res[name] = {M: {'det': v['deterministic'], 'err': v['maxrel_vs_fp64'], 'row0_equals_M1': torch.equal(v['row0'], base)} for M, v in out.items()}
        import time
        x = X[:1].contiguous(); torch.xpu.synchronize(); t = time.perf_counter()
        for _ in range(200): f(x)
        torch.xpu.synchronize(); res[name]['us_M1'] = (time.perf_counter() - t) / 200 * 1e6
    except Exception as e:
        res[name] = repr(e)[:300]
    print(name, json.dumps(res[name], default=str), flush=True)
(root / 'gemmtest.json').write_text(json.dumps(res, default=str))
(root / 'offline.done').write_text('ok\n')
