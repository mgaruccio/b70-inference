import faulthandler, json, pathlib, sys, time
faulthandler.dump_traceback_later(170, exit=True)
import torch
import torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).parent / 'ktest'))
import xpu_fused as xf
root = pathlib.Path(sys.argv[1]); which = sys.argv[2]
dev = 'xpu'; torch.manual_seed(0); eps = 1e-6
res = {}
def ulps(a, b):
    d = (a.float() - b.float()).abs()
    return {'equal': torch.equal(a, b), 'frac_diff': (d > 0).float().mean().item(), 'maxabs': d.max().item()}
def eager_norm(x, w):
    x = x.float(); return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)) * w.float()
def bench(f, n=200):
    f(); torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.xpu.synchronize()
    return s.elapsed_time(e) / n * 1000  # us
for M in (1, 8, 95):
    r = {}
    if which == 'norm':
        x = torch.randn(M, 2048, device=dev, dtype=torch.bfloat16) * 3
        res_ = torch.randn(M, 2048, device=dev, dtype=torch.bfloat16)
        w = (1 + 0.1 * torch.randn(2048, device=dev)).to(torch.bfloat16)
        r['rms'] = ulps(xf.rms_norm(x, w, eps), eager_norm(x, w).to(torch.bfloat16))
        acc = x.float() + res_.float()
        ref_o, ref_r = eager_norm(acc, w).to(torch.bfloat16), acc.to(torch.bfloat16)
        xo, ro = x.clone(), res_.clone(); xf.fused_add_rms_norm(xo, ro, w, eps)
        r['fadd_out'] = ulps(xo, ref_o); r['fadd_res'] = ulps(ro, ref_r)
        r['us_rms'] = bench(lambda: xf.rms_norm(x, w, eps)); r['us_rms_eager'] = bench(lambda: eager_norm(x, w).to(torch.bfloat16))
    elif which == 'silu':
        x = torch.randn(M, 2048, device=dev, dtype=torch.bfloat16) * 2
        g, u = x.float().chunk(2, -1)
        r['silu'] = ulps(xf.silu_and_mul(x), (F.silu(g) * u).to(torch.bfloat16))
        x2 = torch.randn(M, 12288, device=dev, dtype=torch.bfloat16) * 2
        g, u = x2.float().chunk(2, -1)
        r['silu_dense'] = ulps(xf.silu_and_mul(x2), (F.silu(g) * u).to(torch.bfloat16))
        r['us'] = bench(lambda: xf.silu_and_mul(x))
    elif which == 'qk':
        NQ, NKV, HD = 32, 4, 128
        qkv = torch.randn(M, (NQ + 2 * NKV) * HD, device=dev, dtype=torch.bfloat16) * 2
        pos = torch.randint(0, 4000, (M,), device=dev, dtype=torch.int64)
        qw = (1 + 0.1 * torch.randn(HD, device=dev)).to(torch.bfloat16); kw = (1 + 0.1 * torch.randn(HD, device=dev)).to(torch.bfloat16)
        inv = 1.0 / (10000 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        t = torch.arange(4096, dtype=torch.float32)[:, None] * inv[None]
        cs = torch.cat([t.cos(), t.sin()], -1).to(dev).contiguous()
        for rope in (False, True):
            q, k, v = qkv.split((NQ * HD, NKV * HD, NKV * HD), -1)
            c = cs.index_select(0, pos)[:, None, :] if rope else None
            def nr(x, w):
                x = x.unflatten(-1, (-1, HD)).float(); x = eager_norm(x, w).to(torch.bfloat16)
                if c is not None:
                    cos, sin = c.chunk(2, -1); a, b = x.float().chunk(2, -1)
                    x = torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(torch.bfloat16)
                return x.flatten(-2)
            rq, rk = nr(q, qw), nr(k, kw)
            oq, ok, ov = xf.qk_norm_rope(qkv, pos, qw, kw, eps, cs if rope else None, NQ, NKV, HD)
            r[f'q_rope{rope}'] = ulps(oq, rq); r[f'k_rope{rope}'] = ulps(ok, rk); r[f'v_rope{rope}'] = ulps(ov, v.contiguous())
        r['us'] = bench(lambda: xf.qk_norm_rope(qkv, pos, qw, kw, eps, cs, NQ, NKV, HD))
    torch.xpu.synchronize()
    res[M] = r
    print(which, M, json.dumps(r), flush=True)
(root / f'fusedtest-{which}.json').write_text(json.dumps(res))
(root / 'offline.done').write_text('ok\n')
