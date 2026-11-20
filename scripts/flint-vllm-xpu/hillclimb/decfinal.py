"""Unit gate for ops/xpu_decode.py: router invariance/accuracy/top-k, MoE vs fp64."""
import json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import xpu_decode as xd
root = pathlib.Path(sys.argv[1]); dev = 'xpu'; torch.manual_seed(1)
R = {}
H, E, TOPK, I = 2048, 256, 6, 512
w = (torch.randn(E, H, device=dev) * 0.02).to(torch.bfloat16)
x = torch.randn(100, H, device=dev).to(torch.bfloat16)
lg, ids, wts = xd.router(x, w, TOPK)
ref = x.double().cpu() @ w.double().cpu().T
R['router_err'] = (lg.double().cpu() - ref).abs().max().item()
R['router_inv'] = all(torch.equal(xd.router(x[i:i + 1], w, TOPK)[0], lg[i:i + 1]) for i in range(100)) and \
    torch.equal(xd.router(x[:37], w, TOPK)[0], lg[:37]) and torch.equal(xd.router(x[:8], w, TOPK)[0], lg[:8]) and \
    torch.equal(xd.router(x[3:20], w, TOPK)[0], lg[3:20])
R['ids_inv'] = all(torch.equal(xd.router(x[i:i + 1], w, TOPK)[1], ids[i:i + 1]) for i in range(100))
R['router_det'] = all(torch.equal(xd.router(x, w, TOPK)[0], lg) for _ in range(3))
rs, ri = torch.sort(ref, dim=1, descending=True, stable=True)
R['topk_order_match'] = sum(int(ids[i].tolist() == ri[i, :TOPK].tolist()) for i in range(100))
sg = torch.sigmoid(rs[:, :TOPK]); R['wts_err'] = (wts.double().cpu() - sg / sg.sum(1, keepdim=True)).abs().max().item()
w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
s2 = (torch.rand(E, H, I // 64, device=dev) * 0.01).to(torch.bfloat16)
xs = (torch.randn(3, H, device=dev) * 0.3).to(torch.bfloat16)
sh = torch.randn(3, H, device=dev).to(torch.bfloat16)
_, i3, w3 = xd.router(xs, w, TOPK)
out = xd.moe_decode(xs, i3, w3, w13, s13, w2, s2, shared=sh)
def deq(q, s):
    lo = ((q.to(torch.int16) << 12) >> 12).double(); hi = (q.to(torch.int16) >> 4).double()
    return torch.stack([lo, hi], -1).flatten(-2) * s.double().repeat_interleave(64, -1)
errs = []
for t in range(3):
    e = i3[t].long().cpu()
    W13 = deq(w13[e].cpu(), s13[e].cpu()); W2 = deq(w2[e].cpu(), s2[e].cpu())
    gu = torch.einsum('eik,k->ei', W13, xs[t].double().cpu())
    a = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    r = (torch.einsum('ehi,ei->eh', W2, a) * w3[t].double().cpu()[:, None]).sum(0) + sh[t].double().cpu()
    errs.append((out[t].double().cpu() - r).abs().max().item() / r.abs().max().item())
R['moe_relerr'] = errs
R['moe_inv'] = all(torch.equal(xd.moe_decode(xs[t:t + 1], i3[t:t + 1], w3[t:t + 1], w13, s13, w2, s2, shared=sh[t:t + 1]), out[t:t + 1]) for t in range(3))
R['moe_det'] = torch.equal(xd.moe_decode(xs, i3, w3, w13, s13, w2, s2, shared=sh), out)
# shared pseudo-expert path (synthetic GPTQ shared MLP, IS = 2 I)
IS = 2 * I
gq = torch.randint(-2**31, 2**31 - 1, (2 * IS, H // 8), dtype=torch.int32, device=dev)
gs = (torch.rand(H // 64, 2 * IS, device=dev) * 0.01).to(torch.bfloat16)
dq = torch.randint(-2**31, 2**31 - 1, (H, IS // 8), dtype=torch.int32, device=dev)
ds = (torch.rand(IS // 64, H, device=dev) * 0.01).to(torch.bfloat16)
pk = xd.pack_shared_expert(gq, gs, dq, ds, I)
def gdeq(q, s):
    q = q.cpu().to(torch.int64); N, K = q.shape[0], q.shape[1] * 8
    nib = torch.stack([(q >> (4 * i)) & 15 for i in range(8)], -1).reshape(N, K).double() - 8
    return nib * s.cpu().double().t().repeat_interleave(64, 1)
Wgu = gdeq(gq, gs); Wd = gdeq(dq, ds)
osh = xd.moe_decode_sh(xs, i3, w3, w13, s13, w2, s2, pk)
errs2 = []
for t in range(3):
    e = i3[t].long().cpu()
    W13 = deq(w13[e].cpu(), s13[e].cpu()); W2 = deq(w2[e].cpu(), s2[e].cpu())
    gu = torch.einsum('eik,k->ei', W13, xs[t].double().cpu())
    a = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    r = (torch.einsum('ehi,ei->eh', W2, a) * w3[t].double().cpu()[:, None]).sum(0)
    sg = Wgu @ xs[t].double().cpu()
    r = r + Wd @ (torch.nn.functional.silu(sg[:IS]) * sg[IS:])
    errs2.append((osh[t].double().cpu() - r).abs().max().item() / r.abs().max().item())
R['moe_sh_relerr'] = errs2
R['moe_sh_inv'] = all(torch.equal(xd.moe_decode_sh(xs[t:t + 1], i3[t:t + 1], w3[t:t + 1], w13, s13, w2, s2, pk), osh[t:t + 1]) for t in range(3))
R['moe_sh_det'] = torch.equal(xd.moe_decode_sh(xs, i3, w3, w13, s13, w2, s2, pk), osh)
# round 3: permuted norm (bitwise vs xpu_fused) and GPTQ permuted GEMV
import types
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import xpu_fused as xf
xr = torch.randn(5, H, device=dev).to(torch.bfloat16); rr = torch.randn(5, H, device=dev).to(torch.bfloat16)
nw = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
o_ref, r_ref = xf.fused_add_rms_norm(xr, rr, nw, 1e-6)
o1, op1, r1 = xd.rms_norm_perm(xr, rr, nw, 1e-6, True)
_, op2, _ = xd.rms_norm_perm(xr, rr, nw, 1e-6, False)
o3 = xf.rms_norm(xr, nw, 1e-6); _, op3, _ = xd.rms_norm_perm(xr, None, nw, 1e-6, False)
permref = lambda t: t.view(t.shape[0], H // 8, 8).transpose(1, 2).reshape(t.shape[0], H)
R['norm_perm_bitwise'] = (torch.equal(o1, o_ref) and torch.equal(r1, r_ref) and torch.equal(op1, permref(o_ref))
                          and torch.equal(op2, op1) and torch.equal(op3, permref(o3)))
N2 = 5120
qw = torch.randint(-2**31, 2**31 - 1, (N2, H // 8), dtype=torch.int32, device=dev)
qs = (torch.rand(H // 64, N2, device=dev) * 0.01).to(torch.bfloat16)
lay = types.SimpleNamespace(qweight=qw.t(), scales=qs, qzeros=torch.tensor([8], dtype=torch.int8, device=dev))
R['gptq_ok'] = xd.gptq_ok(lay)
xg = (torch.randn(3, H, device=dev) * 0.5).to(torch.bfloat16)
yg = xd.gptq_linear_perm(permref(xg).contiguous(), lay)
Wg = gdeq(qw, qs)
refg = xg.double().cpu() @ Wg.t()
R['gptq_relerr'] = ((yg.double().cpu() - refg).abs().max() / refg.abs().max()).item()
R['gptq_inv'] = all(torch.equal(xd.gptq_linear_perm(permref(xg[t:t + 1]).contiguous(), lay), yg[t:t + 1]) for t in range(3))
R['gptq_det'] = torch.equal(xd.gptq_linear_perm(permref(xg).contiguous(), lay), yg)
ok = R['router_err'] < 2e-6 and R['router_inv'] and R['ids_inv'] and R['router_det'] and R['topk_order_match'] >= 98 \
    and R['wts_err'] < 1e-6 and max(errs) < 5e-3 and R['moe_inv'] and R['moe_det'] and max(errs2) < 5e-3 and R['moe_sh_inv'] and R['moe_sh_det'] and R['norm_perm_bitwise'] and R['gptq_ok'] and R['gptq_relerr'] < 5e-3 and R['gptq_inv'] and R['gptq_det']
R['PASS'] = ok
print(json.dumps(R), flush=True)
(root / 'decfinal.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
