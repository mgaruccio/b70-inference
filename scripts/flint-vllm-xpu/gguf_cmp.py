import json, struct, sys
import numpy as np
from safetensors import safe_open
gg, gdir = sys.argv[1], sys.argv[2]
f = open(gg, 'rb'); f.read(8); nt, nkv = struct.unpack('<QQ', f.read(16))
def rs():
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n).decode('utf-8', 'replace')
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def skip(t):
    if t == 8: rs(); return
    if t == 9:
        it = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
        if it == 8:
            for _ in range(n): rs()
        else: f.seek(SZ[it] * n, 1)
        return
    f.seek(SZ[t], 1)
align = 32
for _ in range(nkv):
    k = rs(); t = struct.unpack('<I', f.read(4))[0]
    if k == 'general.alignment': align = struct.unpack('<I', f.read(4))[0]
    else: skip(t)
tens = []
for _ in range(nt):
    name = rs(); nd = struct.unpack('<I', f.read(4))[0]; dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
    typ = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
    tens.append((name, dims, typ, off))
base = (f.tell() + align - 1) // align * align
print('types of blk.2/3 tensors:', [(n, d, t) for n, d, t, o in tens if n.startswith('blk.3.')])
idx = json.loads(open(gdir + '/model.safetensors.index.json').read())['weight_map']
def G(n): return safe_open(gdir + '/' + idx[n], 'np', device='cpu').get_tensor(n) if False else safe_open(gdir + '/' + idx[n], 'pt', device='cpu').get_tensor(n).float().numpy()
def F(name):
    for n, d, t, o in tens:
        if n == name:
            cnt = int(np.prod(d))
            dt = {0: np.float32, 1: np.float16}.get(t)
            if dt is None: return None, t
            return np.fromfile(gg, dtype=dt, count=cnt, offset=base + o).astype(np.float32).reshape(tuple(reversed(d))), t
    return None, 'missing'
pairs = []
for l in (0, 2, 3, 24, 48):
    pairs += [(f'blk.{l}.attn_norm.weight', f'model.layers.{l}.input_layernorm.weight'),
              (f'blk.{l}.ffn_norm.weight', f'model.layers.{l}.post_attention_layernorm.weight'),
              (f'blk.{l}.attn_q_norm.weight', f'model.layers.{l}.self_attn.q_norm.weight'),
              (f'blk.{l}.attn_k_norm.weight', f'model.layers.{l}.self_attn.k_norm.weight'),
              (f'blk.{l}.attn_sinks.weight', f'model.layers.{l}.self_attn.softmax_offset'),
              (f'blk.{l}.ffn_gate_inp.weight', f'model.layers.{l}.mlp.gate.weight')]
pairs += [('output_norm.weight', 'model.norm.weight')]
for a, b in pairs:
    x, t = F(a)
    try: y = G(b)
    except KeyError: y = None
    if x is None or y is None:
        print(f'{a:28s} gguf={t} gptq={"missing" if y is None else y.shape}')
        continue
    if x.shape != y.shape and x.size == y.size: x = x.reshape(y.shape)
    d = np.abs(x - y).max(); r = np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30)
    print(f'{a:28s} shape {y.shape} maxabs {d:.3g} rel {r:.3g}  gguf[:4] {np.round(x.ravel()[:4], 4)} gptq[:4] {np.round(y.ravel()[:4], 4)}')
names = [n for n, d, t, o in tens]
print('sink-like names:', [n for n in names if 'sink' in n or 'offset' in n][:5])

def q8_0(name):
    for n, d, t, o in tens:
        if n == name:
            assert t == 8, t
            K, N = d  # ggml dims: ne0 = K (input), ne1 = N (output)
            nb = K // 32
            raw = np.fromfile(gg, dtype=np.uint8, count=N * nb * 34, offset=base + o).reshape(N, nb, 34)
            sc = raw[:, :, :2].copy().view(np.float16).astype(np.float32)  # [N, nb, 1]
            q = raw[:, :, 2:].view(np.int8).astype(np.float32)
            return (q * sc).reshape(N, K)
SH = np.arange(0, 32, 4, dtype=np.int32)
def deq(p):
    qw = G2(p + '.qweight'); qz = G2(p + '.qzeros'); sc = G(p + '.scales')
    K8, N = qw.shape
    q = ((qw[:, None, :] >> SH.reshape(1, 8, 1)) & 0xF).reshape(K8 * 8, N).astype(np.float32)
    z = ((qz[:, :, None] >> SH.reshape(1, 1, 8)) & 0xF).reshape(qz.shape[0], -1).astype(np.float32)
    g = (K8 * 8) // sc.shape[0]
    return ((q - np.repeat(z + 1, g, 0)) * np.repeat(sc, g, 0)).T  # [N, K]
def G2(n):
    return safe_open(gdir + '/' + idx[n], 'pt', device='cpu').get_tensor(n).numpy()
for l, kind in ((3, 'sliding/rope'), (5, 'full/nope')):
    for proj, gname in (('q', 'attn_q'), ('k', 'attn_k')):
        A = q8_0(f'blk.{l}.{gname}.weight'); B = deq(f'model.layers.{l}.self_attn.{proj}_proj')
        rel = np.linalg.norm(A - B) / np.linalg.norm(A)
        # row-matching within head 0: for each GGUF row find best-correlated GPTQ row
        An = A[:128] / np.linalg.norm(A[:128], axis=1, keepdims=True); Bn = B[:128] / np.linalg.norm(B[:128], axis=1, keepdims=True)
        C = An @ Bn.T; best = C.argmax(1)
        print(f'layer {l} ({kind}) {proj}: shape {A.shape} rel(A-B) {rel:.3f}; head0 best-match rows[:16] {best[:16].tolist()} diagcorr {np.diag(C).mean():.3f} bestcorr {C.max(1).mean():.3f}')

def q4k_expert(name, e):
    for n, d, t, o in tens:
        if n == name:
            assert t == 12, t
            K, N, E = d
            nbr = K // 256  # blocks per row
            bpe = N * nbr * 144
            raw = np.fromfile(gg, dtype=np.uint8, count=bpe, offset=base + o + e * bpe).reshape(N * nbr, 144)
            dd = raw[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
            dm = raw[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
            s = raw[:, 4:16].astype(np.int32); qs = raw[:, 16:144]
            sc = np.zeros((raw.shape[0], 8), np.float32); mn = np.zeros_like(sc)
            for j in range(8):
                if j < 4:
                    sc[:, j] = s[:, j] & 63; mn[:, j] = s[:, j + 4] & 63
                else:
                    sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
                    mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
            out = np.empty((raw.shape[0], 256), np.float32)
            for jj in range(4):
                q = qs[:, jj * 32:(jj + 1) * 32].astype(np.float32)
                lo = qs[:, jj * 32:(jj + 1) * 32] & 0xF; hi = qs[:, jj * 32:(jj + 1) * 32] >> 4
                out[:, jj * 64: jj * 64 + 32] = (dd * sc[:, 2 * jj])[:, None] * lo - (dm * mn[:, 2 * jj])[:, None]
                out[:, jj * 64 + 32: jj * 64 + 64] = (dd * sc[:, 2 * jj + 1])[:, None] * hi - (dm * mn[:, 2 * jj + 1])[:, None]
            return out.reshape(N, K)
import time
for l in [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else []:
    t0 = time.time(); worst = []
    for proj, gname in (('gate', 'ffn_gate_exps'), ('up', 'ffn_up_exps'), ('down', 'ffn_down_exps')):
        rels = []
        for e in range(256):
            A = q4k_expert(f'blk.{l}.{gname}.weight', e)
            B = deq(f'model.layers.{l}.mlp.experts.{e}.{proj}_proj')
            if A.shape != B.shape: A = A.reshape(B.shape)
            rels.append(float(np.linalg.norm(A - B) / np.linalg.norm(A)))
        r = np.array(rels)
        bad = np.argsort(-r)[:5]
        print(f'layer {l} {proj}: rel median {np.median(r):.3f} p99 {np.percentile(r, 99):.3f} max {r.max():.3f} worst experts {[(int(b), round(float(r[b]), 3)) for b in bad]}', flush=True)
    print(f'layer {l} took {time.time() - t0:.0f}s', flush=True)
