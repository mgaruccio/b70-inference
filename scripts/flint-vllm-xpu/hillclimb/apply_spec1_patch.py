"""spec1: route M <= FLINT_XPU_DECODE_MAXM (default 8) rows through the batch-invariant
decode path (permuted norms, Triton qkv GEMV, router, MoE GEMVs), not only M == 1, so
spec-decode verify uses the same per-row kernels as single-token decode."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = '_XPU_PERM = os.environ.get("FLINT_XPU_PERM", "1") != "0"\n'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '_XPU_DECODE_MAXM = int(os.environ.get("FLINT_XPU_DECODE_MAXM", "8"))\n')
for tail in ('            and self.self_attn._xpu_qkv_perm_ok()\n', '            and self.mlp._xpu_perm_ready()\n'):
    old = '            and hidden_states.shape[0] == 1\n' + tail
    assert s.count(old) == 1, tail
    s = s.replace(old, '            and 1 <= hidden_states.shape[0] <= _XPU_DECODE_MAXM\n' + tail)
p.write_text(s)
print('spec1 patch applied')
