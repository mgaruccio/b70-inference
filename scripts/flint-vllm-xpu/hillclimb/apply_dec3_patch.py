"""Install round-3 decode kernels into the Flint model (idempotent)."""
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
if '_XPU_DECODE_V3' in s:
    print('already patched'); sys.exit(0)

old_env = '_XPU_MOE_DEBUG = os.environ.get("FLINT_XPU_MOE_DEBUG", "0") == "1"\n'
assert s.count(old_env) == 1
s = s.replace(old_env, old_env + '_XPU_DECODE_V3 = os.environ.get("FLINT_XPU_DECODE_V3", "1") != "0"\n')

old_gate = '''    def forward(self, x: torch.Tensor):
        from .ops.xpu_router import flint_router_logits

        return self._return(flint_router_logits(x, self.weight))
'''
new_gate = '''    flint_top_k: int = 6

    def forward(self, x: torch.Tensor):
        if _XPU_DECODE_V3:
            from .ops.xpu_decode import router

            return self._return(router(x, self.weight, self.flint_top_k)[0])
        from .ops.xpu_router import flint_router_logits

        return self._return(flint_router_logits(x, self.weight))
'''
assert s.count(old_gate) == 1
s = s.replace(old_gate, new_gate)

old_init = '''            prefix=f"{prefix}.gate",
        )
        self.shared_expert = FlintMLP('''
new_init = '''            prefix=f"{prefix}.gate",
        )
        if isinstance(self.gate, FlintXpuGateLinear):
            self.gate.flint_top_k = self.top_k
        self.shared_expert = FlintMLP('''
assert s.count(old_init) == 1
s = s.replace(old_init, new_init)

old_dec = '''        if w13 is None or w13.dtype != torch.int8 or w13.dim() != 3:
            return None
'''
new_dec = '''        if w13 is None or w13.dtype != torch.int8 or w13.dim() != 3:
            return None
        if _XPU_DECODE_V3:
            # Router + top-k (always-valid ids), shared expert (oneDNN), then the
            # routed experts with the shared output added in fp32 before rounding.
            from .ops.xpu_decode import moe_decode, router

            _, ids, wts = router(hidden_states, self.gate.weight, self.top_k)
            shared = self.shared_expert(hidden_states)
            return moe_decode(hidden_states, ids, wts, w13, ex.w13_scales,
                              ex.w2_qweight, ex.w2_scales, shared=shared)
'''
assert s.count(old_dec) == 1
s = s.replace(old_dec, new_dec)
p.write_text(s)
print('patched')
