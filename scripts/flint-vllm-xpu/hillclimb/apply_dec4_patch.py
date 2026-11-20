"""Round 2: shared expert folded into the MoE decode kernels (idempotent; needs dec3)."""
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
if '_XPU_SHARED_FOLD' in s:
    print('already patched'); sys.exit(0)
assert '_XPU_DECODE_V3' in s

old_env = '_XPU_DECODE_V3 = os.environ.get("FLINT_XPU_DECODE_V3", "1") != "0"\n'
assert s.count(old_env) == 1
s = s.replace(old_env, old_env + '_XPU_SHARED_FOLD = os.environ.get("FLINT_XPU_SHARED_FOLD", "1") != "0"\n')

old_dec = '''        if _XPU_DECODE_V3:
            # Router + top-k (always-valid ids), shared expert (oneDNN), then the
            # routed experts with the shared output added in fp32 before rounding.
            from .ops.xpu_decode import moe_decode, router

            _, ids, wts = router(hidden_states, self.gate.weight, self.top_k)
'''
new_dec = '''        if _XPU_DECODE_V3:
            # Router + top-k (always-valid ids), then the routed experts plus the
            # shared expert (as packed pseudo-experts when available; otherwise the
            # oneDNN shared output added in fp32 before rounding).
            from .ops.xpu_decode import moe_decode, moe_decode_sh, router

            _, ids, wts = router(hidden_states, self.gate.weight, self.top_k)
            sh = self._xpu_shared_pack_get() if _XPU_SHARED_FOLD else None
            if sh is not None:
                return moe_decode_sh(hidden_states, ids, wts, w13, ex.w13_scales,
                                     ex.w2_qweight, ex.w2_scales, sh)
'''
assert s.count(old_dec) == 1
s = s.replace(old_dec, new_dec)

old_fwd = '''    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_flint_kernel:
            return self.experts(hidden_states, self.gate.weight, self.shared_expert)
'''
new_fwd = '''    def _xpu_shared_pack_get(self):
        """Shared expert repacked as pseudo-experts for the decode kernels, built
        eagerly on the first forward (the memory-profiling run, before graph
        capture). Returns None if unavailable."""
        sh = getattr(self, "_xpu_shared_pack", None)
        if sh is None:
            if torch.xpu.is_current_stream_capturing():
                return None
            sh = False
            ex = getattr(self.experts, "routed_experts", None)
            w13 = getattr(ex, "w13_qweight", None)
            gu, dn = self.shared_expert.gate_up_proj, self.shared_expert.down_proj
            if w13 is not None and w13.dtype == torch.int8 and w13.dim() == 3 and all(
                getattr(m, "qweight", None) is not None
                and m.qweight.dtype == torch.int32
                and m.qweight.t().is_contiguous()
                and getattr(m, "scales", None) is not None
                and getattr(m, "qzeros", None) is not None
                and m.qzeros.numel() == 1 and int(m.qzeros.item()) == 8
                for m in (gu, dn)
            ):
                from .ops.xpu_decode import pack_shared_expert

                sh = pack_shared_expert(gu.qweight.t(), gu.scales, dn.qweight.t(),
                                        dn.scales, w13.shape[1] // 2)
            self._xpu_shared_pack = sh
        return sh or None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_flint_kernel:
            return self.experts(hidden_states, self.gate.weight, self.shared_expert)
        if current_platform.is_xpu() and _XPU_DECODE_V3 and _XPU_SHARED_FOLD:
            self._xpu_shared_pack_get()
'''
assert s.count(old_fwd) == 1
s = s.replace(old_fwd, new_fwd)
p.write_text(s)
print('patched')
