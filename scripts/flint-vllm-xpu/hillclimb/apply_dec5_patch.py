"""Round 3: permuted-output RMSNorms + Triton GPTQ qkv GEMV at M=1 (idempotent; needs dec4)."""
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
if '_XPU_PERM' in s:
    print('already patched'); sys.exit(0)
assert '_XPU_SHARED_FOLD' in s


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


sub('_XPU_SHARED_FOLD = os.environ.get("FLINT_XPU_SHARED_FOLD", "1") != "0"\n',
    '_XPU_SHARED_FOLD = os.environ.get("FLINT_XPU_SHARED_FOLD", "1") != "0"\n'
    '_XPU_PERM = os.environ.get("FLINT_XPU_PERM", "1") != "0"\n')

# MoE block: decode entry taking the post-attention norm's normal + permuted outputs.
sub('''    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_flint_kernel:
            return self.experts(hidden_states, self.gate.weight, self.shared_expert)
''', '''    def _xpu_perm_ready(self) -> bool:
        if not (_XPU_PERM and _XPU_DECODE_V3 and _XPU_SHARED_FOLD and _XPU_MOE_DECODE):
            return False
        ex = getattr(self.experts, "routed_experts", None)
        w13 = getattr(ex, "w13_qweight", None)
        return (getattr(self, "_xpu_shared_pack", None) is not None and w13 is not None
                and w13.dtype == torch.int8 and w13.dim() == 3)

    def forward_xpu_decode_perm(self, h: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
        """M=1 decode with ``xp`` = nibble-major permuted ``h`` (from the norm)."""
        from .ops.xpu_decode import moe_decode_sh, router

        ex = self.experts.routed_experts
        _, ids, wts = router(h, self.gate.weight, self.top_k)
        return moe_decode_sh(h, ids, wts, ex.w13_qweight, ex.w13_scales, ex.w2_qweight,
                             ex.w2_scales, self._xpu_shared_pack, xp=xp)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_flint_kernel:
            return self.experts(hidden_states, self.gate.weight, self.shared_expert)
''')

# Attention: qkv from the permuted norm output through the Triton GPTQ GEMV.
sub('''    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if current_platform.is_xpu():
            return self._forward_xpu(positions, qkv)
''', '''    def _xpu_qkv_perm_ok(self) -> bool:
        """Cached eligibility of the Triton qkv GEMV (decided outside graph capture;
        first evaluated in the memory-profiling forward)."""
        ok = getattr(self, "_xpu_qkv_perm", None)
        if ok is None:
            if torch.xpu.is_current_stream_capturing():
                return False
            from .ops.xpu_decode import gptq_ok

            ok = self._xpu_qkv_perm = bool(_XPU_PERM and _XPU_DECODE_V3
                                           and gptq_ok(self.qkv_proj))
        return ok

    def forward_xpu_perm(self, positions: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
        from .ops.xpu_decode import gptq_linear_perm

        return self._forward_xpu(positions, gptq_linear_perm(xp, self.qkv_proj))

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if current_platform.is_xpu():
            self._xpu_qkv_perm_ok()
            return self._forward_xpu(positions, qkv)
''')

# Decoder layer: M=1 XPU decode uses the permuted norms.
sub('''        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        return self._post_attention(hidden_states, residual)
''', '''        if (
            current_platform.is_xpu()
            and hidden_states.shape[0] == 1
            and self.self_attn._xpu_qkv_perm_ok()
        ):
            from .ops.xpu_decode import rms_norm_perm

            norm = self.input_layernorm
            _, xp, res = rms_norm_perm(hidden_states, residual, norm.weight,
                                       norm.variance_epsilon, False)
            residual = hidden_states if residual is None else res
            hidden_states = self.self_attn.forward_xpu_perm(positions, xp)
            return self._post_attention(hidden_states, residual)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        return self._post_attention(hidden_states, residual)
''')
sub('''        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
''', '''        if (
            isinstance(self.mlp, FlintSparseMoeBlock)
            and current_platform.is_xpu()
            and hidden_states.shape[0] == 1
            and self.mlp._xpu_perm_ready()
        ):
            from .ops.xpu_decode import rms_norm_perm

            norm = self.post_attention_layernorm
            h, xp, residual = rms_norm_perm(hidden_states, residual, norm.weight,
                                            norm.variance_epsilon, True)
            return self.mlp.forward_xpu_decode_perm(h, xp), residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
''')
p.write_text(s)
print('patched')
