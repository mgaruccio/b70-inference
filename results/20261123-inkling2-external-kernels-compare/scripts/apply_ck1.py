"""Port SergiioB's INK2_CUTLASS_MOE opt-in hook (cookbook f10f667) into an hc tree's nvidia/moe.py."""
import pathlib
import sys

moe = pathlib.Path(sys.argv[1]) / 'vllm/models/inkling2/nvidia/moe.py'
s = moe.read_text()
assert 'INK2_CUTLASS_MOE' not in s
if not s.startswith('import os\n'):
    s = 'import os\n' + s

old_pwl = """        if self.xpu:
            from .ops import moe_xpu

            if self.nvfp4 and moe_xpu.MOE_V3:"""
new_pwl = """        if self.xpu:
            self._use_cutlass = os.getenv("INK2_CUTLASS_MOE", "0") == "1"
            if self._use_cutlass:
                if self.nvfp4:
                    self._w13_esc = (self.w13_weight_amax / self.w13_weight_outer_scale_denominator).to(torch.float32).contiguous()
                    self._w2_esc = (self.w2_weight_amax / self.w2_weight_outer_scale_denominator).to(torch.float32).contiguous()
                    self._w13_w = self.w13_weight.contiguous()
                    self._w2_w = self.w2_weight.contiguous()
                else:
                    self._w13_esc = None
                    self._w2_esc = None
                    self.w13_weight.data = self.w13_weight.data.transpose(-1, -2).contiguous()
                    self.w2_weight.data = self.w2_weight.data.transpose(-1, -2).contiguous()
                    self._w13_w = self.w13_weight
                    self._w2_w = self.w2_weight
                self._w13_s = self.w13_weight_scale.contiguous()
                self._w2_s = self.w2_weight_scale.contiguous()
                return
            from .ops import moe_xpu

            if self.nvfp4 and moe_xpu.MOE_V3:"""
assert s.count(old_pwl) == 1
s = s.replace(old_pwl, new_pwl)

old_fwd = """            from .ops.moe_xpu import packed_experts

            ids, weights, is_padding = routing
"""
new_fwd = """            ids, weights, is_padding = routing
            if getattr(self, "_use_cutlass", False):
                from .ops.moe_cutlass import cutlass_experts

                return cutlass_experts(
                    x, ids, weights, is_padding,
                    self._w13_w, self._w13_s, self._w13_esc,
                    self._w2_w, self._w2_s, self._w2_esc,
                    self.nvfp4, self.num_experts, self.gate_cap, self.up_cap,
                )
            from .ops.moe_xpu import packed_experts

"""
assert s.count(old_fwd) == 1
s = s.replace(old_fwd, new_fwd)
moe.write_text(s)
print('patched', moe)
