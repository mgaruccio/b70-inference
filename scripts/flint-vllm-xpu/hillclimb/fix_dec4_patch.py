"""Fix round-2 pack timing: retry until eligible; also pack after the first MoE forward."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
if '_xpu_shared_pack_v2' in s:
    print('already fixed'); sys.exit(0)
old = '''            if torch.xpu.is_current_stream_capturing():
                return None
            sh = False
'''
new = '''            if torch.xpu.is_current_stream_capturing():
                return None
            sh = None  # _xpu_shared_pack_v2: retry until the expert layout is final
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''            self._xpu_shared_pack = sh
        return sh or None
'''
new = '''            self._xpu_shared_pack = sh
        return sh
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''        if current_platform.is_xpu() and _XPU_DECODE_V3 and _XPU_SHARED_FOLD:
            self._xpu_shared_pack_get()
'''
assert s.count(old) == 1; s = s.replace(old, '')
old = '''        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)
'''
new = '''        out = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        if current_platform.is_xpu() and _XPU_DECODE_V3 and _XPU_SHARED_FOLD:
            # Pack eagerly (first call: the memory-profiling run, before capture),
            # after the expert weights have their final layout.
            self._xpu_shared_pack_get()
        return out
'''
assert s.count(old) == 1; s = s.replace(old, new)
p.write_text(s); print('fixed')
