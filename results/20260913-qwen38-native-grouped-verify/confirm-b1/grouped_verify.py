"""Pre-expansion seam for this experiment, not a serving installation hook."""
import math
import torch


def unsupported_reason(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                       max_seqlen_k, k_descale, v_descale, softmax_scale,
                       s_aux, window_size, softcap):
    if not isinstance(q, torch.Tensor) or q.device.type != "xpu":
        return "XPU Q required"
    tensors = (k, v, cu_seqlens_q, seqused_k, block_table, k_descale, v_descale)
    if any(not isinstance(t, torch.Tensor) or t.device != q.device for t in tensors):
        return "all inputs/descales must share Q's XPU device"
    if q.dtype != torch.float16 or tuple(q.shape) != (5, 24, 256):
        return "only five FP16 queries, 24 heads, D256"
    if q.stride(-1) != 1 or any(s <= 0 or s > 2**31 - 1 for s in q.stride()):
        return "positive pitched Q with contiguous D required"
    supported_layouts = {
        ((176, 1664, 4, 256), (1664 * 4 * 256, 256, 1664 * 256, 1)),
        ((152, 1664, 4, 256), (1664 * 4 * 512, 4 * 512, 512, 1)),
    }
    for cache in (k, v):
        if (cache.dtype != torch.float8_e4m3fn
                or (tuple(cache.shape), tuple(cache.stride())) not in supported_layouts):
            return "unsupported FP8 KV shape/strides"
    for t, shape in ((cu_seqlens_q, (2,)), (seqused_k, (1,)), (block_table, (1, 128))):
        if t.dtype != torch.int32 or tuple(t.shape) != shape or not t.is_contiguous():
            return "unexpanded contiguous device int32 C1 metadata required"
    for t in (k_descale, v_descale):
        if t.dtype != torch.float32 or t.numel() == 0 or any(
                n > 1 and stride != 0 for n, stride in zip(t.shape, t.stride())):
            return "scalar broadcast float32 descales required"
    if out is not None and (not isinstance(out, torch.Tensor) or out.device != q.device
                            or out.dtype != q.dtype or tuple(out.shape) != tuple(q.shape)
                            or not out.is_contiguous()):
        return "native-compatible contiguous output required"
    if (type(max_seqlen_k) is not int or not 1 <= max_seqlen_k <= 212992
            or not isinstance(softmax_scale, (int, float))
            or not math.isfinite(softmax_scale) or softmax_scale <= 0):
        return "invalid static capacity bound or scale"
    if s_aux is not None or tuple(window_size) != (-1, -1) or softcap != 0.0:
        return "sink, local attention and softcap unsupported"
    return None


def packed_forward(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                   max_seqlen_k, k_descale, v_descale, softmax_scale,
                   s_aux, window_size, softcap):
    # [t,kv,group,d] -> [kv,t,group,d]. This copy handles pitched Q too.
    # packed[0, kv*30 + t*6 + group, d] = q[t, kv*6 + group, d].
    packed_q = q.reshape(5, 4, 6, 256).permute(1, 0, 2, 3).contiguous().view(1, 120, 256)
    packed_cu = torch.arange(2, dtype=torch.int32, device=q.device)
    # The native helper ignores cu CONTENTS for uniform C1 decode, including
    # padded cu=[0,0]. All five rows attend at least key zero, even for used=0.
    # Clamp the shared scheduling length on DEVICE. For used<=1 all five
    # max(used-4+t,1) limits equal 1, so this preserves exact short semantics.
    packed_used = seqused_k.clamp_min(1)
    y = torch.ops.b70_grouped_verify.forward(
        packed_q, k, v, packed_used, block_table, packed_cu,
        k_descale, v_descale, max_seqlen_k, softmax_scale)
    if out is None:
        out = torch.empty((5, 24, 256), dtype=q.dtype, device=q.device)
    # One strided copy, rather than a second temporary contiguous permutation.
    out.view(5, 4, 6, 256).copy_(y.view(4, 5, 6, 256).permute(1, 0, 2, 3))
    return out


def dispatch(native, *args, **kwargs):
    """Only unsupported source metadata falls back. Eligible failures propagate."""
    reason = unsupported_reason(*args, **kwargs)
    if reason is not None:
        return native(*args, **kwargs)
    return packed_forward(*args, **kwargs)
