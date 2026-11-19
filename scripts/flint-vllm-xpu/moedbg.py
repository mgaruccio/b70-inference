import faulthandler, json, os, pathlib, sys
faulthandler.dump_traceback_later(240, exit=True)
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
os.environ['FLINT_XPU_MOE_DECODE'] = '0'   # model uses FusedMoE; we call pieces by hand
import torch
from vllm import LLM
root = pathlib.Path(sys.argv[1]); here = pathlib.Path.cwd()
step = sys.argv[2] if len(sys.argv) > 2 else 'all'
llm = LLM(model=str(here / 'gptq'), dtype='bfloat16', attention_backend='TRITON_ATTN', enforce_eager=True,
          max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=4096, enable_prefix_caching=False,
          gpu_memory_utilization=0.90)
def probe(model):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
    from vllm.models.flint.ops import xpu_moe_decode as md
    mlp = model.model.layers[3].mlp; ex = mlp.experts.routed_experts
    x = (torch.randn(1, 2048, device='xpu') * 0.5).to(torch.bfloat16)
    info = {k: (list(t.shape), str(t.dtype), t.is_contiguous(), t.data_ptr() % 256) for k, t in
            (('w13', ex.w13_qweight), ('s13', ex.w13_scales), ('w2', ex.w2_qweight), ('s2', ex.w2_scales))}
    print('tensors', info, flush=True)
    lg = mlp.gate(x); lg = lg[0] if isinstance(lg, tuple) else lg
    torch.xpu.synchronize(); print('gate ok', lg.dtype, lg.shape, flush=True)
    w, ids, _ = fused_topk(x, lg, mlp.top_k, True, scoring_func='sigmoid')
    torch.xpu.synchronize(); print('topk ok', ids.dtype, ids.tolist(), w.tolist(), flush=True)
    if step in ('v1', 'all'):
        o = md.moe_decode(x, ids, w, ex.w13_qweight, ex.w13_scales, ex.w2_qweight, ex.w2_scales)
        torch.xpu.synchronize(); print('v1 ok', o.float().norm().item(), flush=True)
    if step in ('v2small', 'all'):
        o2 = md.moe_decode_v2(x, ids, w, ex.w13_qweight, ex.w13_scales, ex.w2_qweight, ex.w2_scales, 8, 8, 2, 16, 8, 2)
        torch.xpu.synchronize(); print('v2small ok', o2.float().norm().item(), flush=True)
    if step in ('v2', 'all'):
        o3 = md.moe_decode_v2(x, ids, w, ex.w13_qweight, ex.w13_scales, ex.w2_qweight, ex.w2_scales)
        torch.xpu.synchronize(); print('v2 ok', o3.float().norm().item(), flush=True)
    return 'done'
print(llm.apply_model(probe), flush=True)
(root / 'offline.done').write_text('ok\n')
