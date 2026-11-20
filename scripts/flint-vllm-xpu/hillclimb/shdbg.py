import json, os, pathlib, sys
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
import torch
from vllm import LLM
root = pathlib.Path(sys.argv[1])
llm = LLM(model=str(pathlib.Path.cwd() / 'gptq'), dtype='bfloat16', attention_backend='TRITON_ATTN',
          enforce_eager=True, max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=512,
          enable_prefix_caching=False, gpu_memory_utilization=0.90)
def work(model):
    b = [l.mlp for l in model.model.layers if hasattr(l.mlp, 'shared_expert')][0]
    import vllm.models.flint.model as fm
    out = {'fold': fm._XPU_SHARED_FOLD, 'v3': fm._XPU_DECODE_V3, 'attr': repr(getattr(b, '_xpu_shared_pack', 'MISSING'))[:80]}
    ex = getattr(b.experts, 'routed_experts', None); w13 = getattr(ex, 'w13_qweight', None)
    out['w13'] = None if w13 is None else (str(w13.dtype), w13.dim())
    for nm in ('gate_up_proj', 'down_proj'):
        m = getattr(b.shared_expert, nm)
        out[nm] = {k: (repr(getattr(m, k, None).dtype), tuple(getattr(m, k).shape)) if getattr(m, k, None) is not None else None for k in ('qweight', 'scales', 'qzeros')}
        out[nm]['t_contig'] = m.qweight.t().is_contiguous()
        out[nm]['qz'] = m.qzeros.tolist()
    out['attr_after_load'] = out['attr']
    out['capturing'] = torch.xpu.is_current_stream_capturing()
    try:
        r = b._xpu_shared_pack_get(); out['get'] = None if r is None else [tuple(t.shape) for t in r]
    except Exception as e:
        import traceback; out['get_err'] = traceback.format_exc()[-600:]
    return out
res = llm.apply_model(work)
print(res, flush=True)
(root / 'shdbg.json').write_text(json.dumps(res, default=str, indent=1))
(root / 'offline.done').write_text('ok\n')
