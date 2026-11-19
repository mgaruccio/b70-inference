"""Bucket XPU kernel time from a torch profiler chrome trace; per decode step."""
import gzip, json, re, sys, collections
path = sys.argv[1]
tr = json.load(gzip.open(path) if path.endswith('.gz') else open(path))
ev = tr['traceEvents'] if isinstance(tr, dict) else tr
kern = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_op', 'Kernel')]
ann = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('user_annotation', 'gpu_user_annotation', 'python_function', 'cpu_op')
       and re.search(r'execute_model|_model_forward|model_forward', e.get('name', ''))]
cats = [
    ('moe_grouped_gemm', r'grouped_gemm|GroupedGemm|cutlass.*group|moe_gemm|xe_gemm.*group'),
    ('moe_misc', r'remap_hidden|moe_gather|topk_sigmoid|topk|moe_align|count_and_sort|silu_and_mul|act_and_mul|swiglu'),
    ('w4a16_gemm(oneDNN)', r'gemm_kernel|onednn|dnnl|int4|woq|matmul'),
    ('attention', r'unified_attention|attn|reshape_and_cache|kv_cache'),
    ('router_triton', r'_router_kernel'),
    ('copy/cast', r'copy|Copy|cast|to_copy|convert|contiguous'),
    ('elementwise', r'elementwise|vectorized|unrolled|reduce|Reduce|mean|rsqrt|pow|mul|add|cat|index|gather|scatter|fill'),
]
def cat_of(n):
    for c, rx in cats:
        if re.search(rx, n):
            return c
    return 'other'
kern.sort(key=lambda e: e['ts'])
# steps: cluster kernels by CPU annotation windows when available, else by gaps > 2ms
steps = []
if ann:
    ann = sorted(ann, key=lambda e: e['ts'])
    print('annotation windows', len(ann), 'example', ann[0]['name'])
cur = [kern[0]]
for a, b in zip(kern, kern[1:]):
    if b['ts'] - (a['ts'] + a.get('dur', 0)) > 2000:
        steps.append(cur); cur = []
    cur.append(b)
steps.append(cur)
print('kernels', len(kern), 'gap-clustered steps', len(steps), 'kernels/step', [len(s) for s in steps][:6], '...')
def summarize(group, label):
    tot = collections.Counter(); cnt = collections.Counter(); names = collections.Counter()
    for s in group:
        for e in s:
            c = cat_of(e['name']); tot[c] += e.get('dur', 0); cnt[c] += 1; names[(c, e['name'][:70])] += e.get('dur', 0)
    n = len(group)
    span = sum((s[-1]['ts'] + s[-1].get('dur', 0) - s[0]['ts']) for s in group) / max(n, 1)
    busy = sum(tot.values()) / max(n, 1)
    print(f'== {label}: steps={n} per-step kernel busy {busy/1000:.2f} ms, span {span/1000:.2f} ms, kernels/step {sum(cnt.values())/max(n,1):.0f}')
    for c, t in tot.most_common():
        print(f'   {c:22s} {t/n/1000:7.3f} ms  ({cnt[c]/n:.0f} launches)')
    print('   top kernels:')
    for (c, nm), t in names.most_common(14):
        print(f'     {t/n/1000:7.3f} ms  [{c}] {nm}')
summarize(steps[:1], 'first step (prefill)')
summarize(steps[1:], 'decode steps')
