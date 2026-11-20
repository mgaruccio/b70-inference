"""Teacher-forced decode capture: replay a reference trajectory (parity.jsonl) through the
normal greedy decode path (graphs unless TF_EAGER=1) and record raw logprobs of the
forced tokens. Also records an unforced greedy run of case 0 as a sanity check."""
import json, os, pathlib, sys
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from forcetok import ForceTokens

root = pathlib.Path(sys.argv[1]); here = pathlib.Path.cwd()
ref = [json.loads(l) for l in open(here / os.environ.get('TF_REF', 'tf-ref-moe.jsonl'))]
prompts = json.loads((here / 'benchmark-prompts.json').read_text())
long_prompt = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
cases = prompts + [long_prompt]
llm = LLM(model=str(here / 'gptq'), dtype='bfloat16', attention_backend='TRITON_ATTN',
          enforce_eager=os.environ.get('TF_EAGER', '0') == '1', max_model_len=4096, max_num_seqs=1,
          max_num_batched_tokens=4096, enable_prefix_caching=False, gpu_memory_utilization=0.90,
          logits_processors=[ForceTokens])
tok = llm.get_tokenizer()
with (root / 'tf.jsonl').open('w') as out:
    for r in ref:
        ids = tok(cases[r['case']], add_special_tokens=True)['input_ids']
        assert len(ids) == r['prompt_len'], (len(ids), r['prompt_len'])
        sp = SamplingParams(temperature=0, max_tokens=len(r['token_ids']), ignore_eos=True, logprobs=1,
                            extra_args={'force': r['token_ids'], 'prompt_len': len(ids)})
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0].outputs[0]
        got = list(o.token_ids)
        lp = [d[t].logprob for d, t in zip(o.logprobs, got)]
        print(json.dumps({'case': r['case'], 'prompt_len': len(ids), 'token_ids': got, 'lp': lp,
                          'forced_ok': got == r['token_ids']}), file=out, flush=True)
    r = ref[0]
    ids = tok(cases[r['case']], add_special_tokens=True)['input_ids']
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                     SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, logprobs=1), use_tqdm=False)[0].outputs[0]
    (root / 'free0.json').write_text(json.dumps({'token_ids': list(o.token_ids),
                                                 'lp': [d[t].logprob for d, t in zip(o.logprobs, o.token_ids)]}))
(root / 'offline.done').write_text('ok\n')
