import copy, importlib.util, json, sys
from pathlib import Path
import torch
root=Path.home()
spec=importlib.util.spec_from_file_location('pilot', root/'mtp-pilot-code/scripts/experiments/glimmer_recursive_mtp.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
report=json.loads((root/'mtp-pilot-run/evaluate.json').read_text())
pair=report['pairs'][-1]
ck=torch.load(root/'mtp-pilot-run/heads/fixed-ce.pt',map_location='cpu',weights_only=True)
backend=sys.argv[1] if len(sys.argv)>1 else 'sdpa'
target=p.GlimmerTarget(backend)
head=p.make_head(6656,ck['rank'],False); head.load_state_dict(ck['head']); head.eval().to(target.device)
rows=[]
with torch.inference_mode():
    prompt=pair['prompt_token_ids']
    states,_,single=target.forward(prompt,None,logits_to_keep=1)
    states_b,_,branch=target.forward(prompt,None,logits_to_keep=1)
    branch_root=states_b[-1]
    for i,token in enumerate(pair['baseline']['token_ids'][:53]):
        before=branch.get_seq_length()
        drafts,_=p.Drafter(head,target).draft(branch_root,token,1,target.eos)
        same_before=copy.deepcopy(single)
        hs,ls,single=target.forward([token],single)
        hb,lb,branch=target.forward([token,*drafts],branch)
        target.crop(branch,before+1)
        branch_root=hb[0]
        top_s=int(ls[0].argmax()); top_b=int(lb[0].argmax())
        row={'position_consumed':i,'consumed_token':token,'draft':drafts,'single_next':top_s,'branch_next':top_b,'hidden_max_abs_diff':float((hs[0].float()-hb[0].float()).abs().max()),'logit_max_abs_diff':float((ls[0].float()-lb[0].float()).abs().max()),'single_top5':[(int(a),float(b)) for a,b in zip(ls[0].topk(5).indices,ls[0].topk(5).values)],'branch_top5':[(int(a),float(b)) for a,b in zip(lb[0].topk(5).indices,lb[0].topk(5).values)]}
        if top_s!=top_b:
            hf,lf,_=target.forward([token,*drafts],same_before)
            row['same_prior_cache_block_next']=int(lf[0].argmax())
            row['same_prior_cache_block_hidden_max_abs_diff']=float((hs[0].float()-hf[0].float()).abs().max())
            _,lp,_=target.forward([*prompt,*pair['baseline']['token_ids'][:i+1]],None,logits_to_keep=1)
            row['fresh_prefill_next']=int(lp[-1].argmax())
        rows.append(row)
    output={'purpose':'same chosen-token prefix diagnostic only; not decoding/acceptance/speed evidence','environment':target.environment,'id':pair['id'],'rows':rows}
    (root/f'mtp-pilot-run/identity-diagnostic-{backend}.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'id':pair['id'],'mismatches':[r for r in rows if r['single_next']!=r['branch_next']],'max_hidden_diff':max(r['hidden_max_abs_diff'] for r in rows)},indent=2),flush=True)
