#!/usr/bin/env python3
"""Exercise the implementation's own greedy/teacher replay on real diagnostic records."""
import json
import os
from pathlib import Path
import random
import sys
import time

os.umask(0o077)
HOME=Path.home(); CODE=HOME/'qwen-mtp-code'; RUN=HOME/'qwen-mtp-run'; MODEL=HOME/'qwen-model'
sys.path.insert(0,str(CODE/'scripts/experiments'))
import qwen38_train_mtp as trainer

def main():
    torch=trainer.runtime().torch; device=torch.device('cuda')
    checkpoint=trainer.Checkpoint(MODEL)
    teacher=trainer.FrozenTarget(checkpoint,device)
    core=trainer.build_native_mtp(checkpoint.config,checkpoint.mtp_state(),device).eval()
    destination=RUN/'teacher-histories'; destination.mkdir(mode=0o700,exist_ok=False)
    manifest=[]; times=[]
    inventories=json.loads((RUN/'diagnostic-inputs.json').read_text())['records']
    for split in ('train','dev'):
        for number,entry in enumerate(inventories[split]):
            record=trainer.load_record(RUN/(split+'-diagnostics')/entry['file'],checkpoint.config,2048)
            roots=trainer.sample_roots(record,4,8,random.Random(42))
            with torch.no_grad():
                states,proposals=trainer.greedy_depths(core,teacher.embedding,teacher.head,record,device,roots)
            torch.cuda.synchronize(); started=time.perf_counter()
            rows=teacher.replay(record,roots,proposals)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-started
            assert rows.shape==(len(roots),4,checkpoint.config.hidden_size)
            prefix_tokens=0
            for index,root in enumerate(roots):
                name=f'{split}-q{number}-r{root:04d}'
                prefix=torch.cat([record['input_ids'][:root+2],proposals[index,:3].cpu()])
                prefix_tokens+=prefix.numel()
                mask=[False]*(prefix.numel()-3)+[True]*3
                assert len(mask)==prefix.numel() and mask[:2]==[False,False]
                row={'name':name,'source_id':record['prompt_id'],'split':split,'root':root,'input_ids':prefix.tolist(),'loss_mask':mask,'proposals':proposals[index].tolist(),'scored_teacher_positions':list(range(root+1,root+5)),'fresh_prefix':True}
                assert row['scored_teacher_positions']==list(range(prefix.numel()-4,prefix.numel()))
                torch.save({'teacher_rows':rows[index].cpu(),'student_rows':torch.stack([state[index].detach().cpu() for state in states]),'proposals':proposals[index].cpu(),'prefix':prefix,'root':root},destination/(name+'.pt'))
                manifest.append(row)
            times.append({'split':split,'source_id':record['prompt_id'],'record_tokens':record['input_ids'].numel(),'roots':len(roots),'teacher_prefix_tokens':prefix_tokens,'teacher_seconds':elapsed,'seconds_per_root':elapsed/len(roots),'seconds_per_prefix_token':elapsed/prefix_tokens})
            del rows,states,proposals
    native_references=json.loads((RUN/'verifier-inputs.json').read_text())
    for entry in native_references:
        prefix=torch.tensor(entry['input_ids'],dtype=torch.long)
        root=entry['root']
        assert root==prefix.numel()-5 and root>=0
        record={'input_ids':prefix[:-3]}
        proposals=torch.cat([prefix[-3:],torch.zeros(1,dtype=torch.long)]).reshape(1,4).to(device)
        rows=teacher.replay(record,[root],proposals)
        assert rows.shape==(1,4,checkpoint.config.hidden_size)
        torch.save({'teacher_rows':rows[0].cpu(),'prefix':prefix,'root':root},destination/(entry['name']+'.pt'))
        manifest.append(entry)
    assert 0<len(manifest)<=128 and sum(len(r['input_ids']) for r in manifest)<=1048576
    report={'status':'completed','optimizer_updates':0,'full_training':False,'backend':'implementation FrozenTarget.replay','records':6,'branches':len(manifest),'teacher_rows':len(manifest)*4,'timings':times,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'weight_loading':'strict original BF16 multimodal checkpoint; only text forward','rough_teacher_only_29920_branch_hours_range':[29920*min(x['seconds_per_root'] for x in times)/3600,29920*max(x['seconds_per_root'] for x in times)/3600],'estimate_caveat':'diagnostic range only; excludes CE/backward/optimizer, control arm, dev checkpoint selection and serving repeats'}
    report['native_verifier_histories']=len(native_references)
    (RUN/'teacher-inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (RUN/'teacher-profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REAL_GREEDY_PREFIX_TEACHER_REPLAY_BRANCHES='+str(len(manifest))+' ZERO_UPDATES',flush=True)

if __name__=='__main__': main()
