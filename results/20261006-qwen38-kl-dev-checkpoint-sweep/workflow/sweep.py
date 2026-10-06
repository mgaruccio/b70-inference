#!/usr/bin/env python3
"""Dev-only D8 checkpoint sweep using the existing disposable serving path."""
import importlib.util,json,os,signal,traceback
from pathlib import Path
CODE=Path.home()/'qwen-mtp-code'
spec=importlib.util.spec_from_file_location('existing_workflow',CODE/'run-pilot.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

def main():
    os.umask(0o077)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt('terminated')))
    protocol=json.loads((w.RUN/'pilot-protocol.json').read_text())
    assert protocol['optimizer_updates']==0 and protocol['split']=='dev' and protocol['requests_per_cell']==64
    assert len(w.records('dev'))==64 and not (w.RUN/'test-requests.jsonl').exists()
    restored=json.loads((w.RUN/'cache-restore.json').read_text())
    inventory={x['step']:x for x in restored['checkpoints']}
    assert set(inventory)==set(range(0,3741,374)) and restored['optimizer_updates_this_run']==0
    assert not restored['test_data_restored'] and not restored['recaptured']
    cells={};epoch_cells={}
    def evaluate(label,depth,step=None):
        weights=None if step is None else w.RUN/'training'/inventory[step]['file']
        cells[label]=w.evaluate(label,depth,'dev',weights)
        w.save(w.RUN/'sweep-progress.json',dict(status='in_progress',split='dev',cells=cells,optimizer_updates=0))
        w.archive('evaluation')
    try:
        evaluate('dev-no-spec',0)
        evaluate('D8-stock-start',8)
        for step in protocol['checkpoint_order']:
            label='D8-step'+str(step).zfill(4);evaluate(label,8,step);epoch_cells[step]=label
        evaluate('D8-stock-end',8)
        def score(step):
            row=cells[epoch_cells[step]]
            return (row['accepted']['accepted_per_draft_pass'],row['median_e2e_tps'],-step)
        winner=max(epoch_cells,key=score)
        reference=2618
        confirmation=[]
        for tag,step in [('A1-ce-selected',reference),('B1-dev-acceptance',winner),('B2-dev-acceptance',winner),('A2-ce-selected',reference)]:
            label='D8-confirm-'+tag;evaluate(label,8,step);confirmation.append(dict(cell=label,step=step))
        matches={}
        for label in cells:
            if label=='dev-no-spec':continue
            comparisons=[]
            for record in w.records('dev'):
                key=record['prompt_id'];ref=json.loads((w.RUN/'dev-no-spec'/(key+'-result.json')).read_text());row=json.loads((w.RUN/label/(key+'-result.json')).read_text())
                assert ref['prompt_token_ids']==row['prompt_token_ids'],'Rendered prompt mismatch'
                common=min(len(ref['token_ids']),len(row['token_ids']))
                differences=[i for i in range(common) if ref['token_ids'][i]!=row['token_ids'][i]]
                first=differences[0] if differences else (common if len(ref['token_ids'])!=len(row['token_ids']) else None)
                comparisons.append(dict(id=key,exact=first is None,first_divergence=first,reference_tokens=len(ref['token_ids']),candidate_tokens=len(row['token_ids']),reference_pass=ref['functional']['pass'],candidate_pass=row['functional']['pass']))
            matches[label]=comparisons
        ranking=[dict(step=step,cell=epoch_cells[step],**cells[epoch_cells[step]]) for step in sorted(epoch_cells,key=score,reverse=True)]
        report=dict(status='completed',tier='development',split='dev',optimizer_updates=0,test_data_used=False,promotion=False,cells=cells,ranking=ranking,dev_selection=dict(step=winner,metric='accepted_draft_tokens_per_verification_pass_bonus_excluded',tie_break=['median_e2e_tps','earlier_step'],previous_ce_selected_step=reference),confirmation=confirmation,token_matches_vs_no_spec=matches,limitations=['Dev-selected exploratory result, not blind test confirmation or production qualification.','Prior strict-parity and baseline fidelity failures remain; this run measures and reports fidelity, not a strict parity closure.','Warmup and sampling unchanged; cold model load excluded from request timing.'])
        w.save(w.RUN/'sweep.json',report)
        print('ALL_TEN_EPOCHS_DEV64_D8_SWEEP_AND_ABBA_CONFIRMATION_COMPLETED_ZERO_TRAINING winner='+str(winner),flush=True)
    except BaseException as error:
        w.save(w.RUN/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()));raise
    finally:w.archive('final')
if __name__=='__main__':main()
