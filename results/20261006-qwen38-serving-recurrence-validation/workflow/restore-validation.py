#!/usr/bin/env python3
"""Restore the existing full corpus and two reference heads; no recapture/test set."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tarfile

os.umask(0o077)
HOME=Path.home(); RUN=HOME/'qwen-mtp-run'; CODE=HOME/'qwen-mtp-code'
EXPECTED='27ba5a05d762c90a6fbba164486dce636dac125214fb62ca40aaaad17c950458'
spec=importlib.util.spec_from_file_location('parts',Path(__file__).with_name('restore-cache.py'))
parts=importlib.util.module_from_spec(spec); spec.loader.exec_module(parts); parts.EXPECTED=EXPECTED

def main():
    index=json.loads((HOME/'qwen-mtp-upload/restore-private.json').read_text())
    assert index['sha256']==EXPECTED and index['bytes']==14070067200 and len(index['parts'])==7
    assert [p['path'] for p in index['parts']]==['final.tgz.part'+str(i).zfill(2) for i in range(7)]
    exact={'training/tuned-mtp.step0000.safetensors','training/tuned-mtp.step2618.safetensors','training/tuned-mtp.json'}
    stage=HOME/'qwen-mtp-validation-restore'; stage.mkdir(mode=0o700,exist_ok=False)
    seen=set()
    with parts.Parts(index) as stream:
        with tarfile.open(fileobj=stream,mode='r|') as archive:
            for member in archive:
                name=member.name.removeprefix('./')
                if name.startswith(('capture-train/','capture-dev/')) or name in exact:
                    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                        raise ValueError('Unsupported selected archive member')
                    archive.extract(member,path=stage,filter='data')
                    if name in exact:
                        assert name not in seen,'Duplicate head/metadata'
                        seen.add(name)
        stream.verify()
    assert seen==exact,'Missing exact reference inventory'
    spec=importlib.util.spec_from_file_location('trainer',CODE/'scripts/experiments/qwen38_train_mtp.py')
    trainer=importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    config=trainer.native_config(json.loads((HOME/'qwen-model/config.json').read_text()))
    torch=trainer.runtime().torch
    counts={}; lengths={}; positions_checked=0
    for split,folder,expected in [('train','capture-train/train',374),('dev','capture-dev/heldout',64)]:
        source=stage/folder; paths=sorted(source.glob('*.pt')); assert len(paths)==expected
        values=[]
        for path in paths:
            record=torch.load(path,weights_only=True,map_location='cpu')
            trainer.validate_record(record,config,2048)
            assert torch.equal(record['positions'],torch.arange(record['positions'].numel())), 'Noncontiguous capture positions'
            values.append({'id':record['prompt_id'],'file':path.name,'tokens':record['input_ids'].numel(),'hidden_rows':record['positions'].numel()})
            positions_checked+=1
        counts[split]=len(paths); lengths[split]=values
        target=RUN/folder; target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        assert not target.exists(); shutil.move(str(source),str(target))
    inventory=[]
    for name in sorted(exact):
        path=stage/name
        sha=hashlib.sha256()
        with path.open('rb') as handle:
            while block:=handle.read(8*1024**2): sha.update(block)
        if path.suffix=='.safetensors':
            with trainer.runtime().safe_open(str(path),framework='pt',device='cpu') as tensors:
                state={k:tensors.get_tensor(k) for k in tensors.keys()}
                trainer.validate_mtp_state(state,trainer.expected_mtp_shapes(config))
                assert all(v.dtype==torch.bfloat16 and torch.isfinite(v).all().item() for v in state.values())
        inventory.append({'file':name,'bytes':path.stat().st_size,'sha256':sha.hexdigest()})
    metadata=json.loads((stage/'training/tuned-mtp.json').read_text())
    assert metadata['optimizer_steps']==3740 and metadata['config']['epochs']==10
    assert metadata['precision']['regime']=='native_bf16'
    target=RUN/'reference-heads'; assert not target.exists(); shutil.move(str(stage/'training'),str(target))
    report={'source_archive_sha256':EXPECTED,'bytes_verified':index['bytes'],'parts_verified':7,'restored_sequences':counts,'contiguous_position_records_checked':positions_checked,'reference_inventory':inventory,'optimizer_updates_this_run':0,'recaptured':False,'test_data_restored':False,'lengths':lengths}
    (RUN/'cache-restore.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FULL374_64_CAPTURE_AND_TWO_REFERENCE_HEAD_RESTORE_VERIFIED',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        print('VALIDATION_RESTORE_FAILED='+type(error).__name__,flush=True)
        raise SystemExit(1)
