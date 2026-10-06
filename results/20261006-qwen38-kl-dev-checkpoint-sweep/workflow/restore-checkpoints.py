#!/usr/bin/env python3
"""Restore only eleven saved BF16 heads from the verified KL archive."""
import hashlib, importlib.util, json, os, re, shutil, tarfile
from pathlib import Path
os.umask(0o077)
HOME=Path.home()
EXPECTED='27ba5a05d762c90a6fbba164486dce636dac125214fb62ca40aaaad17c950458'
spec=importlib.util.spec_from_file_location('parts',Path(__file__).with_name('restore-cache.py'))
parts=importlib.util.module_from_spec(spec);spec.loader.exec_module(parts);parts.EXPECTED=EXPECTED

def main():
    index=json.loads((HOME/'qwen-mtp-upload/restore-private.json').read_text())
    assert index['sha256']==EXPECTED and index['bytes']==14070067200 and len(index['parts'])==7
    for i,part in enumerate(index['parts']):
        assert part['path']=='final.tgz.part'+str(i).zfill(2)
        assert re.fullmatch('[0-9a-f]{64}',part['sha256'])
    names=['tuned-mtp.step'+str(step).zfill(4)+'.safetensors' for step in range(0,3740,374)]+['tuned-mtp.safetensors']
    allowed={'training/'+name for name in names}|{'training/tuned-mtp.json'}
    dest=HOME/'qwen-mtp-checkpoint-restore';dest.mkdir(mode=0o700,exist_ok=False)
    with parts.Parts(index) as stream:
        with tarfile.open(fileobj=stream,mode='r|') as archive:
            seen=set()
            for member in archive:
                name=member.name.removeprefix('./')
                if name in allowed:
                    if not member.isfile() or name in seen:raise ValueError('Checkpoint link or duplicate forbidden')
                    archive.extract(member,path=dest,filter='data');seen.add(name)
        stream.verify()
    assert seen==allowed,'Missing full saved checkpoint inventory'
    report=json.loads((dest/'training/tuned-mtp.json').read_text())
    assert report['optimizer_steps']==3740 and report['config']['epochs']==10
    assert {Path(x['path']).name for x in report['checkpoints']}==set(names)
    from safetensors import safe_open
    schema=None;inventory=[]
    for step,name in zip(range(0,3741,374),names):
        path=dest/'training'/name
        with safe_open(path,framework='pt',device='cpu') as weights:
            current={key:(weights.get_slice(key).get_shape(),weights.get_slice(key).get_dtype()) for key in weights.keys()}
        assert current and all(dtype=='BF16' for shape,dtype in current.values())
        if schema is None:schema=current
        assert current==schema,'Checkpoint tensor schema differs'
        sha=hashlib.sha256()
        with path.open('rb') as f:
            while block:=f.read(8*1024**2):sha.update(block)
        inventory.append(dict(step=step,file=name,bytes=path.stat().st_size,sha256=sha.hexdigest()))
    target=HOME/'qwen-mtp-run/training';assert not target.exists(),'Refuse checkpoint overwrite'
    shutil.move(str(dest/'training'),str(target))
    result=dict(source_archive_sha256=EXPECTED,bytes_verified=index['bytes'],parts_verified=7,checkpoints=inventory,optimizer_updates_this_run=0,recaptured=False,test_data_restored=False)
    (HOME/'qwen-mtp-run/cache-restore.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL_ELEVEN_SAVED_BF16_HEADS_VERIFIED_ZERO_TRAINING',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:
        print('CHECKPOINT_RESTORE_FAILED='+type(error).__name__,flush=True)
        raise SystemExit(1)
