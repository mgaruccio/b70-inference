#!/usr/bin/env python3
"""Full R2 archive readback without requiring another local multi-GB copy."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
root=Path(__file__).resolve().parent
prefix=(root/'r2-location.txt').read_text().strip()
subprocess.run(['rclone','copyto',prefix+'/final.tgz.parts.json',str(root/'final.tgz.parts.json'),'--contimeout','15s','--timeout','180s'],check=True,timeout=900)
index=json.loads((root/'final.tgz.parts.json').read_text())
assert index['archive']=='final.tgz' and re.fullmatch('[0-9a-f]{64}',index['sha256'])
assert 1<=len(index['parts'])<=32
whole=hashlib.sha256();total=0
for position,part in enumerate(index['parts']):
    name=part['path']
    assert name=='final.tgz' if len(index['parts'])==1 else name=='final.tgz.part'+str(position).zfill(2)
    sha=hashlib.sha256();size=0
    with subprocess.Popen(['timeout','900','rclone','cat',prefix+'/'+name,'--contimeout','15s','--timeout','180s'],stdout=subprocess.PIPE) as process:
        while block:=process.stdout.read(8*1024**2):
            sha.update(block);whole.update(block);size+=len(block)
        assert process.wait()==0,'R2 readback failed; preserve the owned capped lease'
    assert size==part['bytes'] and sha.hexdigest()==part['sha256']
    total+=size
assert whole.hexdigest()==index['sha256'] and total==index['bytes']
if '--allow-incomplete' in sys.argv:
    print('INCOMPLETE_RUN_FINAL_ARCHIVE_FULL_READBACK_VERIFIED '+index['sha256'],flush=True)
else:
    report=json.loads((root/'comparison.json').read_text())
    assert report['status']=='completed' and len(report['test_cells'])==9
    assert all(c['requests']==64 for c in report['test_cells'].values())
    training=json.loads((root/'training-report.json').read_text())
    assert training['optimizer_steps']==len(training['train_steps'])==3740
    assert training['counts']['train_sequences']==374 and training['counts']['dev_sequences']==64
    assert training['counts']['sequences_seen']==3740 and training['config']['epochs']==10
    assert training['config']['kl_weight']==1 and training['config']['kl_temperature']==1
    restored=json.loads((root/'cache-restore.json').read_text())
    assert restored['restored_sequences']=={'train':374,'dev':64} and restored['recaptured'] is False
    print('FULL_CACHED_CORPUS374_64_KL_TEN_EPOCHS3740_UPDATES_NINE_TEST_CELLS_AND_FULL_R2_READBACK_VERIFIED '+index['sha256'],flush=True)
