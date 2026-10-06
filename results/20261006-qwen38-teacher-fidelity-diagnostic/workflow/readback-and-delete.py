#!/usr/bin/env python3
"""Verify the existing final multipart archive, then delete only the recorded owned lease."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import uuid

os.umask(0o077)
ROOT=Path(__file__).resolve().parent

def main():
    prefix=(ROOT/'r2-location.txt').read_text().strip()
    assert prefix=='r2:ml-archive/2026-10-06/cache/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic'
    raw=subprocess.check_output(['rclone','cat',prefix+'/final.tgz.parts.json'],timeout=90)
    index=json.loads(raw); (ROOT/'final.tgz.parts.json').write_bytes(raw)
    assert index['archive']=='final.tgz' and 1<=len(index['parts'])<=32 and re.fullmatch('[0-9a-f]{64}',index['sha256'])
    whole=hashlib.sha256(); total=0; verified=[]
    for number,part in enumerate(index['parts']):
        assert part['path']=='final.tgz.part'+str(number).zfill(2) or (len(index['parts'])==1 and part['path']=='final.tgz' and part['bytes']==index['bytes'] and part['sha256']==index['sha256'])
        assert part['bytes']>0 and re.fullmatch('[0-9a-f]{64}',part['sha256'])
        proc=subprocess.Popen(['rclone','cat',prefix+'/'+part['path']],stdout=subprocess.PIPE)
        sha=hashlib.sha256(); size=0
        try:
            while block:=proc.stdout.read(8*1024**2):
                sha.update(block); whole.update(block); size+=len(block); total+=len(block)
            assert proc.wait(timeout=120)==0,'Part download failed'
        finally:
            proc.stdout.close()
            if proc.poll() is None: proc.kill(); proc.wait()
        assert size==part['bytes'] and sha.hexdigest()==part['sha256'],'Archive part readback mismatch'
        verified.append({'path':part['path'],'bytes':size,'sha256':sha.hexdigest()})
        print('FINAL_PART_READBACK_VERIFIED='+part['path'],flush=True)
    assert total==index['bytes'] and whole.hexdigest()==index['sha256'],'Whole archive mismatch'
    report={'archive':prefix,'bytes':total,'sha256':whole.hexdigest(),'parts':verified,'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (ROOT/'archive-readback.json').write_text(json.dumps(report,indent=2)+'\n')
    saved=json.loads((ROOT/'provider-instance.json').read_text()); ident=str(uuid.UUID(saved['id']))
    protocol=json.loads((ROOT/'pilot-protocol.json').read_text())
    assert saved['name']==protocol['lease_name'] and saved['cloud']=='scaleway'
    key=(Path.home()/'.shadeform/api_key').read_text().strip()
    def api(method,path):
        request=urllib.request.Request('https://api.shadeform.ai/v1'+path,method=method,headers={'X-API-KEY':key})
        with urllib.request.urlopen(request,timeout=45) as response:
            data=response.read(); return json.loads(data) if data else {}
    data=api('GET','/instances'); assert isinstance(data.get('instances'),list),'Unexpected provider schema'
    matching=[x for x in data['instances'] if x['id']==ident]
    assert len(matching)<=1
    if matching:
        info=matching[0]
        assert info['name']==protocol['lease_name'] and info['cloud']=='scaleway' and info['hourly_price']==330
        api('POST','/instances/'+ident+'/delete')
        print('VERIFIED_ARCHIVE_OWNED_LEASE_DELETE_REQUESTED='+ident,flush=True)
    absent=False
    for attempt in range(12):
        data=api('GET','/instances'); assert isinstance(data.get('instances'),list),'Unexpected provider schema'
        if not any(x['id']==ident for x in data['instances']): absent=True; break
        time.sleep(5)
    assert absent,'Provider deletion not yet confirmed; do not claim cleanup'
    cleanup={'instance_id':ident,'name':protocol['lease_name'],'provider':'shadeform/scaleway','provider_api_absence_confirmed':True,'confirmed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_verified':True,'explicit_delete_requested_after_readback':bool(matching),'provider_already_absent_when_cleanup_started':not bool(matching),'backup_verified_before_explicit_delete':bool(matching),'lease_cap_hours':2,'lease_cap_usd':6.6,'optimizer_updates_authorized':0}
    (ROOT/'cleanup-confirmation.json').write_text(json.dumps(cleanup,indent=2)+'\n')
    print('OWNED_PROVIDER_ABSENCE_CONFIRMED_AFTER_FULL_ARCHIVE_READBACK',flush=True)

if __name__=='__main__': main()
