#!/usr/bin/env python3
"""One fresh Scaleway live-stock request; require physical readiness, no provider retries."""
import argparse
import datetime
import ipaddress
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
NAME='qwen38-teacher-fidelity-check-20261006'
CLOUD='scaleway'
REGION='paris-france-1'
SKU='H100'
PRICE=330
KEY=(Path.home()/'.shadeform/api_key').read_text().strip()
parser=argparse.ArgumentParser();parser.add_argument('--delete',action='store_true');args=parser.parse_args()
def api(method,path,data=None):
    request=urllib.request.Request('https://api.shadeform.ai/v1'+path,data=json.dumps(data).encode() if data is not None else None,method=method,headers={'X-API-KEY':KEY,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=45) as response:
            raw=response.read();return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        # Retain the diagnostic privately; never print a response that might
        # reflect credential/account data. No automatic creation retries.
        (ROOT/'provider-error-private.json').write_text(json.dumps({'status':error.code,'method':method,'path':path,'body':error.read().decode(errors='replace')}))
        print('PROVIDER_HTTP_REJECTION='+str(error.code)+' PRIVATE_DIAGNOSTIC_RETAINED',flush=True)
        raise
def owned():
    return [x for x in api('GET','/instances').get('instances',[]) if x.get('name')==NAME]
def save(info):
    (ROOT/'provider-instance.json').write_text(json.dumps({k:info.get(k) for k in ('id','name','status','cloud','region','hourly_price')},indent=2))
if args.delete:
    matches=owned();assert len(matches)<=1
    for item in matches:
        assert item['name']==NAME
        api('POST','/instances/'+item['id']+'/delete')
        print('Deletion requested for owned full-corpus lease:',item['id'],flush=True)
    raise SystemExit(0)
assert not owned(),'Owned full-corpus lease exists; inspect/reuse, never duplicate create'
keys=api('GET','/sshkeys')['ssh_keys']
public=' '.join(Path('/home/mike/.sky/clients/983975c5/ssh/sky-key.pub').read_text().split()[:2])
matched=[x for x in keys if ' '.join(x.get('public_key','').split()[:2])==public];assert len(matched)==1
sku=next(x for x in api('GET','/instances/types')['instance_types'] if x['cloud']==CLOUD and x['shade_instance_type']==SKU)
assert sku['hourly_price']==PRICE and sku['configuration']['num_gpus']==1 and sku['configuration']['vram_per_gpu_in_gb']==80
assert any(x['region']==REGION and x['available'] and x['rental_type']=='on_demand' for x in sku['availability'])
deadline=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=2)
request={'cloud':CLOUD,'region':REGION,'shade_instance_type':SKU,'shade_cloud':True,'name':NAME,'ssh_key_id':matched[0]['id'],'rental_type':'on_demand','auto_delete':{'date_threshold':deadline.isoformat().replace('+00:00','Z'),'spend_threshold':'6.60'}}
(ROOT/'provider-create-request.json').write_text(json.dumps(request,indent=2))
try:
    created=api('POST','/instances/create',request)
    save({'id':created['id'],'name':NAME,'status':'creating','cloud':CLOUD,'region':REGION,'hourly_price':PRICE})
    print('ONE_SCALEWAY_LIVE_STOCK_REQUEST_ACCEPTED'+': '+created['id'],flush=True)
    started=time.monotonic()
    while time.monotonic()-started<600:
        info=api('GET','/instances/'+created['id']+'/info')
        assert info['name']==NAME and info['cloud']==CLOUD and info['region']==REGION and info['shade_instance_type']==SKU
        assert info['hourly_price']==PRICE and info['configuration']['num_gpus']==1
        save(info)
        print('ALLOCATION_STATE='+info['status']+' ELAPSED_S='+str(round(time.monotonic()-started)),flush=True)
        details=str(info.get('status_details','')).lower()
        if any(reason in details for reason in ('insufficient capacity','out of capacity','no capacity','sold out')):
            raise RuntimeError('Provider explicitly reports unavailable capacity; cancel immediately')
        if info['status']=='active' and info.get('ip'):
            host=info['ip'];ipaddress.ip_address(host);port=int(info.get('ssh_port',22));user=info['ssh_user']
            assert re.fullmatch(r'[a-z][a-z0-9_-]*',user) and 1<=port<=65535
            alias='shadeform-'+str(uuid.UUID(created['id']))
            text=f'Host {NAME}\n  HostName {host}\n  HostKeyAlias {alias}\n  User {user}\n  Port {port}\n  IdentityFile /home/mike/.sky/clients/983975c5/ssh/sky-key\n  IdentitiesOnly yes\n  BatchMode yes\n  StrictHostKeyChecking accept-new\n  UserKnownHostsFile /home/mike/.ssh/known_hosts\n'
            directory=Path.home()/'.sky/generated/ssh';directory.mkdir(parents=True,exist_ok=True)
            (directory/NAME).write_text(text)
            effective=subprocess.check_output(['ssh','-G',NAME],text=True)
            assert effective.split('hostname ',1)[1].splitlines()[0]==host and effective.split('hostkeyalias ',1)[1].splitlines()[0]==alias
            ready=False
            for _ in range(3):
                check=subprocess.run(['ssh','-o','ConnectTimeout=10',NAME,'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'],capture_output=True,text=True,timeout=20)
                if check.returncode==0:
                    assert 'H100' in check.stdout and any(int(m)>=75000 for m in re.findall(r'(\d+) MiB',check.stdout))
                    print('PHYSICAL_H10080_SSH_VERIFIED '+check.stdout.strip(),flush=True);ready=True;break
                time.sleep(5)
            assert ready,'Provider active but physical GPU SSH readiness failed'
            print('WORKING_OWNED_GPU_ALIAS='+NAME,flush=True)
            break
        if info['status'] in ('error','failed','deleted','deleting'):
            raise RuntimeError('Provider boot failed; do not retry unavailable capacity')
        time.sleep(5)
    else:
        raise TimeoutError('Asynchronous boot did not produce a ready GPU within10min; cancel,no provider retries')
except BaseException:
    if 'created' in globals():
        instance_id=str(uuid.UUID(created['id']))
        api('POST','/instances/'+instance_id+'/delete')
        print('Canceled unready owned Scaleway lease:',instance_id,flush=True)
    raise
