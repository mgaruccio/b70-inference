import argparse,concurrent.futures,hashlib,json,os,time,urllib.request
from pathlib import Path
from urllib.parse import urlsplit
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--manifest',required=True);p.add_argument('--upload',action='store_true');a=p.parse_args();root=Path(a.run_dir).resolve();manifest=Path(a.manifest)
EXPECTED={'recursive-depth4-20k/checkpoint-last.pt','recursive-depth4-20k/train-manifest.json',
          'alternating-depth4-20k/checkpoint-last.pt','alternating-depth4-20k/train-manifest.json',
          'prior-control-depth4.json'}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def transfer(item):
    name=item['path'];dest=(root/name).resolve();url=urlsplit(item['url'])
    if not dest.is_relative_to(root) or url.scheme!='https' or not url.hostname.endswith('.r2.cloudflarestorage.com'):raise ValueError('Unsafe R2 item')
    if a.upload:
        if not dest.is_file():return None
        size=dest.stat().st_size;sha=digest(dest)
        # File streaming with explicit size: no credential file installed on the VM.
        with dest.open('rb') as f:
            req=urllib.request.Request(item['url'],data=f,method='PUT',headers={'Content-Length':str(size)})
            with urllib.request.urlopen(req,timeout=900) as r:
                if r.status not in (200,201,204):raise RuntimeError('R2 PUT did not succeed')
                r.read()
        return {'path':name,'bytes':size,'sha256':sha}
    if name not in EXPECTED:raise ValueError('Unexpected baseline restore path')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size==item['bytes'] and digest(dest)==item['sha256']:return None
    tmp=dest.with_name('.'+dest.name+'.r2part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(item['url'],timeout=60) as r,tmp.open('wb') as f:
                if r.status!=200:raise RuntimeError('R2 GET did not succeed')
                while True:
                    chunk=r.read(1024*1024)
                    if not chunk:break
                    f.write(chunk)
            if tmp.stat().st_size!=item['bytes'] or digest(tmp)!=item['sha256']:raise ValueError('R2 checksum mismatch')
            os.replace(tmp,dest);print('restored and SHA256 verified',name,flush=True);return None
        except Exception as e:print('Restore attempt failed:',name,attempt+1,type(e).__name__,flush=True)
    raise RuntimeError('Restore failed: '+name)
try:
    items=json.loads(manifest.read_text());started=time.monotonic()
    if not a.upload:assert len(items)==len(EXPECTED) and {i['path'] for i in items}==EXPECTED
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 if a.upload else 4) as pool:reports=list(pool.map(transfer,items))
    if a.upload:print(json.dumps({'uploaded':[r for r in reports if r], 'seconds':time.monotonic()-started}),flush=True)
    else:print('R2 restore complete, all SHA256 verified;',len(items),'objects;',round(time.monotonic()-started,2),'seconds',flush=True)
except Exception as e:
    # Never expose presigned query parameters in exception text/tracebacks.
    print('R2 transfer failed:',type(e).__name__,flush=True);raise SystemExit(1)
finally:manifest.unlink(missing_ok=True)
