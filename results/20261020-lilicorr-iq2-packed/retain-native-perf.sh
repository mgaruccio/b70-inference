#!/usr/bin/env bash
set -euo pipefail
case=/home/mike/code/b70-inference/results/20261020-lilicorr-iq2-packed
ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@fd7a:115c:a1e0::c933:4f38 'bash -s' >"$case/retention-native-perf-r1.remote.json" <<'REMOTE'
set -euo pipefail
python3 -B - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/home/mike/b70-evals/20261020-lilicorr-iq2-packed-r2/native-perf-r1')
rows=[]
for p in sorted(root.rglob('*')):
 if not p.is_file():continue
 assert not p.is_symlink(),p
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 rows.append(dict(path=str(p.relative_to(root)),raw_bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(remote_root=str(root),files=rows),indent=2,sort_keys=True))
PY
REMOTE
python3 -B - "$case" <<'PY'
import gzip,hashlib,json,shutil,sys
from pathlib import Path
case=Path(sys.argv[1]);root=case/'native-perf-r1';manifest=json.loads((case/'retention-native-perf-r1.remote.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
rows=[]
for row in manifest['files']:
 p=root/row['path'];assert p.stat().st_size==row['raw_bytes'] and sha(p)==row['sha256'],p
 result=dict(row)
 if p.stat().st_size>1<<20:
  gz=Path(str(p)+'.gz')
  if not gz.exists():
   with p.open('rb') as source,gz.open('wb') as target,gzip.GzipFile(fileobj=target,mode='wb',filename='',mtime=0) as output:shutil.copyfileobj(source,output)
  h=hashlib.sha256()
  with gzip.open(gz,'rb') as f:
   for block in iter(lambda:f.read(4<<20),b''):h.update(block)
  assert h.hexdigest()==row['sha256']
  result.update(gzip_path=str(gz.relative_to(root)),gzip_bytes=gz.stat().st_size,gzip_sha256=sha(gz))
 rows.append(result)
record=dict(local_root=str(root),remote_root=manifest['remote_root'],all_local_sha256_equal=True,files=rows,
 raw_file_count=len(rows),raw_bytes=sum(r['raw_bytes'] for r in rows),gzip_sidecars=sum('gzip_path' in r for r in rows),
 raw_originals='retained unchanged locally and remotely; gzip sidecars are additional')
(case/'retention-native-perf-r1.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print('NATIVE_RETENTION_SHA256_PASSED',record['raw_file_count'],record['raw_bytes'],record['gzip_sidecars'])
PY
