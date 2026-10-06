#!/usr/bin/env python3
"""Upload this run's tar in bounded R2 PUTs using only staged signed URLs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root=Path(sys.argv[1]); phase=sys.argv[2]
archive=root/(phase+'.tgz')
manifest=json.loads((root/'backup-private.json').read_text())
limit=2*1024**3
files=[]
if archive.stat().st_size <= 4*1024**3:
    files=[archive]
else:
    with archive.open('rb') as source:
        for index in range(32):
            first=source.read(8*1024**2)
            if not first:break
            part=root/(archive.name+'.part'+str(index).zfill(2))
            with part.open('wb') as target:
                target.write(first);remaining=limit-len(first)
                while remaining:
                    block=source.read(min(8*1024**2,remaining))
                    if not block:break
                    target.write(block);remaining-=len(block)
            files.append(part)
        assert not source.read(1),'Run archive exceeds staged64GiB bound; original retained'
with archive.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
parts=[]
for path in files:
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    parts.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha))
index=root/(archive.name+'.parts.json')
index.write_text(json.dumps(dict(archive=archive.name,bytes=archive.stat().st_size,sha256=digest,parts=parts),indent=2)+'\n')
for path in [*files,index]:
    selected=[next(item for item in manifest if item['path']==path.name)]
    (root/'phase-private.json').write_text(json.dumps(selected))
    subprocess.run(['python3',str(Path.home()/'qwen-mtp-code/r2-transfer.py'),'--run-dir',str(root),'--manifest',str(root/'phase-private.json'),'--upload'],check=True)
    print('ARCHIVE_OBJECT_UPLOADED='+path.name,flush=True)
print('ARCHIVE_COMPLETE='+phase+' SHA256='+digest,flush=True)
# Temporary parts are redundant copies; preserve the original archive and public index.
for path in files:
    if path != archive:path.unlink()
