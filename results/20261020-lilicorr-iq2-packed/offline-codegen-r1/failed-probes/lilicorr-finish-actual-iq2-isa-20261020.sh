#!/usr/bin/env bash
set -euo pipefail
ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'BASH'
set -euo pipefail
ROOT=/tmp/lilicorr-actual-iq2-codegen-20261020
OUTPUT=$ROOT/native-comparison
# Extracted subdirectories belong to the disposable SDK container's root.
# Keep fresh host-owned outputs separate; do not change their permissions.
test ! -e "$OUTPUT"
mkdir "$OUTPUT"
sha256sum -c "$ROOT/objects-before.sha256"
for name in base packed; do
 mkdir -p "$OUTPUT/$name/native"
 /usr/bin/ocloc compile -file "$ROOT/$name/selected-sg.spv" -spirv_input -device 0xe223 -out_dir "$OUTPUT/$name/native" >"$OUTPUT/$name/ocloc-compile.log" 2>&1
done
python3 -B - "$OUTPUT" <<'PY'
from pathlib import Path
import subprocess,sys,re,json
root=Path(sys.argv[1]);report={}
for name in ('base','packed'):
 d=root/name;print('OFFLINE_COMPILER',name,(d/'ocloc-compile.log').read_text()[-6000:])
 images=list((d/'native').glob('*.bin'));assert len(images)==1,images
 p=subprocess.run(['/usr/bin/ocloc','disasm','-file',str(images[0]),'-device','0xe223','-dump',str(d/'disasm')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (d/'ocloc-disasm.log').write_text(p.stdout);assert p.returncode==0,(name,p.stdout)
 assemblies=[p for p in (d/'disasm').rglob('*.asm')];assert len(assemblies)==1,assemblies
 text=assemblies[0].read_text();counts={op:sum(bool(re.search(r'\b'+op+r'\b',line,re.I)) for line in text.splitlines()) for op in ('dp4a','dpas','mul','mad','send','load','store')}
 metadata=[]
 for q in (d/'disasm').rglob('*'):
  if q.is_file() and ('ze_info' in q.name or q.suffix in ('.yaml','.yml')):
   for line in q.read_text(errors='replace').splitlines():
    if any(k in line for k in ('grf_count','spill_mem_size','private_size','simd_size','slm_size')):metadata.append(line.strip())
 report[name]=dict(instruction_line_counts=counts,metadata=metadata,assembly_bytes=len(text))
 print('ACTUAL_KERNEL_OFFLINE_ISA',name,json.dumps(report[name]),flush=True)
(root/'offline-isa-summary.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print('ACTUAL_OBJECT_OFFLINE_ANALYSIS_ONLY_NOT_RUNTIME_TIMING')
PY
sha256sum -c "$ROOT/objects-before.sha256"
BASH
