#!/usr/bin/env bash
set -euo pipefail
ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'BASH'
set -euo pipefail
ROOT=/tmp/lilicorr-actual-iq2-codegen-20261020
test -f "$ROOT/base/modules_300.bc"
sha256sum -c "$ROOT/objects-before.sha256"
docker run --rm --pull=never --network none --name bolt-lilicorr-actual-iq2-spirv-sg-r1 --cpus=1 -v "$ROOT:/inspect" -i intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0 bash -s <<'SDK'
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
python3 -B - <<'PY'
from pathlib import Path
import subprocess
root=Path('/inspect');tool='/opt/intel/oneapi/compiler/2026.0/bin/compiler/llvm-spirv'
for name in ('base','packed'):
 hits=[p for p in (root/name).glob('*.sym') if 'ggml_sycl_mul_mat_vec_q_id_iq2_batch4' in p.read_text()]
 assert len(hits)==1,(name,hits)
 p=hits[0];dest=root/name/'selected-sg.spv';assert not dest.exists()
 selected=root/name/'selected-kernel.txt'
 if selected.exists():assert selected.read_text()==p.read_text()
 else:selected.write_text(p.read_text())
 # Match the subgroup feature already required by the extracted real kernel.
 # This is an offline translation option, not an application/build flag change.
 subprocess.run([tool,'--spirv-ext=+SPV_INTEL_subgroups',str(p.with_suffix('.bc')),'-o',str(dest)],check=True)
 print('ACTUAL_N4_SPIRV_TRANSLATED',name,p.name,flush=True)
PY
SDK
for name in base packed; do
 mkdir "$ROOT/$name/native"
 /usr/bin/ocloc compile -file "$ROOT/$name/selected-sg.spv" -spirv_input -device 0xe223 -out_dir "$ROOT/$name/native" >"$ROOT/$name/ocloc-compile.log" 2>&1
done
python3 -B - "$ROOT" <<'PY'
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
