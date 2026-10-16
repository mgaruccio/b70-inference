#!/usr/bin/env bash
# Host-only, CPU-only, one-shot Q6_K experiment after Q8 guard validation.
set -euo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
S=/home/mike/inference/src/llama.cpp-bolt-b6334e37-requant
BASE=/home/mike/inference/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
OUT=${BASE}-ssm-q6_k
FILE=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
sha256sum -c "$R/requant-q8_0-binaries.sha256"
test ! -e "$OUT"
python3 - "$BASE" <<'PY'
import os,sys
s=os.statvfs(sys.argv[1]); assert s.f_bavail*s.f_frsize >= 23148331008+2*1024**3
PY
mkdir "$OUT"
cpu() {
  docker run --rm --name bolt-requant-cpu --cpus=4 -v "$S:/src:ro" \
    -v "$BASE:/models:ro" -v "$OUT:/converted" -v "$R:/evidence" \
    "$IMAGE" /src/build-requant-cpu/bin/llama-quantize "$@"
}
common=(--allow-requantize --tensor-type-file /evidence/selective-types-q6_k.txt)
cpu --dry-run "${common[@]}" "/models/$FILE" /evidence/unused-q6-dry-run.gguf IQ2_XXS 8 > "$R/requant-q6_k-dry.log" 2>&1
python3 - "$R" <<'PY'
import json,pathlib,re,sys
r=pathlib.Path(sys.argv[1]); text=(r/'requant-q6_k-dry.log').read_text(errors='replace')
assert 'will require an imatrix' not in text and 'requires an importance matrix' not in text
expected={t['name'] for t in json.loads((r/'requant-original-structure.json').read_text())['tensors'] if t['type']=='NVFP4'}
changed=[]
for line in text.splitlines():
 m=re.search(r'\[\s*\d+/\s*\d+\]\s+(\S+)',line)
 if m and ' -> ' in line:
  assert 'nvfp4' in line.lower() and '(q6_k)' in line.lower(),line
  changed.append(m.group(1))
assert len(changed)==80 and set(changed)==expected,(len(changed),set(changed)^expected)
print('Q6_K dry-run: exact80 conversions, no imatrix requirement.')
PY
cpu "${common[@]}" "/models/$FILE" "/converted/$FILE" IQ2_XXS 8 > "$R/requant-q6_k-convert.log" 2>&1
docker run --rm --name bolt-requant-verify --cpus=4 \
  -v /home/mike/inference/src/llama.cpp-bolt-b6334e37:/src:ro \
  -v "$BASE:/models:ro" -v "$OUT:/converted:ro" -v "$R:$R" -e PYTHONPATH=/src/gguf-py \
  --entrypoint /opt/venv/bin/python3 \
  sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  -B "$R/check-requant.py" --target q6_k --converted "/converted/$FILE" > "$R/requant-q6_k-verify.log" 2>&1
python3 - "$R" "$OUT/$FILE" <<'PY'
import hashlib,json,pathlib,sys
r,output=map(pathlib.Path,sys.argv[1:]); v=json.loads((r/'requant-q6_k-verification.json').read_text())
assert v['structure_and_untouched_payloads_pass'] and v['changed_tensors']==80
with output.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
manifest={'pass_':True,'path':str(output),'bytes':output.stat().st_size,'sha256':digest,'original_sha256':'91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75','verification':'requant-q6_k-verification.json','note':'Artifact checks only; API quality and speed pending.'}
(r/'model-verified-ssm-q6_k.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
PY
