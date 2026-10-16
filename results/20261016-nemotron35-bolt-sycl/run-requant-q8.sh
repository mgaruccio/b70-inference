#!/usr/bin/env bash
# Host-only one-shot experiment. Do not edit/restart while running.
set -euo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
S=/home/mike/inference/src/llama.cpp-bolt-b6334e37-requant
BASE=/home/mike/inference/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
OUT=${BASE}-ssm-q8_0
FILE=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
test ! -e "$OUT"
# Full output plus 2 GiB reserve, before creating or writing model data.
python3 - "$BASE" <<'PY'
import os,sys
s=os.statvfs(sys.argv[1]); assert s.f_bavail*s.f_frsize >= 24209850368+2*1024**3
PY
mkdir "$OUT"
cpu() {
  docker run --rm --name bolt-requant-cpu --cpus=4 -v "$S:/src:ro" \
    -v "$BASE:/models:ro" -v "$OUT:/converted" -v "$R:/evidence" \
    "$IMAGE" /src/build-requant-cpu/bin/llama-quantize "$@"
}
common=(--allow-requantize --tensor-type-file /evidence/selective-types-q8_0.txt)
cpu --dry-run "${common[@]}" "/models/$FILE" /evidence/unused-dry-run.gguf IQ2_XXS 8 > "$R/requant-q8_0-stock-dry.log" 2>&1
python3 - "$R" stock <<'PY'
import json,pathlib,re,sys
r=pathlib.Path(sys.argv[1]); text=(r/'requant-q8_0-stock-dry.log').read_text()
assert 'requires an importance matrix' in text or 'requires an imatrix' in text
expected={t['name'] for t in json.loads((r/'requant-original-structure.json').read_text())['tensors'] if t['type']=='NVFP4'}
changed=[]
for line in text.splitlines():
 m=re.search(r'\[\s*\d+/\s*\d+\]\s+(\S+)',line)
 if m and ' -> ' in line:
  assert 'nvfp4' in line.lower() and '(q8_0)' in line.lower(),line
  changed.append(m.group(1))
assert len(changed)==80 and set(changed)==expected,(len(changed),set(changed)^expected)
print('Stock dry-run: exact80 conversions; spurious copied-IQ2 imatrix warning confirmed.')
PY
(cd "$S" && git apply --check "$R/selective-requant-imatrix.patch" && git apply "$R/selective-requant-imatrix.patch")
docker run --rm --name bolt-requant-guard-build -v "$S:/src" -w /src "$IMAGE" bash -lc '
  set -eo pipefail
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
  cmake --build /src/build-requant-cpu --config Release -j2 --target llama-quantize
' > "$R/requant-guard-build.log" 2>&1
cpu --dry-run "${common[@]}" "/models/$FILE" /evidence/unused-dry-run.gguf IQ2_XXS 8 > "$R/requant-q8_0-patched-dry.log" 2>&1
python3 - "$R" <<'PY'
import json,pathlib,re,sys
r=pathlib.Path(sys.argv[1]); text=(r/'requant-q8_0-patched-dry.log').read_text()
assert 'requires an importance matrix' not in text and 'requires an imatrix' not in text
expected={t['name'] for t in json.loads((r/'requant-original-structure.json').read_text())['tensors'] if t['type']=='NVFP4'}
changed=[]
for line in text.splitlines():
 m=re.search(r'\[\s*\d+/\s*\d+\]\s+(\S+)',line)
 if m and ' -> ' in line:
  assert 'nvfp4' in line.lower() and '(q8_0)' in line.lower(),line
  changed.append(m.group(1))
assert len(changed)==80 and set(changed)==expected,(len(changed),set(changed)^expected)
print('Patched dry-run: exact80 conversions; no imatrix requirement.')
PY
set +e
cpu --tensor-type '^blk\.0\.ssm_in\.weight$=iq2_xxs' "${common[@]}" "/models/$FILE" \
  /converted/negative-MUST-NOT-EXIST.gguf IQ2_XXS 8 > "$R/requant-imatrix-negative.log" 2>&1
rc=$?
set -e
test "$rc" -ne 0
grep -F 'this quantization requires an importance matrix' "$R/requant-imatrix-negative.log"
test ! -e "$OUT/negative-MUST-NOT-EXIST.gguf"
echo 'Negative conversion: new IQ2 still refused without imatrix; no output created.'
cpu "${common[@]}" "/models/$FILE" "/converted/$FILE" IQ2_XXS 8 > "$R/requant-q8_0-convert.log" 2>&1
# Reuse installed NumPy from the existing production image; no GPU device access.
docker run --rm --name bolt-requant-verify --cpus=4 \
  -v /home/mike/inference/src/llama.cpp-bolt-b6334e37:/src:ro \
  -v "$BASE:/models:ro" -v "$OUT:/converted:ro" -v "$R:$R" -e PYTHONPATH=/src/gguf-py \
  --entrypoint /opt/venv/bin/python3 \
  sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  -B "$R/check-requant.py" --target q8_0 --converted "/converted/$FILE" > "$R/requant-q8_0-verify.log" 2>&1
python3 - "$R" "$BASE/$FILE" "$OUT/$FILE" <<'PY'
import hashlib,json,pathlib,sys
r,original,output=map(pathlib.Path,sys.argv[1:])
v=json.loads((r/'requant-q8_0-verification.json').read_text()); assert v['structure_and_untouched_payloads_pass'] and v['changed_tensors']==80
with original.open('rb') as f: before=hashlib.file_digest(f,'sha256').hexdigest()
assert before=='91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75'
with output.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
manifest={'pass_':True,'path':str(output),'bytes':output.stat().st_size,'sha256':digest,'original_sha256':before,'verification':'requant-q8_0-verification.json','note':'Artifact structure/payload verification only; model API quality and speed remain pending.'}
(r/'model-verified-ssm-q8_0.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
PY
sha256sum "$S/build-requant-cpu/bin/llama-quantize" "$S/build-requant-cpu/bin/libllama.so.0" "$S/src/llama-quant.cpp" > "$R/requant-q8_0-binaries.sha256"
echo 'Selective Q8_0 artifact verified; no inference quality/speed claim yet.'
