#!/usr/bin/env bash
set -euo pipefail
umask 077
RUN="$HOME/qwen-mtp-run"
CODE="$HOME/qwen-mtp-code"
UPLOAD="$HOME/qwen-mtp-upload"
IMAGE='vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967'
mkdir -p "$RUN" "$UPLOAD"
archive() {
  local phase="$1"
  tar --exclude='*private*.json' --exclude='__pycache__' -cf "$UPLOAD/$phase.tgz" -C "$RUN" .
  python3 - "$UPLOAD" "$phase" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);name=sys.argv[2]+'.tgz'
items=json.loads((root/'backup-private.json').read_text());item=next(x for x in items if x['path']==name)
(root/'phase-private.json').write_text(json.dumps([item]))
PY
  python3 "$CODE/r2-transfer.py" --run-dir "$UPLOAD" --manifest "$UPLOAD/phase-private.json" --upload
}
finish() {
  local code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$RUN/bootstrap-exit.txt"
  date -u > "$RUN/bootstrap-finished-at.txt"
  cp "$HOME/qwen-bootstrap.log" "$RUN/bootstrap.log" || true
  archive bootstrap || code=1
  exit "$code"
}
trap finish EXIT
printf 'BOOTSTRAP_START\n'
date -u > "$RUN/bootstrap-started-at.txt"
nvidia-smi -q > "$RUN/nvidia-smi.txt"
lscpu > "$RUN/lscpu.txt"
free -h > "$RUN/memory.txt"
df -h > "$RUN/disk.txt"
python3 - "$UPLOAD" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);items=json.loads((root/'backup-private.json').read_text())
(root/'preflight.json').write_text(json.dumps({'phase':'remote_upload_preflight','pilot':'qwen38-cuda-mtp-expanded'}))
(root/'phase-private.json').write_text(json.dumps([next(x for x in items if x['path']=='preflight.json')]))
PY
python3 "$CODE/r2-transfer.py" --run-dir "$UPLOAD" --manifest "$UPLOAD/phase-private.json" --upload
printf 'REMOTE_R2_UPLOAD_PREFLIGHT_PASSED\n'
command -v docker >/dev/null
sudo -n docker info > "$RUN/docker-info.txt"
sudo -n docker pull "$IMAGE" > "$RUN/image-pull.log" 2>&1
sudo -n docker image inspect "$IMAGE" > "$RUN/image.json"
printf 'PINNED_CUDA_IMAGE_READY\n'
sudo -n apt-get install -y --no-install-recommends cuda-compat-13-0 > "$RUN/cuda-compat-install.log" 2>&1
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
dpkg-query -W cuda-compat-13-0 > "$RUN/cuda-compat-version.txt"
if ! python3 -m venv "$HOME/qwen-mtp-env"; then
  sudo -n apt-get update > "$RUN/apt.log" 2>&1
  sudo -n apt-get install -y python3-venv >> "$RUN/apt.log" 2>&1
  python3 -m venv "$HOME/qwen-mtp-env"
fi
PY="$HOME/qwen-mtp-env/bin/python"
"$PY" -m pip install --upgrade pip > "$RUN/pip.log" 2>&1
"$PY" -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu130 >> "$RUN/pip.log" 2>&1
"$PY" -m pip install 'transformers==5.15.1' 'accelerate==1.15.0' 'safetensors' 'pytest' 'huggingface_hub' >> "$RUN/pip.log" 2>&1
"$PY" -m pip freeze > "$RUN/pip-freeze.txt"
"$PY" - <<'PY'
import json,torch,transformers
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
p=torch.cuda.get_device_properties(0)
assert p.total_memory >= 75*1024**3
x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)
y=x@x;torch.cuda.synchronize();assert torch.isfinite(y).all().item()
report={'torch':torch.__version__,'transformers':transformers.__version__,'device':str(p),'bf16':torch.cuda.is_bf16_supported()}
(Path.home()/'qwen-mtp-run/cuda-readiness.json').write_text(json.dumps(report,indent=2))
print('CUDA_BF16_READY',json.dumps(report),flush=True)
PY
"$PY" -m pytest -q "$CODE/tests/test_qwen38_train_mtp.py" > "$RUN/trainer-tests.log" 2>&1
sudo -n docker run --rm --gpus all --network none -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 -v "$CODE/patches:/patches:ro" --entrypoint bash "$IMAGE" -c 'set -e; python3 /patches/patch_mtp_training.py; python3 /patches/patch_mtp_native_capture.py; python3 -c "import vllm,torch; assert torch.cuda.is_available(); x=torch.randn(64,64,device=\"cuda\",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all().item(); print(vllm.__version__); print(torch.__version__)"' > "$RUN/cuda-source-patch-preflight.log" 2>&1
printf 'CUDA_SOURCE_PATCH_PREFLIGHT_PASSED\n'
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import json
repo='Qwen/Qwen3.8-27B';revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
print('BF16_MODEL_DOWNLOAD_START',repo,revision,flush=True)
path=snapshot_download(repo,revision=revision,local_dir=Path.home()/'qwen-model',allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer*','*.txt'],max_workers=4)
config=json.loads((Path(path)/'config.json').read_text())
assert not config.get('quantization_config') and config['text_config']['mtp_num_hidden_layers']==1
(Path.home()/'qwen-mtp-run/model.json').write_text(json.dumps({'repository':repo,'revision':revision,'path':path,'dtype':config.get('dtype',config.get('torch_dtype')),'mtp_layers':1},indent=2))
print('BF16_MODEL_READY',flush=True)
PY
"$PY" "$CODE/restore-cache.py"
printf 'BOOTSTRAP_COMPLETE_FULL_CACHED_CORPUS_NO_OPTIMIZER_UPDATES\n'
