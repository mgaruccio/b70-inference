#!/usr/bin/env bash
# Host-only preparation. Never stops or starts a serving container.
set -euo pipefail
ROOT=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
MODEL=/home/mike/inference/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
REV=e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246
REPO=nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
FILE=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
SRC=/home/mike/inference/src/llama.cpp-bolt-b6334e37
COMMIT=b6334e37e6ef286fc5091a41b6561b2113d915fb
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
mkdir -p "$ROOT"
case "${1:?download or build}" in
  download)
    mkdir -p "$MODEL"
    curl --fail --location --retry 3 --max-time 90 "https://huggingface.co/api/models/$REPO/revision/$REV?blobs=true" -o "$ROOT/model-metadata.json"
    curl --fail --location --retry 3 --max-time 90 "https://huggingface.co/$REPO/raw/$REV/README.md" -o "$ROOT/model-card.md"
    curl --fail --location --retry 5 --connect-timeout 30 --max-time 7200 --continue-at - "https://huggingface.co/$REPO/resolve/$REV/$FILE" -o "$MODEL/$FILE"
    python3 - "$ROOT/model-metadata.json" "$MODEL/$FILE" <<'PY'
import hashlib, json, pathlib, sys
meta = json.loads(pathlib.Path(sys.argv[1]).read_text())
p = pathlib.Path(sys.argv[2])
lfs = next(x['lfs'] for x in meta['siblings'] if x['rfilename'] == p.name)
assert p.stat().st_size == lfs['size'], (p.stat().st_size, lfs)
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
        h.update(block)
assert h.hexdigest() == lfs['sha256'], (h.hexdigest(), lfs)
result = dict(path=str(p), revision=meta['sha'], bytes=p.stat().st_size, sha256=h.hexdigest(), pass_=True)
pathlib.Path(sys.argv[1]).with_name('model-verified.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
PY
    ;;
  build)
    test ! -e "$SRC"
    curl --fail --location --retry 3 --max-time 300 "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$COMMIT" -o "$ROOT/llama-source.tar.gz"
    mkdir "$SRC"
    tar -xzf "$ROOT/llama-source.tar.gz" --strip-components=1 -C "$SRC"
    printf '%s\n' "$COMMIT" > "$SRC/UPSTREAM_COMMIT"
    docker image inspect "$IMAGE" --format '{{.Id}}' > "$ROOT/build-image.txt"
    docker run --rm --name bolt-sycl-build -v "$SRC:/src" -w /src "$IMAGE" bash -lc '
      set -eo pipefail
      source /opt/intel/oneapi/setvars.sh --force
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
      cmake -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_VULKAN=OFF \
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
        -DLLAMA_BUILD_TESTS=OFF
      cmake --build build-sycl --config Release -j6 --target llama-server llama-bench
      LD_LIBRARY_PATH=/src/build-sycl/bin:$LD_LIBRARY_PATH ./build-sycl/bin/llama-server --version
    '
    cp "$SRC/build-sycl/CMakeCache.txt" "$ROOT/CMakeCache.txt"
    ;;
  *) exit 2;;
esac
printf '\n--- final storage/serving state ---\n'
df -h /home/mike
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
sha256sum /home/mike/inference/launchers/start-qwen38.sh
