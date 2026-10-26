#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/mike/b70-evals/20261026-eagle4-profile
CANDIDATE="$ROOT/draft-graphs"
OLD=/home/mike/b70-evals/20261025-intel15-eagle4
IMAGE=sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
[[ -z $(docker ps -q) ]]
mkdir "$CANDIDATE/native"
trap 'docker stop -t 15 intel15-draft-graph-parity >/dev/null 2>&1 || true' EXIT
docker run --rm --name intel15-draft-graph-parity --network none \
 --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/renderD128)" --shm-size 8g --ulimit core=0 \
 -e ZE_AFFINITY_MASK=0 -e TRITON_INTERPRET=0 -e PYTHONDONTWRITEBYTECODE=1 \
 -e EAGLE4_RUN_XPU_DRAFT_SPLIT_BENCH=1 \
 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -e PYTHONPATH=/test-deps:/candidate \
 -v "$CANDIDATE/eagle-source:/candidate:ro" -v "$OLD/pytest-deps:/test-deps:ro" \
 -v "$CANDIDATE/engine-6cae86c.patch:/tmp/engine.patch:ro" -v "$CANDIDATE/native:/out" \
 --entrypoint /bin/bash "$IMAGE" -lc '
set -Eeo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force >/dev/null
patch --fuzz=0 -p1 --directory=/opt/venv/lib/python3.12/site-packages < /tmp/engine.patch
cd /candidate
/opt/venv/bin/python -c "import torch,triton; print(torch.__version__, triton.__version__); assert torch.xpu.is_available(); print(torch.xpu.get_device_name(0))"
/opt/venv/bin/python -X faulthandler -m pytest -q -s -rs -p no:cacheprovider --junitxml=/out/native.xml \
 tests/test_xpu_vllm_tree.py::test_xpu_adaptive_step_graph_replay_real_device \
 tests/test_native_intel_fp8_flash_attention.py \
 tests/test_native_intel_fp8_tree.py::test_native_b70_fp8_tree \
 tests/test_native_intel_fp8_tree.py::test_native_b70_fp8_tree_long_context \
 tests/test_xpu_draft_attention_splits.py::test_native_selected_launch_graph_replays_live_inputs \
 tests/test_fp8_target_cache.py tests/test_verify_attention.py
/opt/venv/bin/python -c "import xml.etree.ElementTree as E; r=E.parse(\"/out/native.xml\").getroot(); assert sum(int(s.get(\"skipped\",0)) for s in r.iter(\"testsuite\")) == 0"
'
