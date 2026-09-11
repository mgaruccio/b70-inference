#!/usr/bin/env bash
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-reference-parity
H=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
IMAGE=vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4
TARGET=/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
name="dspark-rmsnorm-probe-$$"
check_host() {
  python3 -c 'import json,runpy,sys; from pathlib import Path; d=runpy.run_path(sys.argv[1]); h=d["host_invariants"](); Path(sys.argv[2]).write_text(json.dumps(h,indent=2)); assert h==d["expected_host_invariants"](), h' \
    "$R/_dependencies/run-acceptance-diagnostics.py" "$R/$1.json"
}
check_host before-rmsnorm-probe
cleanup() {
  docker rm -f "$name" > "$R/rmsnorm-probe-cleanup.txt" 2>&1 || true
  check_host after-rmsnorm-probe
}
trap cleanup EXIT
cd "$R"
timeout --signal=INT --kill-after=60s 900s docker run --rm --pull=never --name "$name" \
  --network=none --read-only --tmpfs /tmp --ipc=host \
  --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v /dev/dri:/dev/dri:ro -v "$R:/parity:ro" -v "$R:/output" \
  -v "$H/draft:/draft:ro" -v "$TARGET:/target:ro" \
  --entrypoint /opt/venv/bin/python "$IMAGE" -P /parity/probe-grouped-rmsnorm.py 2>&1 | tee rmsnorm-probe-driver.log
