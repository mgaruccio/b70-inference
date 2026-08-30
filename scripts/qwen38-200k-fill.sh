#!/usr/bin/env bash
set -euo pipefail

# Disposable 200k filled-completion soak. Run on inference-host only.
# Does not touch start-qwen38.sh.
# Method and results: docs/qwen38-b70-200k-context-20260830.md

ROOT=/home/mike/inference
MODEL_DIR="$ROOT/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
COOKBOOK="$ROOT/src/intel-arc-pro-b70-inference-cookbook"
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/${STAMP}-200k-fill"
mkdir -p "$OUT"
LOG="$OUT/vllm.log"
export KV_XFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","secondary_tiers":[{"type":"fs","root_dir":"/kv-offload"}]}}'

echo on | sudo tee /sys/class/drm/card0/device/power/control >/dev/null

cleanup() {
  docker rm -f qwen38-ctx >/dev/null 2>&1 || true
  sudo rm -f /dev/shm/vllm_offload_*.mmap
}
trap cleanup EXIT
cleanup
df -h /dev/shm | tee "$OUT/shm-before.txt"

docker run -d --name qwen38-ctx --ipc=host \
  -p 127.0.0.1:18000:8000 \
  --device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -v /home/mike/b70-evals/kv-offload-ssd:/kv-offload \
  -e KV_XFER_CONFIG \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 200000 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 8192 --enable-prefix-caching --mamba-cache-mode align --kv-offloading-backend native --kv-offloading-size 8 --kv-transfer-config "$KV_XFER_CONFIG" --served-model-name qwen38 --language-model-only --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}" --enable-auto-tool-choice --tool-call-parser qwen3_xml' \
  >/dev/null

i=0
health=fail
while [ "$i" -lt 90 ]; do
  docker logs qwen38-ctx >"$LOG" 2>&1 || true
  if curl -fsS http://127.0.0.1:18000/health >/dev/null 2>&1; then
    health=ok
    break
  fi
  status="$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' qwen38-ctx 2>/dev/null || echo 'missing 0')"
  case "$status" in
    "running 0") ;;
    *) echo "container $status" | tee "$OUT/fail.txt"; exit 1 ;;
  esac
  i=$((i + 1))
  sleep 10
done
docker logs qwen38-ctx >"$LOG" 2>&1 || true
if [ "$health" != ok ]; then
  echo health_timeout | tee "$OUT/fail.txt"
  exit 1
fi

python3 - "$OUT" <<'PY'
import json, time, urllib.error, urllib.request, sys
from pathlib import Path

out = Path(sys.argv[1])
url = "http://127.0.0.1:18000/v1/chat/completions"
unit = "The quick brown fox jumps over the lazy dog. "

def chat(content, max_tokens=16, timeout=900):
    body = json.dumps({
        "model": "qwen38",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 12345,
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")
        (out / "http_error.json").write_text(json.dumps({
            "status": e.code, "body": err[:8000],
            "elapsed_s": time.perf_counter() - t0,
        }, indent=2))
        raise SystemExit(f"HTTP {e.code}: {err[:500]}")
    elapsed = time.perf_counter() - t0
    data = json.loads(raw)
    usage = data.get("usage") or {}
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    return {
        "http_status": status,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "content": text[:500],
    }

cal = chat(unit * 200, max_tokens=8, timeout=120)
cal_tokens = cal["prompt_tokens"]
# 200 copies plus chat template. Isolate per-copy cost with a second point.
cal2 = chat(unit * 400, max_tokens=8, timeout=120)
delta = cal2["prompt_tokens"] - cal["prompt_tokens"]
per = delta / 200.0
overhead = cal["prompt_tokens"] - 200 * per
target_prompt = 196000
n = max(1, int((target_prompt - overhead) / per))
receipt = {
    "cal_200": cal,
    "cal_400": cal2,
    "tokens_per_unit": per,
    "template_overhead": overhead,
    "target_prompt": target_prompt,
    "n_units": n,
}
(out / "calibrate.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2), flush=True)

fill = chat(unit * n, max_tokens=32, timeout=900)
(out / "fill.json").write_text(json.dumps(fill, indent=2))
print(json.dumps(fill, indent=2), flush=True)
if fill["http_status"] != 200 or not fill["prompt_tokens"] or fill["prompt_tokens"] < 180000:
    raise SystemExit(f"fill too small or failed: {fill}")
if fill["finish_reason"] not in ("stop", "length") or fill["completion_tokens"] in (None, 0):
    raise SystemExit(f"no completion: {fill}")
print("FILL_OK", fill["prompt_tokens"], fill["completion_tokens"], fill["elapsed_s"])
PY

docker logs qwen38-ctx >"$LOG" 2>&1 || true
rg -n "GPU KV cache size|Available KV cache memory|preempt|OutOfMemory|ERROR" "$LOG" | tail -40 | tee "$OUT/vllm-kv.txt"
echo DONE | tee "$OUT/done.txt"
