# Observed native/reference result — development investigation

No performance promotion or completed optimization claim. All runtime work used the pinned image on inference-host's Intel Arc Pro B70; model/tensor files remain host-local.

## Executed journey and retained failure

- `bash run-preflight-capture.sh`: official source import and CPU model checks passed; real API capture-01 completed with exact capture-on/off output IDs.
- `bash run-reference-command.sh`: the official GPU forward ran, but comparison failed with `KeyError: final_hidden`. The module post-hook missed vLLM's explicit `.forward()` call. Failure/log and originally executed source retained under `reference-01/` and `capture-01/`.
- Changed the capture to wrap the actual backbone `forward` method without replacing its output, and require critical observations before capture success. Added a direct-call regression. `lead-tests-after-capture-fix.txt`: 15 tests passed.
- `bash run-retry-02.sh`: capture-02 and reference-02 completed. Capture-on/off matched the same prompt and 16 output token IDs. All 62 draft weight identities were verified, context positions were 0–67, query positions 68–74, and native attention exposed exactly 75 positions with noncausal attention and BF16 KV. The official model ran twice independently; its observed stages and proposals were bitwise repeatable.

Exact launch commands, source hashes, raw API outputs, comparison metrics, logs and host-invariant snapshots are retained in this directory. Before/after snapshots confirm the production launcher hash, 275 W power cap and stopped Glimmer state were unchanged. Containers were removed.

## Native/official numerical mismatch

`reference-02/comparison.json` reports measurements, not a threshold-certified parity pass:

- FC output is bitwise identical.
- Context normalization is close (RMSE 0.002934); layer-0 context K normalization RMSE 0.004074.
- Raw K projections remain very close in every layer, but layer-1 context K normalization RMSE rises to 0.55872. Layers 2–4 have similarly large errors. Layer-1 attention cosine falls to 0.321716.
- Final hidden cosine is 0.506508, RMSE 1.84517.
- All seven proposals differ: native `[5141,25,271,550,2500,310,220]`, official `[30057,286,264,436,34810,18887,303]`.

Small initial BF16 differences are not the primary finding: the abrupt per-layer context-K error is.

## Confirmed XPU grouped RMSNorm weight-selection bug

`bash run-rmsnorm-probe.sh` ran the actual pinned native `vllm._custom_ops.rms_norm` on B70, first using the captured `[5,68,8,128]` BF16 K tensor and exact checkpoint K-norm weights, then synthetic BF16 and FP16 data.

`grouped-rmsnorm-probe.json` proves:

- Grouped native output equals the captured context K normalization **bitwise**.
- Grouped native output equals five separate calls deliberately using **weight row 0 for every layer**, bitwise.
- Correct separate calls with each layer's own weights differ: max absolute error 5.09375, RMSE 0.510383.
- Synthetic ones with weight rows `[1,2,3,4,5]` produce grouped per-layer means `[1,1,1,1,1]`; correct separate calls produce `[1,2,3,4,5]`, for both BF16 and FP16.

Primary caller source confirms the stacked weights are intended to be per-layer:
https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/model_executor/models/qwen3_dflash.py

The next required validation is a DSpark-opt-in per-layer native-call correction, repeat official comparison, then real API acceptance/performance measurements. This evidence establishes a correctness bug; it does not yet quantify the speedup or exclude additional differences, including target quantization.
