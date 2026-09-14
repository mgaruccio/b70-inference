# Matched four-way serving speed comparison

Status: **completed successfully**, all eight cells exited0 (elapsed1h54m). Development tier, best fixed configurations, no production promotion. User requested comparable speed numbers for native MTP4, custom grouped MTP4, DFlash2 INT4 and DSpark Split-K; mixing historical runs is explicitly excluded.

## Frozen design / public-boundary process

- Remote `inference-host`, one B70 at275W, C1, same target FP16 GPTQ/FP8 KV, greedy seed42, thinking off, prefix cache off.
- Exact same rendered 512/8192/32768/65536-token prompts and128 forced output tokens for every arm and replicate. Use the unchanged existing HTTP long-context client through `/tokenize` and streaming `/v1/completions`.
- Eight fresh-container cells in order: MTP4, custom, DFlash, DSpark, DSpark, DFlash, custom, MTP4. This forward/reverse order balances linear temporal drift. Each cell runs one warmup+six measured requests per context:12 measured samples/arm/context,192 measured requests total,32 warmups. No historical samples or removed outliers.
- Before/after each cell: require idle GPU/no running containers, unchanged production launcher SHA `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`,275000000-microwatt power cap, existing lifecycle cleanup. Original basic public API gates remain prerequisites; this is not a repeat of the full quality evaluation.
- Check observed DSpark Split-K dispatch and custom build06 hash plus actual eligible/FULL-graph execution. A native fallback does not qualify as a custom result.
- Retain exact launch argv, image/environment, source hashes, HTTP requests/SSE/counters, failures, summaries and console/exits. Analyze all eight cells only after all pass; assert identical measured request payloads across all cells, actual lengths and complete streams.
- Report medians and inclusive IQRs for decode proxy, TTFT and whole-request latency. The decode metric is the unchanged client's post-first streaming burst rate `(128-1)/(stream_end-first_nonempty)`, not device-only throughput or single-token ITL.

## Bundle differences (intentional, not normalized algorithms)

| Arm | Configured capacity | Prefill batch | Runtime | Speculation |
|---|---:|---:|---|---|
|Native MTP4|212992|8192|ac7509e2b / f01e24f6 image|K4 INT4 S+M1 draft|
|Custom MTP4|212992|8192|same MTP4 image and serve args|same K4 plus qualified grouped build06|
|DFlash2 INT4|180224|2048|73029d424 / 7a558f63 image|K7, cache group8, partial INT4 draft|
|DSpark Split-K|65664|2048|73029d424 / 7a558f63 image|K7 corrected BF16 draft/KV, native noncausal Split-K|

Native/DSpark/DFlash reference launches are byte copies of the selected best-three campaign references. Custom is the exact tested quality-campaign candidate launch. Only names/output directories/device render group are redirected; historical input directories stay read-only. All four configurations are freshly measured in THIS campaign. Differences in runtime/capacity/batch remain disclosed: conclusions concern the best fixed bundles, not isolated algorithms under one normalized runtime. No capacity or precision is silently lowered to boost a score. No statistical robustness beyond these two cells per arm is implied.

## Research and execution

Fresh official source: https://docs.vllm.ai/en/latest/features/per_request_metrics/ — TTFT, generation interval and prefill-inclusive throughput have different timing boundaries. Preserve the same existing streaming client metric across all arms, do not mix it with GPU or newer server counters. No new metrics flag is added to the older pinned runtimes. Two attempted benchmark-CLI URLs returned404; the per-request source resolved successfully.

```bash
ssh inference-host bash /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-qwen38-four-way-speed/run-all.sh
python3 results/20260914-qwen38-four-way-speed/analyze.py > results/20260914-qwen38-four-way-speed/comparison.json
```

All inference remains outside Pi. Native Lab preview is unavailable (`ENOENT lab.sock`); previously authorized CLI fallback applies. Production is unchanged. Full quality findings remain in `../20260913-qwen38-grouped-quality/`; this request is speed-only.


## Matched results

All192 measured requests and32 warmups completed; measured payloads match across all eight cells. All eight API gates and host-unchanged checks passed; both custom cells passed eligible-dispatch/FULL-graph evidence. Zero excluded samples. Local rerun of `analyze.py` reproduced `comparison.json` exactly.

Median decode tok/s (inclusive IQR), twelve measured samples per configuration/context pair pooled over two independent server loads:

| Context | Native MTP4 | Custom MTP4 | DFlash2 INT4 | DSpark Split-K |
|---:|---:|---:|---:|---:|
|512|70.77 (1.95)|69.48 (2.31)|76.29 (5.79)|52.55 (5.86)|
|8192|63.97 (3.07)|63.45 (1.71)|60.76 (9.50)|42.95 (7.43)|
|32768|61.36 (2.85)|63.27 (2.71)|50.28 (1.19)|37.47 (2.86)|
|65536|55.12 (4.35)|57.93 (2.65)|45.13 (4.95)|31.14 (1.27)|

Custom/native changes: −1.82% at512, −0.82% at8K, +3.11% at32K, **+5.11% at64K**. This replaces the mixed-campaign comparison: it does not reuse the previous +8.85% observation. Two server loads per arm and a narrow prompt set do not establish cross-day statistical superiority, particularly for small short-context differences.

Median TTFT (seconds):

| Context | Native MTP4 | Custom MTP4 | DFlash2 INT4 | DSpark Split-K |
|---:|---:|---:|---:|---:|
|512|0.290|0.291|0.283|0.324|
|8192|4.204|4.206|4.359|4.950|
|32768|21.138|21.135|21.192|23.503|
|65536|53.907|53.903|52.518|57.126|

Median whole-request latency for128 output tokens (seconds, lower is better):

| Context | Native MTP4 | Custom MTP4 | DFlash2 INT4 | DSpark Split-K |
|---:|---:|---:|---:|---:|
|512|2.083|2.117|1.948|2.740|
|8192|6.188|6.206|6.450|7.910|
|32768|23.202|23.143|23.720|26.888|
|65536|56.207|56.097|55.327|61.205|

Thus custom MTP4 has the highest observed32K/64K decode throughput, but DFlash has the lowest64K whole-request latency because of faster prefill. Keep that timing-boundary distinction when interpreting “fastest.” Full per-cell medians, all samples, IQRs, launch metadata and native-relative deltas are retained in `comparison.json`; raw requests/SSE/metrics and cleanup are under each `ARM-0{1,2}` directory. No production or accuracy claim is made by this speed-only campaign.
