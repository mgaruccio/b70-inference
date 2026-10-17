# Continuing Bolt prefill optimization

## Declared experiment and test process

Development-only. Baseline is the validated staged-SSM winner, source archive `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`, upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`. At session start the resident `/src` mount and server/backend binary hashes matched the accepted artifacts; health was OK. Scoped evidence directories were clean at commit `2b34e6e0`; unrelated deletions are not part of this work.

Before selecting another kernel change:

1. Run `prefill-ssm-r1-baseline.sh` on `mike@inference-host` with Bash. The frozen gate-v2 API suite must pass20/20 before timing. Keep55/8270 prompt tokens,128 forced outputs, warm-up plus three measured repetitions, `cache_n=0`, and exact8238-token `BOLT-7319-CEDAR` retrieval.
2. Run `run-prefill-profile-ssm-r1.py`: reuse the current winner's exact create argv with existing isolated unitrace. Pause collection during load and warm-up; collect one8270-token/output1 uncached request. The output budget is deliberately different for attribution, not a performance comparison.
3. Preserve raw requests/results, container configuration, hardware before/after, unitrace report and Chrome trace. Losslessly compress logs with `gzip -n`; analyze with `analyze-prefill-profile-ssm-r1.py`.
4. Restore the original Bolt container in `finally`, require public API `READY`, verify hardware equality. Never restore another model or modify persistent launchers.
5. Select one bounded measured hotspot. Any candidate requires native numerical tests without relaxed thresholds, debug actual-model activation, then a fresh non-debug20/20 API gate **before** matched timing/retrieval. Preserve failures; no identical retry-until-pass.

All builds, native tests and GPU/API workloads run on inference-host. Keep original weights, q8 KV, F32 recurrent state, context65536, batch/microbatch1024, single sequence, ESIMD1/FUSION1/graphs0,275W, boost0 and CPU caps unchanged. No hardware-metrics permission, driver or network changes. Public test boundary is `http://127.0.0.1:18098/v1/chat/completions` on that host.

## Fresh primary-source research

- [Intel PTI unitrace documentation](https://github.com/intel/pti-gpu/blob/pti-1.0.0/tools/unitrace/README.md), freshly retrieved: named-session `--start-paused`, `--resume`, `--stop` provide external collection control without application changes. Device and submission timings are diagnostic; instrumentation is not compared with uninstrumented speed results.
- [Pinned NVFP4 dequant kernel](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/dequantize.hpp) and [dispatch](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/convert.cpp): live archive inspection confirms one32-workitem group per64 values. Four16-value sub-blocks share four scale bytes. Scale reuse or workgroup packing are hypotheses, not measured improvements. Preserve nibble mapping, scale semantics and destination casts; native coverage must include all256 scale bytes.
- [SYCL NVFP4 support PR21227](https://github.com/ggml-org/llama.cpp/pull/21227) and [MoE prefill PR23142](https://github.com/ggml-org/llama.cpp/pull/23142): baseline already contains counting-sort/grouped GEMM. Do not blindly backport newer upstream code or transfer another device/model's reported gain.

The Lab preview/patch UI is unavailable (`ENOENT`); ordinary scoped file edits and retained commands are used. No repair of that unrelated UI is attempted.


## Current-winner results and selected next candidate

`prefill-ssm-r1-baseline`:20/20 API checks pass, exact8238-token retrieval passes. Short server prefill repetitions469.352,468.345,467.905ms; long8198.543,8205.200,8220.171ms. Median client TTFT0.531356s short and8.291637s long. No samples excluded.

`prefill-ssm-r1-unitrace`: profile/warm-up counts8270/1/cache_n0 pass; original container restoration returns `READY`; hardware-before/after match.53,772 device events, summed8.195399s, union7.996289s, span8.945703s, gaps0.949415s. Grouped IQ2_XXS dequant/GEMM808calls1902.307ms23.21%; generic GEMM5276calls1205.043ms14.70%; staged SSM440calls1073.296ms13.10%; **NVFP4→F32 720calls785.961ms9.59%**; IQ2_XXS→F16 3188calls618.192ms7.54%. These are instrumented attribution, not optimized/baseline speed comparisons or bandwidth-saturation evidence.

Selected candidate: pack eight consecutive NVFP4 blocks into each256-workitem group instead of one64-value block per32-workitem group. Keep original scale conversion, nibble mapping, multiply and F32 output; no scale broadcast, arithmetic or precision change bundled. Restrict to fusion-enabled non-reordered F32 conversion with supported device/group size; preserve original fallback. Test tails, all256 scale bytes/all16 nibbles, bitwise equality against the original same-device kernel, and canaries before real-model activation. The larger grouped-GEMM hotspot is deferred because this conversion has a simpler exactness contract. No improvement is claimed yet.
