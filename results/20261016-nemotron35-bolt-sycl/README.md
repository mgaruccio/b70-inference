# Nemotron 3.5 Bolt on Arc Pro B70 — development probe

User request: “another gear shift time. lets get nemotron 3.5 bolt running on the b70 and see how fast we can get it”.

## Scope and authorization

Development tier only, not a standard publishable comparison. Establish Bolt compatibility and a baseline, then test only supported optimizations with paired identical workloads. Qwen is not a same-model baseline.

User approved deletion of **only** `/home/mike/inference/models/Muse-Glimmer-30B-GGUF`, and a temporary Qwen interruption for Bolt compatibility and initial tuning, restoring Qwen afterward. Preserve the Qwen launcher and existing runtime. Do not change drivers/network/power settings. No multimodal or speculative decoding initially.

Subsequent user instructions explicitly requested continued Bolt optimization/integer-format investigation, authorized the named unused benchmark-image and obsolete KV-backing-image cleanup recorded in `execution.md`, and said “yes, do whatever, stop asking”. Continue bounded temporary test windows under that existing authorization, preserve original Qwen/Bolt models, and restore unchanged Qwen after every GPU cell. The unavailable Lab preview socket is not an approval channel; no approval is claimed from its ENOENT failures.

## Latest verified development result

Original NVFP4 weights with the native reorder prototype reproducibly deliver **39.45–39.47 tok/s short decode** and **38.13–38.16 tok/s after 8K**, versus the same 1024/1024 unmodified-backend baseline **28.55 / 27.89 tok/s**. Each of two independent candidate launches passes 20/20 API checks and exact 8K retrieval; the first also passes 56 native numerical cases with optimization on and 56 with it off. Cold prefill remains about 955–956 tok/s. These are small development samples, not full-context or comprehensive quality validation. Q8_0 and Q6_K conversions passed artifact checks but each passed only 19/20 model checks, so timing was gated off and neither is accepted. Full raw results and limitations: [`execution.md`](execution.md).

## Research and chosen path

Fresh primary-source research on 2026-10-16:

- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF — official 74B total/9B active hybrid MoE, 256K native context, mixed IQ2/NVFP4/Q8 GGUF. Model revision `e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246`. Download text model only (22,018,326,528 bytes), not projector/drafter.
- https://docs.vllm.ai/en/stable/features/quantization/b12x/ — official native quant recipe uses NVIDIA-specific CUDA kernels; do not transplant it to XPU.
- https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/ — B-Series hardware support does not validate this model.
- https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md — SYCL supports Intel GPUs and IQ2; build separately with existing oneAPI development image. Pin inspected upstream revision `b6334e37e6ef286fc5091a41b6561b2113d915fb`. The current docs recommend oneAPI 2026.1; existing image is 2026.0, so build compatibility must be observed, not assumed.
- https://github.com/ggml-org/llama.cpp/issues/29183 — open B70 long-prompt corruption report includes a different Nemotron quant. Its DeltaNet root-cause hypothesis does not establish Bolt's Mamba-2 mechanism. Treat as a correctness warning, not proof that this model fails.

## Preconditions and controlled baseline

One B70 (32 GB), Ryzen 5800XT, 32 GB host RAM. Root filesystem initially 15 GB free. Existing Qwen container/image and launcher SHA256 `5701b40e355e99b65c776dd6c9e499980ccc3be758f17288c30f840f86e99a1d` unchanged. Qwen baseline stack: vLLM `0.27.2rc1.dev77+gac7509e2b.xpu`, Torch `2.13.0+xpu`, transformers `5.15.0`, XPU kernels `0.1.12.3`.

Initial Bolt baseline: official GGUF, text-only, one sequence, no speculation, greedy temperature 0 for validation, embedded chat template with thinking disabled. Conservative context/batch first; distinguish compatibility-only limits from eventual usable capacity. Establish exact working settings from runtime support before testing changes. No claimed optimum from a tiny-context run.

## End-to-end test process (declared before execution)

1. Prepare/download/build while Qwen remains live. Retain pinned metadata, exact commands, build logs, and failed attempts. Verify model byte size and upstream LFS SHA256 before loading.
2. Before stopping Qwen, verify immutable original launcher fingerprint and record API health/models/canary, container state, environment, and kernel log. Run host-side detached bounded controller with unconditional restoration after exclusive GPU ownership is established.
3. Start Bolt on an isolated localhost API. Check `/health`, `/v1/models`, then `/v1/chat/completions` through the actual loaded runtime. Validate nonempty coherent output on arithmetic, literal instruction, and simple coding tasks, including finish reason and token usage; no GPU microbenchmark substitutes.
4. Only if compatibility passes: warm up, then retain repeated streaming/API measurements of TTFT, actual input/output tokens, decode tokens/sec, end-to-end time, and server timings. Use distinct prompts/cache disabled, fixed output budget (128 tokens for timing), one request at a time. Begin short, then test longer prompts (at least 8K when runtime permits), checking anchor retrieval rather than speed alone. Preserve early EOS rather than treating fewer tokens as a faster fixed-length run.
5. Any tuning comparison changes one supported setting, keeps model/runtime/context/quant/prompt/output policy constant, includes warmup and repeated baseline/candidate results, and retains failures/OOMs. Label bundled changes and capacity tradeoffs explicitly. No speedup assertion without paired data.
6. Stop experimental process, freshly verify no surviving GPU owner, restore the original Qwen launcher, then check public health/models and completion. Keep Bolt weights/build for follow-up; no persistent service promotion.

## Follow-on controlled prefill test (declared after compatibility passed)

Bolt's first successful 2K-context probe measured 28.5603 client decode tok/s and 0.62947 s median TTFT for 55-token prompts and 128 generated tokens (three repetitions). This is not a context-capacity test or optimization comparison.

Next baseline: 65,536 configured context, logical batch 512, physical microbatch 128, q8_0 K/V, no speculation, same weights/runtime, one sequence. Enable verbosity 4 solely to retain actual offload/device/memory evidence, identically in compared cells. Repeat short smoke/benchmark, then require exact secret-code retrieval with an actual >8K-token chat prompt before three cold >8K/128 timing requests (one warmup). Cache reuse must be zero; rendered usage and server timings retained.

One candidate, only if that baseline passes: increase physical microbatch from 128 to 512, holding logical batch 512, context 65,536 and everything else fixed. Hypothesis: larger prefill GEMMs amortize per-expert scheduling/host synchronization and improve cold prefill. This is not expected to increase single-token decode materially; evaluate TTFT/prompt processing, numerical correctness, and scratch memory. Source at the pinned revision explicitly excludes MUL_MAT_ID from SYCL command-graph capture, so forcing graphs is not a supported next candidate.

User explicitly requested continued optimization after the first microbatch comparison. Next repeat `ctx64k-ub512` with identical settings/workloads, additionally replay the exact same short greedy request three times (`--repeatability`) to distinguish same-config variability from microbatch-induced output changes. Do not infer bit identity or a quality regression from the first A/B text mismatch alone. Subsequent tuning must retain the winning 512-microbatch setup as the baseline, not the original 128 setup.

After the 512-microbatch repeat reproduced ~797 tok/s prefill and showed same-config greedy wording variability, test one larger prefill bundle: logical batch 1024 plus physical microbatch 1024, still context 65,536 and otherwise unchanged. This deliberately bundles both batch limits; do not attribute its delta to physical microbatch alone. Hypothesis: more tokens per routed expert improve GEMM occupancy and amortize host scheduling. Require the same smoke, >8K exact retrieval, three short/long timing requests and Qwen restoration. No precision/graph/speculation change.

The 1024/1024 bundle passed smoke/retrieval and measured ~949 tok/s cold prefill (8.843 s TTFT), so it becomes the next development baseline. Compare default `GGML_SYCL_ENABLE_ESIMD=1` against `0` only, retaining fusion/VMM/graph/precision defaults, q8 KV, 64K configured context and 1024 batch limits. Pinned source lines 5192–5238 show the Q8_0 dense decode dispatch changes from ESIMD/DMMV to reordered MMVQ; `MMVQ_WIDE` remains its default 1. Some activation/accumulation arithmetic changes, so label this quality-sensitive development work: run a fixed 20-case arithmetic/instruction/sorting/JSON/coding check on baseline and candidate as well as >8K retrieval and identical timing prompts. This is not the 500-prompt divergence/full quality suite required for publishable quality-sensitive claims.

Runtime research found UR immediate-command-list controls, but current-main documentation is not enough: installed oneAPI2026.0 ships UR0.12.0 legacy and v2 adapters and a no-device loader trace loads both. Active adapter/default mode is not yet established, so do not force queue/event knobs or call them a measured optimization.

All model/runtime workloads run on `inference-host`, outside interactive Pi. Raw artifacts mirror `/home/mike/b70-evals/20261016-nemotron35-bolt-sycl/`. The Lab preview tool failed with ENOENT; native text diffs are the fallback.

## BF16 prefill feasibility research (not yet implemented)

Fresh official sources document GPU BF16×BF16→F32 MatMul and default strict F32 accumulation: [MatMul support/layout](https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_matmul.html), [accumulation mode](https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_attributes_accumulation_mode.html), [Xe2 BF16 datatype support](https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_data_types.html), and [SYCL BF16 round-to-nearest-even conversion](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_bfloat16.asciidoc). [Intel's B70 datasheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf) identifies Xe2-HPG/XMX hardware. These establish feasibility, not a speedup or the capabilities of the installed binary.

Version caution: v3.14 docs may describe an unreleased/development snapshot. Before implementation/acceptance, inspect the exact installed library version/hash, and require actual oneDNN verbose GPU dispatch with BF16 operands and F32 destination—not documentation or the presence of a shared library alone. [Intel's 2026 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onednn/2026.html) also flag sporadic F32 MatMul correctness issues on Arc B-series; use numerical references and existing real API gates rather than assuming either GPU path is a gold standard. Do not upgrade the library/driver, enable relaxed accumulation, or change global math settings. The precise implementation and E2E process remain contingent on source inspection.


The read-only native C API probe (`bash inspect-onednn.sh`, no GPU mounted) reports **3.14.0**, hash `3f27bec21fdce6b550f233bf6e0afd086ee8d543`, CPU runtime2/GPU runtime512. Loaded library SHA256 is `114f9c84cde4283123bbce3306db9b4a80c29cf3512afcfd871c85461ddfab11`. This identifies the installed build, not its release status or BF16 execution support.

### Diagnostic E2E process, before precision changes

Run `python3 -B run-cell.py --name nvfp4-r1-onednn-diag --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1 --context 65536 --batch 1024 --ubatch 1024 --esimd 1 --dnnl-verbose --quality --long-bench` on the same B70 with original verified NVFP4 weights. The only runtime difference is `ONEDNN_VERBOSE=1`. Require existing smoke/20-case quality and exact8K retrieval; collect actual1024-token-chunk prefill GPU MatMul implementation, descriptors, math attributes and library banner in `server.log`. Retain frozen controller/create argv, raw responses, pre/post state and canary in the named cell. Any recorded timings are **diagnostic-only, not comparable performance results**, because verbose tracing can perturb execution. Inspect the real selected GPU implementation before deciding whether a BF16 prototype is justified; do not infer it from loaded-library presence. Reuse the reviewed exclusive ownership/cleanup and unchanged-Qwen restoration process. No parallel CPU build/conversion or GPU workload during this window.

Observed diagnostic result: 20/20 checks, exact 8K retrieval and Qwen restoration pass. `server.log` confirms the same oneDNN commit on the B70 Level Zero device and real GPU `jit:gemm:any` MatMul calls. Representative SSM-in descriptor: `src:f32 wei:f32 dst:f32`, `attr-fpmath:f16`, problem `1x18560x4096:1x4096x1024`; SSM-out similarly uses `1x4096x8192:1x8192x1024`. This proves an F32-operand GPU path with permission for internal F16 math, **not** strict F32 arithmetic, explicit BF16, native XMX instructions, or a bottleneck. Reported per-primitive execution times are zero, so these are dispatch records, not an operation timing profile. Diagnostic timings are excluded from accepted comparisons.


### Next isolated BF16 prototype: declared verification process

Hypothesis only: explicit BF16 operands with strict F32 accumulation may improve NVFP4 SSM prefill and honor its BF16 precision request. Preserve the working reorder kernel and all default/F32 routes. Limit the new branch to dense named NVFP4 SSM projections with a BF16 request, full-row single-device library GEMM and supported oneDNN execution; keep all unrelated operations unchanged. Activate the existing compiled-out BF16 support only in the experimental build. Never relax the F16 eligibility condition, change global math settings, quantize model data again or promote a launcher.

Environment/preconditions: same host, original verified GGUF, existing pinned oneAPI/oneDNN library and unchanged Qwen image/launcher; a separate source/build derived from the winning native prototype. No CPU conversion/build during GPU measurements. Before stopping Qwen, require compilation/dependency checks and the reviewed ownership preflight.

Native test data/steps: retain the existing 56-case default/F32 NVFP4 suite and its OPT=1/0 controls, unchanged 5e-4 NMSE threshold; add separately identifiable BF16-request coverage for original/reordered layouts, odd tails, small and real SSM dimensions, prefill→decode→prefill transitions and explicit-F32/non-SSM controls. Verify BF16 RNE conversion on halfway-even/odd vectors and compare BF16 MatMul against an F32-accumulation reference using BF16-rounded operands. Fix selectors and expected case counts from the actual test construction before launch; no skipped or loosened failures count as a pass.

Real application boundary: first use a diagnostic window on the new source with the same 65,536 context, 1024 batch/microbatch, ESIMD=1, original weights, q8 KV and F32 recurrent state. Require all native checks, 20/20 API cases and exact 8K retrieval. Capture oneDNN verbose output from the **actual model prefill** proving BF16 source/weights and F32 destination with no F16/relaxed math permission for that branch. Only after correctness/dispatch pass, run uninstrumented matched short55/128 and long8270/128 requests, one warmup plus three repetitions; compare against the two recorded native-R1 launches and independently repeat an apparent gain. Record actual tokens/cache usage, TTFT, server timings and client decode, not a proxy microbenchmark speed.

Retain source patch, exact build/launch commands, library/binary hashes, native counts/errors, frozen controllers, raw API responses/dispatch logs and pre/post device state in the experimental artifact/cell directories. Every failure or success removes only the known Bolt container and restores unchanged Qwen with observed 7+1 usage/finite logprobs. A failed numerical/API gate rejects the candidate; a no-gain result leaves the verified native kernel as winner. Development only, not full-quality or full-context validation.
