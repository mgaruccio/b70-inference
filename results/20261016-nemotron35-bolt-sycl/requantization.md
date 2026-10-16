# Selective integer-format experiment (development)

User requested investigation of another integer format, then approved cleanup and proceeding without further prompts. Preserve original Bolt/Qwen models and production launcher. No blanket conversion of QAD-trained experts.

## Source findings and hypotheses

Pinned upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb` supports NVFP4 dequantization and Q8_0/Q6_K quantization. Stock `--tensor-type-file` accepts regex/type overrides; first regex-search match wins. Generate a fully anchored entry for **every existing tensor**, keeping its original type except exactly 80 `^blk\.[0-9]+\.ssm_(in|out)\.weight$` NVFP4 tensors. Require the dry-run to show only those 80 conversions; do not depend on global quantization heuristics. Global IQ2_XXS keeps the original dominant file-type label; explicit entries preserve the existing Q8_0 tensors too.

The preliminary imatrix guard incorrectly rejects even already-IQ2 tensors that the actual conversion loop copies unchanged. `selective-requant-imatrix.patch` restricts that early guard to changed types. The later per-conversion imatrix requirement remains intact. No fake, unrelated or empty calibration artifact is acceptable.

40 × (4096×18560 + 8192×4096) weights give:

- NVFP4: 2,465,464,320 bytes / 2.29614 GiB.
- Q8_0: 4,656,988,160 bytes / 4.33716 GiB; +2.04102 GiB. Expected complete output ~24,209,850,368 bytes / 22.5472 GiB.
- Q6_K: 3,595,468,800 bytes / 3.34854 GiB; +1.05240 GiB. Expected complete output ~23,148,331,008 bytes / 21.5586 GiB.

Output sizes are estimates until native dry-run, including alignment/metadata. Both should fit 32 GiB VRAM, but actual peak VRAM is not established. More bytes may erase any benefit from faster integer kernels. Requantization cannot recover original lost information.

The graph loads `.scale` tensors independently of weight type and multiplies the matmul result by them. The quantizer dequantizes only the packed weight, without folding these external scales: retain the scales and their application. `build_lora_mm` also sets BF16 accumulation specifically for NVFP4; changing the type may change default backend dispatch/accumulation as well as weight representation. Treat this as quality-sensitive, not a lossless layout-only change.

Primary sources freshly consulted:
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/llama-quant.cpp — early guard, copy branch, quantization and per-conversion imatrix check.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/quantize/quantize.cpp — CLI and regex/type-file syntax.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/llama-model.cpp — generic scale loading.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/models/mamba-base.cpp and `src/llama-graph.cpp` — SSM scale consumers and precision policy.
- https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## Defined real-boundary test process, before implementation

All execution on inference-host, CPU-only containers for conversion/inspection; GPU only in approved exclusive Bolt windows. Use separate experimental source/build, original verified GGUF as immutable input, existing upstream GGUF reader, and native `llama-quantize` CLI. No custom GGUF writer or service.

1. Record pristine stock dry-run: full exact per-tensor type map, proposed sizes and spurious imatrix warning.
2. Apply only the one-condition guard patch to experimental source, rebuild quantizer. Repeat native dry-run; require exactly the 80 intended conversions, no other type/shape changes and no imatrix requirement.
3. Negative native CLI: force one NVFP4 tensor to newly generated IQ2_XXS without imatrix. Require nonzero exit at the imatrix guard and no output file. This must not bypass real low-bit calibration requirements.
4. With free capacity exceeding full output plus 2 GiB reserve, run native full conversion: `llama-quantize --allow-requantize --tensor-type-file EXACT_TYPES.txt INPUT.gguf OUTPUT.gguf IQ2_XXS 8`. Retain full argv/stdout/stderr and binary/source hashes. Start with Q8_0; Q6_K only if useful.
5. Inspect complete input/output metadata, names, shapes and types. Require byte-identical payloads for **every** non-target tensor, including experts and external scales; compare per-tensor hashes. Allow native quantization-version/type/offset metadata rewrites, but require chat template and architecture metadata intact. Measure changed-tensor dequantization error against original NVFP4 values; this is not full model-quality validation.
6. Use original baseline inference backend (ESIMD=1, unchanged fusion/VMM/graphs), q8 KV, F32 recurrent state, 65536 context, batch/ubatch1024, no speculation. Through actual `/v1/chat/completions`, require 20 fixed checks and exact 8238-token retrieval. Then same warmup/three-repeat short55/128 and cold8270/128 requests, cache disabled; retain raw requests/results/timings. Report numerical/quality failures, not merely speed.
7. Remove only owned transient containers, restore unchanged Qwen via reviewed controller, independently check models/7+1 canary/finite logprobs/launcher hash. Keep original weights and conversion evidence. No persistent launcher promotion. Twenty checks are development guards, not the 500-prompt/full quality suite needed for publication.

## Observed artifact validation

The native stock dry-run confirmed exactly 80 changes but emitted an unnecessary imatrix warning for copied IQ2 tensors. After the one-condition patch, the same dry-run still selected exactly 80 changes and required no imatrix. A real non-dry-run negative CLI request to newly convert `blk.0.ssm_in.weight` to IQ2_XXS still failed at the imatrix guard before creating output.

The complete Q8_0 output is **24,209,850,368 bytes**, SHA256 `d037cbbba6b0c0b27b67c585062ae3bd52fe602d5eacc19b94e8b6efb476163b`. All **763 non-target tensor payloads** are byte-identical; all names/shapes match; metadata changes are **zero**, including unchanged external SSM scales and chat template. Across all values of the 80 changed tensors, median per-tensor NMSE is **3.5623183e-5**, maximum **3.8118993e-5**, versus dequantized original NVFP4—not original full-precision weights. The original file's upstream SHA256 was rechecked unchanged. The real API subsequently passed smoke and 19/20 expanded checks, failing `drawer` → `reward` by returning `rewrad`. Timing and long-retrieval stages were gated off; Qwen was restored. This is not an accepted quality/performance candidate, and the small nondeterministic suite does not establish a causal or comprehensive quality regression.

The Q6_K copy is **23,148,331,008 bytes**, SHA256 `ab6e5f1bae4ec9bc80f10a02da5eff0fbf654b638b9de032f725eaaafbde81a8`. It also changes exactly 80 tensors, preserves all 763 other payloads and every metadata value, and rechecks the original SHA256. Median/max per-tensor NMSE versus dequantized original NVFP4 is **0.00033747968 / 0.00035876001**. Real API smoke passes but the expanded gate is **19/20**, again `drawer` → `rewrad`; no timing or long retrieval runs. Qwen is restored. Neither integer-format candidate is accepted on these development guards.

After both tests, available disk fell to 3.4 GiB. The rejected **Q8 scratch copy only** was removed after verifying its exact path, manifest, sole regular-file contents, size and absence from every container mount. Original Bolt/Qwen weights, the Q6 copy and all conversion logs/manifests remain. `cleanup-rejected-q8.py` and its log record the action; approximately 23.9 GiB became available. The retained Q8 manifest records historical verification, not current artifact availability.

Raw logs contain one non-UTF-8 byte in a tokenizer preview. The first parser stopped before conversion; revision2 preserves that raw log and uses replacement decoding only to parse ASCII tensor/status records. Native numerical/payload comparisons are not text-decoded or relaxed.

## Storage

Only two explicitly approved unused benchmark image tags were removed. Their logical unique size overstated physical recovery on the compressed Btrfs filesystem. User then authorized normal unmount and deletion of only `/home/mike/b70-evals/kv-offload.img`, an obsolete 32 GiB backing image. Current Qwen had no mount/use of that volume. Cleanup refused lazy/forced unmount and preserved all model weights; free space became about 48 GiB. Raw cleanup logs remain in the host result directory and will be mirrored here.
