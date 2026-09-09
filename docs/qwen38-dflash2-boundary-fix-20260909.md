# Qwen3.8 DFlash2 shortened-block fix and validation

Status: **verified fixed for the pinned C1 DFlash K7 configuration**. Native FP16/BF16 state validation and both BF16/partial RTN INT4 public-API journeys pass. Full [fixed-runtime artifacts](../results/20260909-qwen38-dflash2-boundary-fix/) preserve the commands, source, failed first test oracle and successful results. No full fixed-runtime performance comparison or production promotion is claimed.

## Confirmed failure

Both original BF16 and partial RTN INT4 DFlash2 failed in the pinned XPU runtime at 32640 rendered input tokens plus 128 forced output tokens, with `max_model_len=32768`, K7, legacy V1 and server C1. Both scheduler dumps show 32760 computed tokens, 121 output tokens, seven scheduled query tokens and six speculative tokens. The XPU GDN kernel rejects that seven-token list against its eight-column speculative-state layout:

```text
RuntimeError: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1) to be true, but got false.
```

The `/v1/completions` stream fails and `/health` subsequently returns 503. This is not an OOM. The separate 32000-token sweeps pass on both drafters; they avoid the edge case rather than fix it. Original failure evidence remains in [the benchmark campaign](../results/20260909-qwen38-dflash2-rtn-standard/).

## Fresh research and implementation constraints

Primary source fetched on 2026-09-09:

- [Pinned vLLM GDN metadata builder](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/attention/backends/gdn_attn.py), lines 293–313: the pure-speculative path clamps `spec_token_indx` to the actual scheduled token count but retains `self.num_spec + 1` state columns. This confirms the scheduler/kernel contract mismatch; retaining the original state width is important because the previous accepted count can still be eight.
- The actual `_xpu_ops.py`, GDN model/backend and scheduler files were copied from a **stopped** container of the exact serving image, without modifying or loading the active server. Source inspection is not a replacement for the fresh external research above.
- [Upstream vLLM issue #54740](https://github.com/vllm-project/vllm/issues/54740) reports the same XPU-kernels 0.1.14.1 fixed-width assertion for ragged speculative blocks with a different drafting algorithm. It supports the kernel-contract diagnosis, not validation of this local fix.
- [XPU-kernels 0.1.14 convolution source](https://github.com/vllm-project/vllm-xpu-kernels/blob/6d92b1bfbf32767ecda8e819613eb151e70030ad/csrc/xpu/gdn_attn/causal_conv1d.hpp) uses **one rolling convolution line**, selected by cache column 0; accepted counts select history windows within that line. [The SSM kernel](https://github.com/vllm-project/vllm-xpu-kernels/blob/6d92b1bfbf32767ecda8e819613eb151e70030ad/csrc/xpu/gdn_attn/gated_delta_rule.hpp) still uses per-token checkpoint slots. This tag is not claimed to be the exact installed 0.1.14.1 wheel commit; the tests execute that installed wheel directly.

Implemented approach: adapt only partial, pure-speculative C1 GDN calls to the existing fixed-width kernel, using temporary internal projected-input/output buffers. Keep the original full-width state-slot table and previous accepted-token count; copy only actual query outputs back. Internal dummy future rows must not become prompt tokens, KV-attention positions, scheduler tokens, verifier candidates or selectable accepted states. Ordinary full-width calls remain unchanged. The patch does not remove the assertion, reduce the context limit, slice off previous state slots, or reclassify unverified speculative tokens as ordinary prefill.

The native-state test below must establish that any internal padding preserves real causal outputs and accepted-prefix state. If that fails, the candidate approach is rejected even if the HTTP request no longer crashes.

## Defined real-system end-to-end process

### Environment and preconditions

- Host: `inference-host`, one Intel Arc Pro B70, no other GPU workload. The existing MTP4 benchmark completed before testing this fix; it was not interrupted.
- Image: `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`; vLLM commit `73029d42441321b631779db3475031f5ec26dd6c`, XPU kernels 0.1.14.1.
- Identical target checkpoint, FP16 GPTQ compute, FP8 target KV, strict DFlash K7, max context 32768, server C1, graphs `[1,2,4,8]`, prefix cache off and thinking off. Test original BF16 and the existing partial RTN INT4 drafter separately.
- Persistent launcher SHA256 remains `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; power cap remains 275000000 microwatts; `glimmer-tb21-prefix-c8` remains stopped. Model mounts stay read-only.
- Python/XPU and benchmark dependencies run only on the inference host/in the disposable image, outside the interactive Pi environment.

### Executable journeys and expected results

1. Run focused patch-transform/runner tests and syntax checks. Verify exact-anchor drift rejection, idempotence and that only DFlash cells apply the patch. These are supplemental checks, not E2E evidence.
2. In isolated GPU containers of the pinned image, run the native `_xpu_C.gdn_attention` differential test with FP16 (actual target compute) and BF16 projections. Cover real block lengths 1–7, previous accepted counts 1–8 and four physical/metadata padding variants; compare real outputs, normalization inputs, all selectable rolling convolution windows and real per-token SSM checkpoints against full-width native execution with differing dummy suffixes. Continue with another full speculative step after every valid accepted prefix, including zero accepted drafts (count 1 includes the target/bonus token), to detect reads of dummy states. Require exact equality (`rtol=0`, `atol=0`); retain argv, code and assertions. Any mismatch blocks API validation/promotion.
3. Start a disposable patched BF16 server with the existing `qwen38_standard_bench.py`/`DFlashCell` path. Exercise its existing natural-stop, finite-output and functional checks unchanged. Save launcher/argv, source snapshots, environment and raw server logs.
4. Through `/tokenize` and streamed `/v1/completions`, run `qwen38_long_context_bench.py` with the exact original 32640-input + 128-output point (one warmup and six measured trials), plus 512/8192/16384 controls. Use the same deterministic prompt construction, temperature 0, seed 42, ignored EOS and unique trial content as the original campaign. Require exact prompt/output usage, `[DONE]`, `finish_reason=length`, no SSE errors and healthy server after every request. Preserve the standard's higher-context HTTP rejections rather than omitting them.
5. Exercise nearby valid prompt lengths 32634–32639 with 128 forced output tokens through the same public API, retaining every request/result. Require healthy, correctly terminated 128-token responses; no changing the configured context ceiling to make them fit.
6. Repeat steps 3–5 with the INT4 drafter and identical target/runtime/settings. Retain per-request raw speculative counters; the strict rejection configuration must remain unchanged. Mere success is not a claim of universal greedy-output identity or quality equivalence.
7. Stop only the owned disposable containers on success or failure. Verify the launcher hash, power cap and stopped Glimmer state. Copy bounded raw artifacts into the repository and record the exact executed commands, paths and outcomes below. Keep all original failed cells; never overwrite them with fixed results.

No deployment or unrelated kernel/hosting tuning is authorized by this fix. Earlier BF16/INT4 throughput ratios describe the pre-fix stack and must not silently be relabeled as measurements of the fixed stack.

## Observed verification

- The first native FP16 run stopped at `L=1 previous=1 suffix independence: conv`, before any API server launched. Its test oracle incorrectly compared the whole rolling convolution line as if each slot were an accepted-token checkpoint. Dummy future rows may legitimately differ. The failed source and log are preserved under the host campaign's `code/` and `native-float16.log`.
- Corrected the oracle to compare precisely the union of convolution windows selectable by accepted prefixes 1–L, while retaining exact output/z/SSM comparisons, poisoned-padding checks, untouched-buffer assertions and every-prefix continuation checks. The runtime overlay itself was **not changed**; no numerical tolerance was relaxed.
- Corrected native gates: **FP16 and BF16 each PASS 224 partial cases and 896 continuations**, actual B70 and pinned 0.1.14.1 native kernel. Source, exact commands and logs: [native-v2](../results/20260909-qwen38-dflash2-boundary-fix/native-v2/).
- Real BF16 and partial RTN INT4 APIs: each passes one warmup and six measurements at 512/8192/16384/**32640** input tokens with 128 forced output tokens, plus all six nearby lengths 32634–32639. Each originally failing warmup payload is replayed unchanged. All corresponding fixed-arm request payloads match across arms. Exact usage, normal stream termination and health checks pass: **34 successful streams per arm**, plus four expected context-limit HTTP 400 probes each.
- Both arms also pass the existing 3 canaries, 131 finite boundaries and 8 functional checks. Graph settings remain `[1,2,4,8]`; both server logs show successful graph capture and neither contains the former assertion or a traceback. No acceptance-policy relaxation was made.
- The integrated source passes **35 focused CPU tests with no skips** and syntax checks for all ten requested Python source/test files. The archived overlay, corrected native test and both serving runners match the delivered source byte-for-byte.
- Resumed pipeline: **exit 0, 23m18s**. [Per-cell exit codes](../results/20260909-qwen38-dflash2-boundary-fix/cell-exits.tsv) and [cleanup evidence](../results/20260909-qwen38-dflash2-boundary-fix/cleanup.txt) confirm the unchanged launcher hash, 275 W cap, stopped Glimmer and no remaining disposable containers.
- Canonical host campaign: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-boundary-fix/`. [Repository artifact index, exact commands and reproduction notes](../results/20260909-qwen38-dflash2-boundary-fix/README.md). Failed first-run evidence remains separate from corrected evidence; pre-fix performance results remain separate from this correctness campaign.
