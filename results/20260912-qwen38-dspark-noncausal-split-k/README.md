# DSpark native noncausal Split-K — development experiment

Hypothesis: the measured five-layer Q7 noncausal BF16 attention cost (~21.7 ms/draft step) can be reduced by treating each query as an independent native Q1 decode row. Each row must retain the **same full parent KV length**. The existing causal helper's length decrement is wrong for this input class. No custom Triton kernel, new acceptance rule, or production promotion.

## Research and pinned contract

Fresh official sources consulted before implementation:

- [Native XPU attention interface](https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/vllm_xpu_kernels/flash_attn_interface.py): describes the causal speculative-query expansion into native Split-K. Used for the transformation concept, not as the installed-version authority.
- [PyTorch SDPA reference semantics](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html): full noncausal scaled QK/softmax/V with dropout zero; reference here uses explicit independent CPU FP32 math.

Actual source was read from the pinned DSpark image: **vllm-xpu-kernels 0.1.14.1**, torch 2.13.0+xpu, vLLM 73029d424. Its public interface only applies small-query expansion when `causal=True`; normal Q1 paged decode accepts full per-row sequence lengths. The experiment calls that original public interface with expanded metadata, not a reimplementation. `vllm._xpu_ops` imports the function by value, so installation updates that one known binding as well as the module attribute; unexpected existing bindings/version fail closed.

## Predeclared qualification and real E2E

All ML execution is inside the pinned image on `inference-host`:
`vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`.

1. Start idle, 275 W, Glimmer stopped. Record launcher SHA256 and device/container state. Use `/dev/dri`, its render group, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `ZE_AFFINITY_MASK=0`.
2. Operator correctness: C1/Q7/H32/KV8/D128, page1664, BF16 Q/K/V, scale `128**-0.5`. Compare candidate, unmodified native and independent CPU FP32 attention. Predeclared tolerance `rtol=0.02, atol=0.002`; require finite results. Include lengths 1,7,1663,1664,1665,8192,65562, permuted pages/padded-tail sentinels, pitched Q/combined HND KV, caller output and allocated output, final-key-sensitive noncausal fixture, unsupported-route fallthrough and zero-length dummy behavior. Never loosen tolerances silently.
3. Capture the complete candidate wrapper in an XPU graph. Mutate Q, KV values, parent sequence length and block-table contents in place, replay and compare reference/native to detect frozen metadata. Require observed candidate dispatch. Only after numerical gates pass, measure paired native/candidate graph replays at64K (warm3, 12 intervals ×16 replays), retaining every sample.
4. Public E2E: reuse the existing corrected-DSpark API driver and shared gates. Baseline then candidate, same model/draft/overlay, depth7, greedy draft sampling, standard rejection, adaptive verification disabled, target INT4/FP16/FP8-KV, BF16 draft/KV, C1, capacity65664, batch2048, FULL_DECODE_ONLY graphs with capture sizes7/8, prefix caching off. Only candidate enables the temporary native route.
5. Before timing each serving cell: existing three canaries, 131 finite-logprob checks, eight sandboxed functional checks and repeatability/output-ID collection. Then warmup1 plus six measured real65536-input/128-output streamed requests, temperature0, seed42, ignore EOS. Same request identities across cells. Retain raw SSE, metrics, accepted/proposed token counts, TTFT and per-request timing, median/IQR and output divergence. No profiler in throughput runs.
6. Cleanup after every run: remove only owned temporary containers; verify launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`, power275000000 µW, stopped Glimmer and no competing device workload. Persist exact commands, sources and all failures. No persistent launcher changes.

The route is opt-in through `B70_DSPARK_NONCAUSAL_SPLIT_K=1`, limited to the measured BF16 C1/Q7/full-attention geometry, and assumes valid packed metadata from vLLM. Unsupported shapes/options go to the original implementation. Metadata expansion is on-device and captured, not host-read or cached across requests. Source lives in `scripts/experiments/qwen38_noncausal_split_k.py`; stage it beside the probe/driver. Reuse `scripts/experiments/qwen38_step_timing_patch.py` staged at `../20260911-qwen38-target-verification/qwen38_step_timing_patch.py` and the existing sibling drivers.

Remote campaign: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260912-qwen38-dspark-noncausal-split-k`.

Native Lab preview was attempted but unavailable (`lab.sock` ENOENT). The user's explicit approval authorizes this bounded disposable experiment. Development results only; not the full publication suite, not a distribution/fidelity proof.

## Qualification history (failures retained)

- Initial tester commits `d250fa6` / `3e56749` were not accepted as GPU evidence. Lead review corrected Q7 versus Q1 metadata, reference head grouping/scale, query width, guards, final-key visibility assertions and mutable-graph checks before executing them. Both revisions are preserved in Git history; the tested file hash is recorded per run.
- `install-check.log`: actual pinned-image import check passed: opt-in disabled leaves native unchanged; enabled installation updates the real `vllm._xpu_ops` binding; repeated installation is idempotent. A separate stdlib fake-tensor test against the installed interface signature confirmed full-length repetition and unsupported-mode fallthrough without importing an ML runtime into Pi.
- `operator-01`: **failed**, retained unchanged. All 14 nonempty cases (seven lengths × two layouts), eight fallthrough guards and explicit final-key fixture passed at the predeclared tolerance. The native empty-KV output was nonfinite; candidate output was finite. Native comparison therefore failed before graph checks/timing. Host settings were unchanged and the temporary container was removed.

### Explicit empty-KV test-contract correction before operator-02

This is a test-domain correction, not a relaxed numeric tolerance or a candidate algorithm change. The original empty-KV native-comparison gate remains a failed diagnostic in `operator-01`; do not report that run as passing.

Pinned vLLM `v1/worker/gpu/spec_decode/dflash/cudagraph.py::_prepare_dflash_inputs_to_capture` calls `InputBatch.make_dummy(num_reqs, num_tokens, ...)`. In `v1/worker/gpu/input_batch.py:113–155`, active dummy sequence lengths equal query lengths, and the returned sequence-length tensor is sliced to `num_reqs`. Thus this experiment's active C1/Q7 capture row has length **7**, not zero; zero padding is beyond the active slice. The route rejects multi-parent batches, so there is no mixture of padded zero parents and live parents inside its seven pseudo-rows.

The artificial `used=0` with seven queries has no keys over which to normalize attention and is outside this measured serving input class. Its native NaNs are retained, not replaced or called correct. Revised qualification still runs it, but requires **exact candidate zero output**, plus candidate graph transitions `65562 → 0 → 8192 → 0 → 1665` with in-place Q/K/V/table mutations. Each nonempty replay must agree with independent FP32 and current native output; empty replays must be zero. This explicitly checks that the artificial empty state cannot leave stale data in subsequent live results. All nonempty tolerances remain `rtol=.02, atol=.002`; the serving API gates remain unchanged.

Staged execution (inside the remote campaign):

```sh
bash run-operator.sh operator-01  # original empty-KV comparison failure
bash run-operator.sh operator-02  # revised empty-domain diagnostic and replay isolation
bash run-operator.sh operator-03  # initialization passes eager zero; captured empty output still NaN
bash run-operator.sh operator-04  # post-mask fixes empty graph replay; full qualification passes
# Only after operator qualification passes:
python3 run-noncausal.py --out baseline-01
python3 run-noncausal.py --candidate --out candidate-01
```

Exact Docker commands/source hashes and before/after host state are retained in each operator directory. All serving commands/configuration and raw HTTP evidence are retained by the existing driver.

`operator-02` also failed and is retained: all nonempty cases and final-key visibility passed, but candidate empty-KV output preserved the caller's sentinel value 3 (max absolute error from zero 3.0). Finiteness alone was insufficient. The subsequent candidate change explicitly zero-initializes its output before every native call/replay, allocating a contiguous zero output when no caller buffer is supplied. Nonempty native computation overwrites that buffer normally; there is no host read, length clamp, acceptance change or tolerance change. `operator-03` reruns the entire qualification, including zero-to-live graph isolation, before the automatic serving A/B pipeline may proceed.

`operator-03` failed in the candidate graph's first empty replay. Initialization fixed eager empty output, but captured native kernels still execute for an empty row and can write NaNs. Its exact tested module is retained at `operator-03/tested-module.py`. The final candidate therefore also masks empty-row output **after** native computation using an on-device predicate; it does not multiply NaNs by zero or branch on host-read lengths. The mask is false for every real nonempty row.

## Passed operator qualification

`operator-04` passed: 14 nonempty shape/layout cases, all four comparisons per case, eight unsupported-route fallthrough checks, explicit final-key visibility, caller/allocation output checks, exact eager zero output, three native mutable replays and five candidate mutable replays including both zero-to-live transitions. Original empty-KV native comparisons remain nonfinite diagnostics, not claimed passes. Tolerances are unchanged.

Paired, warmed graph replay at 65562 KV tokens, **per attention call**, including candidate expansion, output initialization and final mask:

- Native median **4.436893 ms**, inclusive IQR **0.001960 ms**.
- Candidate median **0.997580 ms**, inclusive IQR **0.011386 ms**.
- Operator speedup **4.4477×**; all 12 interleaved intervals ×16 replays retained, compile excluded.

No native fallback warning occurred. Before starting real serving, the pipeline verifies the passing JSON, full candidate replay count, successful host cleanup, and SHA256 equality between the qualified module and staged serving module. These are operator development results, not an end-to-end speedup claim.

## Real serving result and decision

Both `baseline-01` and `candidate-01` passed the real public-API process. Their serving argument arrays are identical. Each used one warmup followed by six measured 65536-input/128-output streams, with identical request JSON per paired trial and no invalid measurements.

- **Decode median: 26.2415 → 30.6599 tokens/s (+16.8375%)**.
- Decode inclusive IQR: **0.9089 → 1.0887 tokens/s**.
- Median decode interval: **4.8401 → 4.1423 s**.
- Median TTFT: **57.3496 → 57.0937 s**; median whole-request time **62.1673 → 61.2372 s**. The decode improvement is not a claim of a 16.8% whole-request gain.
- Both cells passed three canaries, 131 finite-logprob boundaries and eight sandboxed functional checks.
- Fixed output-ID files match **17/19**. Differences are `parity-code-1-output-ids.json` and `parity-code-2-output-ids.json`. Baseline code repeatability already produced two unique variants; candidate produced one. Prose repeatability produced one in both.
- Long output text matches **3/6** identical requests (trials1,3,5). Thus this is **not bitwise-output-qualified**. No repeated-baseline control exists for these exact long requests, so divergence cannot be assigned solely to the attention change.
- Measured speculative counters: baseline **283 steps / 1847 drafted / 472 accepted**; candidate **288 / 1881 / 466**. Per-request deltas and acceptance ratios are in `comparison.json`; do not substitute nominal depth×steps for observed draft counts.
- Both runtime logs report **7.22 GiB available KV memory, 74,528 KV tokens, 0.08 GiB graph capture memory**. These are runtime reports, not peak-memory or near-limit qualification.

Candidate `server.log` shows the eligible native-route dispatch during full DSpark graph capture. Neither serving cell logged a missing native kernel/fallback. Qualified, served and repository module SHA256 agree: `e19d3fa94a11942aeb9db4ca4552ab4357373c14d7095f7612761a05b1d924ff`.

`comparison.json` contains all six timing samples, median/inclusive-IQR values, output comparisons, exact source identity and pooled/per-request speculative deltas. Raw requests, SSE, metrics, checks, launch arguments, runtime identity and logs are retained in each cell. All samples—including the faster fifth trial in both cells—remain included. These are sequential development cells, not interleaved serving A/B with a confidence interval. Operator layout label `hnd` denotes the contiguous NHD-shaped fixture; `combined` exercises the actual combined-HND backing storage and pitched Q.

Final checks: actual-image installation/binding smoke passed; Python AST checks and shell syntax passed; numerical and mutable-graph qualification passed as described above; real API and source-hash consistency passed; authored-file whitespace checks passed. Raw logs retain their original whitespace. Read-only review found no blocking contract break in its scoped probe review; the lead additionally executed the final post-mask graph and API tests. `host-final.txt` and both cell cleanup records confirm unchanged launcher SHA256, 275 W, stopped Glimmer, no running containers and no observed render-device holders.

**Decision:** retain the opt-in native route as a useful DSpark development improvement. No persistent launcher change, no production promotion, no bitwise-equivalence claim, and no full BetterBench/bench-serve/context-sweep qualification. Failed operator runs01–03 remain failures in the retained evidence.
