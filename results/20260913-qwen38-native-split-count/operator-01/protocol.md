# Bounded native causal Split-K count qualification

Development-only experiment; no promotion, production launcher or power changes.
Baseline: best MTP4 at commit 966a93e8; target FP16 GPTQ, FP8 target KV,
INT4 S+M1 draft, mixed-split-v5 GDN, K4, context212992/batch8192/C1,
graph sizes[1,2,4,8], prefix off. Only native verifier split count may change.
Pinned image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.

## Predeclared process

All torch/ML execution is on inference-host in a disposable pinned-image container.
Preconditions: no running containers; production launcher SHA256
`63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`,
power275000000 microwatts, no other GPU workload. No install or runtime rebuild.

1. Inspect installed interface, version and native binary hashes. The installed
   0.1.12.3 interface's speculative helper hardcodes `num_splits=None`; its public
   `num_splits_kv` argument is bypassed by this route. Do not change dispatch or
   causal semantics. For the probe replace only this literal in an in-memory
   copy of the installed helper; the public facade still selects the helper.
2. Qualify automatic selection and explicit1/4/8/16/32 (bounded to upstream
   heuristic's hard cap32). Actual binary support and honored override must be
   observed, not inferred from accepting a Python argument. Save CPU/XPU traces
   with shapes to inspect temporary allocation/reduction dimensions.
3. Synthetic real operator input class: Q[5,24,256] FP16, K/V[176,1664,4,256]
   FP8 E4M3, table[1,128], uniform causal Q5 with per-position adjusted lengths,
   softmax scale1/16, scalar KV descale views. Fixed seed42. Check finite outputs,
   output-buffer identity, noncontiguous/pitched inputs, shuffled pages,
   nonunit scales, and live lengths across page boundaries1663/1664/1665 and
   representative512/8192/32768/65536 contexts plus5 verification tokens.
4. Numerical gate fixed before execution: `rtol=0.02, atol=0.0001`, inherited
   from prior native attention-path qualification. Compare every candidate with
   automatic output and an independent FP32 paged causal attention reference;
   preserve max error/RMSE and failures. Replay captured graphs after mutating
   live-length/table metadata and compare to fresh eager and reference output.
5. Separate compilation/capture from measurement. Warm each captured route,
   then alternate/reverse count order across12 rounds,4 replays per event batch.
   Retain every timing, report median/IQR. Require at least5% median operator
   gain at32K and64K with no greater than5% regression at8K before serving A/B.
   This is a screening threshold, not a statistical throughput claim.
6. Only a passing winner proceeds to fixed-stack real HTTP API A/B through
   existing best-three/step-profile lifecycle and acceptance diagnostics.
   Baseline and candidate keep all listed best-MTP settings fixed; deterministic
   greedy seed42, C1, unique cold512/8192/32768/65536-token prompts,128 output,
   one warmup and six measured requests per point, paired A/B bundle order.
   Require successful canaries, finite-boundary and functional gates, valid SSE
   completion/usage, capture token/text divergence and speculative counters.
   Preserve exact argv, source hashes, environment, all raw requests/responses,
   failures and timings. No output-equivalence or publishable quality claim
   without the standards' full quality suite. No serving trial on operator fail.
7. Remove only owned containers; recheck launcher, power and host idle state.
   Commit raw bounded evidence and report, including no-win or blockers.

## Fresh primary-source research

- https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/csrc/flash_attn/flash_api.cpp
  Retrieved this session: automatic count balances WG occupancy, KV tile work
  and reduction volume; hard cap32. Explicit optional count feeds allocation
  shapes and paged decode dispatch. Public tag is not proof of installed binary
  identity. Image contains no C++ source or direct_url metadata; v0.1.12.3 raw
  source URL returned404. Verify installed behavior directly.
- https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/vllm_xpu_kernels/flash_attn_interface.py
  Retrieved this session; installed helper also observed directly.
- https://raw.githubusercontent.com/pytorch/pytorch/main/torch/xpu/graphs.py
  Retrieved this session for XPU capture/replay semantics; installed APIs remain
  authoritative. Rendered stable XPUGraph docs fetch failed; not counted as read.

Native Lab previews attempted but socket unavailable (`connect ENOENT .../lab.sock`).
Proceed through scoped CLI; no Lab-managed job/config change.
