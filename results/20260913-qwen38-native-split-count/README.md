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

## Executed outcome: no winner

Executed locally (Bash explicitly required because background shell is fish):

```bash
bash results/20260913-qwen38-native-split-count/run-operator.sh operator-01
python3 results/20260913-qwen38-native-split-count/analyze.py
```

The first inline launcher failed locally with exit127 before remote staging/GPU
execution; `launch-00-failed.log` preserves it. The explicit Bash rerun completed
with exit0, 2026-09-12T23:51:56–23:52:04-04:00 on inference-host.
The runner preserves its own exact command and protocol alongside the executed
probe. Remote artifacts also remain under
`/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-native-split-count/operator-01/`.

All49 correctness records passed: seven independent baseline checks and42
candidate/automatic graph cases. These exercise seven live lengths, shuffled
physical pages, pitched Q token stride12288, nonunit descales0.75/1.25, output
buffer identity, and captured replay after table/length mutation. K/V are
contiguous; this does not qualify every possible KV pitch/layout. Maximum
absolute error versus independent FP32 reference was0.000120342 (the gate
includes the predeclared relative tolerance); versus automatic was0.0000610352.
Mutated graph versus fresh eager was bit-identical in all42 cases. No numerical
gates were loosened. This is synthetic native-operator correctness, not model
output-equivalence qualification.

Median graph replay time in milliseconds (12 alternating-order batches ×4
replays; all288 batches retained, none excluded):

| Live KV tokens | Auto | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
|517|0.111797|0.111732|0.116348|0.115150|0.115710|0.114518|
|8197|0.236693|0.461660|0.412025|0.274961|0.239759|0.249596|
|32773|0.581198|1.752780|1.380436|0.765664|0.576250|0.576589|
|65541|0.962728|3.453346|2.717129|1.424551|1.010241|0.962018|

At65541, auto/explicit32 IQRs were0.002513/0.002308ms. Their0.000710ms
median difference is not evidence of a useful gain. No explicit count passed
the predeclared5% screening improvement at both32K and64K. `analysis.json`
contains every cell's median/IQR and latency reduction, recomputed from raw
samples. Event intervals include graph submission effects; these are neither
GPU-kernel-only durations nor full-model serving throughput. No confidence
intervals or full standards suite were run.

### Installed dispatch proof

Six retained `dispatch-*.json` traces establish native argument24 (explicit
split count), Q/K/V geometry, and actual internal allocations. At65541 live
KV/max bound65552, automatic creates partial output[5,768,256] and two
statistics buffers[5,24,32]: **the installed binary selects32 splits**.
Explicit4/8/16/32 allocate matching partial/statistics shapes and execute a
Split-K kernel plus one ReduceSplitK kernel. Explicit1 creates statistics
[5,24,1], aliases output instead of allocating partial output and executes no
reduction. Thus the overrides were honored, not merely accepted by Python.
Automatic count at other contexts was not independently profiled; no claim is
made that the entire public heuristic is byte-for-byte installed.

Installed interface SHA256:
`2a8ce07e2839232bc9f0e9cc9969a4410099c0c75ce616e72d4583e950864942`.
Native facade binary SHA256:
`a0d3bddd4175e1e22d5f1bd067720c72f84d83a9dd504def5aaaeaa221043524`.
Device kernels SHA256:
`76afaae0ac8844a07aeb91932ed8ace960bbf942300eb544d9f632a1465e43c0`.
Runtime: Torch2.13.0+xpu, native kernels0.1.12.3, Arc Pro B70,
Level-Zero driver1.15.39122+11; full exposed device properties in `result.json`.

### Decision and cleanup

**Keep automatic native split selection and best MTP4 unchanged.** No serving
candidate qualified, so conditional HTTP A/B was deliberately not launched.
No serving throughput, output identity, capacity improvement or universal
optimality claim is made. Counts above32, other input distributions, higher
contexts and Q8 are outside this bounded trial. No tuner or serving overlay
was added. Before/after checks show no running containers, no render-device
users reported by `fuser`, unchanged launcher SHA and275W. Owned container was
removed. Raw traces, all samples, executed source/protocol and cleanup logs
are in `operator-01/`.

Final validation: analyzer exit0; all retained JSON parses and Python AST checks
passed; runner `bash -n` passed; executed probe/runner byte-match source.
Authored-file staged whitespace check passed. Native profiler JSON has trailing
whitespace; unrestricted `git diff --cached --check` therefore exits2. Preserve
these raw traces unchanged rather than reformatting benchmark evidence.
Final host recheck at23:56:45-04:00 confirmed idle containers/device, launcher
SHA and275W again. Advisor reported missing tool outputs despite the completed
local validation; this visibility conflict is not a failed benchmark gate.
