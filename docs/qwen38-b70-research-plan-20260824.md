# Qwen3.8 B70 research plan — 2026-08-24

This plan turns the GPT-pro B70/Qwen 3.8 backlog into executable research
tracks. It is grounded in the 2026-08-23 handoff, not a blank-slate Yukon
port.

Context ceiling measured 2026-08-30 (live launcher still 131k @ 0.88):
[`qwen38-b70-200k-context-20260830.md`](qwen38-b70-200k-context-20260830.md).

## Authority

- Original request: produce a research plan for each GPT-pro possibility.
- Last-session facts live in
  `docs/qwen38-b70-next-session-handoff-20260823.md` and
  `docs/qwen38-b70-mlxfast-research-progress-20260823.md`.
- The evaluation contract is immutable: Pi `0.83.0`, Harbor TB2.1 pinned
  five-task cohort, Docker runtime, high thinking, **98,304 / 32,768**, C1
  unless a later track explicitly studies concurrency.
- Live service stays on the original BF16 MTP-4 launcher until a candidate
  passes token-level greedy parity.
- Do not port MLX Metal, `xsums`, or the MLXFast proposal head blindly.
- Transfer the *questions* and *measurement method*, not the Apple kernels.

## Current measured baseline

| Fact | Value | Consequence |
| --- | --- | --- |
| Live service | BF16 MTP-4 | Experimental overlays must be isolated and torn down |
| Strongest speed row | Draft-INT4, graph cap 64, MTP-4 | 54.67 / 17.83 / 0.644 tok/s short / medium / ~98k |
| Quality gate | Failed greedy parity at ~8k | Draft-INT4 is **not** deployable and is a suspect research platform until explained |
| MTP-2 vs MTP-4 | MTP-4 faster; MTP-3 startup fail | Static depth is already partly measured; adaptive depth is not |
| Graph capture | Cap 64 stable; cap 128 fails; no-graph much slower | Graph memory/padding is already a first-order issue |
| GDN metadata overlay | Stable, not a speed win | Keep as correctness/hygiene, not a throughput bet |
| Mixed C2 + v5 | One surviving smoke | Correctness only; no throughput claim |
| Kernel source | `vllm-xpu-kernels 0.1.12.3` is compiled-only | Any kernel study needs an upstream checkout/build path |
| Agent eval | 3/5, 17.10 median tok/s | Agent rate is not the model-card decode number |

The ~98k row is already ~85× slower than the short warm row. Long-context
attention/GDN will dominate agent-eval wall time even if draft-head work
wins short decode.

GPT-pro's "200K context" is a research horizon. The **binding product
target** is still 98,304 in / 32,768 out on a 131,072 serving ceiling.
Treat 128k–200k as a separate stress ladder, not a reason to change the
eval contract.

## Execution model

| Role | Model | Owns |
| --- | --- | --- |
| Lead | current session | Sequence, quality gates, integration, what is allowed onto the live box |
| Researcher / scout / worker / tester | `openai-codex/gpt-5.6-luna`, `thinking: max` | Source audits, experiment design, profiling interpretation, kernel/readout analysis, writeups |
| Mechanical implementer | `cursor/composer-2-5:slow` | Logging hooks, microbench scripts, overlay scaffolding, fixture tests, doc/table updates after a Luna spec |

Rules:

- At most three concurrent descendants.
- Luna writes the question, method, and kill criteria **before** Composer
  writes code.
- No worker writes in the lead checkout; use `pi-worktree`.
- Do not replace the live BF16 server for research. Use disposable
  launchers and restore health after every run.
- A track is "explored" only when it has: a written claim, a measurement,
  a keep/kill decision, and an artifact path.

## Shared method

Every track uses the same evidence contract.

1. **Read first.** Name the exact vLLM / XPU-kernel / MLXFast files and
   functions. Do not hypothesize from Yukon blog tone.
2. **Measure independently.** Isolate the component before changing it.
3. **Compare against the frozen BF16 C1 controls**, not against the
   previous experimental overlay.
4. **Keep target verification numerics unless the track is explicitly a
   draft-only proposal change.** Draft-only changes still must pass the
   greedy ~8k gate before any agent claim.
5. **Record** image digest, vLLM SHA, model revision, launch flags, graph
   cap, MTP depth, KV tokens, TTFT, output tok/s, acceptance length, and
   host/vLLM logs outside `--rm`.
6. **Kill** a track when the isolated cost is <5% of the round, when the
   change cannot preserve the 131k ceiling, or when it fails greedy parity.

Context ladder for any end-to-end claim:

- 4k
- 8k (also the current parity failure point)
- 32k
- 98k (eval-contract length)
- 128k / 200k only as optional stress, never as the eval substitute

## Phase 0 — unblockers

These are not in the GPT-pro list. They still go first.

### P0. Greedy ~8k divergence

**Why first.** Last session's winning speed row is quality-rejected. Until
the first divergent target/draft decision is identified, draft-INT4 cannot
be used as a research platform for items 1, 4, 7, or 12.

**Questions.**

- At which generated token do BF16 and draft-INT4 first disagree?
- Is the disagreement in draft proposal, MTP acceptance, target logits, or
  GDN/KV state after a partial accept?
- Does it reproduce with graph cap 64, graphs off, and MTP-2?

**Method.**

- Luna: design a token-id / acceptance-event tracer and name the exact
  vLLM hooks (`gpu_model_runner`, MTP propose/verify, sampler).
- Composer: implement the narrow logging overlay and a comparison script
  over the existing artifacts:
  - `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-bf16/`
  - `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-gdn/`
  - `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-comparison.json`
- Re-run only the failing ~8k greedy pair with traces. Do not start a
  cohort.

**Keep / kill.**

- Keep draft-INT4 as a research vehicle only if the mismatch is explained
  and either fixed or proven not to affect sampled agent traffic.
- If the mismatch is inherent to changing draft logits, freeze draft-INT4
  as a short-decode curiosity and do later draft-head work on a
  proposal-only path that cannot change target tokens.

### P1. Measurement harness (GPT-pro 14, started now)

**Why now.** Items 1–13 are not research without component timers. Last
session has service-level C1/C2 rows and no kernel-attributed MTP-round
breakdown.

**Questions.**

- What fraction of a speculative round is draft head, MTP layer, verify
  GEMM, GDN, attention, LM head, accept/commit, and host sync?
- How do those fractions move from 4k → 98k?

**Method.**

- Luna: specify microbench matrix and required counters.
- Composer: implement benches as disposable scripts against a research
  container, not the live server.

Required isolated benches:

- draft head only
- one MTP layer
- verify forward at `M=1…9`
- target LM head at `M=1…9`
- GDN spec-decode
- attention vs context length at `M=1` and `M=4`
- accept/commit path, including `.item()` / `tolist()` / D2H

Then keep the existing public-API C1 suite and add enough repeats for
intervals. C2 stays a correctness smoke until P0 is closed.

**Keep / kill.** The harness itself is not optional. Individual benches
can be dropped if they cannot be run without kernel source.

### P2. Upstream kernel checkout

**Why now.** Items 5, 6, 8, 10, 11, and 13 cannot proceed past Python
overlays without `vllm-xpu-kernels` source.

**Method.** Luna locates the kernel tag matching `0.1.12.3` / image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
documents build, and lists GDN / GPTQ / graph-padding files. No kernel
edit until a later track has a measured target.

---

## Track map

GPT-pro ordered 1–8 as “make MTP good on B70” and 10 as a parallel
long-context project. That split is right, with one correction: **10 is
already the agent-eval bottleneck**. It is parallel, not later.

```text
P0 greedy 8k ──► P1 harness ──► short-MTP tracks 1,3,4,7,8,9,12,13
        │              │
        │              └──► 5,6,11 (need kernel source from P2)
        │
        └──► 2 (state semantics; blocks 3 and 4)
                 │
                 └──► 10 (long-context attention; parallel after P1)
```

---

## Item 1 — Draft LM-head elimination / shortlist reranking

**Hypothesis.** The 248,320 × 5,120 draft projection is a large, mostly
wasted cost. Coarse scoring plus exact rerank of ~32 candidates can
replace the full vocab GEMM.

**What we already know.**

- vLLM Qwen MTP still does a conventional LM-head/logits pass.
- Draft-INT4 already changes draft logits and failed ~8k greedy parity.
- MLXFast's winning head is `amal-david/qwen38-mtp-head-q2-q4-rerank-v1`
  (~427 MB) over a compact ~98,336-token draft space. That format is not
  GPTQ/XPU drop-in.

**Questions.**

1. What is the isolated cost of `Qwen3_5MTP.compute_logits()` at `M=1`
   versus the rest of the draft step?
2. Can we score a coarse subset (cluster, prefix, INT2/INT3) and rerank
   ≤32 rows with the accurate head without changing **target** tokens?
3. Is converting the MLXFast q2-q4-rerank artifact cheaper than training a
   B70-native proposal head against this GPTQ target?

**Method.**

- Luna: profile `compute_logits()`; read
  `vllm/model_executor/models/qwen3_5_mtp.py` and the MLXFast head
  runtime; write a conversion-vs-retrain note.
- Composer: only after Luna specifies the probe, add a draft-logits
  dump / top-n overlap tool.
- Do **not** load the MLXFast head into vLLM as a first experiment.

**Success.** Isolated draft-head time drops materially and greedy target
tokens remain identical to BF16 at short, coding, and ~8k.

**Kill.** If draft-head time is <10% of the round at 8k+, or if any
proposal-head change cannot be proven target-neutral.

## Item 2 — MTP committed-history behavior

**Hypothesis.** Fresh-per-round MTP/GDN state is leaving accepted prefix
information on the table. Persistent committed history was important
enough that MLXFast redesigned its session around it.

**What we already know.**

- Native Xe2 speculative GDN exists in current vLLM-XPU kernels.
- Installed Python path is one `_xpu_C.gdn_attention(...)` call.
- Mixed-batch v5 is a Python workaround, not the CUDA/XPU state machine.

**Questions.**

1. After accept `k` of `n` drafted tokens, what happens to MTP hidden
   state, GDN conv/delta state, and KV?
2. Which of these does the XPU path implement: fresh head state, committed
   prefix, speculative write+rollback, replay of accepted trunk hiddens?
3. Does Xe2 GDN rollback match CUDA, or does it recompute?

**Method.**

- Luna-only source audit first:
  - `qwen3_5_mtp.py`
  - `gpu_model_runner.py` propose/verify/commit
  - `vllm/v1/attention/backends/gdn_attn.py`
  - matching files in `vllm-xpu-kernels`
- Then a single-request acceptance trace at MTP-4 with forced reject,
  partial accept, and full accept.

**Success.** A written state machine: which buffers persist, which roll
back, which are replayed. This is a prerequisite for items 3, 4, and 11.

**Kill.** None. This is a correctness map. Implementation only follows if
the map shows wasted recomputation.

## Item 3 — Separate MTP hidden-state update from vocabulary projection

**Hypothesis.** History/priming rows update attention/GDN state but do not
need the giant vocab head. Only the proposal row should project.

**Depends on.** Item 2.

**Questions.**

- Does the current forward always project every MTP row?
- Can we expose `mtp_forward_hidden` / `mtp_update_cache` /
  `mtp_project_last_row` without changing GDN alignment?
- How many vocab projections happen per speculative round today?

**Method.**

- Luna: count projection calls per round from source + a torch/XPU
  profiler.
- Composer: add counters only.
- Prototype last-row projection only after item 2 says rollback is safe.

**Success.** History rows stop touching the vocab head; greedy target
tokens unchanged; measurable drop in draft-step time.

**Kill.** If the implementation already projects only the last row, or if
splitting the graph blows cap-64 memory.

## Item 4 — Adaptive speculative depth

**Hypothesis.** Fixed `num_speculative_tokens=4` wastes draft/verify work
on low-acceptance steps and leaves throughput on the table during
full-accept streaks.

**What we already know.**

- Static MTP-4 beat MTP-2 on the natural C1 suite.
- MTP-3 failed startup; MTP-5 was never tried.
- Long-context acceptance was ~2.65–2.82 mean tokens in the C2 smokes.
- MLXFast moved to streak/adaptive 0–8.

**Questions.**

1. Acceptance probability by depth 1…8 on **natural** Pi-like text, not
   the repetitive short prompt.
2. Cost of draft token `d` and verify width `M`.
3. Rollback cost after a reject at depth `d`.
4. Why MTP-3 fails startup, and whether that blocks odd depths.

**Method.**

- Luna: design the expected-committed-tokens-per-second objective and the
  depth scheduler (streak, recent accept rate, context length).
- First collect a **static** sweep at depths that start: 1, 2, 4, and 8
  if memory allows. Do not begin with a closed-loop controller.
- Only after item 2, try a reactive depth that never desynchronizes GDN.

**Success.** Higher committed tok/s than static MTP-4 at 4k and 8k
without a 98k regression or parity failure.

**Kill.** If accept-by-depth is flat, if scheduler overhead exceeds the
gain, or if changing depth mid-session corrupts GDN/KV.

## Item 5 — Small-M GPTQ/XPU kernel specialization

**Hypothesis.** Generic W4A16 GEMM at `M=2…5` launches many idle groups
on Xe2, the same class of bug MLXFast found (67–80% useless groups).

**Depends on.** P1 timers and P2 kernel source.

**Questions.**

- Submitted vs active workgroups, SIMD occupancy, XMX use, GRF pressure,
  and memory transactions at `M=1…9`.
- Are verify widths 2–5 on a GEMM plan that assumes large `M`?
- Would dedicated plans for 2, 3, 4, 5 beat the generic path?

**Method.**

- Luna: read the GPTQ-G128 Xe2 GEMM launch geometry; specify the
  counters.
- Hardware profiling on a research container only.
- No kernel rewrite until a width shows idle-group or XMX-starvation
  evidence.

**Success.** A measured idle-launch or occupancy smoking gun at a common
verify width, plus an estimated tok/s gain.

**Kill.** If occupancy is already high at `M=2…5`, or if a specialized
plan cannot be built against the installed XPU stack.

## Item 6 — Graph-padding overhead in speculative execution

**Hypothesis.** XPU graph replay launches over captured capacity, not
active speculative rows. There is already an open XPU-kernel change for
graph-padded speculative GDN metadata.

**What we already know.**

- Full graph capture used 5.31 GiB and the first overlay OOMed at long
  context.
- Cap 64 is the stability boundary; cap 128 fails startup.
- Our metadata overlay (PR #43955 static half) did not move C1 speed.

**Questions.**

1. On a cap-64 graph, how many rows do token tensors, state indices,
   accepted-token arrays, GDN metadata, and attention metadata actually
   use versus launch?
2. Is the remaining waste in Python metadata or in the compiled GDN
   graph?
3. Can we raise the useful capture set without crossing the cap-128
   failure?

**Method.**

- Luna: audit active-prefix narrowing in `gdn_attn.py`, model runner, and
  the XPU-kernel GDN spec-decode metadata path.
- Compare cap 32 vs 64 vs no-graph at 4k and 8k with P1 timers, not just
  end-to-end tok/s.
- Follow the upstream graph-padding GDN patch; do not reinvent it.

**Success.** Proof that replay work scales with captured capacity rather
than accepted rows, **or** proof that it does not.

**Kill.** If timers show graph padding <5% after the existing overlay, or
if any capture-shape change reintroduces the long-context OOM.

## Item 7 — Device-side acceptance and synchronization removal

**Hypothesis.** `.item()`, `tolist()`, CPU top-k/argmax, or explicit sync
inside the speculative round is chopping Xe2 throughput.

**Questions.**

- Every host sync in propose → verify → accept → commit.
- Which of verify-reduction, draft compare, accepted-prefix length,
  bonus-token select, and cache-commit info can stay on device?
- Does greedy decode already have top-1/top-2 on device that acceptance
  recomputes?

**Method.**

- Luna: grep/audit the installed vLLM XPU speculate path; list each sync
  with estimated cost.
- Composer: add NVTX/XPU-equivalent ranges or `torch.xpu.synchronize()`
  brackets only where Luna named them.
- Prototype a single small D2H result last, not first.

**Success.** A counted sync list and at least one sync whose removal is
safe and visible in round time.

**Kill.** If the round is already device-chained, or if a fused accept
kernel would require an XPU-kernel ABI change before P2 is ready.

## Item 8 — Producer → consumer kernel fusion

**Hypothesis.** Verification widths 4–9 recompute statistics or rewrite
temporaries that the previous kernel already had. MLXFast deleted 127
standalone dispatches per verify round this way.

**What we already know.**

- MLXFast fused quantized-matvec aux generation into residual+RMSNorm.
- B70 is symmetric GPTQ G128 W4A16, not affine G64. There may be no
  `xsums` equivalent.
- Last session already asked this as expert question 1.

**Highest-interest chains.**

- residual add → RMSNorm → GPTQ projection
- RMSNorm → quant/prepare
- QKV → Q/K norm → RoPE
- GDN project/reorder/state update
- gate/up → SiLU×gate → down

**Method.**

- Luna: from P1, attribute verify-round time to those chains; then read
  the corresponding Xe2 kernels for recomputed row sums, scales, or
  layout transforms.
- Prototype only the hottest measured producer-side reuse.
- Numerics gate: bit-exact or documented-and-accepted ULP vs BF16 target
  verify.

**Success.** Fewer launches or fewer bytes reread at `M=4` with unchanged
target tokens.

**Kill.** If the GPTQ path already fuses the analogue, or if fusion
increases GRF spill / register pressure enough to lose the launch win.

## Item 9 — Wide-vocabulary reductions

**Hypothesis.** Argmax/top-k over 248k tokens is its own bottleneck, and
greedy draft should never materialize full logits.

**Depends on.** Item 1's isolated draft-head profile. If the GEMM
dominates and the reduction is cheap, this collapses into item 1.

**Questions.**

- Cost of materializing full draft logits vs fused project+top-k.
- Hierarchical reduction vs local top-k during GEMM vs direct top-1.
- Does verify also need a 248k reduction, or only draft?

**Method.** Luna profiles reduction vs GEMM first. Composer implements a
top-k dump only if reduction is ≥20% of draft-head time.

**Keep / kill.** Merge into item 1 unless the reduction is independently
large.

## Item 10 — Long-context multi-row verification attention

**Hypothesis.** At 100k–200k, an `M>1` verify that reloads K/V per query
row will dominate every short-context MTP win. A kernel that loads each
historical tile once and evaluates all M queries while resident should
win.

**What we already know.**

- ~98k C1 is 0.64 tok/s versus 44–55 tok/s short.
- Yukon's 512-token fixture does not represent this.
- Agent eval median 17.10 tok/s is a mixture of prefill, thinking, and
  mid-context decode; the 98k tail is the contract risk.

**Questions.**

1. For verify width `M=4`, is attention closer to 1× or 4× a decode
   attention at 32k / 98k / 128k?
2. Do FP8 KV and GDN change the reuse opportunity?
3. Is the current XPU path already batched across speculative queries?

**Method.**

- Start as soon as P1 has an attention-vs-context bench. Do not wait for
  short-MTP wins.
- Luna: compare decode `M=1` vs verify `M=4` bandwidth and time at 4k,
  32k, 98k.
- Kernel work only after the bench shows near-independent per-row loads.

**Success.** Verify attention time grows much slower than `M` at ≥32k
without breaking FP8 KV or GDN rollback.

**Kill.** If the current kernel already shares K/V across the M rows, or
if GDN—not attention—owns the 98k time.

This track is allowed to run **in parallel** with items 1–8 after P1.

## Item 11 — GDN speculative kernels

**Hypothesis.** Native Xe2 speculative GDN is new, has special-purpose
kernels for speculative sequences / accepted-token counts / cache state,
and may still do extra conv-state traffic or padded launches.

**Depends on.** Item 2 and P2. Closely coupled to item 6.

**Questions.**

- Causal-conv state read/write cost vs attention/GDN math.
- Rollback layout: contiguous vs interleaved speculative buffers.
- Token-index indirection cost.
- Whether state update can fuse with the neighboring projection.

**Method.**

- Luna: profile GDN spec-decode separately from normal attention.
- Follow upstream graph-padding / partition-merge work rather than
  extending Python v5.
- v5 stays a C2 correctness crutch, not the research destination.

**Success.** A GDN-spec cost breakdown and one evidenced kernel or layout
change.

**Kill.** If GDN is a small fraction until 32k+, defer and let item 10
lead.

## Item 12 — Weight-format co-design for Xe2

**Hypothesis.** The verifier's GPTQ-G128 W4A16 layout is not the best
physical format for `M=1` draft, `M=2…8` verify, or selected-row rerank.

**Depends on.** Items 1 and 5. Do not start format work before those
profiles.

**Interesting formats.**

- INT2/INT3 coarse draft head
- INT4 selected-row rerank
- prepacked XMX-friendly small-M matrices
- duplicated/reordered LM-head subsets if memory is modest

**Constraints.**

- Target verification weights stay as they are unless a format change is
  proven numerically safe.
- Draft-INT4 already showed that “just quantize the draft tensors”
  improves short tok/s and fails quality. Any new format needs the P0
  tracer.

**Keep / kill.** Start only if item 1 or 5 shows the current layout, not
just the algorithm, is the limit.

## Item 13 — Quantization-side auxiliary work

**Hypothesis.** GPTQ-G128 kernels recompute zero-point-like terms, row
sums, scales, or activation conversions that RMSNorm or a previous
projection could emit once. This is the closest B70 analogue to MLXFast
`xsums`.

**Depends on.** Item 8's chain inventory.

**Questions.**

- What does each W4A16 kernel recompute from activations every call?
- Can QKV and gate-up share prepared activations?
- Can RMSNorm emit the next kernel's aux without changing numerics?

**Method.** Luna answers from kernel source after P2. No Composer work
until a specific aux tensor is named.

**Keep / kill.** Merge into item 8 unless the aux work is large and
shared across several consumers.

## Item 14 — Benchmarking methodology

Covered by P1. Additional rules:

- Never promote a Yukon-style short-context win that regresses 32k or
  98k.
- Never compare agent-trace tok/s to model-card decode tok/s.
- Repeat C1 enough for intervals before claiming a win. One-sample C2 is
  not a throughput result.
- Prefix cache remains an experiment, not a default.
- Parser, port, and firewall changes are not speed experiments.

---

## Suggested first three Luna slices

These can run with the global descendant cap of three. They do not
require live-server replacement.

1. **P0 design + item 2 source map** — greedy-divergence tracer spec and
   XPU MTP/GDN committed-history state machine.
2. **P1 + item 10 bench spec** — microbench matrix and the
   attention-vs-context / `M=1` vs `M=4` ladder.
3. **P2 + items 5/6/11 file map** — `vllm-xpu-kernels` checkout/build
   notes and the exact GDN / GPTQ / graph-padding surfaces.

Composer starts only after slice 1 names the tracer hooks and slice 2
names the bench CLI.

## Out of scope until a later decision

- Replacing the live BF16 service.
- Full Harbor cohort or C5.
- Training a new proposal head before item 1's isolated profile and P0
  are done.
- Copying MLXFast Metal kernels or the q2-q4-rerank weights into vLLM.
- Upstream SYCL rewrites before P1 attributes time.
- Changing the 98,304 / 32,768 eval contract.

## Decision rule

After each track:

- **Adopt** if it improves committed tokens/s on the context ladder
  without breaking greedy target parity or the 131k ceiling.
- **Park** if the isolated cost is real but smaller than item 10's
  long-context term.
- **Kill** if it fails parity, OOMs the serving ceiling, or cannot be
  measured on this XPU stack.

The first adopt/park/kill we need is P0. Until that exists, treat every
draft-side idea as contaminated by the ~8k mismatch.
