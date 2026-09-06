# Glimmer B70: four-path optimization screen

## Scope and retained baseline

Requested paths: (1) draft-only full-vocabulary INT4 head, (2) draft-only
32768-token shortlist, (3) genuinely different INT4 MLP kernels/layout/fusion,
(4) speculation-depth tuning after the normalization fix and cheaper drafting.
Target quality, C8, native 131072 context, and graph-enabled normal streaming
remain fixed. No target-head quantization or target vocabulary restriction.
Native context is per request, not eight reserved full-context slots.

Starting commit: `95e6540`. Fresh exact-repository C8/K3 baseline:
**320.266 aggregate tok/s median**, five waves 319.573–320.887; **8/8** existing
completed-answer smoke checks passed. The experimental shortlist later exceeded
350 in the short C8 screen; full-capacity validation is recorded separately below.
These smoke checks are not a broad model-quality evaluation.

**Final outcome:** the shortlist is the strongest experimental candidate, but
its independent cold restart measured **347.464 tok/s**, below the initial
356.293. **A reproducible 350 tok/s result is not established.** Both draft-head
representations passed the retained queue/native/C8-capacity workloads, with
less total KV headroom. Keep the default recipe unchanged.

Remote artifacts (`R`):
`/home/mike/b70-evals/muse-glimmer/four-paths-20260905` on `mike@100.75.79.54`.
Baseline: `R/baseline/{throughput.json,quality.json,metrics-before.txt,metrics-after.txt,server.log}`.
All GPU experiments are serialized. Only session-owned disposable containers
are used; original checkpoints, drivers, power settings and services unchanged.

Pinned image:
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
Kernels 0.1.13.2, FP16 activations, symmetric GPTQ G128 target/draft, FP8 KV,
`DFLASH_KV_MODE=none`, memory utilization .90, batch tokens 2048, prefix cache off.
The default launcher is **not modified** by these experimental tools.

## Correctness boundary: pinned V2 worker

The retained server log explicitly says `Using V2 Model Runner`.
Actual installed source was inspected in no-GPU disposable containers:

- `v1/worker/gpu/spec_decode/dflash/utils.py:76–85`: target/draft head sharing.
- `v1/worker/gpu/spec_decode/speculator.py:132–141,309–340`: only probabilistic
  drafting caches draft logits; greedy drafting returns an argmax, with
  `draft_logits=None`.
- `v1/worker/gpu/spec_decode/rejection_sampler.py:146–165`: target sampling
  parameters are applied before rejection.
- `rejection_sampler_utils.py:145–189,628–665,811–823`: standard rejection
  accepts a deterministic proposal `d` with probability `p(d)`, otherwise
  samples the full target distribution with `d` excluded. For any `y != d`,
  `(1-p(d))*p(y)/(1-p(d)) = p(y)`. Here `p` includes target temperature,
  top-k/top-p, and model softcapping. This also covers stochastic target sampling,
  not merely target-greedy equality checks.
- `model_executor/models/muse_glimmer.py:1637–1645`: target multiplier and
  softcap remain untouched. Softcap changes probabilities even though
  mathematically monotonic; finite-precision ties also matter.

Required guards: greedy **draft**, standard rejection, no synthetic acceptance,
full unchanged target logits, native target token IDs before verification.
Probabilistic **target** sampling remains enabled. Compact shortlist indices
must never reach target embeddings/KV. EOS inclusion helps acceptance but full
residual/bonus target sampling already preserves termination correctness.

The opt-in `scripts/experimental/patch_glimmer_draft_head.py` changes only draft
loading and `_greedy_sample_draft`, not target computation or rejection. Four
source-hook unit tests pass. Applied twice to actual pinned source in an isolated
container: second application was a no-op; model logits, logits processor and
rejection sampler source remained byte-identical. This is not an end-to-end
candidate result.

## Test process

`C=/home/mike/b70-evals/muse-glimmer/20260905-concurrency`.
For each serving cell, use the real OpenAI-compatible SSE boundary at
`http://127.0.0.1:18080/v1`, served model `muse-glimmer-gptq`:

1. Start the exact baseline or explicitly isolated candidate launcher; require
   `/v1/models` healthy and advertised context 131072.
2. C8 shape warmup: `python "$C/instrument.py" --base http://127.0.0.1:18080/v1
   --model muse-glimmer-gptq --concurrency 8 --reps 1 --max-tokens 256
   --label CELL-warm --out "$R/CELL/warm.json" --log ""`.
3. Same command with `--reps 5`, label `CELL`, output `throughput.json`.
   Existing sky prompt, temperature 1, top_p .95, top_k 64, seed 42.
   Aggregate rate = summed completion tokens / concurrent wave wall time.
   Require eight successful streams per wave; record each wave and median.
4. Save `/metrics` before/after measured waves for acceptance deltas; retain
   startup log KV capacity. Run `python "$C/quality.py" 8 "$R/CELL/quality.json"`.
5. Before promoting a candidate, run `context-probe.py queue 2048 12 LABEL`,
   `context-probe.py retrieval 129000 1 LABEL`, and
   `context-probe.py stress 65536 8 LABEL 2048`. Require successful streams,
   retrieval, queue/replacement behavior, clean drain and no preemptions.
   The existing context client writes JSON, raw SSE and prompts under `C`.
6. Save `docker logs`, remove only the uniquely named owned container. Preserve
   experiment artifacts. Capture-only runs disable graphs and are never used as
   serving-throughput evidence. Unit/microkernel checks are filters, not
   substitutes for this public-boundary test.

Campaign-local `R/run-cell.sh` executes warmup, five waves, metrics, answer checks,
cleanup, and optionally these stress tests with `STRESS=1`.
K2/K4 copies change only `num_speculative_tokens` from the retained K3 launcher.

## Norm-fixed dense-head depth sweep

Five measured waves and 8/8 answer checks per cell:

- K2: **319.600** median (314.776–319.893), mean emitted per draft step
  `1 + 5365/4860 = 2.1039`, reported KV 609229 tokens.
- K3 control: **320.266** (319.573–320.887),
  `1 + 5682/4541 = 2.2513`, KV 609211.
- K4: **296.601** (295.550–297.008),
  `1 + 5836/4391 = 2.3291`, KV 609193.

K2/K3 are practically tied in this screen; K4's extra acceptance does not cover
its cost. Keep K3 as the control, then repeat on viable cheaper-head candidates.
These counters are per-draft emissions, not measured speedups or exact output
counts (request boundaries can truncate). Artifacts: `R/{k2,k4}/` as for baseline.
No context stress was run on these non-promoted dense-head K variants.

## Real draft activations and offline head screen

Opt-in helper: `scripts/experimental/glimmer_draft_head.py`. Capture-only
C8/K3 used the same target/draft with graph replay explicitly disabled; its
latency is not benchmark evidence. Fixed prompts in `R/capture/client.py`:
24 calibration and 16 disjoint held-out requests covering code, math, technical
explanation, prose and multiple languages. An additional **32758-token**
held-out retrieval passed, as did 8/8 existing completed-answer checks.

Bounded captures: 342 files, **6978 hidden-state rows**, including 2892 held-out
rows. Top-2048 dense draft-head candidates were saved with original IDs;
64 captures per label maximum, stride 4. Calibration votes are top-k occurrences
plus one extra vote for top-1, **not target-generated token frequency** or a
SpecVocab implementation. Selection used `cal-a/cal-b/cal-c` only; held-out
labels were excluded and frozen before testing. Selected 32768 sorted IDs;
EOS IDs 200001 and 200008 were protected (displacing ID 144277). This small
calibration set does not establish broad-domain acceptance or optimal pruning.

Validated same-process XPU screen with the original read-only BF16 checkpoint
head cast to FP16 exactly like serving:

- Dense reproduced saved draft top-1 **100%**.
- Symmetric G128 round-to-nearest INT4: **89.3093%** top-1 agreement vs dense.
  Against FP32 dequantized reference: max logit error **.0320435**, mean
  **.00441809**, top-1 agreement **99.5701%** (finite-precision near-ties).
  All rows passed finite checks and mixed tolerance `.05 + .01*abs(reference)`.
- Frozen shortlist: held-out dense top-1 coverage/agreement **92.2545%**.
  These are proposal agreement metrics, not target-model quality scores.
- Separate INT4 resident buffers **693428737 bytes** (661.3 MiB). Shortlist
  weight plus ID map **436469760 bytes** (416.25 MiB). Both retain the original
  dense target head; these counts exclude initialization/temporary workspace.

50 warmups + 100 measured calls per shape, using captured hidden states and the
same timing boundary (head + argmax, synchronized around repeated calls):

- M16: dense **4608.847 µs**, INT4 **1285.196**, shortlist **765.855**.
- M24: dense **4658.079 µs**, INT4 **1607.352**, shortlist **769.209**.
- M32: dense **4717.147 µs**, INT4 **1409.053**, shortlist **768.832**.

These are standalone means, not graph replay or end-to-end improvements.
The unusually slower M24 INT4 shape is measured, not extrapolated from M32.
No target-head operation or sampler was replaced. Only greedy draft proposals
use the new head and native-ID mapping.

Lead integration fixes: capture labels are refreshed after initialization,
per-label caps persist across changes, offline loader casts BF16 to serving
FP16, and guards use actual pinned mapping/config fields. Probe fails on empty,
nonfinite or malformed captures, invalid dense references and out-of-tolerance
INT4 numerical results. Proposal disagreement itself is recorded, not rejected
as a target-quality failure. **20/20 targeted tests pass in the pinned image**;
local Python without Torch runs 11 and skips 9.

Artifacts: `R/capture/{data,client.py,*-requests.json,long.log,quality.json,
server.log}`, frozen `R/head-probe/shortlist.json`, and
`R/head-probe/{run.sh,validate.sh,probe.json,probe-validated.json}`.
The first report predates fail-closed checks and the dense timing control;
`probe-validated.json` is the final numerical/timing screen. Commands:
`python glimmer_draft_head.py calibrate --captures /capture/data
--calibration-label-file /capture/calibration-labels.txt
--heldout-label-file /capture/heldout-labels.txt --output /artifacts/shortlist.json`,
then protect EOS as recorded in `run.sh`; run `probe --model-index
/model/model.safetensors.index.json --captures /capture/data --shortlist
/artifacts/shortlist.json --heldout-label-file /capture/heldout-labels.txt
--device xpu --m-values 16,24,32 --k-values 1 --warmup 50 --repeats 100`.
Here `k-values` means repeated timing calls, **not speculation depth**.


## Graph-enabled K3 head A/B

Normal retained settings with only the opt-in draft head added:

- INT4: **343.942 aggregate tok/s median**, five waves
  343.794, 354.914, 343.942, 355.007, 343.414; 8/8 answer checks.
  Mean emitted per draft step `1 + 5842/4414 = 2.3235`. Loading 19.31 GiB;
  KV capacity **528504 tokens**; captured graphs .83 GiB.
- Frozen shortlist: **356.293 median**, waves
  356.661, 356.550, 356.293, 355.564, 355.450; 8/8 answer checks.
  Mean emitted `1 + 5805/4440 = 2.3074`. Loading 19.07 GiB;
  KV capacity **560980 tokens**; captured graphs .84 GiB.

The shortlist exceeds 350 on this exact short streaming workload. Both preserve
C8 and the advertised 131072 per-request limit, but **total KV capacity falls**
from 609211 because the draft representation is additional resident memory.
Do not describe this as unchanged total context capacity. The default launcher
remains unchanged; promoting the memory/capacity tradeoff requires user approval.

Acceptance counters increased on the sky workload even though held-out draft
agreement fell. This is not evidence of broad acceptance improvement: altered
proposal choices also change sampled trajectories. The online distribution
correction remains authoritative. Capped throughput probes are not completed
answers; the separate eight-request smoke suite (three repeated tasks) passed.

Artifacts: `R/{int4-k3,shortlist-k3}/{throughput.json,quality.json,server.log,
metrics-before.txt,metrics-after.txt,memory-summary-*.json}`. All requests
succeeded. K2/K4 retuning and cold-start capacity stress follow this screen.


## Cheaper-head depth sweep and cold confirmation

K2/K3/K4 all preserve C8, native context and normal graph replay:

- Shortlist K2: **325.263** median (325.232–325.409), emitted/step 2.0328,
  8/8 automated answer checks.
- Shortlist K4: **301.681** (300.874–302.534), emitted/step 2.2197, 8/8.
- INT4 K2: **331.542** (331.528–331.752), emitted/step 2.0532, **6/8**
  automated checks. Two correct $18/day responses ended with 9 eggs; the
  existing grader extracted the last integer 9.
- INT4 K4: **322.472** (322.205–323.035), emitted/step 2.3410, 8/8.

Thus **K3 remains best** for both cheaper-head variants. No K increase or
concurrency/context shortcut was used to get the initial 350+ shortlist screen.

Independent cold restart, same code, settings and frozen shortlist:

- Shortlist K3: **347.464** median, five waves 347.920, 347.827, 347.464,
  347.004, 346.111; emitted/step 2.3521; KV 560980 unchanged from first launch.
- INT4 K3: **339.179** median, waves 336.345, 340.378, 339.631, 339.179,
  338.876; emitted/step 2.2170; KV 528504 unchanged.

The cause of the first-launch/cold-restart throughput spread was **not isolated**.
Do not promote the initial 356.293 as a stable 350 result. These are medians of
short fixed-prompt waves, not a workload-wide throughput guarantee.

Both cold runs scored **7/8** automatically. The shortlist response correctly
said $18/day then ended with 9 eggs; INT4 correctly said $18/day then $126/week.
`vllm-dflash-share-suite.py:209–214` uses the last integer when no `####` answer
marker is present. These are observed extraction false negatives; the raw
automated failures and nonzero exit codes are preserved. No grader was changed,
and manual inspection is not relabeled an automated pass.

## Cold graph-enabled queue and context results

Both K3 candidates passed all three real HTTP/SSE capacity journeys:

- **12 staggered 2042-token requests:** eight active at peak, waiting/replacement
  verified, all streams successful, zero preemptions and clean drain.
  Shortlist 27.95 s; INT4 25.98 s.
- **128994-token native retrieval:** all beginning/middle/end codes correct,
  natural stop, zero preemptions and clean drain. Shortlist 93.27 s;
  INT4 92.73 s.
- **8 × 65532 prompt + 2048 output:** all eight produced the full output budget,
  peak eight active, zero preemptions and clean drain. Shortlist **429.23 s**,
  peak KV **91.303%**; INT4 **411.35 s**, peak KV **97.070%**.

The last test retains the prior stress envelope; it does not establish support
for eight simultaneous full 131072-token requests. The extra head consumes KV
headroom: shortlist loses 48231 reported tokens (~7.9%); INT4 loses 80707
(~13.2%). This tradeoff remains unapproved for the default serving recipe.

Exact final orchestration: `bash "$R/run-cell.sh" int4-k4`, then
`STRESS=1 bash "$R/run-cell.sh" shortlist-k3-confirm` and the equivalent
`int4-k3-confirm`. The runner continued capacity checks after recording answer
grading failures, then preserved those nonzero quality exit codes. Final codes
were 0, 1, 1 respectively; capacity checks themselves passed.

Server/throughput/answer artifacts: `R/{shortlist-k3-confirm,int4-k3-confirm}/`.
Raw capacity evidence: `$C/fourpaths-{shortlist-k3-confirm,int4-k3-confirm}-
{queue,native,c8-64k}.json` plus matching raw SSE/prompt files. Final orchestration
is `R/final-checks.sh`. All session-owned containers were removed; final
`docker ps` was empty.

No genuine C>8 active-request experiment was run on this vLLM setup. The
12-request tests exercise an eight-active queue. Older llama.cpp C10/C12/C16
results are a different backend/recipe and cannot substitute for a new sweep.


## ESIMD INT4 MLP screen: rejected, no serving integration

Built `scripts/experimental/glimmer_esimd_probe.cpp` against the unmodified
pinned header, using `icpx -O2 -std=c++17 -fsycl -I/work/upstream
/work/glimmer_esimd_probe.cpp -o /work/glimmer_esimd_probe` in
`intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04`
(local image ID `aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`).
Device `Intel(R) Graphics [0xe223]`, driver `1.14.37020+3`, compiler 2026.0.0.
No extra compiler installation or host configuration was needed.

All six numerical cases passed: tiny M2/N16/K128; M2/N32/K256 distinct groups;
signed extremes M17 tail; all-zp8 exactly zero; exact gate/down M32 shapes.
All output elements were finite. Large shapes checked 2048 sampled outputs
against FP32 dequantized reference: maximum absolute errors **0.0003945 gate**
and **0.0004498 down**. Mixed screen tolerance .05 + .02*abs(reference); this
synthetic numerical filter is not proof of full-model equivalence.

After 500 warmups and 500 timed iterations:

- One bank: gate **420.175 µs**, down **328.247 µs**.
- Three resident weight banks: gate **421.545 µs** (MAD 7.205),
  down **328.627 µs** (MAD .451). Preprocessing excluded.
- Same-session exact-revision oneDNN `benchdnn` control: gate **278.024 µs**
  average, down **146.804 µs** average (minima 270.104/142.916).

ESIMD timings include host dispatch plus synchronized wait; benchdnn uses its
own performance measurement. These are not identical timing boundaries and
must not be quoted as a precise serving speedup ratio. The large regression,
also versus prior synchronized Torch oneDNN results, rejects this unmodified
kernel for integration. In particular, the down projection's unsplit K path
does not exploit an obvious extra parallel dimension. This screen did not
implement a custom retuned ESIMD kernel, reorder the serving model, or fuse
MLP activations. It adds a distinct negative result beyond the prior tile sweep.

Runs: `/work/glimmer_esimd_probe --warmup 500 --iters 500 --weight-banks 1`
and `--weight-banks 3`. Same device, sequential, normal clocks. oneDNN command:
`benchdnn --matmul --mode=P --engine=gpu --dt=f16:u4:f16 --stag=ab --wtag=ba
--dtag=ab --attr-scales=wei:3:f16:128x1 --attr-zero-points=wei:common:8:s8
--attr-fpmath=f16:true --attr-scratchpad=user 32x6656:6656x39936
32x19968:19968x6656`. P-mode summary is not a fresh oneDNN numerical check;
its exact-revision correctness was established in the prior campaign.

Artifacts: `R/esimd/{build.log,results.jsonl,warm-one-bank.jsonl,
warm-three-bank.jsonl,onednn-control.log,repeat.sh}`. All containers removed.

## Fresh primary-source research affecting the approach

- [vLLM speculative decoding documentation](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/speculative_decoding/README.md):
  memory-bound performance is workload-dependent; mathematical distribution
  preservation does not imply bitwise equality across numerical/batch changes.
- [VocabTrim](https://arxiv.org/html/2506.22694),
  [frequency calibration](https://github.com/SamsungLabs/SpecVocab/blob/main/src/specvocab/utils/compute_frequencies.py),
  [pruning implementation](https://github.com/SamsungLabs/SpecVocab/blob/main/src/specvocab/utils/prune_vocabulary.py):
  frequency-selected draft rows with explicit ID mappings and protected EOS.
  Target-generated calibration is the paper baseline; any draft-activation
  frequency proxy in this experiment must be identified as such, not called a
  reproduction. Freeze selection before held-out evaluation.
- [Intel INT4 ESIMD header, pinned revision](https://github.com/intel/llm-scaler/blob/ede4320a24a67f664fb53081d2623f9efe9a75b7/vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/int4_GEMM.h):
  Apache-2.0, unsigned nibbles minus 8, G128, FP16 dequantization/DPAS operands,
  FP32 accumulation. It is not integer DPAS. Standard GPTQ requires byte-layout
  and scale-layout conversion; no arbitrary zero points or act-order support.
  Both exact M32 projections dispatch K_THREADS=1. Large accumulator register
  footprint and dequantization may erase benefits; benchmark, do not assume.
- [Pinned build integration](https://github.com/intel/llm-scaler/blob/ede4320a24a67f664fb53081d2623f9efe9a75b7/vllm/custom-esimd-kernels-vllm/setup_gemm_only.py):
  the kernel header itself is Torch-independent, permitting a smaller standalone
  SYCL screen before considering a serving extension. Build inputs and upstream
  license are retained under `R/esimd/upstream`.
