# Glimmer B70: actual active-concurrency sweep

## Scope

User request: “let's do a sweep and figure out how high we can get token rates
by boosting concurrency”. This is separate from the C8/350 tok/s goal.
No current-vLLM C>8 result existed before this campaign; twelve queued clients
with eight active slots did not constitute C12 throughput.

Starting repository commit: `75e37b9`. The default launcher remains unchanged.
The sweep uses the strongest experimental draft-only candidate from
[the four-path screen](glimmer-b70-four-paths-20260905.md): frozen 32768-ID
shortlist, DFlash K3. Target dense logits, softcap and standard one-hot rejection
remain unchanged. The shortlist costs additional resident memory; no default
promotion of that capacity tradeoff is implied.

Remote campaign (`R`) on `mike@100.75.79.54`:
`/home/mike/b70-evals/muse-glimmer/concurrency-sweep-20260906T002815Z`.

Fixed settings: pinned image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
kernels 0.1.13.2, GPTQ symmetric G128 target/draft, FP16 activations, FP8 KV,
XPU graphs, `DFLASH_KV_MODE=none`, K3, **131072 native per-request context**,
memory utilization .90, batch tokens 2048, prefix cache off.
Only `--max-num-seqs C` and matching client concurrency change between cells.
The original checkpoint files are read-only; no host drivers, services, power
settings or hardware allocation change.

## Fresh external guidance and test process

Fresh primary-source research:
[vLLM V1 optimization guide](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/configuration/optimization.md).
Relevant conclusions: KV exhaustion can preempt/recompute requests and hurt
latency; monitor preemption metrics; larger concurrency can expose CPU input,
scheduling and streamed-output bottlenecks. Batch-token tuning changes
throughput/TTFT/ITL tradeoffs, so it is held fixed in this concurrency-only sweep.
These recommendations are not evidence of B70 speedups.

Each cell uses the real OpenAI-compatible streaming boundary at
`http://127.0.0.1:18080/v1`, model `muse-glimmer-gptq`:

1. Start a uniquely named disposable container via `R/cC/start.sh`; require
   healthy `/v1/models` and advertised max context 131072. Exit early if the
   owned container dies during readiness; never silently reduce context or
   change graph/memory settings to make a cell start.
2. Existing `C=/home/mike/b70-evals/muse-glimmer/20260905-concurrency` clients:
   `python "$C/instrument.py" --base http://127.0.0.1:18080/v1
   --model muse-glimmer-gptq --concurrency C --max-tokens 256 --log ""
   --reps 1 --label cC-warm --out "$R/cC/warm.json"`.
3. Repeat with `--reps 5`, label `cC`, output `throughput.json`. Same sky prompt,
   temperature 1, top_p .95, top_k 64, seed 42. Aggregate throughput is summed
   completion tokens / concurrent wave wall time, **including reasoning tokens**.
   These capped streams are not completed-answer quality scores.
4. `R/measure.py` samples public `/metrics` every .2 s. Require the observed
   scheduler-running peak to equal C, all C×5 streams to return 256 tokens with
   visible content or reasoning, no errors/preemptions, and a clean drain.
   Scheduler-running includes prefill; this does not prove every device step
   executed one fully occupied C-sized decode batch.
5. Run `python "$C/quality.py" C "$R/cC/quality.json"`: C simultaneous
   completed-answer requests, using the existing three repeated tasks. Preserve
   raw grader failures. Stop the sweep for incomplete/empty answers, but record
   scoring failures separately rather than changing the grader or discarding
   successful performance waves.
6. Save server logs, advertised KV capacity and graph allocation, then remove
   only that owned container. `R/run-cell.sh C` orchestrates steps 1–6.
7. Planned follow-up: cold-repeat the best region and test native retrieval
   and longer concurrent inputs through HTTP/SSE. **Not run at session close.**
   Native max context is not C reserved full-context slots. Longer-input results
   must be reported separately from the short throughput workload.

Artifacts per cell: `R/cC/{start.sh,models.json,warm.json,throughput.json,
summary.json,scheduler-metrics.json,metrics-before.txt,metrics-after.txt,
quality.json,quality.log,server.log}`. Exact committed experimental code is
copied under `R/probes`; the fixed vocabulary is `R/shortlist.json`.
Latency p95 uses nearest rank over all requests in the five measured waves.
Timing includes the monitoring overhead consistently for every cell.

### Workload interpretation

This is a concurrent **burst / short-prefill / fixed-output-length** test,
not continuous arrivals or a varied coding-agent workload. Every request uses
the same 140-character sky prompt, 83 tokens after chat formatting:

> Explain in detail why the sky is blue. Cover Rayleigh scattering, wavelength dependence, and why sunsets look red. Write complete sentences.

Requests are submitted as a concurrent thread-pool burst, not through an exact
start-time barrier. The next wave starts only after all requests finish.
At C48, each wave produced `48 * 256 = 12288` completion tokens.
One wave took 15.4723 s, giving 794.194 aggregate tok/s. The denominator includes
scheduling/prefill, generation and streaming/client overhead, not isolated
kernel durations. Prefix KV caching is disabled despite the repeated prompt.

**All 2140 measured responses across C8–C128 reached the length limit while
still producing reasoning; none contained final-answer content.** The rate counts reasoning
tokens. Separate completed-answer requests use a 2048-token budget and three
repeated math/code/prose tasks; their scores do not produce the throughput rate.

The native 131072 setting is not the measured prompt length: these throughput
inputs are only 83 tokens. Do not extrapolate these rates to long-context
sessions, varied prompts/lengths, tool loops or uninterrupted arrival traffic.

### Versioned execution recipes

The exact executed scripts are retained in
`scripts/experimental/glimmer-concurrency-sweep/{measure.py,run-cell.sh,start-template.sh}`.
These are historical campaign snapshots with fixed B70 paths, not a new default
launcher or a general installer. The template is the C8 cell. Other cells change
`--max-num-seqs`, the C label and the per-cell `/artifacts` mount; all other
settings remain fixed. Remote `R/cC/start.sh` copies are the exact cell commands.
Inspect paths before replay: reusing a cell directory overwrites its result
files. Model weights, frozen shortlist, raw generations and runtime telemetry
remain in the referenced remote campaign/model paths, outside git.

## Completed initial sweep

All cells below reached the named active-request peak, completed all five
throughput waves, had zero measured-wave preemptions, no metric errors and a
clean drain. Rates are medians of five aggregate wave rates; latency and
individual decode rates are medians over all measured requests.

- **C8: 345.346 tok/s** (342.753–346.291). TTFT .429 s; request 5.644 s;
  individual decode 49.073 tok/s. KV 560980 tokens; graph memory .84 GiB.
  Automated answer checks 8/8.
- **C12: 449.944** (449.489–451.749). TTFT .621 s; request 6.710 s;
  individual decode 42.039. KV 560927; graphs 1.17 GiB. Checks 11/12.
- **C16: 514.712** (513.525–516.654). TTFT .790 s; request 7.712 s;
  individual decode 36.988. KV 560891; graphs 1.43 GiB. Checks 15/16.
- **C24: 482.247** (481.289–484.216). TTFT 1.167 s; request 12.438 s;
  individual decode 22.678. KV 560819; graphs 2.02 GiB. Checks 24/24.
- **C32: 583.812** (583.349–586.580). TTFT 1.452 s; request 13.597 s;
  individual decode 21.146. KV 558415; graphs 2.57 GiB. Checks 32/32.
- **C48: 794.194** (787.973–797.696). TTFT 2.169 s; request 14.938 s;
  individual decode 19.994. KV 555885; graphs 3.17 GiB. Checks 48/48.
- **C64: 823.297** (818.444–834.464). TTFT 2.866 s; request 19.213 s;
  individual decode 15.656. KV 555687; graphs 3.79 GiB. Checks 64/64.
- **C96: 840.810** (836.927–847.619). TTFT 4.308 s; request 28.346 s;
  individual decode 10.586. KV 553050; graphs 3.83 GiB. Checks 95/96.
- **C128: 827.754** (825.181–833.580). TTFT 5.129 s; request 38.516 s;
  individual decode 7.680. KV 548080; graphs 3.89 GiB. Checks 127/128.

C12/C16/C96/C128 each had one observed grader extraction false negative: the
answer correctly stated $18/day but ended with 9 eggs, which the unchanged
last-integer extractor selected. All four responses completed naturally with
visible content.
Raw automated failures remain in `quality.json`; they are not relabeled passes.

Initial mean emitted tokens per draft step stayed near 2.24–2.29 at C8–C32
(2.2850, 2.2471, 2.2584, 2.2413, 2.2604). The C24 throughput dip is therefore not
explained by a large acceptance collapse. Its kernel/scheduling cause was not
isolated; do not assume monotonic concurrency scaling or interpolate a peak.

## Final outcome and closeout

**C96 had the highest observed median: 840.810 aggregate tok/s**, about 2.43×
C8. C128 declined to 827.754 tok/s while median request time rose to 38.516 s
and individual decode fell to 7.680 tok/s. No C>128 cell was attempted.

**C48 is the observed throughput/latency knee for this workload**: 794.194
aggregate tok/s (2.30× C8), only 5.5% below the best median, with individual
decode near 20 tok/s versus 10.6 at C96. C48→C64 adds only about 3.7% aggregate
throughput for 33% more concurrency. This is not a production recommendation
or proof of a global maximum.

All nine cells completed: 2140 measured capped streams and 428 separate
completed-answer requests. Automated answer checks passed 424/428; the four
extraction false negatives described above remain recorded. This repeated
three-task smoke suite is not a broad quality evaluation.

The independent cold repeat, high-concurrency native retrieval, longer-input
stress, mixed-prompt and continuous-arrival follow-ups were **not run**.
The user requested session closeout after the already-running C128 cell;
these follow-ups are deferred, not pending jobs. Native 131072 was configured
throughout but was not exercised by the 83-token throughput inputs.

All experiment containers were removed; the final remote `docker ps` returned
no running containers. No serving defaults were changed or promoted. The
report and execution snapshots are versioned; raw evidence remains at `R`.
