# Qwen3.8 B70 P1 microbench and long-context multi-row verification

**Status:** research specification only. This document does not implement a bench, start a
server, change the evaluation contract, or make a speed claim.

## Scope and evidence labels

This is the P1 measurement-harness slice and GPT-pro backlog item 10. Composer may
implement this specification later in a disposable research container. It must not be
used as a reason to replace the live BF16 service, edit XPU kernels, run Harbor/Prime,
or change the `98,304 / 32,768` evaluation contract.

Use these labels in the eventual run manifest and report:

- **Measured:** a result already present in the local handoffs or read-only control
  artifacts. It is not re-measured by this document.
- **Source:** a behavior or shape read from a pinned upstream file. It is not evidence
  that the installed compiled kernel has the same implementation.
- **Hypothesis:** a question for the bench; never report it as a result.
- **Decision:** a pre-registered keep/park/kill rule applied to the measured rows.

### Existing measured facts that constrain the design

- **Measured:** the frozen BF16-draft versus draft-INT4 C1 rows are
  `44.03 / 16.91 / 0.639` versus `54.67 / 17.83 / 0.644 tok/s` for short,
  medium, and approximately 98k contexts. The cap-64 row is stable, while the
  candidate is still quality-rejected by the approximately 8k greedy parity gate.
  See `docs/qwen38-b70-next-session-handoff-20260823.md:9-31` and
  `docs/qwen38-b70-mlxfast-research-progress-20260823.md:18-49`.
- **Measured:** the approximately 98k row is already about 85x slower than the
  short warm row. The `17.10 tok/s` agent-eval median is a mixed prefill/decode,
  thinking, and API-duration measure, not a decode bench. See
  `docs/qwen38-b70-research-plan-20260824.md:21-42` and
  `docs/qwen38-b70-speed-improvement-handoff.md:25-37`.
- **Measured:** graph capture cap 64 is the known stability boundary; cap 128
  failed startup and no-graph was much slower. This spec therefore treats graph mode
  as an experimental arm, not as a hidden variable. See
  `docs/qwen38-b70-mlxfast-research-progress-20260823.md:32-49`.
- **Measured:** the existing controls and logs under
  `~/b70-evals/qwen38-b70-gptq-int4-mtp4/` are read-only inputs for
  comparison. New runs use sibling directories and never overwrite them.

## Source map and nomenclature

The source references below are the implementation surfaces Composer should audit
before writing a runner. The source map is intentionally explicit because the
installed `vllm-xpu-kernels 0.1.12.3` is compiled-only in the current environment.

1. **MTP model path.** At vLLM commit
   `ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`,
   `vllm/model_executor/models/qwen3_5_mtp.py`:
   `Qwen3_5MultiTokenPredictor.forward` (lines 139-183) selects
   `spec_step_idx % num_mtp_layers`, calls one decoder layer, and normalizes the
   result; `Qwen3_5MTP.forward` (279-291) calls the predictor; and
   `Qwen3_5MTP.compute_logits` (293-298) applies the MTP LM head and logits
   processor. These are **source** boundaries for the draft/MTP benches.
2. **Target model path.** At the same vLLM commit,
   `vllm/model_executor/models/qwen3_next.py`:
   `Qwen3NextDecoderLayer.forward` (485-556),
   `Qwen3NextForCausalLM.forward` (788-800), and
   `Qwen3NextForCausalLM.compute_logits` (839-843). The target LM-head bench must
   invoke the loaded target model's head, not silently substitute the MTP head.
3. **Runtime GDN wrapper.** At the same vLLM commit,
   `vllm/_xpu_ops.py:_gdn_attention_core_xpu_impl` (116-201) collects
   `GDNAttentionMetadata` and calls `torch.ops._xpu_C.gdn_attention` with the
   spec/non-spec indices, accepted-token tensor, states, and dimensions. The
   metadata shape contract is in
   `vllm/v1/attention/backends/gdn_attn.py:43-73,210-524,526-547`.
4. **Upstream XPU GDN entry points.** The fetched `main` tree of
   `https://github.com/vllm-project/vllm-xpu-kernels` had tree SHA
   `4543b580fecca68a7dd54ddaf6e444dc5f11a6a4`. Relevant source names are
   `csrc/xpu/gdn_attn/gdn_attn_interface.cpp:causal_conv1d_spec` and
   `gdn_attention`, `csrc/xpu/gdn_attn/gated_delta_rule.hpp:kernel_launcher_spec`
   and `gated_delta_rule`, and
   `benchmark/benchmark_gdn_attn.py:Workload`, `_make_spec_inputs`, and its
   `mode="spec"` workload. The upstream benchmark explicitly distinguishes
   `decode` (one new token) from `spec` (drafts per sequence). This is **source
   context only**; the runner must record the exact installed checkout/package
   and not assume current `main` equals 0.1.12.3.
5. **XPU documentation.** The fetched developer-preview page
   `https://docs.vllm.ai/en/latest/models/hardware_supported_models/xpu/` lists
   Intel Arc Pro B-Series as validated hardware and lists Qwen3 and GPTQ model
   entries. It does not establish this B70 model's MTP or long-context performance.

In this document, `B` is logical batch size and is `1` for isolated component rows.
`C` is the number of historical context tokens present before the active query rows.
`M` is the number of active query rows passed to the measured operation, including the
current row: `M=1` is ordinary decode and `M=4` is a four-row verify. Every artifact
must also record `num_speculative_tokens`, `num_accepted_tokens`, and the actual
metadata-derived token count; never infer `M` from a marketing name such as MTP-4.
For GDN, the upstream shape uses `num_spec_tokens = num_speculative_tokens + 1`, so
both values must be written explicitly when translating the harness convention.

## Part A — P1 microbench matrix

### A0. Harness contract

The later runner has one process per row family and one visible B70. It loads the exact
pinned model revision once, preallocates all timed inputs and outputs, and runs the
component without an HTTP server. A disposable server is allowed only for the separate
public-API C1 arm. No timed row may include model download, tokenizer setup, memory
allocation, prompt construction, graph capture, or an accidental host read.

The frozen BF16-draft row is mandatory. Where the disposable image can load the measured
draft-INT4 overlay, rerun the draft-head and MTP rows as a separate `weight_variant`
with the target GPTQ weights unchanged; never pool BF16 and draft-INT4 samples. The
draft-INT4 quality rejection remains a gate for deployment, not a reason to omit its
isolated attribution row.

The primary context ladder is:

- `C=4,096`, `8,192`, `32,768`, and `98,304` tokenizer-verified tokens;
- `C=131,072` may be an optional ceiling/stress row only when the disposable runtime
  has the required headroom;
- `C=200,000` is an optional separate stress experiment, not a product or eval row and
  not comparable to the 131,072 serving ceiling without an explicit configuration note.

For context-independent projections, use `C=0` with fixed captured activations and
record a separate `context_replay` field if the activation was captured after a long
context. For context-sensitive attention and GDN rows, fill real cache/state storage
at each `C`; do not synthesize a short cache and label it 98k.

Default timing policy (overridable only by a recorded CLI argument): use 10 warmups,
30 samples at 4k/8k, 10 samples at 32k, and 5 samples at 98k/optional stress lengths.
Warmups must include one post-input-preparation synchronization. Use XPU device events
for device elapsed time and a monotonic host timer for the enclosing call. Synchronize
only at the declared timing boundaries. Record every sample, not only a median.

Each component is run in two graph arms when the path supports graph execution:

- `no-graph`: eager/enforce-eager, with no graph capture;
- `cap64`: the existing cap-64 graph configuration, with the exact effective vLLM
  compilation settings recorded.

`cap64` is a capture-size limit, not the active `M` and must not be treated as one.
Direct isolated kernels default to `no-graph`; a graph replay variant is a separate
row. For cap-64 rows, exclude capture and allocation from steady-state device timing,
but save capture time, captured shape, active tokens, padded tokens, and graph replay
count. Never average cap-64 and no-graph samples together.

### A1. Matrix summary

| Bench ID | Isolated boundary | `M` / `C` rows | Inputs and pinning | Required counters | Graph arms and primary artifact |
| --- | --- | --- | --- | --- | --- |
| `draft_head` | Loaded draft/MTP projection and its reduction, before target verification | `M=1..9`; `C=0`; optional captured activations from each primary `C` | Frozen BF16 draft weights; optional separately labeled draft-INT4 overlay; contiguous `[M,H]` hidden states; fixed seed plus one captured-hidden replay | GEMM, logits allocation, reduction/top-k separately; device/host time; launches; bytes; peak memory; `weight_variant` | Eager/no-graph component row; cap-64 replay only as a separately captured full-path variant. `components/draft_head/samples.jsonl` |
| `mtp_layer` | One selected `Qwen3_5DecoderLayer`/MTP layer, including its declared state updates but excluding the following LM head | `M=1,4,9`; `C=4k,32k,98k`; `spec_step_idx` fixed and recorded | Actual layer weights; captured hidden/input embeddings, positions, attention/GDN metadata, and states for `B=1` | Layer total plus attention/GDN, norm, MLP, residual, metadata, state movement, launches and syncs | Eager first; cap-64 only as a separate full-layer replay. `components/mtp_layer/samples.jsonl` |
| `verify_forward` | Full MTP verify forward through the predictor, without target LM-head timing | `M=1..9`; `C=4k,8k,32k,98k`; fixed `spec_step_idx` sequence | Same token/hidden/state fixture for each width; active rows exactly `M`; no hidden padding in eager arm | Per-layer and total device time; active/padded tokens; layer launches; GDN/attention time; output checksum; acceptance input metadata | `no-graph` and `cap64`, capture excluded. `components/verify_forward/samples.jsonl` |
| `target_lm_head` | Target model `lm_head`/`compute_logits`, excluding sampler and accept/commit | `M=1..9`; `C=0`; optional hidden captures from 4k/32k/98k | Actual target head, contiguous `[M,H]` hidden states, output `[M,V]`; target head is not the MTP head | Projection versus logits materialization; reduction if included; output allocation; launches; bytes; peak memory | Eager component row; graph replay only if the real target path captures it. `components/target_lm_head/samples.jsonl` |
| `gdn_spec` | XPU GDN spec-decode call, with conv and delta stages separated when possible | `M=1,4,9`; `C=4k,32k,98k` state fixtures; `B=1`, `num_spec_decodes=1` | `projected_states_qkvz`, `projected_states_ba`, `z`, conv state, SSM state, weights, `spec_query_start_loc=[0,M]`, contiguous `spec_token_indx`, `[1,M]` state indices, int32 accepted count | `causal_conv1d_spec` time, `gated_delta_rule_spec` time, fused-call time, state read/write bytes, launches, accepted count, D2H/sync | Eager direct-op row; cap-64 only through the real metadata path. `components/gdn_spec/samples.jsonl` |
| `attention_context` | One representative full-attention layer's paged attention, not GDN | `M=1,4`; `C=4k,32k,98k`; optional 128k; `B=1` | Real Q/K/V shapes and page table from the target model; prefilled FP8 KV cache and scales; same historical pages for M=1/M=4 | Device/host time, logical and allocated K/V bytes, page count, scale bytes, estimated bandwidth, profiler memory reads if available, launches | `no-graph` and cap-64 replay separately. `components/attention_context/samples.jsonl` |
| `accept_commit` | Device accept decision, device commit/rollback, and host-observable result as separate stages | `M=1..9`; `C=0` for token decisions plus long-context captured metadata; `B=1` and one observed-acceptance replay | Captured target/draft token IDs/logits and probabilities; device accepted-count/index tensors; state-index and GDN state snapshots | Each stage time; `.item()` calls, `.tolist()` calls, `cpu()`/`.to(cpu)` D2H calls and bytes, explicit `synchronize` calls, state-copy bytes, accepted count | Eager only for the first isolation; full graph replay as a separate commit row. `components/accept_commit/samples.jsonl` |

The table's files are JSONL sample streams; the common schema below is mandatory for
every row. A bench may add fields, but it may not omit the dimensions needed to compare
`M`, context, graph mode, and source/runtime identity.

### A2. Per-bench protocol

#### `draft_head`

- Load the actual draft/MTP projection from the pinned checkpoint. Feed contiguous
  hidden states of shape `[M,H]` for each `M=1..9`, first from a fixed deterministic
  tensor and then from a captured real MTP activation. Keep the hidden tensor, dtype,
  weights, and seed fixed across graph arms.
- Time the projection alone, then the projection plus logits materialization, then the
  existing draft reduction/top-k/argmax if it is part of the production proposal path.
  These are separate `stage` values, not one unexplained total.
- Record `projection_time_us`, `reduction_time_us`, `logits_bytes`, `weight_bytes` and
  launch count. This gives item 1 a way to distinguish head work from reduction work;
  it is not a claim that either is currently dominant.

#### `mtp_layer`

- Select one real MTP decoder layer and record its index/name and `spec_step_idx`.
  Supply prebuilt positions, hidden/input embeddings, attention metadata, and cache or
  GDN states. Do not run the whole model and call the result a one-layer time.
- Use the same fixture at `M=1,4,9` and at 4k/32k/98k. Snapshot state before each
  sample, restore it from a preallocated copy, and hash the post-state. A state hash
  mismatch is a correctness failure, not a timing outlier.
- Instrument child stages where the runtime permits: input normalization, QKV/linear
  projections, full attention or GDN, MLP/MoE, output projection, residual/norm, and
  metadata/state movement. If a child cannot be isolated, record `unattributed` rather
  than assigning its time to another stage.

#### `verify_forward`

- Invoke the real predictor boundary represented by
  `Qwen3_5MultiTokenPredictor.forward`, with active leading dimension exactly `M` in
  the no-graph arm. Repeat `M=1..9`; do not pad an eager tensor and report the padded
  size as active work.
- Record the layer sequence and the per-layer timings. The verify-width ladder is
  independent of static MTP depth: `M` is active token rows, while `spec_step_idx` is
  the selected MTP layer. Both are required fields.
- In cap-64 mode, record `graph_capture_size`, `active_tokens`, `padded_tokens`, and
  whether the run is capture or replay. Compare replay-to-replay only; capture cost is
  its own artifact.

#### `target_lm_head`

- Invoke the target model's `Qwen3NextForCausalLM.compute_logits`/loaded LM head
  boundary with `[M,H]` hidden states. Do not route through
  `Qwen3_5MTP.compute_logits`, which is the draft/MTP head boundary.
- Keep logits on device for the projection row. If a separate reduction or sampler is
  timed, label it as a child stage and do not include it in `target_lm_head` totals.
  Record output shape `[M,V]`, allocation bytes, dtype, and a stable device checksum.
- This row is context-independent in math; a long-context hidden capture is a useful
  distribution check but is not a long-context attention measurement.

#### `gdn_spec`

- Prefer the real Python-to-XPU path through `_gdn_attention_core_xpu_impl` and
  `torch.ops._xpu_C.gdn_attention`. If the installed package exposes split operations,
  time the conv and delta calls separately as well as their fused legacy boundary.
- Construct the source-shaped metadata exactly: `num_spec_decodes=1`,
  `spec_query_start_loc=[0,M]`, `spec_token_indx` length `M`,
  `spec_state_indices_tensor` shape `[1,M]`, and int32 `num_accepted_tokens` shape
  `[1]`. Use a fixed valid accepted count plus one replay of an observed count; record
  both. Restore conv/SSM state between samples.
- The GDN spec operation has recurrent-state and causal-convolution work; it is not a
  historical paged-KV attention scan. Report its time separately so an item-10
  decision does not accidentally attribute GDN work to full attention.

#### `attention_context`

- Choose one representative **full-attention** layer from the target model and record
  the layer index, query heads, KV heads, head dimension, page size, dtype, and KV
  format. Populate exactly `C` historical tokens in the same paged layout as the
  target runtime. Use FP8 KV plus its scale metadata for the baseline arm; a BF16 KV
  diagnostic is optional and must be a separate `kv_format` row.
- For `M=1`, issue one decode query. For `M=4`, issue four causally ordered query
  rows against the same historical pages; restore the cache before each sample so
  one row does not silently become the next sample's history. Record the exact
  effective length for each row (`C`, `C+1`, ...) and the page padding.
- Time the attention operation itself before any target projection, sampler, or commit.
  If the backend cannot be invoked independently, run a full one-layer boundary and
  report the enclosing boundary plus the measured children; do not call the entire
  98k API request "attention time".

#### `accept_commit`

Run four explicitly named variants on the same captured decision inputs:

1. device accept decision only, returning device tensors;
2. device accept plus device commit/rollback, no host read;
3. variant 2 plus one scalar `.item()` for accepted count;
4. variant 2 plus token-ID `.tolist()` and explicit `cpu()`/`.to("cpu")` D2H.

Wrap or otherwise count `.item()`, `.tolist()`, `cpu()`/D2H, and explicit
`synchronize` boundaries. If a framework call cannot be counted without changing its
path, report that limitation and use profiler/API traces rather than replacing the
path with a mock. Save pre/post token/state hashes and the accepted count for every
variant. The difference between variants is a synchronization/D2H observation, not a
claim that a replacement is safe.

### A3. Common counters and graph-confounder controls

Every sample must contain:

- host elapsed seconds and XPU-event elapsed microseconds, with timing boundary names;
- `M`, `B`, `context_tokens`, `active_tokens`, `padded_tokens`,
  `num_speculative_tokens`, `num_accepted_tokens`, layer/stage name, dtype, KV
  format, and graph mode;
- kernel/operation launch count, peak allocated/reserved device memory, allocation
  count if available, and output/state checksums;
- `sync_calls`, `.item_calls`, `.tolist_calls`, D2H call count, D2H bytes, and
  host-visible result size;
- logical/allocated K/V bytes and scale bytes for attention rows, plus
  `kv_bytes_estimate` and `kv_bandwidth_estimate_gib_s` (an estimate, not a hardware
  counter);
- profiler artifact references. If hardware memory-read counters, EU occupancy, or
  cache hit/read counters are unavailable, write `null` and preserve the raw profiler
  limitation rather than inventing a value.

The attention estimate is registered as:

```text
logical_kv_bytes = effective_context_tokens
                   * 2                         # K and V
                   * num_kv_heads * head_dim
                   * kv_bytes_per_element
                   + scale_metadata_bytes

shared_lower_bound = logical_kv_bytes
independent_upper_bound = M * logical_kv_bytes
estimated_bandwidth = selected_estimate / device_time_seconds
```

`effective_context_tokens`, page padding, FP8 scale layout, and the selected lower or
upper bound must be recorded. This formula does not prove what the kernel loaded; a
profiler memory-read counter or source/kernel evidence is needed for that conclusion.

Controls against graph-capture confounding are mandatory:

- allocate and initialize tensors before timing; exclude tokenizer, H2D, allocation,
  graph capture, and first-use compilation from steady-state samples;
- use the same logical tensors and metadata for no-graph and cap-64 arms, restoring
  mutable cache/state before each sample;
- record capture size, active/padded tokens, graph replay count, and graph memory;
- run direct component rows eagerly unless a separately named replay is requested;
- use device events for device work and a host timer for the synchronization boundary;
  never use an implicit host read to stop a device timer;
- keep `C`, `M`, layer, dtype, KV format, seed, and fixture hash constant across the
  two arms. A graph mode that cannot represent a row is `unsupported`, not a dropped
  sample.

### A4. Artifact schema

Each run gets a UTC directory and a manifest before timing starts. The minimum
`manifest.json` shape is:

```json
{
  "schema_version": "b70-p1-1",
  "run_id": "UTC-name",
  "bench": "attention_context",
  "status": "complete|partial|unsupported|failed",
  "source_refs": [{"path": "...", "revision": "...", "symbol": "..."}],
  "environment": {"image_digest": "...", "vllm_sha": "...", "model_revision": "..."},
  "config": {"graph_mode": "no-graph|cap64", "warmup": 10, "repeats": 5},
  "input": {"B": 1, "M": 4, "context_tokens": 32768, "fixture_sha256": "..."},
  "artifacts": {"samples": "samples.jsonl", "stdout": "stdout.log", "profiler": null}
}
```

Each `samples.jsonl` row must add the common counters from A3, including
`sample_index`, `stage`, `host_seconds`, `device_microseconds`, `graph_capture_size`,
`active_tokens`, `padded_tokens`, `num_speculative_tokens`, `num_accepted_tokens`,
`kv_format`, `kv_bytes_logical`, `kv_bytes_allocated`, `kv_bytes_estimate`,
`kv_bandwidth_estimate_gib_s`, `kernel_launches`, `peak_memory_bytes`, `sync_calls`,
`item_calls`, `tolist_calls`, `d2h_calls`, `d2h_bytes`, `output_checksum`,
`state_before_checksum`, `state_after_checksum`, and `error`/`unsupported_reason`.
Use `null` for inapplicable counters and keep units in field names or a manifest
unit table; do not mix milliseconds and seconds.

`summary.json` is derived from the sample stream and contains counts, median and
p10/p90 values, confidence/interval method, failed/unsupported rows, and the exact
source sample paths. It must not discard raw samples. `stdout.log`, `stderr.log`,
XPU profiler output, and server logs (for C1 only) are retained outside a disposable
container's lifecycle.

### A5. Public-API C1 reuse

The component rows answer attribution questions; they do not replace the existing
public endpoint control. Later Composer runs must add a separate `c1/` arm that:

- launches the disposable service with the frozen BF16 C1 configuration first, then
  any experimental row in a fresh container; the live BF16 service is not touched;
- reuses tokenizer-verified, single-request inputs at exactly 4k, 8k, 32k, and 98k;
  uses the same payloads, sampling, seed, `max_tokens`, and one-at-a-time C1 order in
  both arms; and records the prompt fixture hash;
- records per request HTTP status, prompt/completion tokens, TTFT, wall time, output
  tokens/s, finish reason, acceptance length if exposed, and server-side logs;
- repeats enough times for an interval rather than treating one sample as a result.
  The existing short controls use five iterations; the new runner must record its
  actual repeat count and must not silently combine different prompt families;
- keeps 128k (and, only if deliberately configured, 200k) as optional stress rows.
  They are never a substitute for the 98,304 / 32,768 eval contract and never turn
  an API trace into a pure decode rate.

Reuse the existing control field names where possible (`case`, `prompt_tokens`,
`completion_tokens`, `http_status`, `wall_seconds`,
`output_tokens_per_second`, `response`) from the read-only
`20260823T163646Z-c1-service-controls/` artifacts, while adding the manifest and
runtime identity fields below. Do not rewrite those artifacts.

### A6. Exact launch, image, model, and host evidence

The manifest must record the literal command, all environment variables affecting
execution, and these fields for every run (including no-graph and cap-64 as separate
rows):

| Evidence group | Required fields |
| --- | --- |
| Run identity | UTC start/end, run ID, harness git SHA and dirty state, CLI string, seed, fixture/tokenizer SHA256, container ID, exit status |
| Image/runtime | Full image digest `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`; vLLM version/SHA `0.27.2rc1.dev77+gac7509e2b`; `vllm-xpu-kernels` package version and exact source/package identity; Python, PyTorch, Triton, and XPU runtime/oneAPI versions |
| Model | Hugging Face ID `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`; revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`; tokenizer ID/revision; config and quant-config hashes; target GPTQ symmetric G128 W4A16; draft dtype BF16; target dtype; vocab/hidden/KV dimensions |
| Launch | Full server/loader command and flags; MTP method/depth; graph mode and cap; `max_model_len=131072`; `gpu_memory_utilization=0.88`; `max_num_batched_tokens=8192`; FP8 KV setting and scale mode; prefix-cache setting; scheduler/max sequences; tensor parallel/visible device; parser `qwen3_xml`; port/bind address |
| Host | Hostname, OS/kernel, CPU model and RAM, Intel Arc Pro B70 PCI identity, visible-device mapping, GPU memory, driver/firmware, XPU runtime, power/performance state, temperature/clock snapshot, and raw `xpu-smi`/PCI evidence paths |
| Timing | Warmup/repeat counts, event/timer method, synchronization policy, graph capture/replay counts, allocation policy, page size, layer index/type, `B/M/C`, accepted count, KV format, dtype, and profiler command/version |
| Logs | Host-side stdout/stderr, `docker inspect`, `docker logs`, health evidence, profiler output, and any failure/OOM response stored outside `--rm` at paths in the manifest |

If an experimental container cannot preserve a field, mark it `unknown` and stop the
comparison; do not fill it from the live server or from a prior run.

### A7. CLI sketch and output layout

This is a CLI contract for later implementation, not a command to run in this slice:

```text
python -m b70_p1_microbench component \
  --bench {draft_head,mtp_layer,verify_forward,target_lm_head,gdn_spec,attention_context,accept_commit} \
  --model SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --revision 9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e \
  --context 4096 8192 32768 98304 \
  --m 1 4 9 --graph-mode no-graph --seed 12345 \
  --warmup 10 --repeats 5 \
  --output ~/b70-evals/qwen38-b70-p1-microbench/<UTC>/components

python -m b70_p1_microbench c1 \
  --contexts 4096 8192 32768 98304 \
  --config configs/qwen38-b70-gptq-int4-mtp4-pi-local.toml \
  --graph-mode cap64 --repeats 5 \
  --output ~/b70-evals/qwen38-b70-p1-microbench/<UTC>/c1
```

The implementation must expose an equivalent `--graph-mode cap64` arm and print the
effective vLLM compilation settings; it must not hide the mapping behind a generic
"fast" flag. A proposed run layout is:

```text
~/b70-evals/qwen38-b70-p1-microbench/
  <UTC>-<row-name>/
    manifest.json
    summary.json
    environment/
      host.json
      image.json
      launch.txt
      model.json
    components/
      draft_head/{samples.jsonl,summary.json,stdout.log,stderr.log}/
      mtp_layer/{samples.jsonl,summary.json,profiler/}
      verify_forward/{samples.jsonl,summary.json,profiler/}
      target_lm_head/{samples.jsonl,summary.json,profiler/}
      gdn_spec/{samples.jsonl,summary.json,profiler/}
      attention_context/{samples.jsonl,summary.json,profiler/}
      accept_commit/{samples.jsonl,summary.json,profiler/}
    c1/
      context-4k/{requests.jsonl,summary.json,server/}
      context-8k/{requests.jsonl,summary.json,server/}
      context-32k/{requests.jsonl,summary.json,server/}
      context-98k/{requests.jsonl,summary.json,server/}
    logs/{docker-inspect.txt,docker-logs.txt,health.txt}
```

Do not place new output in the existing `qwen38-b70-gptq-int4-mtp4` control tree.
The research output directory is an artifact destination, not a new durable service,
store, schema, or deployment gate.

### A8. What P1 is sufficient to unblock

P1 is sufficient to make the following next measurements well-posed, but not to
approve a kernel change:

- **Item 1 — draft-head elimination/shortlist:** `draft_head` separates projection,
  logits materialization, and reduction over `M=1..9`, with actual vocab/hidden sizes,
  memory, and graph arms.
- **Item 5 — small-M GPTQ/XPU specialization:** `verify_forward`, `mtp_layer`, and
  `target_lm_head` provide real small-`M` shapes, per-stage launches, W4A16/target
  dtype identity, and no-graph versus cap-64 evidence.
- **Item 7 — device-side acceptance/synchronization:** `accept_commit` reports the
  actual `.item()`, `.tolist()`, D2H, and explicit synchronization costs on the
  existing accept/commit path, with state hashes to protect correctness.
- **Item 8 — producer-to-consumer fusion:** `mtp_layer` and `verify_forward` expose
  producer/consumer boundaries, intermediate bytes, launch counts, and stage timers
  needed to name a real reusable tensor rather than porting MLX code by analogy.
- **Item 10 — long-context attention:** `attention_context` supplies the `M=1` versus
  `M=4` context ladder, and `gdn_spec` supplies the separate hybrid-layer attribution
  needed for the decision in Part B.

The harness is a measurement prerequisite. An item remains blocked if its boundary
cannot be isolated on the installed XPU package or if the row fails target-token
correctness/state checks.

## Part B — Item 10 attention ladder

### B1. Concrete comparison

Run a matched ladder on one representative full-attention layer, then repeat on a
second full-attention layer if layer types differ. For each row, use `B=1`, the same
historical KV pages, the same query dtype, and the same FP8 KV format as the frozen
service control:

| Context `C` | Decode row | Verify row | Required output |
| ---: | --- | --- | --- |
| 4,096 | `M=1` | `M=4` | `t1`, `t4`, logical/allocated KV bytes, scale bytes, estimated bandwidth |
| 32,768 | `M=1` | `M=4` | same; this is the first long-context decision row |
| 98,304 | `M=1` | `M=4` | same; this is the binding eval-context risk row |
| 131,072 (optional) | `M=1` | `M=4` | stress only; never substitutes for 98,304 |

For every cell, collect at least the configured warmups and repeats from A0, device
time and enclosing host time, page/padding metadata, and profiler output where
available. The primary ratios are:

```text
R_time(C) = median(t4(C)) / median(t1(C))
R_bytes(C) = measured_or_estimated_KV_bytes(M=4, C)
             / measured_or_estimated_KV_bytes(M=1, C)
```

Use the logical lower bound and independent upper bound from A3 when no hardware
memory-read counter exists, and label the ratio as an estimate. The ladder is a
measurement of an attention boundary, not a claim about end-to-end tok/s. A separate
full one-layer/model-step row may be used to compute the fraction of the 98k step
attributable to attention, GDN, projection, and host synchronization.

### B2. Distinguishing shared K/V from per-row loads

**Hypothesis:** a multi-row verify kernel can keep each historical K/V tile resident
while evaluating all active query rows. The alternative is near-independent K/V loads
for each query row. Timing alone is suggestive but not sufficient.

Call the path **already batched/shared** only when the following evidence agrees at
`C>=32k`:

1. the attention invocation receives a query tensor with an active row dimension
   `M=4` (not four sequential API calls), and the kernel/source or profiler shows a
   query-parallel tile loop/grid;
2. hardware/profiler memory reads or a validated kernel byte model stay close to the
   `M=1` K/V read volume while query/output work grows; and
3. `R_time(C)` grows substantially slower than four, with repeated rows and no
   graph-capture change. A pre-registered operational bound is
   `R_time < 2` and `R_bytes < 2`, corroborated by source/profiler evidence.

Call the path **near-independent per-row loading** only when long-context rows show
both approximately linear K/V bytes and approximately linear time (a pre-registered
signal is `R_bytes > 3` and `R_time > 3` at one or more of 32k/98k, with profiler or
source support). Intermediate values are **ambiguous**: preserve the artifacts and do
not start a kernel rewrite. Optional diagnostic rows `M=2` and `M=8` can resolve a
nonlinear M=4 result, but the required decision comparison remains M=1 versus M=4.

Also check that `M=4` was not accidentally serialized by the Python harness, that
page-table and allocation work are outside the attention timer, and that the same
cache pages were restored before each row. A flat total with four separate launches
is not proof of K/V reuse; a single launch with four queries is not proof either.

### B3. FP8 KV and GDN interpretation

#### FP8 KV

The baseline uses FP8 KV. FP8 changes the byte model, not the reuse question:

- count one byte per stored K/V element only when the actual cache is one-byte FP8;
  add per-page/per-head scale and metadata bytes from the real layout;
- record whether dequantization is fused into attention or is a separate pass; its
  time and reads belong to the attention boundary only if the boundary includes it;
- an FP8 row with half the bytes of BF16 is not evidence of a speedup, and a high
  estimated bandwidth is not evidence that K/V was loaded only once;
- an optional BF16-KV diagnostic uses identical `C`, `M`, pages, and query values and
  is a separate row. It is for interpretation, not a replacement for the FP8 control.

#### GDN

Qwen3.8 is a hybrid model. The GDN route exposed by
`_gdn_attention_core_xpu_impl` calls `gdn_attention` with recurrent conv/SSM states,
while the upstream `gdn_attention` implementation chains
`causal_conv1d_spec` and `gated_delta_rule_spec` for spec rows. That route is not the
same historical paged-KV scan as a full-attention layer.

Therefore:

- use `attention_context` for item 10's K/V reuse question;
- use `gdn_spec` to attribute GDN conv, delta-rule, state movement, accepted-token
  metadata, and any long-prefill/chunk work separately;
- report the number and type of full-attention and GDN layers and their per-layer
  times before making a model-step attribution;
- do not infer full-attention K/V reuse from a GDN time, or infer GDN cost from a
  full-attention KV-bandwidth estimate.

**Hypothesis:** if the approximately 98k model step is dominated by GDN state/chunk
work or its metadata/host boundary rather than full attention, improving full-attention
multi-row reuse cannot address the observed tail even if the attention rows are
independent.

### B4. Item-10 keep/park/kill rule

Apply this rule after the matched ladder and stage attribution; do not use a short
512-token result as the decision:

- **Park item 10's kernel-rewrite work (shared):** at 32k and 98k, source/profiler
  evidence confirms shared historical K/V reads and the pre-registered shared bounds
  (`R_time < 2`, `R_bytes < 2`) hold. Keep the measurement artifact as a baseline,
  but do not spend a kernel slice trying to add reuse that already exists.
- **Keep item 10 for a narrowly scoped kernel investigation (independent):** at least
  one of 32k/98k shows the independent signal (`R_time > 3` and `R_bytes > 3`, or
  equivalent source/profiler evidence) and full attention is a material part of the
  measured 98k model-step time. The next slice must name the exact kernel surface;
  this spec itself makes no speed claim.
- **Park item 10 (GDN-owned):** GDN/conv/SSM or its metadata/state/host boundary is
  the dominant measured contributor to the 98k step (pre-register `>50%` of the
  comparable model-step boundary) while full-attention reuse is not independently a
  large term. Route the question to the GDN track instead.
- **Do not decide / kill the row:** if the path cannot be isolated, the graph arms
  are not comparable, state hashes fail, or the profiler cannot distinguish serial
  calls from one multi-row call. This is a harness failure, not evidence for a speed
  claim. A failed/unsupported row is archived with its reason.

These rules implement the research-plan kill condition: park item 10 if the current
kernel already shares K/V, or if GDN—not attention—owns the 98k time. They do not
permit changing the product ceiling, FP8 policy, MTP depth, or target numerics.

### B5. Why Yukon 512-token fixtures are not a substitute

The Yukon/MLXFast 512-token fixture can be useful for a separate short-context
correctness or microkernel comparison, but it cannot answer item 10:

- a 512-token history does not exercise 32k/98k page count, memory pressure, long
  context cache residency, graph padding, or the context-dependent crossover;
- fixed projection/launch overhead can hide a per-row K/V reload that becomes the
  dominant term at 98k;
- the MLXFast fixture is an Apple MLX/Metal workload with affine G64 4-bit weights,
  whereas this target is vLLM/XPU with symmetric GPTQ G128 W4A16, FP8 KV, and hybrid
  GDN state; the stacks and byte layouts are not interchangeable;
- the B70 contract is a stochastic public-API/agent workload at 98,304 input and
  32,768 output tokens. Its `17.10 tok/s` trace median is explicitly not a decode
  benchmark, and a 512-token synthetic decode cannot stand in for it.

The local research progress records this distinction at
`docs/qwen38-b70-mlxfast-research-progress-20260823.md:109-137`; the research plan
records the item-10 ladder and Yukon warning at
`docs/qwen38-b70-research-plan-20260824.md:474-509`.

## References

- Local authority and measured controls: `docs/qwen38-b70-next-session-handoff-20260823.md`,
  `docs/qwen38-b70-speed-improvement-handoff.md`,
  `docs/qwen38-b70-mlxfast-research-progress-20260823.md`, and
  `docs/qwen38-b70-research-plan-20260824.md` (especially Shared method, P1,
  item 10, and item 14). `docs/qwen38-b70-research-plan-20260824.md` is not edited
  by this slice.
- Pinned vLLM MTP source:
  <https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/models/qwen3_5_mtp.py>
- Pinned vLLM Qwen3-Next target source:
  <https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/models/qwen3_next.py>
- Pinned vLLM XPU GDN wrapper and metadata:
  <https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/_xpu_ops.py> and
  <https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/attention/backends/gdn_attn.py>
- XPU kernel source/benchmark entry points:
  <https://github.com/vllm-project/vllm-xpu-kernels>
- vLLM XPU hardware page:
  <https://docs.vllm.ai/en/latest/models/hardware_supported_models/xpu/>
- MLXFast context warning and challenge references:
  <https://www.yukon.org/mlxfast> and
  <https://github.com/Layr-Labs/qwen-3.8-mtp-challenge>
