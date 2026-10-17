# Bolt grouped IQ2_XXS prefill experiment

**Correctness and activation pass; rejected for lack of useful prefill gain.**
Long median8230.415 → 8216.915ms (−0.164%); accepted SSM remains resident.
Development-only. Baseline is the accepted staged-SSM archive
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`, upstream
`b6334e37e6ef286fc5091a41b6561b2113d915fb`. No persistent launcher or model change.
The preceding NVFP4 packing candidate was rejected: long median prefill
8231.233 → 8248.592 ms; see `../20261017-nemotron35-bolt-prefill-nvfp4/README-r2.md`.

## Research and bounded approach

Freshly fetched primary source:
https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/fused-gemm.hpp

The grouped gate bounds **average** routed rows per active expert by 64, not
individual expert size. The accepted source already contains grouped XMX GEMM;
no blind backport of PR23142/29245 is warranted. Live source confirms current
f16 8×16×16 sg16 uses **BN32**, SG_ROWS16, BK32 and four K splits. Changing the
global `GGML_SYCL_XMX_GATHER_SHAPES` mask also changes dense/other-precision paths,
so a global shape-mask sweep is not this experiment.

Fresh research refresh also fetched the [Intel joint-matrix specification](https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc),
[pinned implementation](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/fused-gemm.cpp),
[PR29245](https://github.com/ggml-org/llama.cpp/pull/29245) and its
[patch](https://github.com/ggml-org/llama.cpp/pull/29245.patch). Conclusions:
TN16 is the hardware matrix width while BN32 is software tiling; BN32 already
reuses A across two N fragments. A narrow tail can retain the legal TN16 matrix
shape, whereas cross-workgroup A reuse is a larger redesign. Keep runtime device
combination/subgroup checks and matrix alignment/stride requirements. A and B
convert to F16 before F32 accumulation: overflow-range inputs require explicit
coverage, not comparison to an assumed equivalent F32 GEMM. The intended native
contract is same-device original/candidate bitwise output equality, not a relaxed
CPU-reference tolerance; nonfinite cases must be reported explicitly.

First measure actual rows/expert, padding, tiles/expert, selected combination and
fallback reasons without changing arithmetic or dispatch. `tiles / active_experts`
is the cross-tile weight decode duplication factor; `tiles / routed_rows` is not.
Choose at most one subsequent optimization based on these observations. Prior
instrumented grouped time share does not establish occupancy, bandwidth saturation,
or a transferable B70 speed claim. No grouped exactness claim exists yet.

Native harness preparation uses the real grouped entry with AoS IQ2_XXS and a
same-device original/replay check before a candidate exists. It must explicitly
assert activation, full writes, inputs/canaries, and an independent simple-value
oracle. This is not yet original/candidate proof. Include empty experts and
1/15/16/17/31/32/33/63/64/65 routed-row boundaries: one expert with65 rows fails
the average gate, whereas counts[65,1] pass and the65-row expert needs **three**
BN32 tiles. A later candidate selector should remain internal and narrowly scoped;
no global XMX mask changes are required.

## Declared end-to-end process

Environment: `mike@inference-host`, explicit Bash; image
`intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04`, IntelLLVM2026.0.0,
six build jobs. Strict SSH options: `-o BatchMode=yes -o ConnectTimeout=8
-o ControlMaster=no -o ControlPath=none -o ForwardAgent=no`.

1. Verify the live `nemotron-bolt-probe` label, accepted source mount, health,
   baseline binaries and upstream marker. Preserve unrelated dirty files.
2. Apply the diagnostic-only patch to a new disposable accepted-source clone;
   build server/backend/native tests remotely with `build-grouped-diagnostic.sh`.
   Preserve all build failures. No kernel changes in this first stage.
3. Run `python3 -B run-grouped-diagnostic.py` remotely. It retains the accepted
   container stopped, reuses its exact recovery create argv with only the source
   mount/name/debug changed, and runs the unchanged native385 gate before serving.
4. At public boundary `http://127.0.0.1:18098/v1/chat/completions`, send the frozen
   8270-token prompt with 128 forced outputs to warm through actual decode. Then
   send the same uncached prompt with output1, preserving timestamped diagnostic
   window separately from native and warmup logs. Require exact prompt/output
   counts and `cache_n=0`. Instrumented latency is not a performance result.
5. Parse all grouped geometry records and validate row totals, tile/padding
   counts and fallback accounting. Select a bounded kernel candidate only if
   there is a material measured opportunity.
6. Before candidate timing: same-device original/candidate full-array exactness
   and canaries, explicit native activation/fallback cases, unchanged native385,
   debug actual-model activation, then fresh **non-debug 20/20 gate-v2 API**.
7. If all gates pass, use original weights, q8 KV, F32 recurrent state, context65536,
   batch/microbatch1024, single sequence, ESIMD1/FUSION1/graphs0. Matched baseline
   and candidate: 55/8270-token prompts,128 forced outputs, warmup plus three
   measured repetitions, and exact8238-token `BOLT-7319-CEDAR` retrieval. Keep all
   samples and failures; no retry-until-lucky. No old `run-cell.py`.
8. Stop diagnostic container and restore original SSM container in `finally`;
   require public `READY`, unchanged baseline binary hashes and hardware equality
   (boost0,16 CPU caps3801000kHz,275W). Rejected candidates return to SSM too.

Artifacts: scripts and patches here; remote root
`/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped/`, including build,
runtime precheck, binary hashes, `diagnostic-r1/{controller.py,create-argv.json,
native-and-startup.log,server.log,diagnostic-window.log,*.json}` and raw API
requests/responses. Large logs are compressed losslessly with `gzip -n`.
No accepted archive, weights, hardware/driver/network settings or persistent
launchers are modified. The Lab preview socket remains unavailable (ENOENT);
plain scoped changes are retained without repairing that unrelated tool.

## Preparation checks and retained corrections

The initial worker parser incorrectly treated row remainders17–31 as invalid.
Lead review removed that restriction and added acceptance coverage for all tail
width classes, empty experts and65-row skew. The corrected parser reports narrow
and wider tails separately, weighted padding and tiles/active-expert reuse.
Five parser tests passed on inference-host (`parser-tests.log`). The worker's
local test claim is not counted as host verification.

The worker's copied-source `git apply --check` was inconclusive because Git
skipped untracked paths. Lead used `patch --dry-run -p1` against the actual remote
accepted archive: both target files checked successfully, retained in
`diagnostic-patch-dry-run.log`. No source mutation in that check.

The first background build launch failed before running the script because nested
shell quoting expanded its remote log path to `/build-diagnostic.log` (permission
denied). Retained `build-launch-failure.log`; corrected launch uses literal
absolute paths. This is a launch failure, not a compiler or numerical result.

## Diagnostic result and selected candidate

`diagnostic-r1` passed the unchanged native385 gate and real API count/cache checks;
accepted SSM container restored, `READY` passed, hardware and baseline hashes
unchanged. The decode-warmed8270/output1 window contains880 IQ2 calls:808 grouped,
72 fallbacks, **all** `average_gt_64`; no outer gate failures. Every grouped call
selected combo0, f16 8×16×16 sg16, BN32/SG_ROWS16.

Grouped geometry across those808 calls:138356 active-expert occurrences,
261386 token tiles,5290848 valid vs8364352 padded columns (36.7453% padding),
1.88923 tiles/active expert.106442 tiles (40.7222%) have nonzero tails≤16;
28604 tails are17–31. Individual experts reach954 rows/30 tiles despite the
average≤64 gate. These are actual observations, not uniform-routing assumptions.

The880 calls reconcile as40 layers ×2 expert matmuls ×11 chunks. Routed-row
counts are9216 on560 calls, and45/657/9180/36 on80 each: token chunk widths
1024×7,5,73,1020,4 sum to8270. They are not nine simple1024-sized batches.

Selected candidate R1: on grouped AoS IQ2_XXS combo0 only, skip the unused second
TN16 B-fragment load, matrix multiply and accumulator store when a BN32 tile has
≤16 valid columns. Retain BN32 packing and schedules, dequant A, existing matrix
shape, F16 conversion, barriers, K traversal and four-way final sum. Dense paths,
other quantizations/combinations, fallbacks and global shape mask remain unchanged.
The internal grouped entry gets an optional per-call selector for native
original/candidate comparison, not a global serving knob. Candidate activation
must show narrow tiles on the actual model before non-debug API/timing gates.
No speedup is claimed from geometry alone; unchanged register/local-memory
allocation and the added branch may offset skipped work. Reject without useful
matched long-prefill improvement or on any correctness/quality failure.

## Native harness build failure

The original/replay harness's first build failed at its input-mismatch diagnostic:
`fprintf("%s", label)` passed a `std::string` through variadic arguments.
`build-native.log` retains the IntelLLVM error; `grouped-native-string-fix.patch`
adds `.c_str()`. The initial `grouped-native.patch` and source copy remain unchanged
as failure evidence. Rebuild uses `build-native-string-fix.log`; no threshold,
fixture or kernel arithmetic changed to address this compilation error.

The corrected original/replay harness builds and passes on B70:
`native-original-r1` reports13 valid full-array exact-repeat cases,6 negative
gates with untouched outputs, guards/input checks and the simple F16 oracle.
The accepted SSM container was restored with `READY`, equal hashes/hardware and
healthy API. This establishes the harness's original path only; candidate
original-vs-optimized proof is still required.

Candidate build succeeded with strict, zero-fuzz patch application and source-copy
comparison. Apply in order: `grouped-tail-r1.patch` (cumulative against accepted
source), `grouped-native.patch`, `grouped-native-string-fix.patch`, then
`grouped-native-candidate.patch`. The last patch changes only native controls:
ORIGINAL passesfalse, CANDIDATE passestrue; both selectors exercise each of the
six rejection fixtures. Lead corrected the all-empty fixture to pass actual
`total_rows=0`, while retaining a nonempty sentinel allocation for overrun checks.
The success gate requires13 valid cases,6 rejection fixtures/12 negative calls,
13 original and13 candidate kernel-selection markers. Historical original/replay
artifacts remain untouched. `build-tail.log` and `tail-binaries.sha256` identify
the build to be gated.

Read-only harness review found no blocking design issue. Its limits remain:
large raw cases primarily establish repeat/candidate exactness, not independent
CPU-oracle fidelity; the simple-value oracle is small. The original overflow
fixture produced50176 NaNs with identical bits in both runs, reported explicitly.
No full-model output-identity guarantee follows from native equality.

## Candidate correctness and activation

`tail-r1-activation` passed native385 plus13 original/candidate full-array bitwise
cases and all six rejection fixtures under both selectors. Shell activation
assertions observed exactly13 original and13 optimized native launch markers.
The debug actual-model window observed808 optimized grouped calls and106436
narrow tiles, agreeing exactly with its own independently parsed row geometry.
The few-row routing difference versus the earlier diagnostic launch is retained;
independent model launches are not assumed bit-identical. Both8270-token requests
had `cache_n=0`, with128 warmup outputs and1 activation output. Restoration,
`READY`, baseline hashes and hardware equality passed.

## Matched non-debug result: reject R1 for serving

Both fresh launches passed20/20 frozen gate-v2 API checks **before** timing, plus
exact8238-token `BOLT-7319-CEDAR` retrieval. Normalized server argv and hardware
match; all20 quality,4 short and5 long request payloads are byte-identical.
All measured requests use55/8270 prompt tokens,128 forced outputs and `cache_n=0`.
One warmup and three measurements per context; no exclusions or retries. The short
warmup is54 tokens in the inherited harness; its three measurements are55 each.

- Short prefill baseline473.149,468.686,471.675ms; candidate468.576,468.315,468.132ms.
  Median471.675 → 468.315ms (−0.712%).
- Long prefill baseline8230.415,8226.255,8230.688ms; candidate8222.669,8216.915,8212.975ms.
  Median8230.415 → 8216.915ms (−0.164%); throughput1004.809600 → 1006.460454tok/s (+0.164%).
- Long client TTFT8.323125 → 8.318624s (−0.054%); short TTFT0.893265 → 0.671591s.
  The much larger short-TTFT change is not matched by server prefill and is not
  attributed to the kernel (RAM prompt-cache overhead varies in this setup).
- Decode medians short40.345126 → 40.478335tok/s; long39.034062 → 39.047878tok/s.
- Full measured-output string identity:0/3 short and0/3 long. Native bitwise
  agreement and passing fixed API checks do not establish full-model equivalence;
  independent same-build launches were already known to vary.

The13.5ms long median difference is not a useful serving gain at this development
sample size. Keep the SSM winner; do not launch a retry-until-positive sweep or
claim unused-column work elimination translated into a meaningful speedup. This candidate skips
unused matrix-fragment work but retains A-stage work, scheduling, packing and
static allocations. No occupancy, register, bandwidth or per-kernel timing
measurement here explains the small result.

`comparison.json` contains every timing sample, medians, deltas, payload checks
and identity counts, generated on inference-host by `summarize-grouped-pair.py`.
`tail-r1-baseline/` and `tail-r1-candidate/` contain raw API evidence and exact
create argv. Their controllers restored the original container in `finally`.
`restored-residency.log` verifies final live SSM mount/label, health, unchanged
three baseline binary hashes and boost0/16caps3801000kHz/275W. No persistent
launcher, model weight, hardware, driver or network change was made.

## Executed commands and limitations

Scripts below were transferred by SSH stdin/tar and run on inference-host using
Bash, strict SSH options declared above. Each stdout/stderr stream is retained
under the corresponding build/run log. Source patch application is zero-fuzz,
and every script is retained in this directory; each run also saves its controller.

```bash
bash build-grouped-diagnostic.sh
python3 -B test-analyze-grouped-diagnostic.py
python3 -B run-grouped-diagnostic.py
# Apply grouped-native.patch; first build fails as retained above.
bash build-grouped-native.sh
# Apply grouped-native-string-fix.patch; corrected build succeeds.
bash build-grouped-native.sh
python3 -B run-grouped-native.py
bash build-grouped-tail.sh
python3 -B run-grouped-cell.py --phase activation
python3 -B run-grouped-cell.py --phase baseline
python3 -B run-grouped-cell.py --phase candidate
python3 -B summarize-grouped-pair.py
```

Retained archives: accepted SSM unchanged, diagnostic clone and rejected tail
clone. Linked development worktrees removed after integration; branches retained.
Large local server/window logs and geometry analyses use lossless `gzip -n`;
remote originals remain available at the declared run root. No background
workloads remain. This is not publication-standard benchmarking: no BetterBench
full20-pass, concurrency sweep, broad fidelity suite, maximum-context or peak-memory
characterization. Next work, if requested, can investigate measured cross-tile
A-dequant duplication (1.889× average, heavy routing skew) rather than globally
changing XMX shapes; this experiment did not implement that larger redesign.
