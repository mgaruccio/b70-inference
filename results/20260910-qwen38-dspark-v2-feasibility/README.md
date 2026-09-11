# Qwen3.8 DSpark V2 feasibility — development only

## Final outcome

**PASS — bounded eager/C1 DSpark V2 feasibility on B70, including 64K input and the exact context boundary. Not promoted to production.** The chronological failures below are retained, not the current outcome.

- Target-only V2 NaN fixed by the existing unchanged short-prefill overlay. Guarded mixed FP16-target/BF16-draft overlay added; native draft cache selector corrected without changing BF16 storage or target FP8 cache.
- Corrected short smoke: 3/3 canaries, 131/131 finite boundaries, 8/8 functional code tasks, eight stable 128-token streams. `python3 compare-smoke.py` passes 19/19 exact output-token matches with identical requests/prompt IDs against target-only control. This is bounded greedy parity, not broad quality equivalence.
- Native boundary reuse: `bash run-native-boundary-probe.sh` on inference-host, pinned image/B70, FP16 actual target dimensions, passes 224 partial cases, 896 continuations and 5408 alternating chained steps exactly (`rtol=atol=0`). See `native-boundary-probe-output.txt`. Existing runtime patch and oracle copied unchanged. CPU routing suite: 10 tests, one installed-source skip locally; actual native installed-source replay passed.
- `python3 -u run-dspark-boundary-64k.py` (task `b7e6d8699`, exit 0) passes all short gates and the existing public API client at **65536 input +128 output**, max length 65664, prefix disabled, one warmup plus **6/6 valid measured requests**, zero invalid/errors. See `dspark-v2-boundary-64k/{launch-argv.json,source-check-argv.json,summary.json,server.log}` and `long-context/{summary.json,points/length-65536/}` for exact commands, request/SSE/metrics evidence. The first near-limit failure is retained separately in `dspark-v2-64k/`.
- Development-only 64K timing: median post-first decode **14.7631 tok/s**, inclusive IQR **0.32546 tok/s**; median TTFT **57.5467 s**, total **66.1532 s**. This eager bundle is **not** an apples-to-apples comparison with historical graph-enabled MTP4; no speedup claim.
- Full pinned CPU DSpark suite: **20/20 passed, zero skips** (`cpu-dspark-cache-fix-tests.txt`); earlier fixture failures retained. Source/client copy comparisons, Python syntax checks and secret-pattern scan passed. Raw logs intentionally retain original whitespace.
- Installed overlay replay checks and before/after host invariants passed. Owned containers removed, Glimmer stopped, power 275 W, persistent launcher SHA unchanged. Draft weights remain host-local and are not committed.

**Not tested/claimed:** graph mode, 160K context, concurrency, stochastic sampling, broad quality/distribution equivalence, or production readiness. A fair performance phase needs a fresh paired control. Stop here at the authorized feasibility scope.

## Predeclared scope and first gate

User authorized moving from DFlash tuning to DSpark V2. This is a bounded compatibility/correctness feasibility experiment, not production promotion or a standard-publishable performance comparison.

Candidate: `RadixArk/Qwen3.8-27B-DSpark`, revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`. Its official card reports V2 results; the checkpoint is 1,857,358,337 BF16 parameters, five full-attention layers, seven proposals (eight target verification positions), confidence and Markov heads. DSpark checkpoint V2 and vLLM model runner V2 are distinct version labels.

First prerequisite: unchanged target through the installed XPU V2 model runner, without speculation. Target `/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`; image `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4` (vLLM `73029d42441321b631779db3475031f5ec26dd6c`). No patches, target FP16 compute/GPTQ, FP8 KV, C1, prefix/thinking off, prefill budget 2048, power 275 W. Intentional smoke-only differences from the last DFlash campaign: V2 instead of legacy runner, no drafter/overlays, eager instead of graphs, 8192 instead of 180224 context. No performance delta will be inferred from this prerequisite.

Operational performance baseline, only if feasibility passes: best existing same-host MTP4; historical 64K/160K medians 55.75/46.39 tok/s are context only, not a contemporaneous control. A performance phase needs a fresh paired control and must label differing runtime stacks as a bundle. Planned points are 512/65536/160000 input tokens, 128 outputs, one warmup and six measurements, median/IQR, acceptance and memory. No throughput phase is authorized by a failed prerequisite.

## Fresh primary-source conclusions

Consulted September 10, 2026:

- [Pinned model card](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/blob/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/README.md): official published results use SGLang on GB300/H200, evaluated NVFP4/FP8 targets. This is not B70, GPTQ-target, or vLLM compatibility evidence. Card gives weight SHA256 `2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7` (3,714,723,322 bytes).
- [Pinned draft configuration](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/blob/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/config.json): BF16, `DSparkDraftModel`, five full-attention layers, no sliding window. Long-context draft memory must be measured; DFlash's sliding-window budget does not transfer.
- [Pinned XPU runner](https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/worker/xpu_model_runner.py): `XPUModelRunnerV2` exists and wraps shared GPU runner APIs. Its existence alone does not establish complete DSpark/GDN support.
- [Pinned DSpark implementation](https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py): V2 DSpark uses parallel backbone plus sequential Markov sampling, with optional confidence-head adaptive verification. Begin with fixed depth/standard rejection if reached, not adaptive verification.
- [Pinned speculative config](https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/config/speculative.py): external drafter inherits target dtype. Actual installed `/opt/venv/lib/python3.12/site-packages/vllm/config/speculative.py` line 1255 confirmed `dtype=self.target_model_config.dtype`, SHA256 `4e3d0a9b93f54fc0f25897e8c7bba5412e3851dd819950964c201ed6376e08b7`. Thus preserving target FP16 and draft BF16 is not a stock launch flag here. Do not silently downcast the draft, change target dtype, or reuse DFlash-specific patches.

## Executable end-to-end process (declared before launch)

Environment: idle `inference-host` B70; Glimmer stopped, 275 W, original persistent launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; installed immutable image and existing read-only target. Driver uses system Python/stdlib and a byte-for-byte copy of `scripts/experiments/qwen38_lossy_probe.py`; no inference dependencies enter interactive Pi.

1. Copy `run-prerequisite.py` and existing probe module to `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility/` without overwriting prior results.
2. On that host, execute `python3 -u run-prerequisite.py` in the campaign directory, through explicit Bash over SSH. The background wrapper bounds runtime and retains stdout/stderr.
3. Driver records invariants, image inspection and complete argv; starts only `qwen38-dspark-v2-prerequisite`, localhost port 8000, target read-only. Wait at most 900 seconds for `/health`; fail immediately on process exit. Verify `/v1/models` exposes `qwen38`/8192.
4. Reuse `Cell.gates`: real `/v1/chat/completions` SSE for arithmetic `19 + 23`, exact JSON and Python addition canaries; require correct content, token IDs/usage, finish reason and `[DONE]`. Send token-ID prompts `[42] * n` to `/v1/completions` for `n=1..128,133,197,261`; require finite logprob, exact input usage and one output token.
5. Force 128 streamed output tokens on the mutable-default-arguments prompt to exercise sustained decode. Retain each request, SSE event, result, boundaries and metrics. Passing this gate does not prove speculative correctness or broad quality.
6. Always remove only the owned container, join its process, record post-run invariants. Existing target weights and launcher are never edited. Draft weights are not downloaded at this stage.

Stop conditions: startup failure, unsupported backend/configuration, incorrect canary/transport, nonfinite boundary, timeout or changed host invariants. Retain the failure; do not start DSpark or benchmark throughput after failure. A substantial custom runtime port requires an explicit follow-up decision.

## Results

**Initial phase: blocked at the unpatched XPU V2 target correctness prerequisite. The authorized follow-up below resolves that target prerequisite with the existing prefill patch. DSpark correctness and throughput remain untested at this point.**

Both executions started the real GPTQ target successfully, selected V2, served `/v1/models` with context 8192, and passed all three SSE chat canaries. Both then failed on the first `/v1/completions` boundary request (`n=1`); zero finite-boundary checks completed and the forced 128-output continuation was not reached.

- `target-v2/`: original run, background task `b07507934`, exit 1. The existing probe raised HTTP 400 without saving the body; retain this incomplete failure evidence rather than replacing it.
- `target-v2-diagnostic/`: exact same launch and API gates, background task `bcb506d97`, exit 1. Driver changed only the output directory and exception-body capture. `http-error-body.txt` records `{"error":{"message":"Out of range float values are not JSON compliant: nan","type":"BadRequestError","param":null,"code":400}}`. This establishes a nonfinite response-serialization failure, not the originating kernel or tensor.
- Both `host-after.json` files equal their respective `host-before.json`: no running containers, Glimmer stopped, original launcher SHA, power `275000000`. No draft weights were downloaded; no image, target weights or application source was patched.

The first failing request is reconstructable exactly from the retained byte-identical `qwen38_lossy_probe.py:262` path (the original helper saves boundary rows only after successful responses):

```json
{"model":"qwen38","prompt":[42],"max_tokens":1,"temperature":0,"seed":42,"ignore_eos":true,"logprobs":1,"cache_salt":"boundary-1"}
```

Exact host execution, after copying the driver and probe module:

```bash
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
set -o pipefail
timeout --signal=INT --kill-after=90s 1800s python3 -u run-prerequisite.py 2>&1 | tee driver-output.txt
```

For the first run, use retained `run-prerequisite-original.py` (the original was named `run-prerequisite.py` when executed). The diagnostic uses the current `run-prerequisite.py` and `driver-diagnostic-output.txt`. Both refuse to reuse an existing output directory. Complete container commands and configuration are in each cell's `launch-argv.json`, `launcher.sh`, `image-inspect.json` and `server.log`; chat requests/SSE/results and `/v1/models` are retained beside them.

### Interpretation and next decision

This does **not** establish that DSpark is slow or inherently unusable on B70. It establishes that the selected unpatched V2 target runtime fails the declared real-input correctness gate, independently of any drafter. The initial canary rates are tiny eager-mode smoke samples, not benchmark results. Do not compare them to MTP/DFlash performance.

The next engineering scope would be to diagnose/fix the V2 target nonfinite output first, then separately validate an independent BF16 DSpark draft with the unchanged FP16 GPTQ target and strict rejection. Earlier DFlash-only legacy-runner patches do not establish either behavior. Adaptive verification, draft quantization, long-context tuning, stochastic convergence, concurrency and production promotion remain untested/out of scope here.

External research also identified [Qwen hybrid DSpark support PR #47377](https://github.com/vllm-project/vllm/pull/47377), [DeepSeek-V4 XPU DSpark PR #47677](https://github.com/vllm-project/vllm/pull/47677), and [B70 GDN nonfinite-output issue #548](https://github.com/vllm-project/vllm-xpu-kernels/issues/548). These are follow-up leads, not proof that the local `n=1` failure has the same cause; the reported `T % 64 == 5` issue is a different input boundary. The RadixArk release is a third-party Qwen DSpark checkpoint, not an official DeepSeek Qwen3.8 release.

## Artifact verification

Lead checks: retained probe copy equals `scripts/experiments/qwen38_lossy_probe.py` by `cmp`; all three campaign Python files parse with `ast.parse`; every JSON/JSONL record parses. Both launch argv lists are identical; both server logs contain `Using V2 Model Runner`; both summaries report three passing canaries and failed gates; both boundary logs are empty; both before/after invariant records match. The diagnostic HTTP body was checked exactly. These are artifact checks, not a passing inference-correctness result. `host-environment.txt` records the post-experiment host/kernel/CPU/RAM/Python/Docker inventory and another clean launcher/power/container check. Credential-pattern scan found no candidates.
Raw `server.log` files intentionally retain vLLM banner/progress whitespace. The full staged whitespace check flags those upstream log lines; the authored-file check excludes only the two raw server logs and passes. No runtime output was normalized to make a formatting check pass.

The post-experiment inventory command was:

```bash
ssh -n inference-host "bash -lc 'date -Is; hostname; uname -a; python3 --version; lscpu; free -b; docker version --format \"{{.Server.Version}}\"; sha256sum /home/mike/inference/launchers/start-qwen38.sh; cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap; docker ps --format \"{{.Names}}\"; docker inspect -f \"{{.State.Running}}\" glimmer-tb21-prefix-c8'"
```

## Follow-up authorized: fix V2 prerequisites

After publication of the blocked phase, the user chose **Fix V2 prerequisites**. First bounded candidate reuses the existing `scripts/patch-vllm-qwen38-xpu-prefill.py` unchanged, rather than inventing a new V2 patch. It directs GDN's existing `split_decodes_and_prefills` helper to honor `is_prefilling` on real XPU metadata, so a new one-token prompt is initialized as prefill rather than consuming recurrent decode state. Fresh pinned-source inspection confirmed the unpatched helper call and original file hashes; upstream classification reference remains [PR #53059](https://github.com/vllm-project/vllm/pull/53059).

Predeclared comparison: the two failing target-only V2 runs above versus `target-v2-prefill-fix/`. Same target/image, dtype, KV, limits, eager mode, prompts and API gates. Only the existing GDN short-prefill patch is added in the disposable container. `apply-prefill.py` verifies whole original GDN and helper SHA256 values plus the unchanged patch SHA, checks compile, exact reverse and replay before applying. No legacy runner guard or DFlash overlay is added. Run `python3 -u run-prefill-fix.py` on the host, with the same outer 1800-second timeout and cleanup. Expected: 3 canaries, 131 finite boundaries (including 133/197/261) and forced 128-token decode all pass; stop otherwise. Three existing CPU dispatch/replay/drift-rejection tests passed (`cpu-prefill-tests.txt`). DSpark itself remains gated on this result and separate mixed-dtype validation.

Observed result: task `be077ded9` exited 0. All three canaries, all 131 finite boundaries and the forced 128-token continuation passed; host before/after invariants matched. Actual installed patched GDN SHA256 is `5173f3394c1385d215bd99f0d12290e8336da844b96da612954da01d62a0b062`, with exact reverse/replay recorded in `target-v2-prefill-fix/effective-prefill-source.json`. This fixes the observed prerequisite failure with one existing classification patch; it does not demonstrate broader V2 or DSpark correctness. No new reusable application patch was necessary.

Fresh follow-up research identifies the directly matching classification defect in [vLLM #51562](https://github.com/vllm-project/vllm/issues/51562) and proposed regression/fix [#51565](https://github.com/vllm-project/vllm/pull/51565): shape-only one-token classification can read uninitialized/reused recurrent state rather than initialize a first prefill. This is consistent with the controlled failure→pass above; no intermediate tensor instrumentation was performed to identify the first NaN-producing operation. V2's eager runner already carries prefill status, so the legacy uniform-decode guard was unnecessary here. The distinct kernels #548 `64n+5` failures were not observed in this patched 1..128/133/197/261 sweep. Mixed concurrency is not established by these C1 tests.

With that prerequisite passing, `download-draft.sh` stages only the pinned `config.json` and `model.safetensors` on inference-host and checks model-card SHA256 values before renaming partial files. No remote custom model code is downloaded or executed; approximately 3.72 GB of weights remain outside Git and interactive Pi. Mixed-dtype source audit is the next gate before any DSpark launch.

Download task `bcda82499` exited 0: both official SHA256 checks passed, config 2448 bytes and weights 3,714,723,322 bytes. `config.json`, `draft-safetensors-header.json` and raw download output are retained; all 62 weight tensors are BF16, with no checkpoint embedding or LM head (the target's FP16 modules must be shared without mutation).

Predeclared expanded smoke control: `run-dspark-control.py` / `target-v2-control/` uses the identical passing target configuration and `dspark-smoke-checks.py`. After 3 canaries and 131 finite boundaries, execute existing `Cell.quality`'s eight functional tasks in the existing restricted CPU/no-network sandbox, then four 128-token repetitions each of existing `probe.CODE` and `probe.PROSE`. Require complete streams and zero speculative counters for the control. Retain repeatability counts; do not assume bitwise stability. The future DSpark cell will use identical payloads, require nonzero proposals/acceptance, compare output IDs against this control and investigate divergence before proceeding. These are bounded correctness checks, not the 500-prompt quality-sensitive publication suite or a throughput benchmark. Execute `timeout --signal=INT --kill-after=90s 1800s python3 -u run-dspark-control.py` through explicit Bash; cleanup and production invariants remain the same.

Expanded control task `b25486970` exited 0: 3/3 canaries, 131/131 boundaries, 8/8 sandboxed functional tasks and all eight forced 128-token streams passed. Each prompt family had one unique output across four repetitions. Speculative counters stayed zero and host cleanup invariants matched.

### Guarded mixed-dtype implementation and native smoke declaration

Worker commit `71ada11d9838b674e040789b1205e663ba576903` was integrated as `e480cf3`: only `scripts/patch-vllm-qwen38-dspark-bf16.py` and its focused tests. `B70_DSPARK_BF16=1` enables a whole-source-pinned overlay for explicit V2, the exact Qwen target/draft family, K7, greedy proposals, standard rejection and explicit BF16 draft KV. Draft backbone/Markov parameters, aux projection, context/query buffers and cache stay BF16; shared target embedding/head stay aliased FP16 with activation-only casts. The confidence projection intentionally remains upstream FP32 and is unused because adaptive verification is disabled. Startup/runtime checks validate actual loaded parameter/cache dtypes and sharing, rather than trusting configuration labels. No target rejection sampler or target weight is modified.

Predeclared native candidate: `run-dspark-smoke.py` / `dspark-v2-smoke/`, the same 8K/eager/C1 target configuration and expanded smoke payloads as `target-v2-control/`. Differences are only the DSpark drafter, its independent BF16 KV, the opt-in precision overlay and standard K7 speculation. Apply existing prefill fix then DSpark overlay in the disposable container. After `/health` and `/v1/models`, verify the installed `/opt/venv/` sources are an exact overlay replay and retain hashes. Require all API gates/functional checks/streams and nonzero draft proposals/acceptance, then compare paired prompt/output token IDs with the stable control before any throughput phase. Retain startup/configuration failures and HTTP bodies. Same bounded timeout, owned-container cleanup and production-invariant checks apply. Graph capture, stochastic-distribution convergence, concurrent workloads and long-context capacity are not proven by this eager smoke.

Lead CPU rerun: first task `be3cdd90d` ran 20 tests with five filesystem errors because the suite creates temporary directories under its root and the verification copy was mounted read-only; the other 15 passed. Retained `cpu-dspark-tests-readonly-failed.txt`. Repeating against the writable dedicated experiment copy (`b036c783f`) passed all 20 with zero skips (`cpu-dspark-tests.txt`). No production or original runtime source was changed by those CPU tests.

The selective reviewer found no blocker for an isolated smoke, while explicitly noting that the backend's advertised BF16 support did not prove native kernel dispatch. Actual native task `b2eaa3a11` exited 1 during warmup: `DSparkSpeculator.propose → precompute_and_store_context_kv → FlashAttentionImpl.do_kv_cache_update → _custom_ops.reshape_and_cache_flash` raised `Unsupported data type of kv cache: bfloat16`. The target/draft loaded and passed the startup dtype guards before this failure, but `/health` and the API checks were not reached. Host cleanup invariants matched. The originally executed overlay and driver are preserved in `dspark-v2-smoke/`; do not confuse them with later follow-up versions.

This directly disproves the initial assumption that advertised explicit BF16 KV support also covers the installed XPU native cache-write selector. A dedicated follow-up is limited to translating the unquantized draft backend selector while preserving BF16 allocations/specs and the unchanged FP8 target cache. Before another server launch, `run-native-cache-probe.sh` executes `native-cache-probe.py` on the actual B70: BF16 key/value tensors and BF16 caches, slot IDs `[0,15,16,-1]`, exact written values and untouched poisoned slots. It expects the `bfloat16` selector to reproduce the error and `auto` to preserve BF16 storage and exact cache contents. This native tensor probe supplements, not replaces, the full API gate. The `_custom_ops.py` wrapper is pinned at SHA256 `eb439f4656903c11c8789cf8fd0eb86300f69540cf07e29a0564a9476aedecb3`.

Native cache probe `b100b7827` exited 0 on `Intel(R) Arc(TM) Pro B70 Graphics`, Torch `2.13.0+xpu`: explicit `bfloat16` dispatch reproduced the exception; `auto` with the same BF16 key/value and cache tensors exactly matched the expected writes, preserved untouched poisoned slots and retained `torch.bfloat16` storage. Launcher/power were unchanged and no container remained. See `native-cache-probe-output.txt`.

Next predeclared server attempt is `dspark-v2-cache-fix-smoke/`, retaining identical model/config/payloads and changing only the guarded draft native selector normalization. No storage dtype, target KV, target quantization, rejection algorithm or sampling setting may change. The first attempt's driver/overlay remain frozen in `dspark-v2-smoke/`; the root driver will target the new output directory and refuse overwrite.

Selector fix worker commit `a6427070190e0e0c7fa852fff0bff3a0a9c389a2` integrated as `29e5ad3`; lead additionally pinned the native wrapper and made the symlink-test fixture independent of manifest order. The first CPU run after adding the wrapper pin passed 19 tests and failed that incomplete fixture (`b1dcbeeb3`, retained fixture-failure log); corrected fixture plus unchanged runtime code passed all 20, zero skips (`b82aa5e45`, `cpu-dspark-cache-fix-tests.txt`). Selective delta review passed for isolated smoke and confirmed context/query cache writes use the same implementation selector while allocations remain BF16.

**Corrected native DSpark smoke passed** (`b0581e85f`, exit 0): 3/3 canaries, 131/131 finite boundaries, 8/8 functional tasks and eight forced 128-token streams. Both code/prose families were stable across four repetitions. Actual installed source replay/hash checks passed. `python3 compare-smoke.py > smoke-comparison.json` verifies identical payloads and rendered prompt IDs for all 19 paired requests, with **19/19 exact output-token matches** (including 8/8 longer parity streams) against `target-v2-control/`. This is bounded greedy smoke evidence, not broad quality/distribution equivalence.

The eight parity streams recorded 543 draft steps, 3801 proposed tokens, 473 accepted draft tokens: `1 + accepted/steps = 1.8710865562` emitted tokens per completed speculative step. Do not compare these eager-mode tiny-prompt rates against historical graph-enabled MTP/DFlash. Both candidate/control cleanup checks passed. All raw successful artifacts and the executed overlay/driver are frozen in `dspark-v2-cache-fix-smoke/`.

### Predeclared 64K capacity gate

Startup reported only approximately 19K KV-token capacity. `run-dspark-64k.py` / `dspark-v2-64k/` therefore changes only maximum model length to **65664** (65536 input + 128 output) while keeping the corrected eager C1 DSpark stack, target, memory utilization and cache layout unchanged. If startup and the same API checks pass, it runs the existing byte-identical `qwen38_long_context_bench.py --base-url http://127.0.0.1:8000 --model qwen38 --out <cell>/long-context --lengths 65536 --near-limit 65536 --confirm-prefix-cache-disabled`, including its one warmup/six measured requests and token/transport validations. Retain any allocation rejection and stop rather than changing quantization/cache layout to force a result. Run the driver through explicit Bash with outer `timeout --signal=INT --kill-after=90s 2400s`; startup remains capped at 900s and long client at 1200s. Cleanup and production invariants are unchanged. No 64K, 160K, graph-mode or MTP comparison success is assumed.

## Tooling failures before launch

- Prime Lab `preview_action` unavailable: `connect ENOENT /tmp/prime-lab-1000/2c31a91a094d7226455d297a/lab.sock`. Side effects were described in chat; user had authorized isolated DSpark work. No Lab run was launched.
- First host inventory printed Bash `printf: --: invalid option` for headings; inventory/launcher/power commands still executed. A source-inspection command failed locally with shell quoting syntax error before SSH; corrected command confirmed installed source. Neither is a workload failure.

### 64K boundary failure and predeclared unchanged-overlay reuse

Task `b9892e7f3` failed during the first long warmup, not allocation: startup reports 74528 KV tokens at max length 65664, and all short checks pass. Thus the earlier ~19K short-config capacity was not a hard memory limit. At 65536 prompt +121 emitted tokens, scheduler supplies seven verification rows; native GDN requires eight (`spec_token == num_spec_decodes * (num_speculative_tokens + 1)`). Stream ends with HTTP/SSE error and health503; zero measured long trials, no valid throughput. Failed artifacts and executed driver/overlay remain in `dspark-v2-64k/`. Host unchanged.

Fresh official pinned [wrapper inspection](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/_xpu_ops.py) confirms the metadata-driven wrapper is shared, without a DFlash-specific branch. Reuse existing byte-identical `patch-vllm-qwen38-xpu-boundary.py`, not new kernel math: internal scratch padding only, previous acceptance counts/state slots preserved, real output prefix only. Before API launch, `bash run-native-boundary-probe.sh` must pass the existing FP16 native prefix/continuation oracle, including alternating lengths. Then `run-dspark-boundary-64k.py` repeats the same real 65536+128 public API journey (warmup +six measured, prefix disabled, token/transport checks), changing only that boundary overlay. New output `dspark-v2-boundary-64k/`, same pinned image, C1/eager, installed-source exact replay, owned-container cleanup and host invariants. No graph/160K or production promotion.
