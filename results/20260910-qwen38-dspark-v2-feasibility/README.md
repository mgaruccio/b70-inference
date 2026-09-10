# Qwen3.8 DSpark V2 feasibility — development only

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

**Blocked at the unpatched XPU V2 target correctness prerequisite. DSpark was not loaded, and no DSpark throughput or acceptance measurement exists.**

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

## Tooling failures before launch

- Prime Lab `preview_action` unavailable: `connect ENOENT /tmp/prime-lab-1000/2c31a91a094d7226455d297a/lab.sock`. Side effects were described in chat; user had authorized isolated DSpark work. No Lab run was launched.
- First host inventory printed Bash `printf: --: invalid option` for headings; inventory/launcher/power commands still executed. A source-inspection command failed locally with shell quoting syntax error before SSH; corrected command confirmed installed source. Neither is a workload failure.
