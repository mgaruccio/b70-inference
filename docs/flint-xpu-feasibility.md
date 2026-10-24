# Flint XPU feasibility: blocked before target serving

**Development diagnostic only. Full-model HTTP E2E: NOT RUN.** No Flint model
port, drafter, conversion, kernel, download, training, or persistent launcher is
included. No speed, quality, or DFlash compatibility claim is made.

## Exact scope and sources

- Runtime: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
  `vllm==0.27.2rc1.dev77+gac7509e2b`, `torch==2.13.0+xpu`.
- Candidate: [mattbucci/Flint-AWQ, revision
  `2cae47a65e71d151a002172d18bcf2589fe09177`](https://huggingface.co/mattbucci/Flint-AWQ/tree/2cae47a65e71d151a002172d18bcf2589fe09177).
  Its [config](https://huggingface.co/mattbucci/Flint-AWQ/raw/2cae47a65e71d151a002172d18bcf2589fe09177/config.json)
  specifies AWQ GEMM, INT4, group size 64, `zero_point: true`. Its
  [README](https://huggingface.co/mattbucci/Flint-AWQ/raw/2cae47a65e71d151a002172d18bcf2589fe09177/README.md)
  describes standard AWQ `qweight/qzeros/scales`, including routed experts;
  CUDA/SGLang validation is not Intel validation.
- Pinned [AutoAWQ](https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/layers/quantization/auto_awq.py#L547-L567)
  selects an XPU WNA16 expert backend. This does **not** establish AWQ MoE support.
- The [XPU converter](https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/layers/fused_moe/oracle/int_wna16.py#L1021-L1081)
  assumes GPTQ input-axis packing, ignores the quantization config, and transposes
  the packed tensors. Its [dispatch](https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/layers/fused_moe/oracle/int_wna16.py#L1574-L1609)
  neither passes nor returns AWQ zero points.
- [XPUExpertsWNA16](https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/layers/fused_moe/experts/xpu_moe.py#L267-L302)
  documents **symmetric** INT4; the native expert invocation has no zero-point
  input. AWQ's noncentral integer zero points cannot simply be discarded.

Reproduced on `inference-host` (Intel Arc Pro B70) on 2026-10-24 using the
already cached pinned image. The checked-in diagnostic returned **2 / INCOMPATIBLE**;
the wrapper accepted that expected reproduction result. This is not model inference.

## Minimal reproduction (no kernel dispatch)

The script calls the real installed converter, unchanged, with tiny CPU AWQ
INT4 tensors: two experts, input 128, intermediate 64, BF16 scales, packed zero
points 5 and 7. Constant weight nibbles are valid in AWQ's interleaved packing.
It does not load weights, construct a serving model, or invoke a MoE kernel.

Observed installed-runtime result (matching the source prediction):

- Input AWQ int32 shapes: `((2, 128, 16), (2, 64, 16))`.
- Required native uint8 shapes: `((2, 128, 64), (2, 128, 32))`.
- Actual converter shapes: `((2, 16, 512), (2, 16, 256))`.
- Both returned zero points: absent; exit **2**, `INCOMPATIBLE`.

The actual run copied the checked-in script to `/tmp/probe-flint-xpu-awq.py` on
`inference-host`. Its retained output is
`/home/mike/b70-evals/flint-xpu-spike/awq-converter.log` (`probe_exit=2`). Python in
the container was 3.12.3. No install or model download was needed; the disposable
container was removed on exit. The GPU was exposed only for XPU-runtime detection;
probe tensors and conversion remained on CPU. Reproduce from a checkout on that
host with the following Bash commands (the actual mount source was the staged
`/tmp/probe-flint-xpu-awq.py` instead of `$PWD/scripts/probe-flint-xpu-awq.py`):

```bash
# From the repository root, in Bash:
mkdir -p "$HOME/b70-evals/flint-xpu-spike"
rc=0
docker run --rm --pull=never --network=none --device=/dev/dri \
  --label pi.flint-xpu-spike=awq-probe --entrypoint=python \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -v "$PWD/scripts/probe-flint-xpu-awq.py:/probe-flint-xpu-awq.py:ro" \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  -u /probe-flint-xpu-awq.py \
  > "$HOME/b70-evals/flint-xpu-spike/awq-converter.log" 2>&1 || rc=$?
printf 'probe_exit=%s\n' "$rc" >> "$HOME/b70-evals/flint-xpu-spike/awq-converter.log"
test "$rc" -eq 2
```

The script has no compatibility-pass exit: **3** means unverified, including
missing dependencies, changed versions/API, or matching metadata without
numerical/kernel verification. `--help` exits normally without torch/vLLM.
The log prints expected/installed versions, converter location/signature,
shapes, zero-point presence, and the no-kernel/no-HTTP boundary.

Local checks (stdlib only, not runtime verification):

```bash
python3 -S scripts/probe-flint-xpu-awq.py --help
python3 -m unittest discover -s tests -p test_flint_xpu_awq_probe.py -v
```

Lead verification: **7 tests passed** on Python 3.14.7; both Python files passed
syntax checks, dependency-free help exited 0, and missing-runtime execution exited
3 / `UNVERIFIED`. These are diagnostic checks, not full-model serving tests.

## Remaining target and DFlash prerequisites

Stop here: a correct asymmetric AWQ expert path or a separately approved target
format is required before a credible B70 target-serving attempt. Merely repacking
AWQ without preserving its zero points is not a solution. No such change is made.

Even after that blocker is resolved, upstream [Flint at
`fc3b190acf42489a3930228e0d09d2dc2e4c11f9`](https://github.com/vllm-project/vllm/blob/fc3b190acf42489a3930228e0d09d2dc2e4c11f9/vllm/models/flint/model.py)
needs a stock-XPU port: its attention unconditionally requires FlashInfer and
CUDA fused QK/cache operations. Pinned Triton attention advertises sinks and
sliding windows, but their correctness on B70 is not established by this probe.

There is no released trained Flint drafter in this spike. Internal model-runner
auxiliary hidden-state taps, layer/residual conventions, multi-token target
verification and KV rollback remain unverified. Flint is pure attention: there
is **no GDN recurrent-state requirement**. Once the actual target works, the lead
must run the planned real HTTP chat/stream/tool/EOS checks before assessing
DFlash verification; n-gram success would not establish DFlash compatibility.
