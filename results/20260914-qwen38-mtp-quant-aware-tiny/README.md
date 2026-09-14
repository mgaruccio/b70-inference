# Native MTP quant-verifier feature tuning — tiny pipeline

**Tier: development. Tiny pipeline completed; acceptance regressed and no meaningful speedup was observed. Stock retained; no production promotion.**

## Scope and fixed baseline

Warm-start the existing 424,699,392-parameter `mtp.*` block from
`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision
`9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`. Use the existing native MTP4
vLLM XPU deployment on `inference-host`, not llama.cpp or an EAGLE architecture.
Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
Runtime: vLLM `0.27.2rc1.dev77+gac7509e2b`, PyTorch `2.13.0+xpu`,
Transformers `5.15.0` (checked in the pinned image).

Keep the target GPTQ tensors, embeddings, target output head, KV dtype, and
existing draft RTN INT4 quantizers unchanged. This tests training on
quantized-verifier features, not fake-quant QAT. The tuned native weights are
packed by the same existing RTN loader as the stock weights.

No persistent launcher edits or stock checkpoint overwrites. Its original SHA256
is `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

## Fresh source research affecting the implementation

- [Pinned vLLM native MTP](https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/models/qwen3_5_mtp.py):
  independent input norms, concatenation `[embedding, hidden]`, FC, one full-attention
  gated-Q decoder, final norm. Capture the target state before MTP's own input norm.
- [Pinned proposer](https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/step3p5.py)
  and [base input construction](https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/llm_base_proposer.py):
  token IDs shift; target hidden states and positions do not. For first-depth CE:
  `ids[1:-1]`, `hidden[:-2]`, `positions[:-2]`, labels/mask `ids[2:]/mask[2:]`.
- [SpecForge native objective](https://github.com/sgl-project/SpecForge/blob/3d64e7a61f5fcc7f7d78ba6164c881f831943947/specforge/core/mtp.py):
  native MTP uses final target hidden states and token CE, not EAGLE3 auxiliary features.
  [XPU PR](https://github.com/sgl-project/SpecForge/pull/769) does not establish native
  MTP support on this exact stack. Use installed PyTorch/Transformers rather than
  replacing the XPU environment with SpecForge's dependencies.
- [Prior B70 head tune](https://huggingface.co/rwmacy/qwen3.8-27b-mtp-head-v8-b70):
  reports approximately 3.2 hours training, but real-workload replay was effectively
  parity (L 3.01 vs 3.00; +0.20 tok/s). It is feasibility evidence, not a promised gain.
- [Official MBPP splits](https://github.com/google-research/google-research/tree/master/mbpp):
  select training IDs 601–700 and disjoint validation IDs 511–534. Only descriptions
  and public tests enter prompts, never reference solutions. This is a Python coding
  pipeline smoke set, not a claim of representative personal-agent coverage.

## Declared real end-to-end process

All inference/training runs execute on `inference-host`, outside interactive Pi.
Preconditions: no running GPU containers, stock model and five cookbook patches
available, 275 W power cap, pinned image cached. Initial disk check: 209 GiB free.
Artifacts root: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-mtp-quant-aware-tiny/`.

1. Stage `scripts/experiments/qwen38_mtp_tune_probe.py` and its existing probe/reference
   imports under `source/`; run `python3 source/qwen38_mtp_tune_probe.py --make-corpus corpus`.
2. Baseline: `python3 -u source/qwen38_mtp_tune_probe.py --out stock-baseline --prompts corpus/heldout.jsonl --tokens 256`.
3. On-policy generation: `python3 -u source/qwen38_mtp_tune_probe.py --out onpolicy-train --prompts corpus/train.jsonl --tokens 256 --generate`.
4. Prove unmodified native weight export/reload before tuning. Capture complete,
   aligned sequences from the exact verifier; verify capture does not perturb outputs.
5. Unload verifier; train only native MTP weights on cached sequences, with separate
   heldout records. First-depth teacher forcing is a smoke test, not recursive MTP4 training.
6. Reload tuned native weights through the existing RTN draft quantizers. Compare
   stock/tuned in forward/reverse order through `/v1/chat/completions` and `/metrics`.
   Retain outputs, latency, decode tok/s, proposed/accepted counts and per-position counters.
   Run output/functional checks; do not infer serving improvement from offline loss.
7. Stop experimental containers; verify original launcher and power remain unchanged.

Serving cells use the existing cold-cache reference adapter: 212,992 context, C1,
FP8 KV, balanced performance mode, graph sizes `[1,2,4,8]`, existing prefill guard,
non-thinking, greedy sampling and seed 42. Timing uses a 64-token warmup excluded
from summaries, then 256 forced output tokens. Generation permits natural EOS.
These deliberate benchmark settings differ from production's thinking/cache defaults
and must be identical across compared heads. All four per-position acceptance
fractions use total draft passes as their denominator; boundary-truncated passes
remain included. A draft-pass counter is not asserted to count every verifier call.

Raw commands, SSE events, token IDs, metrics deltas, launcher copies, server logs,
source/prompt hashes and failures are retained under each remote cell. Completed
small summaries and final observed outcomes will be copied here. No large corpus
or cloud rental is authorized by a positive training-loss change alone.

## Observed first stock runs

`stock-baseline.json`: 24/24 transport-pass requests, 90.3076 median post-first
streaming decode tok/s, 2.61222 accepted draft tokens per draft pass, 65.3055%
draft-token acceptance. Position acceptance per draft pass: 0.94947, 0.86545,
0.43596, 0.36134. These are stock-only smoke results, not an A/B improvement.

`onpolicy-train.json`: 100/100 completed sequences, 10,370 generated tokens,
maximum full prompt+output length 593 tokens. Natural-EOS lengths differ from
the forced-length baseline, so their throughput summaries are not comparable.
Both cells reported unchanged launcher and power and stopped their serving container.

Natural-EOS heldout generation: `python3 -u source/qwen38_mtp_tune_probe.py --out onpolicy-heldout --prompts corpus/heldout.jsonl --tokens 256 --generate`.
Functional check: `python3 -u source/qwen38_mtp_tune_probe.py --check-outputs onpolicy-heldout`.
`stock-functional.json` records 16/24 public-test passes. Five failures hit the
256-token cap and left incomplete fenced answers; three stopped naturally but
failed assertions. This limited-budget baseline is not a general quality score.
The tuned head must be compared with the same budget and extraction policy.

## Integration and identity gates

The integrated runtime/trainer tests passed **100/100** in the pinned CPU-only
image (`check-runtime.sh`), including full stock export/overlay identity and the
actual RTN packing contract. Initial attempts failed before testing (fish parsed
a Bash heredoc) and then at one missing staged reference file (99/100 passed);
the corrected staging passed all tests. No dependencies were installed.
A read-only review found no blockers in the native architecture, shifts, overlay,
or capture seam. It does not replace the live XPU tests.

`stock-overlay.json` / `stock-overlay-identity.json`: stock overlay served all
24 prompts. Only 19/24 forced 256-token outputs were fully identical, but **24/24
matched through the first terminal token**. The five first divergences were at
zero-based positions 102, 150, 165, 56, 57; their first terminal positions were
91, 144, 155, 50, 49 respectively. Thus all observed differences were in forced
post-answer continuation, not the actual answer. Full raw differences remain
retained. Image, quantization, patches, context, sampling and seed matched.

**Protocol refinement before training:** natural-EOS acceptance and functional
checks are primary; forced-length throughput is only a stress measurement.
Stock/stock and stock-overlay/stock-overlay repeats are also being retained to
measure run variability rather than treating one pass as a speedup.

`identity-repeats.json`: stock/stock repeated answers matched 24/24 through the
first terminal token; overlay/overlay and the second stock/overlay comparison
matched 22/24 (mbpp-515 and mbpp-523 differed). Therefore exact serving equivalence
is **not established across runs**, despite bitwise stock export identity. This
uncertainty remains part of the exploratory result, not an excuse to claim a
quality-neutral speedup. Persistent serving defaults are not promoted.

## Observed feature capture and training

`capture-features.sh` completed both 124-sequence target-only replay cells.
`capture-parity.json`: **124/124 identical one-token outputs** with capture off/on.
`dataset.json` describes the converted 100-train/24-heldout complete-sequence
records. Each carries `prompt_id`; the trainer rejects overlapping prompt IDs.
The collector writes raw tensors, and the converter adds the corpus identity
without retokenization. This validates the capture toggle, **not** equality of
prefill and speculative-decode hidden states.

`train-tiny.sh` ran the real native 424.7M-parameter head on the B70 after verifier
unload: 25 AdamW updates, four complete sequences/update, lr 5e-6, seed42,
FP32 masters/FP16 autocast, logits chunks32, no MTP-core fake quantization.
It exported `tuned-mtp.safetensors` (849,400,392 bytes) outside the stock checkpoint.
`training.json`: heldout token CE **0.137184 → 0.142251** over 3,168 positions in
24 sequences. This is a slight regression, not evidence of improved acceptance.

The final serving comparison is `compare-heads.sh`: stock/tuned/tuned/stock,
24 natural-EOS heldout prompts/cell, maximum256 tokens, identical runtime adapter
and RTN quantization for both heads. Each cell also runs the same isolated public
functional tests. No architecture, depth, sampler, or quality-gate changes are
bundled with the new head.

## Final matched natural-EOS result

All four `compare-heads.sh` cells completed through the real public API.
`comparison.json` and each `abba-*/` directory retain the measurements, raw
per-request results/requests, functional results, prompts and exact launch commands.

| Cell | Accepted draft tokens/pass | Draft acceptance | Median decode tok/s | Public functional tests |
|---|---:|---:|---:|---:|
| Stock A1 | 3.27395 | 81.8489% | 106.9215 | 17/24 |
| Tuned B1 | 3.22800 | 80.7000% | 106.9799 | 17/24 |
| Tuned B2 | 3.24364 | 81.0910% | 106.8669 | 17/24 |
| Stock A2 | 3.25772 | 81.4430% | 106.5101 | 17/24 |

Pooled counter ratio: **3.26581 → 3.23580 accepted draft tokens/pass (−0.919%)**;
draft acceptance **81.6454% → 80.8951% (−0.7502 percentage points)**.
The mean of the two run-level decode medians is **106.7158 → 106.9234 tok/s
(+0.195%)**. This is not a statistically established speedup and does not meet
the acceptance objective. The direction of acceptance is worse in both pairs.

Both within-head repeats and both cross-head comparisons matched **23/24 full
natural outputs**; the sole variable prompt was mbpp-515 in every comparison.
All four runs had exactly the same seven failing public-test task IDs. Thus no
functional regression appeared in this small test, but bitwise determinism or
general quality equivalence is not established. The 256-token cap truncates some
answers; this is not a full MBPP+/HumanEval+ quality qualification.

**Decision:** retain stock. Do not scale data or promote this particular tune.
The working export/capture/train/reload loop is the result; the quant-aware
acceptance hypothesis remains unproven. This run used first-depth CE on a very
small coding set, not a BF16-feature control or multi-depth training experiment.
No cloud GPU was rented. Heavy features and both head files remain on
`inference-host` under the artifact root, outside Git and the stock checkpoint.

## Closeout verification

- 100 pinned-image runtime/trainer tests passed; 6 lead harness/protocol tests passed.
- Final artifact checks confirmed identical serving settings across all four ABBA
  cells, 24 raw request results/cell, identical failed functional-task IDs,
  124 capture matches and 25 optimizer steps.
- `cleanup.txt`: no running Docker containers; no render-device holders reported
  by `fuser`; production launcher hash unchanged; power cap remains275000000.
- All experiment containers are stopped. Both writing worktrees were integrated
  and removed cleanly. The original `local-dev-model` working tree was not edited.
- `logs/` retains command output (trailing whitespace normalized), including failed setup/staging attempts;
  heavy cached tensors/checkpoints remain host-local. Small checked-in evidence
  is approximately0.63MB. No model promotion or additional experiment is running.
