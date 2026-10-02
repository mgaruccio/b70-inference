# First frozen-Glimmer recursive MTP pilot

**Development tier only.** This is the approved standalone pilot, not a serving-stack
change, production speedup, DFlash comparison, sampled-decoding claim, or a
standard-compliant benchmark under `BENCHMARKING_STANDARDS.md`. No persistent
launcher changes. The lead provisions/runs the dedicated A100 80GB and enforces
the **$25 total cloud cap**; this script neither provisions nor starts services.

## Fixed experiment contract

Target: `meta-models/Muse-Glimmer-30B`, revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, BF16, text-only
`AutoModelForImageTextToText` / `MuseGlimmerForConditionalGeneration`. All target,
embedding, final-norm and output-head parameters are frozen. Default attention is
SDPA, identically configured for candidate and target-only baseline.

For root final-normalized teacher state `h[t]`, step `d` consumes the **next** token
`x[t+d]` via the real `get_input_embeddings()` module (including Glimmer's embedding
normalization), computes

```
z = SiLU(D_d concat(hhat[t+d-1], embedding(x[t+d])))
hhat[t+d] = hhat[t+d-1] + sigmoid(g_d) * U_d z
p(x[t+d+1]) = frozen_output_head(hhat[t+d]) with Glimmer multiplier and softcap
```

The root is `hhat[t] = h[t]`; thereafter only predicted states in the target's
**post-final-norm representation space** are fed back. The trunk's learned final
RMSNorm is **not applied again**: its signed channel scale breaks residual identity
and produced odd/even state flips in the first prototype. A zero residual update
must preserve the state exactly. Blocks use FP32 parameters/computation; feedback
and LM-head input are cast to target BF16. The baseline has eight different blocks;
both recursive variants reuse one physical block. Checkpoints carry the
`postnorm-gated-residual-v1` contract; old repeated-norm heads are refused and must
be retrained. Saved teacher captures remain reusable because they contain genuine
post-normalized target states.

Three variants: `fixed-ce`, `shared-ce`, `shared-state`. Rank defaults to 64 (128
also supported). All train depth 8; evaluate prefixes 1/2/4/8. The ordered windows,
updates, batch size, learning rate and seed are equal, **not parameter budgets**.
At rank 64: 1,284,608 shared vs 10,276,864 unshared trainable parameters; rank 128:
2,562,560 vs 20,500,480. Checkpoints contain the small head only, plus run metadata.

Each step has CE on token `x[t+d+1]`, weighted `[1,1,.8,.8,.5,.5,.5,.5]` and divided
by the sum of weights. `shared-state` additionally uses default coefficient 0.2
on `(RMS-normalized MSE + cosine distance)` against teacher `h[t+d]`. Tokens are
teacher-forced during training; hidden states never are after the root. Optional
`--kl-weight` computes teacher logits on demand, using the frozen output head;
it defaults to zero. If enabled, label all three as **CE+KL**, not CE-only.
There are no saved full-vocabulary logits and no evaluation-driven checkpoint
selection: only the last prescribed update is saved. Set all budgets before
looking at heldout results.

The JSONL fixture has 24 ordinary text/code training sequences and 12 heldout
prompts: two each of code, prose, reasoning, structured, repetitive and high-entropy.
These are **raw text continuations**, not chat-template prompts. `--data` accepts
user JSONL with `id`, `family`, `split` (`train`/`eval`), `text`, and heldout
`category`. Whole sequences/families are split before tokenization; duplicate IDs,
canonicalized full text, token sequences and cross-split families are rejected.
Users must assign honest family labels; this does not infer semantic near-duplicates.
No truncation or random window-level splitting is performed. Captures store reusable
CPU BF16 final-normalized states and token IDs in `torch.save` files loaded strictly
with `weights_only=True`. Only `train` rows enter optimization.

## Cache and greedy verifier

The prefill returns the actual teacher root and next target-greedy anchor token.
Each pass recursively drafts from that root, consuming the anchor then its own
selected tokens. **One cached target forward on `[anchor, draft1, ...]`** verifies
all positions. Logits at row 0 verify draft1, not the anchor. For `k` accepted drafts,
keep exactly the old prefix plus anchor plus those `k` drafts; crop rejected KV;
reset the root to actual teacher hidden row `k`; carry row `k`'s argmax as the next
correct anchor (mismatch correction or full-acceptance bonus). The pending anchor
is emitted on the next loop; an EOS/final-budget anchor needs no extra forward.
Seeds and bonus/correction tokens are **not accepted drafts**.

Only ordinary Transformers `DynamicCache` with full/sliding dynamic layers is
accepted. The script checks every layer before/after rollback. It passes a negative
removal count to `crop`, supported by both older and current APIs. Every prompt,
output budget and speculative margin must fit inside **1792 tokens**, strictly
below the target's SWA 2048. Saturated, truncated, static or unknown caches are
refused. There is no cache-error retry or full-prefix generation fallback.

The no-spec reference uses the same target and cache loop at depth zero. Each
candidate is paired/interleaved with that baseline. **Strict identity is the default:**
any observed token-list difference fails the run. BF16 batch-shape-dependent argmax
differences were reproduced on the real A100 pilot, including with an identical
cloned prior cache. The explicitly opt-in `--record-divergence` mode reports all
pairs without asserting bitwise fidelity; it does not change drafting, target
verification, rejection, or rollback. It records exact-match rate, zero-based first
divergence position and generated-length differences. Diagnostic replay must still
reproduce the measured **candidate** tokens; genuine chosen-path teacher states
follow that candidate, not a divergent baseline. EOS is honored equally. No sampling
or production/lossless claims are made for diagnostic results.

## Predefined real end-to-end process (lead execution)

Environment/preconditions: dedicated idle A100 80GB; sufficient host RAM/disk for
the official BF16 model; model download authorization if required; CUDA PyTorch;
no other inference workload; isolated GPU container/venv **outside the Pi runtime**.
Use a provider-side spending/time limit before starting, retain billing evidence,
and terminate the instance before the $25 cap. Stop on unavailable model/class,
OOM or identity failure; do not replace Glimmer with a tiny model or alter the
baseline silently. CPU tests below are supplemental, not this E2E.

The inspected official Transformers source is commit
`35dff0957a99d50eaf85d7852a96fd29e59052d6` (Glimmer config requires 5.15-era support).
In the dedicated GPU runtime's Bash shell, retain the exact environment and commands:

```bash
# Install only in the isolated GPU environment, retaining its CUDA PyTorch build.
python -m pip install 'transformers==5.15.1' accelerate safetensors sentencepiece
RUN=/workspace/glimmer-pilot-20261002
mkdir -p "$RUN"
python -m pip freeze > "$RUN/pip-freeze.txt"
python -m torch.utils.collect_env > "$RUN/collect-env.txt"
nvidia-smi -q > "$RUN/nvidia-smi.txt"
git rev-parse HEAD > "$RUN/code-commit.txt"
set -o pipefail
set -x
P=scripts/experiments/glimmer_recursive_mtp.py
python "$P" validate
python "$P" capture --output "$RUN/capture.pt" 2>&1 | tee "$RUN/capture.log"
python "$P" train --capture "$RUN/capture.pt" --output-dir "$RUN/heads" \
  --rank 64 --updates 100 --batch-size 4 --seed 20261002 \
  2>&1 | tee "$RUN/train.log"
python "$P" evaluate --capture "$RUN/capture.pt" \
  --heads "$RUN/heads/fixed-ce.pt" "$RUN/heads/shared-ce.pt" "$RUN/heads/shared-state.pt" \
  --max-new-tokens 64 --repeats 1 --output "$RUN/evaluate.json" \
  2>&1 | tee "$RUN/evaluate.log"
```

Retain the shell command transcript as well as all listed artifacts, fixtures and
code commit. These are deliberately small pilot budgets, not an optimized training
recipe. A 128-token or repeated run must be declared beforehand and remain within
the cloud cap. Preserve failures; an existing output path is refused. The evaluation
JSON records completed/partial pairs and a failure reason if execution fails after
setup. Setup/training errors remain in the shell logs. No automatic retries.

Expected public-boundary results: successful capture and three final checkpoints;
12 prompts x 4 depths x 3 variants = 144 measured A/B pairs; `status: complete`.
Strict mode additionally requires all `exact_token_identity: true`. For the separately
approved numerical diagnostic, add `--record-divergence` and use a new output path;
completion means the matrix ran, **not** that fidelity passed. Acceptance may be zero
and speedup below one—both are valid pilot findings, not checkpoint-selection grounds.
Clean up by downloading artifacts and terminating the GPU instance; do not keep
inference servers or change production launchers. The lead reports the actual
commands, instance/environment, artifact location, elapsed time, cost and failures.

## Measurements and interpretation

JSON retains prompt/output token IDs, decoded text, raw seeds/drafts/accept counts,
per-position proposal/acceptance counts, acceptance histograms, accepted **drafts**
per verification pass, target calls/output token (including prefill, explicitly),
generated tokens excluding prompt, synchronized draft and verify seconds, separate
prefill/decode/total wall time, paired decode ratios, head parameter/checkpoint
sizes, peak allocated/reserved GPU bytes, command/config and environment.
Head transfers and warmup are excluded; the head lives on CPU during the target-only
measurement. CUDA reserved memory can remain high after earlier cases; peak allocated
is the useful live-memory comparison. Peak metrics include prefill. Both paths
request hidden states and execute strict cache checks; synchronized Python/HF
measurements are not a production scheduler benchmark. Raw repetitions are retained,
with median paired ratios in the summary.

Drift is **separate from live timings** and has two deliberately distinct diagnostics:

* `chosen_path_drift_*`: one separate teacher forward on the actual complete chosen
  token sequence, then recursive head rollouts at the measured verification roots,
  consuming **chosen target tokens**. Every step compares against the genuine
  target state for exactly that consumed prefix, including depths beyond the first
  bad head prediction. Hidden feedback is still predicted, never teacher-forced.
* `drift_*`: an untimed replay of actual drafting/verification. Steps through the
  first rejected prediction have a `chosen_token_prefix`; later steps are explicitly
  `conditional_after_rejection`, comparing against target states on the rejected
  draft branch. These are not live chosen-path drift or additional acceptance.

Diagnostic full-prefix recomputation is used only to obtain reference states, never
to repair or accelerate the measured decoding path. Very short EOS outputs can leave
some depths without samples. State metrics are normalized MSE and cosine distance.

## Research and targeted verification

Fresh primary-source research before implementation:

* [Recursive layer reference](https://github.com/IQuestLab/vllm-iquest-q1/blob/1f305c04f6d63b936e118e5aa4a9eb6b20e38a32/src/vllm_iquest_q1/mtp_recursive.py): one physical layer, shifted actual token embedding, normalized hidden feedback. Its training implementation is absent; no inferred training asymmetry is claimed.
* [Official pinned Glimmer config](https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/config.json) and [architecture](https://huggingface.co/blog/muse-glimmer): 52 layers, width 6656, vocabulary 202048, hybrid SWA 2048.
* [Inspected official model implementation](https://github.com/huggingface/transformers/blob/35dff0957a99d50eaf85d7852a96fd29e59052d6/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py): text-only public forward, actual embedding normalization, final hidden norm, output multiplier and tanh softcap. Runtime projection check refuses incompatible hidden-state semantics.
* [Cache docs](https://huggingface.co/docs/transformers/main/en/internal/generation_utils) and [inspected cache implementation](https://github.com/huggingface/transformers/blob/35dff0957a99d50eaf85d7852a96fd29e59052d6/src/transformers/cache_utils.py): rollback below sliding-window saturation; negative crop count avoids the deprecated positive absolute-length API.
* [PyTorch 2.14 numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html), freshly checked for the diagnostic continuation: batched and slice computations are not guaranteed bitwise-identical despite mathematical equivalence. This supports recording fidelity separately, not assuming every observed mismatch is harmless or changing target acceptance.

Targeted CPU tests use an existing isolated Docker image, with the checkout read-only,
no GPU or network, and no packages added to Pi:

```bash
docker run --rm --network none \
  --mount type=bind,src="$PWD",dst=/work,readonly --workdir /work \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=1 \
  --entrypoint python3 nvcr.io/nim/nvidia/kumo-relational:1.0.1 \
  -m pytest -q -p no:cacheprovider tests/test_glimmer_recursive_mtp.py
python3 -S scripts/experiments/glimmer_recursive_mtp.py validate
```

The CPU runtime has PyTorch `2.12.0a0+5aff3928d8.nv26.05`, Transformers `5.8.0`,
pytest `9.0.3`. Tests exercise the real HF hybrid DynamicCache rollback, decoder
control flow against a prefix-dependent oracle, zero/full/first/mid rejection,
EOS/budget/bonus alignment, exact no-spec token identity, split refusal, all-depth
backpropagation and freezing, token shift, recursive hidden feedback, parameter
budgets, softcap projection, safe checkpoint loading and same-prefix drift.
This runtime tests cache/head contracts, **not official 30B model execution**;
the real GPU E2E remains a lead-owned gate before pilot conclusions.
