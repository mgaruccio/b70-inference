# Why the tiny native MTP tune failed to improve acceptance

Follow-up diagnosis only. No optimizer updates, serving changes, or model writes.
The saved stock/tuned heads and cached train/heldout sequences were evaluated in
the same pinned image on inference-host. All diagnostic containers auto-removed;
final `docker ps` reported no running containers.

## Measured distinction: generalization versus quantization

`bash fourway-eval.sh` evaluates the same 10,370 train and 3,168 heldout positions
under four frozen-head conditions. The BF16 exports are loaded as FP32 parameters
with the trainer's FP16 autocast. The RTN conditions apply the existing symmetric
group-128 quantization/effective-weight formula to each MTP matrix, then evaluate
with dense GEMMs. The shared frozen output head is the existing effective RTN
head in all conditions. This isolates weight precision in the offline model;
**it is not an execution of the real serving INT4 GEMM or recursive MTP4**.

| Head / core weights | Train CE | Train argmax agreement | Heldout CE | Heldout argmax agreement |
|---|---:|---:|---:|---:|
| Stock / BF16 export | 0.169616 | 94.3587% | 0.137184 | 95.8333% (3036/3168) |
| Tuned / BF16 export | 0.078927 | 98.2160% | 0.139529 | 95.8649% (3037/3168) |
| Stock / RTN effective | 0.178038 | 94.2816% | 0.141409 | 95.8649% (3037/3168) |
| Tuned / RTN effective | 0.086759 | 97.8303% | 0.142041 | 95.5492% (3027/3168) |

This is strong evidence of fitting the tiny training set without generalization:
train loss more than halved, while heldout loss worsened. It is no longer merely
an inference from noisy per-batch training losses. Exported non-RTN token accuracy
is essentially unchanged on heldout (+1 correct token). RTN turns that into ten
fewer correct tokens versus quantized stock. Most of the training-set gain
survives RTN, so "quantization erased the entire tune" is false.

The original in-training heldout CE after the update was 0.142251 on unsaved FP32
masters. The saved BF16 export scores 0.139529 here. These are distinct precision
stages, not interchangeable scores. Neither beats stock heldout CE. Original
FP32 master weights were not retained, so this diagnostic cannot rerun that exact
post-update master state.

## How much changed after packing?

`bash weight-delta.sh` runs the actual existing MTP packer extracted from its
installer on CPU, with input weights first cast to FP16 as in the serving loader.
The first source-extraction attempt failed before comparison because the helper
is embedded as a string in the installer; its error is retained.

- 188,635,703 / 424,699,392 BF16 parameter values changed (about44.4%).
- 1,410,531 / 424,673,280 INT4 matrix codes changed (about0.332%).
- 21,665 / 3,317,760 group scales changed (about0.653%).
- Final norm, embedding pre-FC norm, and Q/K norm exports were unchanged;
  hidden-state pre-FC norm changed only33/5120 values.

Code-change percentages are not functional importance or a count of all changed
effective weights: changing a scale affects its group. We cannot distinguish a
zero master update from an update rounded away in unchanged BF16 norm exports.
The frozen evaluations above, not code sparsity alone, establish the behavior.

## Other findings and uncertainties

- Serving first-position acceptance fell95.3567%→94.4556%; the failure is not
  solely an inability of first-step CE to optimize late speculative positions.
- The observed pooled acceptance change remains small: −0.919%. A paired
  prompt-cluster bootstrap (10,000 draws, seed42, resample24 prompt IDs with
  replacement, retaining both runs/head within each ID) gives an exploratory
  percentile interval of [−2.514%, +0.738%]. This includes zero and is not a
  publishable significance claim. Excluding variable-output mbpp-515 yields−0.623%.
- First-step teacher forcing never feeds the head its own recursive hidden states;
  later MTP4 positions therefore remain an objective/distribution mismatch.
- Capture-on/off124/124 identity does not establish prefill/decode hidden parity.
- All25 raw gradient norms exceeded clipping threshold1, but clipping with AdamW
  does not imply a proportionally smaller parameter update. Do not raise the clip
  threshold on that evidence alone.
- No proven shift, chunked-gradient, or loader-order bug was found. The low stock
  loss supports correct broad alignment, but is not itself a full equivalence test.
- `B70_MTP_BF16_DRAFT` disables checkpoint quantization during construction; its
  name does not establish BF16 compute. The serving launcher explicitly uses
  float16. Do not change training autocast based on that environment name alone.

## Recommended next attempt, in order

1. **Evaluate the deployed representation during tuning.** At step0 and frequent
   intermediate checkpoints, measure matched train/dev CE and argmax before and
   after BF16 export plus RTN. Retain the stock checkpoint as an eligible winner;
   reject candidates that worsen post-RTN development agreement. Confirm selected
   candidates in the real serving kernel, not only effective dense weights.
2. **Reduce overfitting pressure before scaling compute.** A bounded next ablation
   can train only the existing fusion FC and input norms, freezing attention/MLP,
   versus the current full-head recipe. This preserves native architecture. Use
   early checkpoints and conservative updates rather than blindly adding epochs.
   Lower LR is an experiment, not a guaranteed fix: tiny changes may round away.
3. **Add genuine core quantization-aware training if the precision gap persists.**
   Match FP16 input conversion, group128 symmetric RTN, FP32 code calculation and
   FP16 scales in the forward pass with FP32 masters/appropriate gradient surrogate.
   Current training is quantized-verifier-feature-aware, not core fake-quant QAT.
   Test this as a separate factor rather than simultaneously changing everything.
4. **Then improve coverage and depth.** Use a modest, more representative corpus
   spanning coding, tools, reasoning and actual agent requests; MBPP alone is a
   plumbing smoke set. Once first-position post-quant agreement is sound, add
   recursive-depth training matching the existing MTP4 state/position/cache path.
5. **Protect final evaluation.** The original24 heldout prompts have now informed
   diagnosis/model selection; treat them as development data and reserve a fresh,
   untouched prompt set for the next final ABBA comparison.

Do not generate millions of tokens or switch draft architectures based on this
result. The working pipeline is useful, and the quantized-verifier hypothesis is
not disproven. The immediate defect in experimental design was selecting a final
full-head update without checking generalization and its post-quantized behavior.
