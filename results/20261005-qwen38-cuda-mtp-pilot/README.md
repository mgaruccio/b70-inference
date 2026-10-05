# Qwen3.8 native recursive MTP: completed CUDA training pilot

**Tier: development. Completed training and evaluation; not promoted.** This is a small coding pilot, not a full MBPP leaderboard, a standard publishable performance benchmark, or production qualification.

## Outcome

The native shared MTP head completed **100 actual optimizer updates** with the target trunk frozen. Ten exported BF16 MTP tensors changed. Dev weighted token-CE objective decreased from **1.163755 to 1.099996**. The checkpoint loaded into the real native vLLM serving path and completed four 16-request held-out test cells.

At draft depth four, accepted draft tokens/pass rose from **3.286765 to 3.330855 (+1.34% relative)**. Decode throughput was slightly higher, but **API end-to-end throughput was essentially unchanged**. This does not establish a material speedup. All four test cells produced identical token sequences and the same **14/16** functional passes.

The upstream head is already recursive and strong: its stock depth-eight development result was **4.756989 accepted draft tokens/pass**. Those stock results are not benefits created by this training run. The earlier quantized B70 baseline uses different hardware, precision and prompts and is not pooled with these numbers.

## Model, hardware and precision

- One rented **NVIDIA A100-SXM4-80GB**, Shadeform/massedcompute, $1.38/hour advertised rate. The owned pilot instance was deleted after final archive verification; a subsequent provider GET showed no pilot lease. No persistent B70 launcher was changed.
- Target: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, unquantized BF16.
- Serving image: `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`; vLLM **0.27.1**, Torch **2.13.0+cu130**, Python 3.12.
- Dedicated trainer: Torch **2.14.1+cu130**, Transformers **5.15.1**, Python **3.10.12**. Dependencies and workloads remained outside interactive Pi.
- **424,699,392 trainable parameters** in the native one-layer MTP head; FP32 masters/AdamW, CUDA BF16 autocast, BF16 exports. Original target embedding and LM head remained frozen. No target trunk was constructed in the head trainer. **No RTN or other quantization** was used.
- Depth four, weights `[1, 1, 0.8, 0.8]`, LR `1e-6`, seed 42, roots 8, gradient accumulation 1, maximum sequence 1024, logits chunk 64. Token CE only: **no state-supervision objective** yet.
- Training consumed 100 sequences/29,755 input tokens; supervised loss tokens by depth: `[14150, 800, 800, 800]`. Peak allocated training CUDA memory **13.52 GiB**, reserved **13.87 GiB**. These are head-training peaks, not whole serving memory or incremental deployment overhead.

## Data and defined end-to-end process

Public MBPP only: train tasks **601–664 (64)**, dev **511–526 (16)**, test **11–26 (16)**. Source, exact task IDs and request/check hashes are in `data-protocol.json`. Prompts include task descriptions and public checks, not reference solutions. Native teacher continuations/hidden states were generated for train/dev. Test requests were fixed before training and did not enter optimization or checkpoint selection. Public pretraining exposure and historical benchmark exposure are unknown.

The executed journey was:

1. Run native stock depths **0/1/2/4/8** on dev through the real loopback OpenAI-compatible streaming API.
2. Capture public train/dev teacher states using the existing native vLLM capture path; validate capture counts and alignment.
3. Run a one-update CUDA smoke and verify changed exported weights; then execute 100 updates, exporting at 0/50/100 and selecting on dev BF16-export objective. Step 100 was selected.
4. Measure teacher-forced state drift at depths 1–4 on four dev captures (32 roots/depth).
5. Restart the same native server for test **stock → trained → trained → stock**, all at depth four. Candidate changes only the exported MTP weights, via `B70_MTP_WEIGHTS=/mtp.safetensors`.
6. Warm each serving cell separately, then measure 16 requests at concurrency one, greedy/seed 42, thinking off, natural EOS with a 512-token cap. Incomplete outputs fail functional checks. Evaluate bare Python or one whole Python fence in the existing no-network, read-only, capability-dropped CPU sandbox.
7. Retain requests, SSE events, results, commands, configurations, runtime/server logs, captures and checkpoints. Archive directly to R2, download and checksum-verify the complete final archive before deleting the owned lease.

Serving uses BF16 KV (`auto`), context 8192, batch-token limit 2048, memory utilization 0.92, balanced mode, graph sizes `[1,2,4,8,16]`, aligned Mamba cache, no async scheduling and no prefix caching during timing. Capture deliberately enables prefix caching with unique request salts. Exact server, smoke and training argv are in `commands.json`; the executed driver `./pilot-command.py`, API helper and complete environment are retained in the raw archive.

The exact training command was:

```sh
/home/shadeform/qwen-mtp-env/bin/python /home/shadeform/qwen-mtp-code/scripts/experiments/qwen38_train_mtp.py \
  --model /home/shadeform/qwen-model \
  --train-dir /home/shadeform/qwen-mtp-run/capture-train/train \
  --eval-dir /home/shadeform/qwen-mtp-run/capture-dev/heldout \
  --recursive-depth 4 --roots 8 --depth-weights 1 1 0.8 0.8 \
  --max-length 1024 --grad-accum 1 --logits-chunk 64 \
  --device cuda --seed 42 --lr 0.000001 --steps 100 --checkpoint-every 50 \
  --output /home/shadeform/qwen-mtp-run/training/tuned-mtp.safetensors
```

## Stock upstream baseline: development split

| Draft depth | Accepted drafts/pass | Draft acceptance | Median decode tok/s | Median API E2E tok/s | Functional |
|---|---:|---:|---:|---:|---:|
| 0, no speculation | — | — | 27.682 | 27.440 | 15/16 |
| 1 | 0.964662 | 96.47% | 49.059 | 47.812 | 15/16 |
| 2 | 1.858542 | 92.93% | 66.372 | 64.334 | 15/16 |
| 4 | 3.151181 | 78.78% | 90.842 | 83.062 | 15/16 |
| 8 | 4.756989 | 59.46% | 104.253 | 94.503 | 15/16 |

## Held-out test: matched native depth-four ABBA

| Cell | Accepted drafts/pass | Draft acceptance | Median decode tok/s | Median API E2E tok/s | Functional |
|---|---:|---:|---:|---:|---:|
| A1 stock | 3.286765 | 82.17% | 89.754 | 79.106 | 14/16 |
| B1 trained | 3.330855 | 83.27% | 91.729 | 78.518 | 14/16 |
| B2 trained | 3.330855 | 83.27% | 91.964 | 79.107 | 14/16 |
| A2 stock | 3.286765 | 82.17% | 89.163 | 79.166 | 14/16 |

Acceptance excludes bonus tokens. These are accepted **draft** tokens per recorded draft pass; the counters are not asserted to count every target verification call. Position counts in `comparison.json` are unconditional joint survival, not conditional acceptance probabilities.

All 16 test token sequences matched across repeats and between stock/trained. Tasks 15 and 26 failed functional checks in every test cell. Test does not include a no-spec arm: this sample establishes stock/trained equivalence, **not universal identity with no-spec**. No uncertainty interval or statistically established speedup is claimed.

## State drift: teacher-forced development diagnostic

| Depth | Stock cosine | Trained cosine | Stock relative L2 | Trained relative L2 | Stock KL | Trained KL |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.61201 | 0.61234 | 0.94788 | 0.94759 | 0.06489 | 0.05105 |
| 2 | 0.51610 | 0.51748 | 1.13792 | 1.13446 | 0.11350 | 0.10526 |
| 3 | 0.45964 | 0.46333 | 1.21738 | 1.20985 | 0.22787 | 0.12769 |
| 4 | 0.43975 | 0.44609 | 1.23276 | 1.22422 | 0.99600 | 0.86160 |

KL is teacher-to-head next-token KL. These compare post-final-norm states with true-token feedback, not free-running rollout drift or serving acceptance. Small cosine/L2 improvements and lower KL do not establish that state distance is causal; a useful MTP representation need not equal the trunk representation.

## Artifacts, checks and limitations

R2 directory: `r2:ml-archive/2026-10-05/cache/b70-evals/20261005-qwen38-cuda-mtp-pilot/`.

- Complete `final.tgz` (**4,743,966,720 bytes**), downloaded/read-back verified SHA256: `508f6d405927058aa2ea5b76de837617f9f1395a99ff9056e616932c85d9271b`.
- Final `tuned-mtp.safetensors` (**849,400,392 bytes**), uploader-reported SHA256: `6cd0b069662ca8407745c6bcb9e2bd9438cd08e50501521e6ebd9a1b903abe2f`; also present in the verified full archive.
- Local archive: `/home/mike/b70-evals/20261005-qwen38-cuda-mtp-pilot/final.tgz`. Intermediate `baseline.tgz`, `capture.tgz`, `training.tgz` and `bootstrap.tgz` are also archived in R2.
- Native CUDA trainer tests: **65 passed, 1 skipped** (21.84s), preserved in `trainer-tests.txt`. The skipped test needs an installed vLLM source tree in the trainer environment; the real serving image independently exercised both patches and completed the API comparison.
- Setup failures were corrected on the same lease and retained: two missing test-fixture files, a local fish command-quoting failure, and use of `python` in an image that supplies `python3`. The final corrected workflow exited **0** after **59m8s**, including final evaluation, archival and teardown.

This pilot does not cover prose/reasoning/structured workloads, batching, long context, adaptive depth, state supervision, alternative small heads, alternating recurrence, or a larger held-out sample. BetterBench, `vllm bench serve`, the repository long-context sweep and full quality-regression suite were **not** run. Accordingly, this remains development evidence, not a standards-compliant community performance claim. No production configuration or longer training run was launched based on this small gain.

## Primary sources informing this run

- [Official Qwen config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json): native head configuration and target dimensions.
- [Pinned vLLM native MTP implementation](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/qwen3_5_mtp.py): recursive reuse of the single MTP layer; capture/overlay source anchors verified against this image.
- [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/): native MTP serving configuration.
- [PyTorch autocast source](https://github.com/pytorch/pytorch/blob/main/torch/amp/autocast_mode.py): explicit CUDA BF16 autocast/support checks with FP32 master parameters.
- [Official MBPP source/splits](https://github.com/google-research/google-research/tree/master/mbpp): disjoint public train/dev/test task IDs.
