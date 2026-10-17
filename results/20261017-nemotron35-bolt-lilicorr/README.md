# Bolt LiLiCorr capability and development experiment

Status: **blocked at debug actual-model activation** by an existing SYCL `OUT_PROD` layout assertion. Development tier only; no timing benchmark, source change, promotion, or speed claim. Original accepted Bolt restored and independently verified.

## Frozen baseline and discovery

Accepted source `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`, upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`; repository baseline `c36d438e`. The live `nemotron-bolt-probe` was healthy and its three accepted binary hashes matched on 2026-10-17. Target remains the original 22,018,326,528-byte Bolt GGUF, revision `e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246`, previously verified SHA256 `91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75` (not rehashed in initial discovery).

CPU-only header inspection on inference-host found GGUF v3, 843 tensors, 51 metadata entries, architecture `nemotron_h_moe`, 88 blocks, highest block 87, and zero `nextn`/`mtp` tensor names. Embedded MTP is unavailable. Pinned `src/llama-context.cpp:4103-4107` refuses an MTP context without NextN layers. Pinned `common/speculative.cpp` implements LiLiCorr through `draft-dflash` and `src/models/dflash.cpp` loads its metadata/tensors.

## Fresh external research

- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/raw/main/config.json — no NextN prediction layers, consistent with the installed GGUF header.
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF — official quick-start uses a separate LiLiCorr GGUF with `draft-dflash`, maximum draft depth 6. It does not use embedded MTP.
- https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF?blobs=true — resolved draft revision `182d2987939e12f458298b5d5f6ef98912f4ad6c`, file `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf`, 998,242,560 bytes, SHA256 `080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194`.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/docs/speculative.md — use real target verification, never synthetic acceptance; record drafting and acceptance behavior.
- https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/ — NVIDIA speed/acceptance figures use other hardware; they do not predict B70 performance.

## Declared bounded process

All downloads, CPU analysis, native tests, and GPU/API workloads run on `mike@inference-host` using explicit Bash and SSH options `-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no`. Remote evidence root `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr`. No target weights, accepted source, persistent launchers, drivers, network or hardware settings are changed. Download the pinned official drafter only into this isolated experiment root and verify size/hash. Do not copy the large draft into Git.

1. Inspect draft metadata, tensor shapes/types, target-layer indices and tokenizer compatibility before launching. No build or runtime upgrade.
2. First capability run: unchanged accepted native385 numerical gates, then isolated debug actual-model activation with LiLiCorr. Use `--spec-type draft-dflash --spec-draft-n-max 4 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy --no-spec-draft-backend-sampling --spec-draft-ngl 999`, plus the explicit mounted draft path. Keep original target q8 KV, F32 recurrent state, context65536, batch/ubatch1024, single sequence, ESIMD1/FUSION1/graphs0. Draft-context/default cache settings and recurrent snapshot allocation must be captured; speculation necessarily adds those resources. No synthetic/relaxed verification.
3. Require actual draft proposals, accepted and rejected tokens, finite successful output, and valid recurrent rollback behavior. Retain debug logs and compare deterministic responses against the non-speculative baseline. If unsupported, crash, OOM, numerical failure, or unexplained divergence occurs, stop; no timing and no retry-until-lucky.
4. Only after activation passes: fresh non-debug baseline and candidate must each pass native385 and gate-v2 API20/20 before timing. Compare fixed deterministic greedy outputs (including longer generation and exact long retrieval), recording full-output identity and first divergence. Treat this as a small development fidelity screen, not a 500-prompt publishable quality study.
5. If all gates pass, use inherited API probes with matched 54/8269-token warmups and three 55/8270-token measurements, 128 forced outputs, cache_n=0, and exact8238-token `BOLT-7319-CEDAR` retrieval. Report draft depth, proposals/accepts, acceptance lengths where available, target/draft time split where exposed, decode and end-to-end throughput, and all failures. Do not claim unexposed metrics or transferable performance.
6. Stop each experiment container and retain it. In nested finally, restart the same accepted container and verify ID/config/all mount fields, three binary hashes, unchanged hardware, health and READY. Preserve raw logs remotely, use lossless `gzip -n` copies locally. Scoped commit/push only.

The Lab preview socket is unavailable (`ENOENT`); the explicit textual declaration above is the reviewable plan. No Lab repair will be attempted.


## Preparation and pre-launch review

Pinned draft download passed exact byte-size and SHA256 verification. All 11 target `tokenizer.ggml.*` fields (including complete token and merge arrays) match the drafter. Draft: 5 DFlash layers, block size8, embedding4096, target feature layers `[2,19,36,52,69,86]`, LiLiCorr top-k8, 153 tensors (25 NVFP4/type40, 98 F32/type0, 30 BF16/type30). Feature projection `[24576,4096]` matches six target4096-wide features. `test-static.py` checks syntax, a synthetic GGUF scalar/array/tensor fixture, exact draft flags, padded acceptance logs, and the real saved draft/target shapes.

Read-only pre-launch review caught that INFO acceptance logs pad both counts (`accepted  2/ 4 draft tokens`); the parser now permits whitespace around `/`, and its exact production regex is covered by a fixture. Review's recommendation to require checkpoint replay was not adopted: pinned `common/common.h:401-406` requests `n_rs_seq=draft.n_max` for DFlash, `common/common.cpp:1715` propagates it, and `src/llama-arch.cpp:1252-1265` supports partial recurrent rollback for Nemotron-H-MoE. The activation gate requires logged `n_rs_seq=4`, accepts and rejects, and exact output identity after those operations. This is bounded behavioral coverage, not exhaustive rollback numerical validation. The controller never authorizes timing: `timing_permitted` remains false even after activation passes; fresh non-debug gates remain mandatory.

## Commands

Run from the remote evidence root with explicit Bash and `set -euo pipefail`:

```bash
python3 -B prepare-draft.py 2>&1 | tee prepare.log
python3 -B test-static.py 2>&1 | tee static-tests-r3.log
python3 -B run-activation.py 2>&1 | tee activation.log
```

Preparation refuses an existing draft directory; activation refuses existing run directories/container names. Do not rerun these commands against the same artifacts. Earlier static-check logs are retained. Activation failed on its first short request; the long candidate request, fresh non-debug candidate API20/20, fidelity completion, and all performance measurements were not run.


## Observed result and blocker

- Both fresh launches passed the inherited native385 gates (56+56+20+20+82+82+34+34+1). Each success line appears twice because the launcher validates `tee` output with `grep`; this is not770 tests. These existing gates cover the frozen target paths, **not the newly encountered strided LiLiCorr OUT_PROD case**.
- Non-speculative reference passed fresh gate-v2 API20/20 and completed the frozen short55/128 and long8270/128 requests with cache_n=0. These responses are fidelity references, not three-repetition timing samples; their incidental timings are not benchmark results.
- Candidate loaded the official draft and registered `draft-dflash`, n_max4/n_min0/p_min0, trained block8, six extracted target layers. Target `n_rs_seq=4`; draft `n_rs_seq=0`. Startup logged target recurrent state823.44MiB (F32 R/S) and draft model944.48MiB/KV10MiB. These are allocation logs, **not peak-memory or context-capacity measurements**.
- The first candidate short request emitted only `Design`, then the process failed before final usage, finish reason, or target acceptance. Captured container exit139, OOMKilled=false. No acceptance event occurred; acceptance rate, accepted length, rollback fidelity and throughput are **unmeasured**.
- At `2026-10-17T09:25:40.246959622Z`, `ggml/src/ggml-sycl/outprod.cpp:12` failed `GGML_ASSERT(ggml_is_contiguous(src0))`.

Actual failing F32 operation:

```text
OUT_PROD dst: ne=[128,56,8,1], nb=[4,512,28672,229376]
src0: ne=[128,56,8,1], nb=[4,12288,512,688128] (strided/permuted)
src1: ne=[56,56,8,1],  nb=[224,4,12544,100352] (transposed)
```

Pinned `src/models/dflash.cpp:743-747` builds `v` as a strided QKV view and passes it to `ggml_out_prod(v, transpose(attn))`. SYCL's support predicate at `ggml/src/ggml-sycl/ggml-sycl.cpp:7122-7127` accepts the F32 types without checking the first input's layout, while its implementation requires contiguous src0. Thus the runtime advertises a layout it cannot execute. This is a concrete capability/implementation mismatch, not an acceptance-rate or GPU-performance finding. `failure-source-diagnostic.log.gz` retains the exact implementation, predicate, graph construction and startup excerpts.

The inherited stream client returns an incomplete result after the server connection ends; the controller consequently reports `KeyError('prompt_tokens')`. The preserved `activation-r1/summary.json` retains that controller failure unchanged. The underlying backend assertion and truncated SSE are retained in `activation-r1/activation/{server.log.gz,short.events.json,short.result.json}`; no retry or benchmark was attempted.

**Next proposed work (not implemented):** an isolated SYCL OUT_PROD layout/capability correctness fix, with native tests reproducing the exact strided geometry before another LiLiCorr activation. Do not remove the assertion without implementing its layout semantics, silently switch the target/quantization, or promote this failed configuration. Further unknown compatibility/fidelity issues may follow this first blocker.

## Restoration and evidence

`verify-results.py` independently passed at `2026-10-17T09:28:46Z`: original container ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, identical complete config and mount fields, all three accepted binary hashes, unchanged boot/boost0/16 CPU caps3801000/275W, live health OK, saved restoration READY response, and both experimental containers stopped. Only the original accepted Bolt is running. `verification.json` records this successful failure-classification/restoration check; it does not turn the experiment into a pass.

Final verification commands, on inference-host in the evidence root:

```bash
set -euo pipefail
python3 -B test-static.py 2>&1 | tee static-tests-final.log
python3 -B verify-results.py 2>&1 | tee verification.log
bash compress-evidence.sh
```

`compress-evidence.sh` preserves remote originals and validates every `gzip -n` copy by decompression and byte comparison. `compressed.sha256` was checked after SSH/tar retrieval. Repository evidence contains compressed logs, complete requests/responses, original/restored/experimental container inspections, header inventories, model provenance, scripts and failure summary; it excludes the998MB draft weights. The verified draft remains isolated under the remote evidence root for a later explicitly selected source candidate. No accepted source or persistent launcher was edited, and no autonomous subagent remains.
