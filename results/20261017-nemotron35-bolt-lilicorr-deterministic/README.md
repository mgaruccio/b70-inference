# Bolt LiLiCorr fidelity after deterministic oneDNN

Development-only continuation of the user-requested Bolt/B70 optimization and LiLiCorr work. The user explicitly said “why stop? don’t do that”; continue bounded follow-ups, but do not run timing or promote before correctness/fidelity. The original accepted container/archive/weights/launchers/hardware remain untouched except transient, restored experimental service switches.

## Research and established facts

Fresh primary model card fetched before integration: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF/raw/182d2987939e12f458298b5d5f6ef98912f4ad6c/README.md . It confirms the separate drafter targets this exact24GB model and that user-visible tokens are verified by the target; it is not standalone or embedded MTP.

The earlier OUT_PROD fix passed native190 numerical and9 layout checks, but initial fidelity attribution was blocked by baseline variability. `../20261017-nemotron35-bolt-dnn-deterministic` (ff98c9b7) now passes native385/API20 per launch and repeated full55/128/prefix94/1 stability with oneDNN's deterministic attribute. Real model matmul descriptors are F32/F32/F32 with fpmath=f16, not the native-test F16 route. Traced and untraced full outputs differ despite within-mode stability, so **all arms here disable tensor callbacks and oneDNN verbosity**.

## Build and configuration declaration

A baseline: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-dnn-deterministic-r1`, drafter off. B/C candidate: new isolated clone `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-deterministic-r1`, applying the exact existing `outprod-layout-r1.patch` and `outprod-tests-r1.patch` from `../20261017-nemotron35-bolt-lilicorr-outprod`. Relative to A, only `ggml/src/ggml-sycl/outprod.cpp`, `outprod.hpp`, `ggml-sycl.cpp`, and `tests/test-backend-ops.cpp` may change. Keep the one-line deterministic GEMM attribute and the same dormant callback code byte-for-byte. Build with the existing Intel2026 image/oneDNN/six jobs. Original and baseline hashes stay unchanged; candidate wrapper/server implementation match baseline, SYCL/test binaries change. Save all four hashes for each tree and source-scope checks. No downloaded weights or new dependency/runtime in Pi.

C additionally mounts the already verified LiLiCorr file (998242560bytes, SHA256080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194) read-only and uses exactly the previous flags: `--spec-type draft-dflash --spec-draft-n-max 4 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy --no-spec-draft-backend-sampling --spec-draft-ngl 999 --spec-draft-model /draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf`.

All arms explicitly set `BOLT_TRACE_TENSORS=0`, `ONEDNN_VERBOSE=0`, `GGML_SYCL_ENABLE_DNN=1`. Keep FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8 KV,F32 recurrent,ctx65536,batch/ubatch1024,parallel1,continuous batching; hardware boost0,16 CPU caps3801000kHz,275W. Preserve the original model bytes and tokenizer. No persistent launcher edits, promotion or timing.

## Predeclared E2E

All builds/tests/analysis/GPU/API calls run on mike@inference-host via explicit local/remote Bash and SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. Commands: `bash build-candidate.sh`; `python3 -B test-controls.py`; `python3 -B run-controls.py`; `python3 -B analyze-controls.py`.

Fixed order **A0,B0,C0,C1,B1,A1**, two fresh launches per arm. Each: native385; additionally native190/9 OUT_PROD checks for B/C; real API20/20; one original frozen55/128 streamed full-decode warmup; three identical measured-content55/128 streamed requests (temp0,seed42,cache_prompt=false,ignore_eos=true); one separate nonstream token-ID/logprob diagnostic copied from the prior fidelity-control process. All use http://127.0.0.1:18098. Count/cache/finish contracts must pass. These requests assess content only: do not summarize latency/throughput. Diagnostic logprobs for speculative output are placeholders at this revision and must not be treated as confidence; diagnostic requests are not replays of the streams.

Require A internally stable across six ordinary responses and both launches; B likewise, and identical to A; C likewise, and identical to A/B. Record exact mismatch positions, within/cross-launch variability, and supplementary diagnostic token-ID comparisons separately. Preserve predeclared results even when fidelity fails; no retries to obtain matching output. Retain actual speculative acceptance/rejection counters only as diagnostics, not speed evidence. Confirm DFlash activation only in C and no tensor/operand/oneDNN verbose debug records in any arm. Stop on native/functional/quality/count failures.

Nested cleanup must stop owned experiments and restore the original `nemotron-bolt-probe` container even if artifact collection fails. Independently verify original identity/config/all mount fields/all four hashes/hardware/health/READY and all experiments stopped. Preserve exact source patches/build output/commands/env/requests/responses/failures and lossless logs in this root; raw originals remain remote. Commit/push scoped evidence. Passing this short fidelity gate permits the next declared correctness stage, not automatic performance timing or promotion.

## Observed result: repeatable, but fidelity failed

The fixed sequence completed without retries. `verification.json` reports **fidelity_passed=false**; the analyzer's exit0 means analysis/restoration succeeded, not that fidelity passed. No timing or promotion was performed.

- Build/source-scope/four-artifact checks passed. Candidate SYCL SHA256: `bdfa0df45ad4614610a678f58aa1682255af574f9cefe21176508dd5972fb77c`; backend test: `f203b60539b207ed236b60bba035beeb4ebecca1d0c3585e6a666784ade136b8`. Wrapper and server implementation remain identical to deterministic baseline. Exact receipts are in `*-binaries.json`.
- Lead reran `python3 -B test-controls.py` on inference-host: **12/12 passed**, including syntax checks (`controller-tests-r1.log.gz`). The worker also reported supplementary local AST/trailing-whitespace checks; those were outside the requested remote-only verification discipline and are not relied on here.
- All six native385 gates passed; all four B/C launches additionally passed190 numerical/9 support checks. All six fresh API20 gates passed (**120/120**). DFlash activated only in C; actual-model tensor callbacks and oneDNN verbosity remained disabled. Count/cache/finish contracts passed.
- A, B and C each produced one unique ordinary output across their six responses and two launches. **A=B in36/36 cross-pairs**,573 characters. C has574 characters and matches A/B in **0/36** pairs each. First ordinary difference is character index519: `primary` versus `original`.
- Separate diagnostics happen to match their own ordinary streamed text in all six launches. A/B and C differ only at generated token ordinal116 (index115): A/B6833 (` primary`), C4618 (` original`); the other127 IDs match. These diagnostics remain supplementary, not execution-equivalent replays.
- A0's genuine diagnostic logprobs at that token are -0.43095794320106506 (` primary`) and -1.0566980838775635 (` original`), a gap of approximately0.626. Do not describe this as an established near-tie. C's0/top[] entries are placeholders, not target confidence.
- Every C warmup/ordinary/diagnostic request recorded86 accepted of163 proposed tokens. Each C launch's five-request window has205 verification events and135 rejection events; API/log counters agree. These are diagnostic counts, not speed evidence.
- Original container `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e` was restored and independently verified: identical config/all mount fields/four binaries/hardware, health OK, exact READY response. All six experimental containers are retained stopped.

Artifacts: `run-r1/declaration.md` preserves the original declaration; each `run-r1/{A0,B0,C0,C1,B1,A1}/` retains exact launch arguments, requests/responses, native/API evidence and losslessly compressed logs. Raw logs remain under the same result basename in `/home/mike/b70-evals/` on inference-host. `analysis-r1.log.gz`, `controller-r1.log.gz` and `verification.json` retain the failed fidelity result.

The baseline nondeterminism confound is removed for this tested sequence. The remaining LiLiCorr mismatch is **not yet attributed** to feature capture, target batch numerics, verification or recurrent rollback. Continue bounded source/width controls; do not relax the fidelity gate or begin timing.
