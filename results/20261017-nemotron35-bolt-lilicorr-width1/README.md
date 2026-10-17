# Bolt LiLiCorr one-token draft control

Development-only continuation of the Bolt/B70 optimization. The preceding proposal-suppression matrix showed stable exact A=B text (draft-off versus n_max4/n_min5) while active depth4 C retained its token116 mismatch. All120 API checks passed and the original runtime was restored. No timing, promotion, persistent launcher or inference-source changes in this experiment.

## Research and interpretation

Fresh pinned primary source was fetched on inference-host for the immediately preceding control and verified byte-identical to the local runtime: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/common/speculative.cpp . See `../20261017-nemotron35-bolt-lilicorr-suppressed/research-urls.txt` and its declaration for conclusions. LiLiCorr's lattice loop honors n_max (line1341), and min5 clears at most4 proposals (1361-1363), while its drafter still evaluates the eight-token block.

Here change only active C's n_max from4 to1. Context n_rs_seq derives from n_max (`common/common.h:401-407`, `common/common.cpp:1715`). Therefore C also changes target output capacity to2, rollback depth to1, trailing prefill reservation to2, and SSM K to2. Its multi-token scan may use the custom prefill kernel (K<=4), unlike the depth4 K5 case. This is a **coupled width/depth control**, not independent proof about rollback versus batch numerics or prefill kernel choice. Do not infer that feature capture or allocations are universally harmless from the preceding single prompt.

## Fixed environment and real E2E

Reuse the unchanged deterministic baseline and combined OUT_PROD+deterministic candidate; four-artifact hashes, source-scope receipts and prior build/precheck evidence must be rechecked. Original source, weights/tokenizer, verified drafter, accepted container and hardware remain unchanged. No new build.

- A: deterministic baseline, drafter off, target n_rs_seq0.
- B: combined candidate, DFlash n_max4/n_min5, target n_rs_seq4; zero proposals.
- C: combined candidate, DFlash n_max1/n_min0, target n_rs_seq1; active proposals.

Keep p_min0, greedy draft sampling, no draft backend sampling, drafter GPU layers999, block_size8; tensor callback0, oneDNN verbosity0, DNN1; FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8 KV,F32 recurrent,ctx65536,batch/ubatch1024,parallel1, continuous batching. Hardware boost0, sixteen3801000kHz caps,275W. Fixed **A0,B0,C0,C1,B1,A1**, no retry-until-match.

All executable analysis, syntax/CPU tests, GPU/API work runs on mike@inference-host using explicit local/remote Bash and SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. No local AST checks. Remote root `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1`; commands `python3 -B test-controls.py`, `python3 -B run-controls.py`, `python3 -B analyze-controls.py`.

Each launch: native385, plus190 numerical/9 support OUT_PROD checks in B/C; actual-model startup/parameter checks and fresh API20/20 on http://127.0.0.1:18098; unchanged frozen55/128 warmup; three ordinary streamed55/128 responses with temp0/seed42/cache_prompt=false/ignore_eos=true; one separate nonstream token-ID diagnostic. Preserve exact previous payload/count/cache/finish contracts. Verify A has no drafter, B/C DFlash block8 with declared n_max/n_min/n_rs_seq and no clamp. B must have zero API proposals/acceptance and zero verification events; C must have nonzero proposals with API/log counters agreeing. Never interpret placeholder speculative logprobs as confidence or summarize speed.

Record each arm's within/across-launch stability, all ordinary A/B/C pairwise comparisons, exact first differences, and separate diagnostic IDs. Preserve every failure. Stop on native/API/activation/count/mechanism failures. Nested cleanup restores original nemotron-bolt-probe and stops only owned experiments; independently verify original ID/config/all mount fields/four hashes/hardware/health/READY and all experiments stopped. Retain declarations, exact launch arguments, requests/responses and lossless logs; raw originals stay remote. Commit/push only this result scope. A short fidelity pass permits the next declared correctness stage, not automatic timing or promotion.

## Observed result: depth1 retains the mismatch

The fixed sequence completed without retries; `verification.json` reports **fidelity_passed=false**, not a timing authorization. Lead remote CPU/syntax tests passed14/14; all six native385 gates, four additional190/9 OUT_PROD gates, and **120/120** fresh API checks passed. Startup verified C n_max1/min0, target n_rs_seq1, draft0 and block8.

All three arms are internally stable across six ordinary responses/two launches. A=B exactly; active C still differs at character519/token ordinal116 (6833 ` primary` versus4618 ` original`). The other127 diagnostic token IDs match. C depth1's entire ordinary text is exactly equal to the preceding depth4 C text. Each launch's diagnostic text matches its own ordinary stream.

C's five requests per launch each accepted58 of69 proposals; each launch recorded345 verification events and55 rejection events, with API/log counters agreeing. B retained zero proposals/acceptance. Reduced width/depth did not restore fidelity; this does not distinguish batch numerical differences from rollback errors.

Original container ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, config/all mounts/four binaries/hardware were independently verified unchanged; health OK, exact READY response, all six experiments stopped. Exact artifacts reside in `run-r1/`, `verification.json`, `controller-r1.log.gz` and `analysis-r1.log.gz`; raw logs remain remote.

Next: bounded paired target-score/verification capture from the first sampled position, with matched state allocation and explicit observer/alignment checks. No further depth sweep or timing until the scoring-versus-selection/state split is established.
