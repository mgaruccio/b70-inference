# Bolt LiLiCorr proposal-suppression control

Development-only continuation after `../20261017-nemotron35-bolt-lilicorr-deterministic` (91cd9b58): all arms repeatable and120 API checks passed, but active LiLiCorr differs from draft-off at generated token116. No timing, promotion, persistent launcher changes, or new runtime build is authorized by this experiment.

## Fresh research and source constraints

Fetched https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/common/speculative.cpp on inference-host before implementation, and verified byte-for-byte equality with the candidate's local source. The LiLiCorr lattice walk emits at most `n_max` tokens (lines1341-1358), then clears the result when its size is below `n_min` (1361-1363). `n_max=4,n_min=5` therefore suppresses all proposals. Neither exceeds the trained block's seven-token cap (1092-1097). Drafter block evaluation and feature capture remain enabled. This is not an acceleration configuration.

Read-only audit: greedy verification compares draft IDs to sampled target IDs exactly (`common/sampling.cpp:690-722`); no synthetic verification flags are enabled. `n_rs_seq` derives from `n_max`, not `n_min` (`common/common.h:401-407`, `common/common.cpp:1715`). Reducing n_max to0 or1 would also change target state allocation, output capacity and prefill splitting, so it is not the first control here.

Additional confound confirmed by lead: `src/models/mamba-base.cpp:172` sets SSM K to `n_rs_seq+1`; `ggml/src/ggml-sycl/ssm_scan.cpp:219-224` excludes K>4 from the custom prefill kernel. Both drafter-on arms here have n_rs_seq4/K5 and use the original scan path; draft-off has n_rs_seq0/K1 and may use custom prefill. Proposal suppression does NOT isolate feature capture from allocation, prefill splitting or kernel selection. A B-versus-C difference would isolate effects requiring actual proposal verification/rollback; it would not distinguish batch numerics from rollback correctness.

## Fixed configuration and E2E declaration

Reuse the exact deterministic baseline and combined OUT_PROD+deterministic binaries from the preceding matrix. Recheck all four artifact hashes before and after; no rebuild. Original archive, weights, drafter SHA256, tokenizer, hardware and accepted container remain unchanged.

- A: deterministic baseline, draft off.
- B: combined candidate, LiLiCorr `--spec-draft-n-max 4 --spec-draft-n-min 5`.
- C: same combined candidate, LiLiCorr `--spec-draft-n-max 4 --spec-draft-n-min 0`.

Keep all other prior flags, including p_min0, greedy draft sampling, no draft backend sampling, drafter GPU layers999; tensor callback0, oneDNN verbosity0, DNN1; FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8 KV,F32 recurrent,ctx65536,batch/ubatch1024,parallel1. Hardware boost0, sixteen3801000kHz CPU caps,275W. Fixed order **A0,B0,C0,C1,B1,A1**; no retries to obtain matching output.

All executable analysis, syntax/CPU tests, GPU and API work runs on mike@inference-host through explicit local/remote Bash and SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. Do not run even supplementary AST tests locally. Commands in remote `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-suppressed`: `python3 -B test-controls.py`, `python3 -B run-controls.py`, `python3 -B analyze-controls.py`.

Each launch: native385; additionally native190/9 OUT_PROD checks for B/C; actual-model activation and fresh API20/20 on http://127.0.0.1:18098; the unchanged frozen55/128 warmup, three ordinary streamed55/128 responses (temp0,seed42,cache_prompt=false,ignore_eos=true), and one separate nonstream token-ID diagnostic. Reuse exactly the preceding matrix's request bytes/count/cache/finish contracts. No latency or throughput interpretation. Speculative logprobs remain placeholders, not confidence.

Require DFlash activation in B/C only. Verify target n_rs_seq0 for A and4 for B/C; startup n_max4/n_min5 in B and4/0 in C with block_size8. B must have zero proposed/accepted API tokens and zero verification acceptance events in the five-request observation window; C must have nonzero proposals with API/log counters agreeing. These are mechanism checks, not a demand that outputs match. Preserve all ordinary output equality/inequality results and diagnostic token differences separately; report within/across-launch stability and all A/B/C pairs without redefining fidelity.

Stop on native/API/activation/count/mechanism failures. Nested finally cleanup stops only owned experimental containers and restores original nemotron-bolt-probe. Independently verify original ID/config/all mount fields/four hashes/hardware/health/READY and all experiments stopped. Retain exact declarations, commands, environments, requests/responses and failures under this result root; preserve raw logs remotely and losslessly gzip repository copies. Commit/push only this result scope. Continue bounded causal follow-ups after interpreting the result; no automatic promotion or timing.
