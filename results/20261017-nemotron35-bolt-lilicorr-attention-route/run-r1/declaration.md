# Bolt first-attention-block route observation

Observational follow-up to5afea801. Latent two-column DMMV passes numerical/route/API checks and preserves S0, but S0/C0 still differ in scores from2 and text at80. Full header inventory shows32 Q8_0 attention projection matrices not covered by the block0/1 logger. First attention block is7: q/output[4096,4096], k/v[4096,256]. Observe before assuming routes or changing computation.

## Research and exact scope

Fresh pinned official model source: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/models/nemotron-h.cpp . Its attention builder passes separate Q/K/V and output weights into attention; actual dispatch still requires runtime evidence. Existing stdlib target GGUF header reader saved attention-header.json. The target's attention matrices are Q8_0, not NVFP4.

Clone `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-latent-dmmv2-r1` to `...-lilicorr-attention-route-r1`. ONLY runtime source edit: add `blk.7.` to bolt_route_selected_weight in ggml/src/ggml-sycl/ggml-sycl.cpp, retaining block0,block1,output. No other source/test/kernel/selector/precision/reorder/callback/synchronization changes. Preserve latent two-column DMMV, two-token IQ2 per-expert fallback, head-MMVQ, NVFP4-width2 and legacy-only SSM controls. This is observational, not another computational control.

CPU build Intel2026/oneDNN3.14/six jobs; fuzz0 patch; source scope exactly one host filter/file. Protect five accepted/parent artifacts; ONLY candidate SYCL changes. Candidate test-backend-ops, wrapper, common and server implementation match parent. Keep all thirteen added native fixtures unchanged.

## Defined E2E

Execution/inspection/analysis/tests/build/GPU/API only on mike@inference-host through explicit local AND remote Bash; SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. Local edits/git only. Remote root `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-attention-route`. Lab preview/show_patch unavailable, do not retry/repair. No accepted-source/weights/launcher/host-package/hardware changes.4GiB capture reserve.

Same fresh S0(max1/min2) then C0(max1/min0), n_rs_seq1/K2,block8,p_min0,greedy draft,no draft backend sampling,ngl999. Same55-token prompt/128 outputs, warmup/ordinary stream/separate nonstream diagnostic at http://127.0.0.1:18098. Context65536,batch/ubatch1024,parallel1,q8KV,F32state,graphs0,FUSION/ESIMD/OPT/VMM/DNN1,callbacks0,oneDNN verbosity0. Hardware boost0/sixteen3801000kHz caps/275W.

Retain per-arm native385 +190 numerical/9 support +six IQ2 +seven Q8 numerical/actual-route gates, API20/20, all startup/cache/count/finish/text/framing/path/size/128-row/history/acceptance/removal checks, exact benign CPU-backend skip handling and timestamp normalization. Keep Q8 gate read-only script mounts and all original restoration guards. Native numerical cases cannot skip or change tolerances.

Keep known first-block SSM/head aligned reordered-MMVQ gates, both-arm latent DMMV_ESIMD (one/two columns), S0 single-expert MMVQ and C0 per-expert fallback/MMVQ with no grouped call. Add required target routes for blk.7.attn_q/k/v/output.weight in selected S0 singleton55/56 and C0 two-row55/56 batches, canonical q8_0 weights/F32 input-output. Do not require a guessed attention route or fabricate missing fusion records. Preserve all dimensions/strides/reorder/precision/result fields. Keep target-backend anchoring at block0 SSM and nested other-backend separation.

Restore strict BOTH-arm observer equality to each latent-dmmv2 parent: all128 raw vectors byte-for-byte, prompt/emitted IDs, ordinary/diagnostic text, input batches and selected positions/rows. Any observer difference blocks attribution; retain failure without inference retry-until-match. No C0 report-only computational exception in this experiment. S0/C0 comparison still stops at divergent prior emitted history, including the first differing output.

Run `bash build-candidate.sh`, `python3 -B test-trace.py`, `python3 -B run-trace.py`, `python3 -B analyze-trace.py`. Stop arms before final logs; nested-finally restore same original service. Independently verify ID/config/HostConfig/all mounts/five hashes/hardware/health/READY; experimental containers stopped. Raw arrays remote only; scoped metadata and deterministic compressed logs in Git. No timing or promotion. Choose a subsequent computational control only from observed evidence.

## Parallel IQ2 source audit and next-control priority

Read-only audit of the latent-dmmv2 parent found no proven arithmetic difference between IQ2_XXS single-token q-id MMVQ and per-expert ordinary MMVQ: both call vec_dot_iq2_xxs_q8_1 through the same codebook/sign tables, quantize_q8_1, and XOR subgroup reduction. The q-id adapter is a wrapper; its blocks-per-warp ceiling equals the ordinary floor for the supported subgroup sizes. Their launch topology and pointer strides differ, so this is not a bitwise-equivalence proof.

A later bounded test could reuse the single-token helper per token with src1.nb[2], ids.nb[1], dst.nb[2] offsets and prevalidate both descriptors before enqueueing either. It would need actual native route coverage because the current six IQ2 fixtures name their weights `as`. Do not implement that speculative alternative now: observe the untraced Q8_0 attention projections first and select only one evidence-backed computational control afterward. No change to this observational experiment's acceptance criteria.
