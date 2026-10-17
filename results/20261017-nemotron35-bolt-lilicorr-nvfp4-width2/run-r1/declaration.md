# Bolt NVFP4 two-column reorder-initialization control

Development-only single-factor follow-up to6655db8e. Aligning the output head to MMVQ did not remove token2 score divergence or token116 output mismatch. Actual S0 first-block NVFP4 SSM projections are reordered; C0 two-column projections remain plain. Hold the aligned head path constant and test only this remaining observed dispatch difference. No timing/promotion.

## Research and exact intervention

Fresh pinned official source fetched before implementation: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/mmvq.cpp . Local experimental source is authoritative for additions. The parent ggml-sycl.cpp5405-5424 permits NVFP4 reorder initialization only at ne[1]==1, retaining immutable .ssm_in/.ssm_out whole-matrix/layout/alignment/SYCL-buffer guards. Its mmvq.cpp2970-2986 already reuses the reordered NVFP4 GEMV for every column, including i>0, once reorder is installed. This control changes initialization eligibility, not that kernel implementation.

Clone `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-head-mmvq-r1` to `...-lilicorr-nvfp4-width2-r1`. In ggml/src/ggml-sycl/ggml-sycl.cpp, replace ONLY the NVFP4 eligibility clause `dst->src[1]->ne[1] != 1` with `(dst->src[1]->ne[1] != 1 && dst->src[1]->ne[1] != 2)`. Permit exactly1/2 columns, not all widths. Keep all other name/type/view/contiguity/shape/alignment/op/device restrictions, reorder implementation, kernels, head-MMVQ control, legacy-only SSM selector, route/capture logging and global flags unchanged. No MoE reorder or GET_ROWS/view exposure.

This is a width1-speculation diagnostic, not a general fix for depth4 or a production promotion. Inherited head alignment remains an experimental control. S0 is expected computationally unchanged for the captured diagnostic and must be checked; C0 may change. Do not rebase accepted fidelity or assume convergence.

## Predeclared E2E

All execution/analysis/tests/build/inference on mike@inference-host via explicit local AND remote Bash; SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. Remote root `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-nvfp4-width2`. Local editing/git only. Preview/show_patch socket unavailable; do not retry or repair.

CPU-only build: Intel2026/oneDNN3.14, six jobs, fuzz-free patch, source scope exactly the one eligibility clause/file. Protect accepted/parent FIVE artifacts. Only candidate SYCL library may change; wrapper/common/server implementation/backend tests must match parent. Four GiB capture reserve; preserve all build/dependency/failure logs.

Fresh S0(max1/min2 suppressed), then C0(max1/min0 active), same candidate,n_rs_seq1/K2,block8,p_min0,greedy draft,no draft backend sampling,ngl999. Same q8KV/F32state,ctx65536,batch/ubatch1024,parallel1,graphs0,FUSION/ESIMD/OPT/VMM/DNN1,callbacks0,oneDNN verbosity0, hardware boost0/sixteen3801000kHz caps/275W. Same frozen55/128 warmup/ordinary stream/nonstream diagnostic, API20 requests and http://127.0.0.1:18098 boundary. No new request class.

Each arm retains native385 plus190 numerical/9 support gates, custom-prefill expectation0, startup, API20/20, count/cache/finish checks, ordinary/diagnostic text consistency, strict128-row capture/history/rollback alignment and API/log proposal-count agreement. Existing NVFP4 native dispatch-presence checks remain required; the existing multi-column reordered implementation must pass actual native numerics after this eligibility change. Do not weaken numerical/activation checks if it fails.

Actual route gate: first selected S0 singleton batches and C0 two-row batch must BOTH show first-block NVFP4 .ssm_in/.ssm_out as MMVQ_REORDERED/reorder1; Q8_0 output head remains MMVQ_REORDERED/reorder1. Canonical runtime dtype names are lowercase. Require the expected1/2 activation-column shapes. Retain target backend partition via SSM anchors and preserve nested other-backend records separately.

Against the head-MMVQ parent: S0 MUST preserve all128 raw score vectors byte-for-byte, emitted IDs/text, prompt IDs, actual input batches and selected row/position metadata. Any S0 difference blocks this control's intended attribution; preserve the failure. C0 historical equality is NOT required: report raw/text/ID/batch differences only through identical prior emitted history, including first differing output. Current S0/C0 comparison likewise stops when prior histories diverge and handles full128 equality. Do not require old C0 acceptance totals or unchanged C0 bootstrap scores.

Commands: `bash build-candidate.sh`; `python3 -B test-trace.py`; `python3 -B run-trace.py`; `python3 -B analyze-trace.py`. No inference retry-until-match. Stop experiments before final log capture. Restore original container and independently verify ID/config/HostConfig/all mounts/FIVE artifacts/hardware/health/READY; experiments remain stopped. Raw captures stay remote, logs compressed losslessly in scoped Git evidence. Accepted source/weights/container definition/persistent launchers/hardware/unrelated dirty files stay untouched. No timing or promotion even if this diagnostic converges.
