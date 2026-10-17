# Bolt target-only reorder and allocator controls

Development diagnosis continuing the exact-prefix experiment (`../20261017-nemotron35-bolt-prefix-replay`, commit `d74c60f0`). That experiment passed all80 quality checks but every fusion/ESIMD cell produced varying score vectors for six identical cold94-token one-token requests. No timing, promotion or source patch is authorized by this diagnosis.

## Research and intended differences

Fresh primary-source verification: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/ggml-sycl.cpp confirms `GGML_SYCL_ENABLE_OPT` and `GGML_SYCL_ENABLE_VMM` default1, are logged at startup, and respectively gate weight reorder and select VMM versus the legacy device pool. Disabling VMM also disables this accepted archive's custom SSM-prefill fast path; it is therefore an allocator/dispatch bundle, not a pure allocator experiment. Neither control guarantees determinism. The previous experiment's fresh API research and exact token-array replay caveats continue to apply.

No new binary or runtime source. Same accepted archive `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`, target GGUF, container image, oneDNN, q8 KV,F32 recurrent state,context65536,batch/ubatch1024,parallel1,graphs0 and existing continuous batching. Keep FUSION=1 and ESIMD=1 throughout; no drafter. Preserve boost0,16 CPU caps3801000kHz,275W. OPT disables more than one potentially eligible reorder path, so a result does not identify a single tensor/kernel.

## Fixed real-API test process

Exactly two fresh launches, in order **O0V1** (OPT0,VMM1) then **O1V0** (OPT1,VMM0). Baseline comparison is the just-completed F1E1 cell with OPT/VMM at defaults1/1; no simultaneous claim of paired timing or process-order independence. Do not adaptively repeat a favorable outcome. An eventual confirmation must be separately declared.

For each launch, inherit the previous controller's complete public-boundary sequence unchanged: native385; fresh API20/20; one frozen streamed55/128 full-decode warmup; template/tokenization/detokenization55-ID round-trip; three saved nonstream chat55/128 diagnostics; three explicit integer-array native55/128 diagnostics; six native94/1 replays using55 original prompt IDs plus the frozen39 generated IDs. All requests cachefalse,temp0,seed42,ignore EOS; diagnostic families use n_probs5,return_tokens,backend_samplingfalse. A replay re-prefills the prefix rather than reproducing incremental execution. Inspect within-family score/token variation before cross-family/control comparisons.

Native385 retains its explicitly declared internal OPT/FUSION/VMM test overrides; model startup occurs in a separate process and must log the selected cell's OPT/VMM plus FUSION1/ESIMD1. The inherited numerical battery therefore remains the existing mixed-configuration gate, not385 new tests all at the cell's values. Every actual model request uses the verified cell settings.

Planned totals:40 quality checks,2 full warmups,6 full chat diagnostics,6 full native diagnostics,12 one-token replays. Stop the whole matrix on a native/API/quality/shape failure, preserving it; output/score variation is an observation, not a retry trigger. No long-context/timing probes.

All execution, tests and analysis on `mike@inference-host`, using SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no with explicit Bash. Remote artifacts `/home/mike/b70-evals/20261017-nemotron35-bolt-state-controls`. The original accepted container is temporarily stopped; owned experiments are retained stopped; nested cleanup restores and verifies the same original ID/config/mounts/binaries/hardware/health/READY. Accepted archive, weights, persistent launcher and hardware settings remain untouched. Retain exact argv,controller,declaration,all requests/results/logs,failed attempts and independent restoration evidence. Compress logs losslessly and commit/push only this evidence. Continue diagnosis from the outcome rather than declaring fidelity cleared.
