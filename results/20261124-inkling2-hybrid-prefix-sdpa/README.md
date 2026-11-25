# Inkling 2 Mini: hybrid XPU flash prefill on the B70

**Development tier, 2026-11-25 UTC (Nov 24 local).** Same-build A/B on one Arc Pro B70 at 275 W. Both arms use FP8 E4M3 KV, CUTLASS Xe2 MoE, and DSpark k=7 with probabilistic draft sampling. Only `INK2_HYBRID_PREFIX_SDPA` changes. The eligible bias-free prefix runs on SYCL-TLA XPU flash SDPA; Triton still handles the relative-bias tail.

## Cold A/B

Three matched requests per depth and flag, after one 16K warmup in each arm. These are batch-one, text-only requests to `/v1/completions` with seeded synthetic token IDs, temperature 0, 128 output tokens, EOS ignored, and prefix caching disabled. vLLM counters confirmed the full prompt was computed in each request; request SHA-256s match between arms. Prefill rates are computed from engine timing, not from client TTFT. Values are medians; parentheses in the prefill columns show IQR (n=3).

| Prompt tokens | FP8 stock prefill tok/s | FP8 hybrid prefill tok/s | Change | TTFT stock → hybrid |
|---:|---:|---:|---:|---:|
| 65,536 | 3,108.0 (13.7) | 3,436.9 (13.6) | **+10.6%** | 21.16 → 19.15 s |
| 131,072 | 2,461.8 (1.0) | 2,851.7 (1.3) | **+15.8%** | 53.34 → 46.06 s |
| 245,760 | 1,812.3 (0.7) | 2,203.5 (0.9) | **+21.6%** | 135.73 → 111.66 s |

| Prompt tokens | Decode tok/s stock → hybrid | Median acceptance length stock → hybrid |
|---:|---:|---:|
| 65,536 | 111.7 → 115.3 | 3.88 → 4.09 |
| 131,072 | 93.1 → 93.1 | 3.76 → 3.85 |
| 245,760 | 66.5 → 68.1 | 3.45 → 3.54 |

Decode and acceptance vary with the generated trajectory. The experiment isolates the prefill flag, not FP8 versus BF16, and does not establish a decode gain or broader FP8 quality equivalence. This fixed-order run did not include BetterBench, a full context/concurrency sweep, or the repo's publication-tier sample count.

## Reproduction and evidence

- Model: `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`, checkpoint NVFP4/MXFP8 experts and BF16 activations. Plugin: [`27474f2`](https://github.com/mgaruccio/vllm-inkling2-xpu/commit/27474f21f0fba583d314deb400e5d052c0194567). Runtime: vLLM `3146a9df41` + #69186, PyTorch 2.15.0+xpu, Triton 3.9.0, compute runtime 26.40.40196.8.
- On the B70 guest, with the live service stopped and the versioned plugin site staged: `TRIALS=3 bash bench.sh "$HOME/inkling-vllm-xpu/plugin-27474f21f0fba583d314deb400e5d052c0194567" /path/to/new/run-dir`. The runner uses loopback `:18080`, 262,144 max context, 2,048-token prefill budget, 64-token pages, batch one, and GPU utilization 0.98. Both benchmark servers exited before production was restarted.
- [Environment](raw/environment.txt), [stock launch and results](raw/flag-0/), and [hybrid launch and results](raw/flag-1/). The guest's `~/inkling-vllm-xpu/pr3-maintenance/run-20261125-013028/` also retains full Prometheus snapshots, SSE and server logs. vLLM reported 338,263 stock / **337,659 hybrid** KV tokens and 8.76 / 8.74 GiB usable KV memory. Peak run-time VRAM was not separately sampled.
- On B70: `python -m pytest -q -rs tests/test_fp8_kv.py tests/test_hybrid_prefix_sdpa.py` — **18 passed**, including both real-kernel page-boundary parity cases.
- Method reference: [vLLM serving benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/). The sweep uses a purpose-built client to capture the per-request prefill counter and TTFT together.

## Live cutover

At this cutover, the authenticated `inkling-inference` service on `:8080` started serving the tested plugin SHA with FP8 KV, `INK2_HYBRID_PREFIX_SDPA=1`, and CUTLASS still enabled. Prefix caching stayed enabled and `/v1/models` still reported a 262,144-token cap. The previous launcher was retained on the guest as `serve-inkling.sh.pre-hybrid-20261125-013028`. The [subsequent concurrency rollout](../20261125-inkling2-concurrency/) raised the live cap to 327,680 tokens and 32 active sequences.

`bash ~/cmp/live/live_check.sh before-hybrid-pr3` and `bash ~/cmp/live/live_check.sh after-hybrid-pr3` both passed the model listing, authenticated chat (`95`), and tool call (`echo_value`) checks. After cutover, an unauthenticated `/v1/models` returned 401, and `/health` returned 200. The service's running process confirmed the versioned plugin path, hybrid flag, and FP8 dtype.

| Live DSpark decode (median tok/s) | BF16 previous launcher | FP8 hybrid launcher |
|---|---:|---:|
| Short, six prompts | 99.0 | 106.9 |
| +30K, three prompts | 91.2 | 90.8 |

The live before/after changes the plugin path, KV dtype, hybrid flag and prefill budget (256 → 2,048); it is a rollout smoke comparison, not attribution. See [before](raw/live-before/) and [after](raw/live-after/) for raw summaries and acceptance lengths.
