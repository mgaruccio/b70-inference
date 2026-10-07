# Qwen3.8 layer localization

**Development only. Zero optimizer updates. Not training-ready.** This localizes the already measured teacher-fidelity gaps. It is not a throughput result, and it does not close the numerical or exact-argmax gates.

Lease `789d1b2e-0b02-48c5-9fac-7b6ad3ba8c6f` was deleted only after the native artifacts were copied. Hugging Face used the same full-prefill versus uncropped single-token cache path as the 2026-10-06 diagnostic. Native compared eager fresh prefill with the depth-8 target forward after correcting vLLM's three identical mRoPE position rows.

## Findings

Layer 0 is not the break. Both stacks start near zero and the error grows with depth. The first gate trips are small max-abs/RMS spikes. Relative L2 crosses 0.03 only late, and only on some positions. Argmax still agrees on the Hugging Face final rows.

| Case | What was compared | Result |
|---|---|---|
| verifier-000 | HF full vs cached, 64 layers | Final relative L2 0.04825, 0.05850, 0.05484, 0.05359. Relative L2 crosses 0.03 at layers 45–50. |
| verifier-008 | HF full vs cached | Final relative L2 0.01105, 0.03826, 0.01459, 0.02960. Relative L2 crosses 0.03 only at positions 209 and 211, layers 54 and 59. |
| verifier-010 | HF full vs cached | Final relative L2 stays under 0.03 at every layer. The final-row failure is max-abs/RMS only. |
| verifier-000 | Native eager prefill vs depth-8 live prefill | Exact match at all 64 layers. Relative L2 0. |
| verifier-008 | Native 212-token eager prefill vs 9-token target forward at 208–211 | First gate at layer 4, max-abs/RMS. Position 209 crosses relative L2 0.03 at layer 51. Final layer 0.01292, 0.05690, 0.01709, 0.02390. |
| verifier-010 | Native 239-token eager prefill vs 9-token target forward at 235–238 | First gate at layer 7, max-abs/RMS. Final layer 0.02522, 0.02787, 0.02664, 0.02978. |

Both linear-attention and full-attention layers fail after the error has propagated. This does not isolate one kernel, and it does not show an indexing or rejection-cache bug. The native rows are decoder residual streams, not the post-norm final hidden state used by the original gate.

**Do not train yet.**
