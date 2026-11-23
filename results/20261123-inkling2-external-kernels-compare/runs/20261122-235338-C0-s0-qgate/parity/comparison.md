# Independent full-model numerical comparison

Corrected CPU reference compares four short prompts in full and the first 2048 tokens of two long prompts (raw-long-00 and chat-long-00).

| Prompt | Compared positions | XPU top-1 agreement | XPU median / p95 / max Δlogprob | CPU bf16 median / p95 / max Δlogprob | Outside noise-floor positions |
|---|---:|---:|---:|---:|---:|
| raw-short-00 | 11 | 1.0000 | 0.0166473 / 0.152558 / 0.154723 | 0.0109509 / 0.120713 / 0.147867 | 5 |
| chat-short-00 | 26 | 0.9615 | 0.0548909 / 0.542631 / 0.723628 | 0.0825133 / 0.707399 / 2.06629 | 12 |
| raw-short-01 | 14 | 1.0000 | 0.00740981 / 0.0695975 / 0.139779 | 0.0201674 / 0.128471 / 0.200066 | 3 |
| chat-short-01 | 29 | 1.0000 | 0.0490179 / 0.385674 / 0.891864 | 0.0679903 / 0.630672 / 0.740498 | 8 |
| raw-long-00 | 2047 | 0.9985 | 0.000400409 / 0.0098573 / 0.136997 | 0.000372576 / 0.00944667 / 0.362145 | 1101 |
| chat-long-00 | 2047 | 0.9902 | 3.62363e-05 / 0.0272382 / 5.54755 | 3.23035e-05 / 0.0372843 / 5.63051 | 1063 |

Overall (4174 scored positions): XPU top-1 agreement 0.9943; CPU bf16 top-1 agreement 0.9938.
XPU-vs-fp32 |Δlogprob| median/p95/max: 0.000170877 / 0.0230309 / 5.54755.
CPU-bf16-vs-fp32 noise floor median/p95/max: 0.000166295 / 0.0285079 / 5.63051.
XPU delta exceeded same-position bf16 noise at 2192 positions.

## Greedy decode first divergences

| Prompt | Ref tokens | Candidate tokens | First divergence index |
|---|---:|---:|---:|
