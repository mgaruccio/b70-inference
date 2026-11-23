# Independent full-model numerical comparison

Corrected CPU reference compares four short prompts in full and the first 2048 tokens of two long prompts (raw-long-00 and chat-long-00).

| Prompt | Compared positions | XPU top-1 agreement | XPU median / p95 / max Δlogprob | CPU bf16 median / p95 / max Δlogprob | Outside noise-floor positions |
|---|---:|---:|---:|---:|---:|
| raw-short-00 | 11 | 1.0000 | 0.0137024 / 0.138674 / 0.23103 | 0.0109509 / 0.120713 / 0.147867 | 6 |
| chat-short-00 | 26 | 0.9615 | 0.0424149 / 0.252854 / 0.318197 | 0.0825133 / 0.707399 / 2.06629 | 6 |
| raw-short-01 | 14 | 1.0000 | 0.0146058 / 0.0985165 / 0.156244 | 0.0201674 / 0.128471 / 0.200066 | 6 |
| chat-short-01 | 29 | 0.9655 | 0.060426 / 0.545112 / 0.923579 | 0.0679903 / 0.630672 / 0.740498 | 14 |
| raw-long-00 | 2047 | 0.9971 | 0.000377988 / 0.00978433 / 0.198291 | 0.000372576 / 0.00944667 / 0.362145 | 1072 |
| chat-long-00 | 2047 | 0.9912 | 3.1348e-05 / 0.0322373 / 5.68923 | 3.23035e-05 / 0.0372843 / 5.63051 | 981 |

Overall (4174 scored positions): XPU top-1 agreement 0.9938; CPU bf16 top-1 agreement 0.9938.
XPU-vs-fp32 |Δlogprob| median/p95/max: 0.00016786 / 0.0227467 / 5.68923.
CPU-bf16-vs-fp32 noise floor median/p95/max: 0.000166295 / 0.0285079 / 5.63051.
XPU delta exceeded same-position bf16 noise at 2085 positions.

## Greedy decode first divergences

| Prompt | Ref tokens | Candidate tokens | First divergence index |
|---|---:|---:|---:|
