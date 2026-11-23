# Independent full-model numerical comparison

Corrected CPU reference compares four short prompts in full and the first 2048 tokens of two long prompts (raw-long-00 and chat-long-00).

| Prompt | Compared positions | XPU top-1 agreement | XPU median / p95 / max Δlogprob | CPU bf16 median / p95 / max Δlogprob | Outside noise-floor positions |
|---|---:|---:|---:|---:|---:|
| raw-short-00 | 11 | 1.0000 | 0.00819206 / 0.118793 / 0.122756 | 0.0109509 / 0.120713 / 0.147867 | 2 |
| chat-short-00 | 26 | 0.9615 | 0.0394456 / 0.532336 / 0.637573 | 0.0825133 / 0.707399 / 2.06629 | 9 |
| raw-short-01 | 14 | 1.0000 | 0.00649565 / 0.0990131 / 0.163248 | 0.0201674 / 0.128471 / 0.200066 | 6 |
| chat-short-01 | 29 | 0.9655 | 0.0700455 / 0.501224 / 0.763505 | 0.0679903 / 0.630672 / 0.740498 | 9 |
| raw-long-00 | 2047 | 0.9990 | 0.000337395 / 0.0074086 / 0.179791 | 0.000372576 / 0.00944667 / 0.362145 | 927 |
| chat-long-00 | 2047 | 0.9893 | 3.05152e-05 / 0.0352169 / 5.83611 | 3.23035e-05 / 0.0372843 / 5.63051 | 991 |

Overall (4174 scored positions): XPU top-1 agreement 0.9938; CPU bf16 top-1 agreement 0.9938.
XPU-vs-fp32 |Δlogprob| median/p95/max: 0.000148677 / 0.0230354 / 5.83611.
CPU-bf16-vs-fp32 noise floor median/p95/max: 0.000166295 / 0.0285079 / 5.63051.
XPU delta exceeded same-position bf16 noise at 1944 positions.

## Greedy decode first divergences

| Prompt | Ref tokens | Candidate tokens | First divergence index |
|---|---:|---:|---:|
