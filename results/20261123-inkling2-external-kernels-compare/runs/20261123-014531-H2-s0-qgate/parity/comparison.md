# Independent full-model numerical comparison

Corrected CPU reference compares four short prompts in full and the first 2048 tokens of two long prompts (raw-long-00 and chat-long-00).

| Prompt | Compared positions | XPU top-1 agreement | XPU median / p95 / max Δlogprob | CPU bf16 median / p95 / max Δlogprob | Outside noise-floor positions |
|---|---:|---:|---:|---:|---:|
| raw-short-00 | 11 | 1.0000 | 0.0059768 / 0.120969 / 0.122831 | 0.0109509 / 0.120713 / 0.147867 | 3 |
| chat-short-00 | 26 | 0.9615 | 0.0709319 / 0.544968 / 0.661452 | 0.0825133 / 0.707399 / 2.06629 | 12 |
| raw-short-01 | 14 | 1.0000 | 0.00724304 / 0.132213 / 0.163431 | 0.0201674 / 0.128471 / 0.200066 | 5 |
| chat-short-01 | 29 | 0.9655 | 0.0305758 / 0.662114 / 0.867156 | 0.0679903 / 0.630672 / 0.740498 | 8 |
| raw-long-00 | 2047 | 0.9990 | 0.000340182 / 0.00751919 / 0.179791 | 0.000372576 / 0.00944667 / 0.362145 | 923 |
| chat-long-00 | 2047 | 0.9888 | 3.05168e-05 / 0.0328494 / 5.8399 | 3.23035e-05 / 0.0372843 / 5.63051 | 990 |

Overall (4174 scored positions): XPU top-1 agreement 0.9935; CPU bf16 top-1 agreement 0.9938.
XPU-vs-fp32 |Δlogprob| median/p95/max: 0.000147566 / 0.022937 / 5.8399.
CPU-bf16-vs-fp32 noise floor median/p95/max: 0.000166295 / 0.0285079 / 5.63051.
XPU delta exceeded same-position bf16 noise at 1941 positions.

## Greedy decode first divergences

| Prompt | Ref tokens | Candidate tokens | First divergence index |
|---|---:|---:|---:|
