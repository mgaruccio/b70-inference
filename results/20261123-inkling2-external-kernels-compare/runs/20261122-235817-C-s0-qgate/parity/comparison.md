# Independent full-model numerical comparison

Corrected CPU reference compares four short prompts in full and the first 2048 tokens of two long prompts (raw-long-00 and chat-long-00).

| Prompt | Compared positions | XPU top-1 agreement | XPU median / p95 / max Δlogprob | CPU bf16 median / p95 / max Δlogprob | Outside noise-floor positions |
|---|---:|---:|---:|---:|---:|
| raw-short-00 | 11 | 1.0000 | 0.0192647 / 0.150847 / 0.153095 | 0.0109509 / 0.120713 / 0.147867 | 7 |
| chat-short-00 | 26 | 0.9231 | 0.0745053 / 0.32143 / 0.725122 | 0.0825133 / 0.707399 / 2.06629 | 12 |
| raw-short-01 | 14 | 1.0000 | 0.0109294 / 0.147272 / 0.213542 | 0.0201674 / 0.128471 / 0.200066 | 7 |
| chat-short-01 | 29 | 0.9310 | 0.0688376 / 0.497765 / 1.02314 | 0.0679903 / 0.630672 / 0.740498 | 9 |
| raw-long-00 | 2047 | 0.9990 | 0.000394321 / 0.00957787 / 0.142232 | 0.000372576 / 0.00944667 / 0.362145 | 1073 |
| chat-long-00 | 2047 | 0.9888 | 3.74305e-05 / 0.0293513 / 5.73954 | 3.23035e-05 / 0.0372843 / 5.63051 | 1068 |

Overall (4174 scored positions): XPU top-1 agreement 0.9931; CPU bf16 top-1 agreement 0.9938.
XPU-vs-fp32 |Δlogprob| median/p95/max: 0.000175403 / 0.0231074 / 5.73954.
CPU-bf16-vs-fp32 noise floor median/p95/max: 0.000166295 / 0.0285079 / 5.63051.
XPU delta exceeded same-position bf16 noise at 2176 positions.

## Greedy decode first divergences

| Prompt | Ref tokens | Candidate tokens | First divergence index |
|---|---:|---:|---:|
