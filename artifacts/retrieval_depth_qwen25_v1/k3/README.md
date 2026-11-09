# Qwen2.5-7B retrieval-depth result: k=3

All values below are recomputed from the final audited 1,000-query artifacts.
Invalid clean answers, zero-executable-domain prefixes, and invalid factual reader outputs are excluded from both metric numerators and denominators.
The retrieval-depth scheduler has no separate control stage, so control is reported as N/A.

| Dataset | Ans-FR | F1-FR | Mean edits | Graph coverage | Nodes | Edges | E2E s/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 93.85% (915/975) | 95.00% (380/400) | 2.757 | 99.00% | 239.45 | 604.26 | 5.166 |
| TimeQA | 90.10% (865/960) | 100.00% (267/267) | 1.411 | 98.50% | 180.04 | 789.06 | 3.049 |
| FinQA | 82.41% (815/989) | 85.71% (60/70) | 2.748 | 99.20% | 497.48 | 193.96 | 11.980 |
| MuSiQue | 93.66% (857/915) | 97.87% (138/141) | 1.949 | 95.80% | 270.76 | 672.24 | 4.592 |
| QuaRTz | 25.00% (249/996) | 23.07% (200/867) | 4.868 | 99.80% | 44.48 | 122.89 | 3.976 |
| TriviaQA | 80.65% (792/982) | 74.61% (523/701) | 4.746 | 99.10% | 778.26 | 782.40 | 17.378 |
| 2Wiki | 87.50% (861/984) | 94.29% (264/280) | 3.168 | 99.50% | 235.89 | 424.66 | 6.112 |
| PopQA | 98.08% (972/991) | 98.23% (444/452) | 1.698 | 99.50% | 182.49 | 493.29 | 3.790 |
| **Micro/weighted** | **81.19%** | **71.62%** | **2.933** | **98.80%** | **303.54** | **509.01** | **7.005** |

`summary.json` contains per-dataset status histograms, reader calls, latency components, pool hashes, and artifact hashes.
`audit.json` is the independent final integrity audit.
