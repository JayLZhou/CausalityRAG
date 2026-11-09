# Qwen2.5-7B retrieval-depth result: k=1

All values below are recomputed from the final audited 1,000-query artifacts.
Invalid clean answers, zero-executable-domain prefixes, and invalid factual reader outputs are excluded from both metric numerators and denominators.
The retrieval-depth scheduler has no separate control stage, so control is reported as N/A.

| Dataset | Ans-FR | F1-FR | Mean edits | Graph coverage | Nodes | Edges | E2E s/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 86.46% (817/945) | 96.25% (257/267) | 2.075 | 98.20% | 70.49 | 369.71 | 2.222 |
| TimeQA | 71.93% (674/937) | 100.00% (148/148) | 1.139 | 96.60% | 56.01 | 357.48 | 1.347 |
| FinQA | 84.69% (835/986) | 94.29% (66/70) | 1.977 | 99.20% | 164.29 | 224.62 | 3.893 |
| MuSiQue | 79.42% (652/821) | 98.78% (81/82) | 1.473 | 88.60% | 86.87 | 359.94 | 1.728 |
| QuaRTz | 32.56% (324/995) | 30.85% (273/885) | 3.220 | 99.70% | 15.55 | 72.24 | 2.604 |
| TriviaQA | 79.04% (773/978) | 77.96% (474/608) | 3.335 | 99.10% | 261.87 | 606.76 | 5.143 |
| 2Wiki | 75.53% (707/936) | 92.31% (180/195) | 2.923 | 98.20% | 79.81 | 233.20 | 2.508 |
| PopQA | 94.51% (895/947) | 98.49% (391/397) | 1.211 | 95.60% | 56.98 | 256.18 | 1.857 |
| **Micro/weighted** | **75.24%** | **70.51%** | **2.192** | **96.90%** | **99.48** | **309.25** | **2.663** |

`summary.json` contains per-dataset status histograms, reader calls, latency components, pool hashes, and artifact hashes.
`audit.json` is the independent final integrity audit.
