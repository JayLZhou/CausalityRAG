# Qwen2.5-7B retrieval-depth result: k=10

All values below are recomputed from the final audited 1,000-query artifacts.
Invalid clean answers and invalid factual reader outputs are excluded from both metric numerators and denominators.
The retrieval-depth scheduler has no separate control stage, so control is reported as N/A.

| Dataset | Ans-FR | F1-FR | Mean edits | Graph coverage | Nodes | Edges | E2E s/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 93.52% (923/987) | 91.89% (408/444) | 3.357 | 99.70% | 920.15 | 705.27 | 17.491 |
| TimeQA | 95.50% (954/999) | 99.04% (310/313) | 1.711 | 100.00% | 613.09 | 1049.68 | 10.475 |
| FinQA | 87.63% (871/994) | 84.21% (64/76) | 3.321 | 99.60% | 1692.94 | 181.63 | 45.628 |
| MuSiQue | 97.53% (949/973) | 97.16% (171/176) | 1.997 | 99.10% | 923.59 | 796.60 | 14.330 |
| QuaRTz | 19.54% (195/998) | 18.41% (160/869) | 6.137 | 100.00% | 145.70 | 114.31 | 8.342 |
| TriviaQA | 74.92% (735/981) | 65.53% (479/731) | 6.537 | 98.80% | 2604.04 | 908.55 | 72.548 |
| 2Wiki | 89.75% (893/995) | 94.72% (287/303) | 3.283 | 99.60% | 807.39 | 510.47 | 15.962 |
| PopQA | 98.30% (982/999) | 96.97% (448/462) | 2.249 | 100.00% | 651.05 | 652.29 | 11.201 |
| **Micro/weighted** | **82.03%** | **68.97%** | **3.573** | **99.60%** | **1042.38** | **614.44** | **24.497** |

`summary.json` contains per-dataset status histograms, reader calls, latency components, pool hashes, and artifact hashes.
`audit.json` is the independent final integrity audit.
