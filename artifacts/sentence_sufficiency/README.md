# Sentence-Sufficiency Case Study

## Question

Do the sentences containing a method's selected tokens retain enough evidence for
the same reader to answer the question correctly?

This experiment replaces the dynamic-cache case study. It does not simulate
request traces, cache eviction, or stale answers.

## Protocol

- `Full`: the original top-5 retrieved chunks.
- `ReFlow`: every distinct sentence containing at least one token in ReFlow's
  final top-level `selected_ids` set.
- `MIRAGE`: sentences reached by the MIRAGE token ranking until it matches the
  number of distinct ReFlow sentences for that query.
- `ARC-JSD`: sentences reached by the ARC-JSD token ranking until it matches the
  same per-query sentence budget.
- Selected sentences are restored to their original chunk/sentence order before
  being given to the reader.
- All four contexts are evaluated by fresh calls to the same
  Qwen2.5-7B-Instruct reader (`temperature=0`, at most 64 generated tokens).
- Metrics are normalized exact match (EM) and answer-token F1.
- Error bars are query-level nonparametric 95% bootstrap intervals with 10,000
  resamples. Paired ReFlow-minus-baseline intervals use the same query population.
- The common evaluation population contains queries with a nonempty final
  ReFlow token set and enough ranked baseline tokens to meet ReFlow's distinct
  sentence budget. Coverage is reported rather than silently imputed.

The plotted legend names are exactly `Full`, `ReFlow`, `MIRAGE`, and `ARC-JSD`.

## Main-text results

| Dataset | Queries | Full EM/F1 | ReFlow EM/F1 | MIRAGE EM/F1 | ARC-JSD EM/F1 | ReFlow sentences | Full/ReFlow tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 983 | 29.81 / 35.55 | 27.16 / 33.34 | 24.82 / 30.57 | 23.80 / 29.63 | 1.95 | 477.5 / 50.7 |
| TimeQA | 931 | 31.36 / 44.20 | 30.61 / 42.51 | 29.00 / 40.30 | 27.93 / 39.49 | 1.21 | 354.4 / 93.9 |

On both main datasets, all four nominal 95% paired bootstrap intervals for
ReFlow versus MIRAGE/ARC-JSD (EM and F1) are above zero. ReFlow preserves most
of Full's reader accuracy while using substantially less context.

## Appendix results

| Dataset | Queries | Full EM/F1 | ReFlow EM/F1 | MIRAGE EM/F1 | ARC-JSD EM/F1 | ReFlow sentences | Full/ReFlow tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 985 | 41.73 / 53.13 | 38.48 / 50.72 | 21.42 / 28.55 | 32.39 / 42.84 | 1.80 | 524.2 / 51.7 |
| FinQA | 918 | 8.71 / 9.06 | 8.93 / 9.11 | 7.41 / 7.49 | 7.84 / 7.87 | 1.75 | 1138.8 / 379.3 |
| MuSiQue | 938 | 15.99 / 24.83 | 15.99 / 24.21 | 11.94 / 19.25 | 13.43 / 20.44 | 1.50 | 550.0 / 48.8 |
| QuARTz | 958 | 72.44 / 75.77 | 68.79 / 71.17 | 68.16 / 70.37 | 66.91 / 68.57 | 2.76 | 82.2 / 45.3 |
| QASPER | 943 | 4.35 / 10.86 | 4.14 / 9.62 | 4.24 / 9.34 | 4.35 / 9.55 | 1.83 | 441.2 / 54.5 |
| PubMedQA | 998 | 6.81 / 7.59 | 9.22 / 10.07 | 8.12 / 8.72 | 8.42 / 9.20 | 1.64 | 1134.9 / 54.1 |

The strongest appendix separation is on HotpotQA, FinQA, and MuSiQue. QuARTz
and QASPER are near-ties under the matched sentence budget; PubMedQA shows a
small ReFlow advantage, but most paired intervals cross zero. These are nominal
intervals and are not adjusted for multiple comparisons.

## Coverage

| Dataset | Included / 1000 | Exclusion reason |
|---|---:|---|
| 2Wiki | 983 | 17 empty ReFlow selections |
| TimeQA | 931 | 69 empty ReFlow selections |
| HotpotQA | 985 | 12 empty ReFlow selections; 3 baseline sentence-budget shortfalls |
| FinQA | 918 | 82 empty ReFlow selections |
| MuSiQue | 938 | 62 empty ReFlow selections |
| QuARTz | 958 | 42 empty ReFlow selections |
| QASPER | 943 | 57 empty ReFlow selections |
| PubMedQA | 998 | 2 empty ReFlow selections |

## Execution record

- Frozen retrieval, token-unit, ReFlow, MIRAGE, and ARC-JSD inputs were copied
  from machine 3.
- Reader evaluation ran only on machine 14 physical GPU0.
- Remote root:
  `/data2/yujia/CausalityRAG_fig8_fig9_20260804/sentence_sufficiency`
- The final audit found no reader failures; all expected per-query rows and all
  eight summaries are present. Port 8101 is closed and GPU0 is released.
