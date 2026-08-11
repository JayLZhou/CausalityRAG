# Experiment Inventory (2026-08-11)

This inventory freezes the publication status before launching the
hyperparameter experiments. The main benchmark suite is:

`hotpotqa, timeqa, finqa, musique, quartz, triviaqa, 2wiki, popqa`.

## Publication-ready

- Final Table 3: complete for 8 datasets x 1,000 queries x 7 methods.
- Final Table 3 execution audit: zero baseline execution gaps.
- Table 3 mean edited-token audit and bar-chart data: complete.
- Query-level token contract: a verified flip uses its verified attempt;
  a failed flip uses the largest saved terminal attempt; an invalid clean
  answer is excluded from both the token numerator and query denominator.
- Ans-FR population: valid clean answers only. A valid-clean query with an
  unsupported or unexecuted intervention remains in the denominator as zero.
- PopQA correctness: matching any gold alias is correct; a correctness flip
  requires the edited answer to miss every gold alias.

Final output root:
`/data1/yujia/CausalityRAG/out_table3_final8_20260811`.

Mean edited tokens over valid-clean queries:

| Dataset | Token sum | Valid clean | Mean |
| --- | ---: | ---: | ---: |
| HQA | 2,664 | 989 | 2.694 |
| TQA | 1,566 | 995 | 1.574 |
| FQA | 3,051 | 992 | 3.076 |
| MSQ | 1,830 | 965 | 1.896 |
| QTZ | 5,321 | 1,000 | 5.321 |
| TrQA | 5,576 | 979 | 5.696 |
| 2Wiki | 3,007 | 994 | 3.025 |
| PopQA | 2,083 | 998 | 2.087 |

Across all eight datasets, the pooled value is `25,098 / 7,912 = 3.1721`.

## Measured but not final for the current suite

- Fixed-budget / budget-to-match experiments: old outputs cover HQA, TQA,
  FQA, MSQ, QTZ, 2Wiki, Qasper, and PubMedQA. They omit TrQA and PopQA and do
  not include the targeted graph/ranking repairs used by final Table 3.
- Exhaustive prefix experiment: same old dataset mismatch and repair gap.
- Runtime/scaling aggregate: must be regenerated against the final eight
  datasets and final graph artifacts.
- Cache, fidelity, frontier, and security case studies have measured outputs,
  but their frozen query and graph SHAs must be checked against the final
  artifacts before the publication gate. Reuse is allowed only on exact SHA
  matches.
- Cross-model sensitivity has factual/control files for the old eight-dataset
  suite, but it is not publication-ready: the added models contain many
  `missing_paraphrase` controls, old no-frontier cases remain, and PubMedQA
  needs the corrected yes/no/maybe reader protocol. Valid factual graphs may
  be reused after a per-artifact SHA and protocol audit.

## Not run

### RUN-K: retrieval depth

- Values: `k = {1, 3, 5, 10}` from one frozen top-10 retrieval result.
- Rebuild for every prefix: clean target, token units, replacement domain,
  contribution graph, frontier, and all matched-budget method outputs.
- Required metrics: Ans-FR, F1-FR, mean edited tokens, graph nodes/edges,
  coverage, and end-to-end latency.
- `k=5` may be reused only if every frozen query, retrieval, pool, model,
  prompt, and decoding SHA exactly matches final Table 3.

### RUN-BEAM: receiver beam

- Values: `d = {12, 24, 48, 96}`.
- Rebuild ReFlow graphs/frontiers and verify factual reader outputs.
- Required metrics: retained contribution mass, graph edges, graph time,
  Ans-FR, mean verified intervention size, and coverage.
- `d=48` may be reused only after the same exact-manifest check.

## Frozen run order for the next session

1. Write and validate one manifest per dataset with query, retrieval, pool,
   clean-target, model, prompt, and decoding hashes.
2. Run RUN-K for `k=1,3,10`; reuse `k=5` only after exact equality checks.
3. Run RUN-BEAM for `d=12,24,96`; reuse `d=48` only after exact equality
   checks.
4. Run no-frontier repair only for affected configurations and queries.
5. Render hyperparameter JSON/TeX from per-query outputs; no manual or
   synthetic coordinates.
6. Repair the cross-model controls and corrected PubMedQA-specific stages.
7. Recompute fixed-budget, budget-to-match, prefix, and scaling outputs for
   the final eight-dataset suite.

Hyperparameter runs were deliberately not started on 2026-08-11.
