# HotpotQA ReFlow: token-label graph with on-demand replacements

This run uses 1,000 HotpotQA queries, top-5 retrieved chunks, the complete
non-punctuation context-token domain, the exact breakpoint Flow-Contract
frontier, and reader verification. Token selection is independent of
replacement availability. Missing counterfactual replacements are generated
after selection, validated with spaCy, and persisted in a shared pool.

## Result

| Metric | Value |
|---|---:|
| Queries | 1,000 |
| Verified flips | 906 |
| Overall flip rate | 90.6% |
| Mean modified tokens on successful queries | 2.21 |
| Mean terminal token count over all queries | 3.46 |
| Mean reader calls | 4.26 |
| Total reader calls | 4,264 |
| Mean supported frontier size | 30.51 |
| Queries without a graph frontier | 1 |

## Server artifacts

- Raw closed message flow (retained):
  `out/hotpotqa/graphs/contribution_graph_closed_beam48_1000.jsonl`
- Contracted Contribution Graph:
  `out/hotpotqa/graphs/contribution_graph_token_label_1000.jsonl`
- Growing replacement pool:
  `out/hotpotqa/inputs/replacement_pool_on_demand_1000.jsonl`
- Final ReFlow rows:
  `out/hotpotqa/methods/reflow/reflow_token_label_on_demand_1000.jsonl`
- Final summary:
  `out/hotpotqa/methods/reflow/reflow_token_label_on_demand_1000.summary.json`

The original strict replacement registry remains unchanged at
`out/hotpotqa/inputs/replacement_registry_strict_1000.jsonl`.
