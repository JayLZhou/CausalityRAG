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
| Reader-evaluated queries | 999 |
| Verified flips | 906 |
| Verified no-flip interventions | 93 |
| Unsupported queries | 1 |
| Overall flip rate | 90.6% |
| Flip rate among reader-evaluated queries | 90.69% |
| Mean modified tokens on successful queries | 2.21 |
| Mean modified tokens over all queries, including failures | 3.46 |
| Mean modified tokens over reader-evaluated queries | 3.46 |
| Mean modified tokens on verified no-flip queries | 15.69 |
| Mean reader calls | 4.26 |
| Total reader calls | 4,264 |
| Mean supported frontier size | 30.51 |
| Queries without a graph frontier | 1 |

A query is counted as a verified failure only when ReFlow applies a nonempty
counterfactual token intervention, invokes the reader on the revised context,
and the answer remains unchanged. A candidate without an executable
counterfactual replacement is not counted as a failure. The single unsupported
query above has no graph frontier and therefore receives no reader evaluation.

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
