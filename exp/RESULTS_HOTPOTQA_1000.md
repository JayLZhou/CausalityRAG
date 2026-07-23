# Historical HotpotQA-1000 Hybrid Result

This report records the first closed-domain evaluation of grouped contribution
flow. It is an experiment artifact, not a venue-readiness claim.

## Frozen Setup

- Dataset: first 1,000 rows of the fixed HotpotQA retrieval file, top-5 chunks.
- Reader: Qwen2.5-7B-Instruct, greedy JSON short-answer prompt.
- Editable domain: non-punctuation chunk surface tokens only.
- Replacement contract: answer-blind, non-deleting, one-word replacement with
  contextual POS, tag, morphology, and entity-slot checks.
- Selector: absorbing direct-activation graph plus ARC-JSD unary support,
  `alpha=0.5`, copy cap 8, `beta=0.25`, `eta=1`, and `K=10`.
- The native bicriteria output therefore targets strict residual support
  `B=0.25` and permits the theorem's relaxed threshold `(1+eta)B=0.50`.
- Primary verifier: local Hugging Face eager attention using the same model
  weights and prompt as graph construction.

The provisional vLLM cohort contained 400 normalized-exact clean-correct
queries. Local eager re-verification retained 397 exact-clean queries. The
native threshold had a valid candidate for 393 of those 397 queries.

## Reader Results

All comparisons are paired and size matched. `Flow` and `Unary` use the same
token count and the exact same frozen replacement for every shared token.

| Setting | Flow | ARC-JSD unary | Difference | Paired 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| Budget 1, n=397 | 198/397 (49.87%) | 198/397 (49.87%) | 0.00 pp | [-3.27, 3.27] | 1.0000 |
| Budget 3, n=397 | 245/397 (61.71%) | 231/397 (58.19%) | +3.53 pp | [0.76, 6.30] | 0.0243 |
| Budget 5, n=397 | 257/397 (64.74%) | 244/397 (61.46%) | +3.27 pp | [0.25, 6.30] | 0.0470 |
| Native threshold, n=393 | 246/393 (62.60%) | 229/393 (58.27%) | +4.33 pp | [1.02, 7.63] | 0.0186 |

The native candidate uses 2.64 tokens on average. Budget 1 shows that the graph
does not improve the best singleton. The statistically supported gain begins
at multi-token budgets, which is the behavior expected from path bottlenecks
rather than another unary ranking.

The round-robin vLLM run gave the same qualitative pattern, but exact agreement
between vLLM and local eager edited answers was only 75.0--83.0% depending on
the setting. Therefore vLLM results are diagnostic and are not the primary
numbers above.

## Optimization Audit

On the first 100 final-registry graphs, HiGHS proved 99 threshold instances
feasible and optimal and proved one instance infeasible. Among the 99 feasible
instances:

- 89 branch-free solutions have the same cardinality as the exact relaxed-
  threshold optimum;
- mean size ratio is 1.05, median is 1.0, and maximum is 2.0;
- 96 instances satisfy the theorem precondition `k* <= K=10`;
- there are zero residual-flow or cardinality-bound violations among those 96.

This audit concerns the graph surrogate. It is not an exact oracle for the
black-box reader objective.

## Registry And Cost

- Fixed-point registry candidates: 7,749.
- Strictly valid replacements: 7,407 (95.6%).
- Invalid and uncuttable tokens: 342.
- Missing replacements in every evaluated flow/unary set: 0.
- Editor calls in the final coherent closure: 4,082.
- Reader calls in the four diagnostic vLLM comparisons: 2,588 after caching
  identical flow/unary edit sets.
- Local eager verification: 2,307 unique edited variants plus 400 clean
  variants, evaluated in 400 per-query batches.
- Mean local eager time per query, including clean and all unique variants:
  0.745 seconds.
- Mean graph construction time: 0.407 seconds per query.
- Mean branch-free solver CPU time: 4.15 seconds per query; optimization of the
  pure-Python residual max-flow implementation remains open.

## Frozen Artifacts On Server A

```text
out/direct_activation_absorbing_flow_hotpotqa_1000.jsonl
out/token_units_hotpotqa_top5_1000.jsonl
out/replacement_registry_absorbing_coherent_i4_1000.jsonl
out/hybrid_absorbing_cap8_gw50_final_beta25_1000.jsonl
out/attack_final_b1_1000.jsonl
out/attack_final_b3_1000.jsonl
out/attack_final_b5_1000.jsonl
out/attack_final_native_1000.jsonl
out/hf_reverify_final_core_1000.jsonl
out/group_flow_final_registry_oracle_first100.jsonl
```

## What Is Still Missing

These results establish a credible method result against the controlled unary
baseline. A VLDB claim still requires at least a second dataset and reader,
MIRAGE/attention/random/YVETTE comparisons, rank and alpha ablations,
restricted exhaustive reader optima, semantic-equivalence and edit-validity
audits, and optimized solver/runtime measurements.
