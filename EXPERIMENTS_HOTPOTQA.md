# HotpotQA Experimental Protocol

This document freezes the first full evaluation protocol for token-level RAG
resilience.  All methods use the same top-5 retrieved chunks, clean reader
trajectory, editable word-token domain, replacement registry, and answer
verifier.

## 1. Research Questions

- **RQ1 Effectiveness.** Does the method find small valid context-token sets
  whose replacement changes a clean RAG answer?
- **RQ2 Attribution fidelity.** Does the sentence-to-token lift predict actual
  singleton and pair intervention effects?
- **RQ3 Interaction value.** Do signed pair interactions improve over unary
  attribution rankings at equal edit cost?
- **RQ4 Optimization.** Does exact max-flow return the same graph optimum faster
  than MILP and improve over greedy selection?
- **RQ5 Efficiency.** How many logical model evaluations, GPU batches, reader
  generations, and editor generations are required relative to YVETTE and
  perturbation-based attribution baselines?
- **RQ6 Failure modes.** When does the method produce large sets, invalid
  revisions, or verified non-flips?

## 2. Frozen Task

For every HotpotQA query, use the first five retrieved chunks in the frozen
retrieval file.  The reader is Qwen2.5-7B-Instruct with greedy decoding and the
repository's short-answer JSON prompt.

The intervention domain contains non-punctuation surface-word tokens in the
retrieved chunks.  Edits must be non-deleting, single-token replacements and
must pass contextual NER, POS, tag, and morphology validation.  Query tokens,
generated answer tokens, model parameters, and retrieval rankings are immutable.

The primary cohort is the fixed first 1,000 records in the retrieval file.  A
manifest must store record IDs, top-5 chunk IDs and hashes, model/tokenizer
identifiers, prompts, decoding configuration, and repository commit.

Report two populations separately:

1. all 1,000 queries;
2. the clean-correct subset under the frozen reader.

Primary intervention conclusions use the clean-correct subset.  Results on all
queries are diagnostic only.

## 3. Shared Replacement Registry

Selection and replacement generation must be separated.

1. Every selector outputs token IDs without seeing candidate replacement text.
2. Build the union of token IDs requested by all methods for a query.
3. Generate and cache one deterministic valid replacement per token.
4. Reuse that exact replacement for every method selecting the token.
5. If a token has no valid replacement, ranking methods skip it and graph
   methods remove it and re-solve.

This registry prevents a method from winning because it received stronger or
more numerous replacements.  It also prevents replacement generation from
seeing the clean answer or method identity.

## 4. Comparison Regimes

### 4.1 Native One-Shot

Each method emits its natural set once and receives one reader verification.
This is the primary end-to-end setting for the unrestricted ratio method and
system baselines.

### 4.2 Size-Matched Selection

For each query, let the proposed method return `k = |S|`.  Every ranking
baseline edits its top `k` valid tokens with the shared registry.  This isolates
selection quality from edit count and uses one reader generation per method.

### 4.3 Budget Curves

Evaluate every token-ranking method at `k in {1, 2, 3, 5, 10}`.  Report valid
flip rate and correctness drop as functions of edit cost.  These are evaluation
budgets, not inputs to the unrestricted proposed algorithm.

### 4.4 Search-to-Flip

As a secondary, call-intensive protocol, verify prefixes in increasing `k` and
stop at the first valid flip.  Report reader calls separately; do not mix this
result with one-shot efficiency.

## 5. Baselines

### 5.1 Main Token-Selection Baselines (1,000 Queries)

1. **Random-Valid.** Uniform random valid tokens, five seeds.
2. **Query-Overlap.** Rank context tokens by lexical/BM25 relevance to the
   query, breaking ties by retrieval rank.
3. **Attention.** Rank by upper-layer response-to-context attention mass using
   the same clean trajectory as the proposed method.
4. **ARC-JSD Unary.** Use sentence ARC-JSD plus the conservative token lift,
   but retain only unary node weights `a_i`; rank by `a_i / c_i`.
5. **MIRAGE.** Gradient-based context-token saliency adapted from the official
   model-internals answer-attribution method.
6. **ALTI-Logit.** Layerwise input-token logit contribution accumulated over
   the generated response.
7. **ContextCite+Attention.** Compute ContextCite segment scores and rank tokens
   within segments using clean attention routing.
8. **Proposed.** Signed sentence-to-token lift, positive pairwise supermodular
   envelope, and exact unrestricted ratio optimization.

ARC-JSD Unary is the most important controlled baseline: it holds the sentence
intervention oracle and lift fixed, isolating the contribution of pair
interactions and ratio optimization.

### 5.2 End-to-End System Baseline

- **YVETTE.** Run its native extraction, rule, selection, revision, and reader
  verification pipeline on the same query/top-5 inputs.  Report its native edit
  unit and call count separately; do not present it as a size-matched token
  selector.

### 5.3 Restricted Oracles and Solver Baselines

- **Restricted Reader Oracle.** On 50 fixed queries, form a method-independent
  pool of at most 10 valid tokens from the union of baseline rankings.  Enumerate
  fixed-replacement subsets in increasing size with batched reader calls.  This
  gives the exact minimum flip set only within that pool.
- **MILP-Ratio.** Optimize the same pairwise graph objective using Dinkelbach
  plus a linearized MILP (`y_ij = x_i AND x_j`) with a 60-second timeout.
- **Greedy-Ratio.** Use a deterministic marginal-density greedy algorithm on
  the same graph.
- **Max-Flow Ratio.** The proposed exact maximum-closure solver.

The full black-box minimum-edit problem is not labeled an ILP oracle: its reader
constraints are unavailable without enumerating interventions.

## 6. Method Ablations

Run all ablations on the same 1,000-query manifest:

- uniform sentence-to-token allocation instead of attention routing;
- absolute shift vectors instead of signed shift vectors;
- signed lift with unary weights only;
- signed lift with pair edges but greedy optimization;
- signed lift with pair edges and exact max-flow;
- no contextual type gate;
- no pair-edge clipping diagnostic (analysis only; not a supermodular method).

Record positive-edge fraction, dropped negative interaction mass, selected-set
median/P90/max, and the fraction of sets larger than 20 tokens.

## 7. Fidelity Experiments

### 7.1 Singleton Fidelity

On 100 fixed queries, sample at least 20 valid tokens per query.  Apply the
shared replacement individually and compute true response JSD.  Compare it with
the lifted unary score using Spearman, Pearson, and top-k overlap.

### 7.2 Pair Fidelity

On the same queries, sample at least 100 valid token pairs per query, stratified
by predicted positive, near-zero, and negative raw inner product.  Compare

```text
true synergy = D({i,j}) - D({i}) - D({j})
```

with the raw signed interaction and positive envelope weight.

### 7.3 Restricted Optimality Gap

For each restricted-oracle query, report:

- exact restricted minimum edit count;
- proposed selected-set size and whether it flips;
- additive and multiplicative edit-count gap;
- whether the proposed set intersects an exact optimum.

## 8. Metrics

### 8.1 Answer Metrics

- HotpotQA exact match and token F1 for clean and edited answers;
- normalized clean-to-edited answer-change rate;
- correct-to-incorrect flip rate on the clean-correct subset;
- manually audited semantic-equivalence accuracy on 200 stratified answer pairs.

Do not use raw answer-string containment as the only primary metric.

### 8.2 Intervention Metrics

- valid one-shot flip rate;
- valid flip rate at matched size;
- success at each evaluation budget;
- median, mean, P90, and maximum selected tokens;
- successful-set edit count as a verified upper bound;
- NER/POS/morphology validity and failed-candidate rate.

### 8.3 Efficiency Metrics

- end-to-end latency (median/P90);
- graph construction and solver latency separately;
- logical model evaluations;
- batched GPU forward calls;
- reader and editor generations;
- peak GPU memory;
- output-token usage for LLM-based stages.

Use paired bootstrap 95% confidence intervals.  Use paired significance tests
for flip outcomes and report five-seed variation for Random-Valid.

## 9. Experiment Matrix

| Experiment | Queries | Methods |
|---|---:|---|
| Main native + size-matched | 1,000 | cheap baselines, attribution baselines, proposed |
| Budget curves | 1,000 | all token rankings + proposed ranking variants |
| YVETTE comparison | 1,000 | YVETTE, proposed |
| Singleton fidelity | 100 | lifted unary vs true intervention |
| Pair fidelity | 100 | lifted edge vs true pair synergy |
| Restricted reader oracle | 50 | exhaustive pool oracle, proposed, rankings |
| Solver comparison | 1,000 graphs | max-flow, MILP timeout, greedy |
| Human validity audit | 200 answer pairs + 200 edits | stratified by method/outcome |

## 10. Pipeline and Artifacts

Run stages independently and never regenerate upstream artifacts implicitly.

```text
00 manifest
01 clean trajectories
02 contextual token annotations
03 sentence interventions and routing
04 method scores/graphs
05 selector outputs
06 shared valid replacement registry
07 edited reader outputs
08 metrics and bootstrap tables
```

Each artifact must contain a configuration hash and parent-artifact hash.  A
resumed stage must reject incompatible parents instead of silently mixing runs.

## 11. Acceptance Criteria Before Main Claims

- at least 95% of applied edits pass contextual validation;
- zero answer-string-to-context matching in every proposed selector;
- all methods use identical top-5 chunks and cached clean responses;
- size-matched baselines use the same replacement registry;
- primary flip claims are reported on the clean-correct subset;
- max-flow and MILP agree on all graphs solved to MILP optimality;
- restricted-oracle and fidelity experiments are complete;
- latency and call counts include every attribution, editing, and verification
  stage.
