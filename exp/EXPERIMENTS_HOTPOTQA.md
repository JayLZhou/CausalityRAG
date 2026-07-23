# Historical HotpotQA Experimental Protocol

This document freezes the first full evaluation protocol for token-level RAG
resilience.  All methods use the same top-5 retrieved chunks, clean reader
trajectory, editable word-token domain, replacement registry, and answer
verifier.

## 1. Research Questions

- **RQ1 Effectiveness.** Does the method find small valid context-token sets
  whose replacement changes a clean RAG answer?
- **RQ2 Graph fidelity.** Does residual contribution flow predict which strict
  token-replacement sets change the reader answer?
- **RQ3 Structural value.** Does path-aware grouped flow improve over the same
  ARC-JSD unary scores at equal edit cost?
- **RQ4 Optimization.** What accuracy/runtime tradeoff is induced by active
  group rank, and how tight is K-guess min-cut rounding against restricted
  exact optima?
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
   methods keep its paths but make all gates with that label uncuttable.
6. Repeat registry construction and selection until no newly selected token is
   missing from the registry.

This registry prevents a method from winning because it received stronger or
more numerous replacements.  It also prevents replacement generation from
seeing the clean answer or method identity.

## 4. Comparison Regimes

### 4.1 Native One-Shot

Each method emits its natural set once and receives one reader verification.
This is the primary end-to-end setting for grouped contribution-flow and system
baselines.

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

### 4.5 Reader Backend Consistency

Attribution and answer-change verification must use the same model weights,
attention implementation, dtype, prompt, and greedy decoding backend.  Results
produced by the round-robin vLLM proxy are diagnostic only: different attention
kernels or replicas can change a greedy trajectory even at temperature zero.
For final flip metrics, re-run every unique edited context with the local HF
`eager` reader used to produce the clean trajectory.  Cache identical edit sets
across methods and use `scripts/verify_hf_results.py`; do not count a
backend-induced answer difference as a token-intervention flip.

## 5. Baselines

### 5.1 Main Token-Selection Baselines (1,000 Queries)

1. **Random-Valid.** Uniform random valid tokens, five seeds.
2. **Query-Overlap.** Rank context tokens by lexical/BM25 relevance to the
   query, breaking ties by retrieval rank.
3. **Attention.** Rank by upper-layer response-to-context attention mass using
   the same clean trajectory as the proposed method.
4. **ARC-JSD Unary.** Use sentence ARC-JSD plus the conservative token lift,
   but retain only unary node weights `a_i`; rank by `a_i / c_i`.
5. **MIRAGE.** Follow the official two-stage model-internals method: select
   response tokens whose full-vs-contextless KL is at least the example mean
   plus one population standard deviation, then score context tokens by the L2
   embedding gradient of the contrastive log-probability difference between
   the clean token and its contextless foil.  Sum saliency across selected
   response tokens.  For editable surface words spanning multiple model
   subwords, take the L2 norm of the concatenated subword gradients.  Evaluate
   the resulting unary ranking at fixed Top-1/3/5 budgets and size-matched
   budgets using `exp/run_mirage_topk_attack.py`.
6. **ALTI-Logit.** Layerwise input-token logit contribution accumulated over
   the generated response.
7. **ContextCite+Attention.** Compute ContextCite segment scores and rank tokens
   within segments using clean attention routing.
8. **Proposed.** Build a path-preserving absorbing contribution DAG from direct
   local target-logit writes. Preserve transformer layers and give every
   editable layer copy its input-token label. Add normalized ARC-JSD unary
   support as independent paths with the same labels, then solve grouped
   residual-flow interdiction using K weighted min-cuts, label rounding, and
   feasibility-preserving label pruning.

ARC-JSD Unary is the most important controlled baseline: it holds unary scores,
the editable domain, replacements, and edit count fixed, isolating the value of
the contribution paths and grouped flow optimizer.

Do not interpret a native threshold result against MIRAGE Top-5 as a
size-controlled comparison. Claims about graph structure must use the
size-matched protocol in Section 4.2.

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
- **Grouped MILP Oracle.** Use one binary variable per token label, cut-side
  variables for layer copies, and finite-edge crossing variables. Enforce all
  copies of a label jointly and solve with HiGHS, using a 60-second timeout on
  graphs not proved optimal.
- **Contracted Min-Cut.** Contract equal token positions across layers and solve
  one ordinary min-cut. This is fast but may create impossible cross-layer
  paths.
- **Copy Min-Cut.** Ignore shared labels and solve the weighted copy relaxation
  without rounding.
- **K-Guess Group Flow.** The proposed weighted copy min-cut plus token-label
  rounding.

The full black-box minimum-edit problem is not labeled an ILP oracle: its reader
constraints are unavailable without enumerating interventions.

## 6. Method Ablations

Run all ablations on the same 1,000-query manifest:

- ARC-JSD unary only (`alpha = 0`);
- contribution graph only (`alpha = 1`);
- `alpha in {0.25, 0.5, 0.75}` fixed globally on development data;
- absorbing flow versus closed flow with synthetic background mass;
- path-preserving DAG versus global edge pruning;
- layer-preserving groups versus layer-copy contraction;
- active-copy cap in `{4, 8, 16, all}`;
- K-guess threshold output versus heuristic reverse-delete budget chains;
- rounded K-guess output versus feasibility-preserving threshold pruning;
- strict contextual registry versus surface-only replacements as a diagnostic;
- one versus three deterministic replacement-registry seeds.

Record realized maximum/mean group rank, excluded copy throughput, absorbed
mass, min-cut calls, candidate sizes, registry misses, and strict validity.

## 7. Fidelity Experiments

### 7.1 Residual-Flow Fidelity

On 100 fixed queries, evaluate the proposed candidate chain plus stratified
random and unary-matched sets. Compare predicted residual flow with clean-answer
logit drop, output JSD, and verified answer change using Spearman, AUROC, and
calibration plots.

### 7.2 Structural Complementarity

At budgets 1, 3, and 5, report paired proposed-only and unary-only flips. Audit
the selected paths for proposed-only wins and test whether their tokens occupy
shared internal bottlenecks that unary scores cannot represent.

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
| Residual-flow fidelity | 100 | graph candidates, unary-matched, random-matched |
| Registry robustness | 1,000 | three fixed strict replacement seeds |
| Restricted reader oracle | 50 | exhaustive pool oracle, proposed, rankings |
| Solver comparison | 1,000 graphs | max-flow, MILP timeout, greedy |
| Human validity audit | 200 answer pairs + 200 edits | stratified by method/outcome |

## 10. Pipeline and Artifacts

Run stages independently and never regenerate upstream artifacts implicitly.

```text
00 manifest
01 clean trajectories
02 contextual token annotations
03 absorbing contribution graphs and ARC-JSD unary scores
04 provisional selector outputs
05 iterative shared strict replacement registry
06 final grouped-flow and baseline outputs on the closed editable domain
07 edited reader outputs
08 local-HF eager re-verification
09 metrics and bootstrap tables
```

Each artifact must contain a configuration hash and parent-artifact hash.  A
resumed stage must reject incompatible parents instead of silently mixing runs.

## 11. Acceptance Criteria Before Main Claims

- at least 95% of applied edits pass contextual validation;
- zero answer-string-to-context matching in every proposed selector;
- all methods use identical top-5 chunks and cached clean responses;
- size-matched baselines use the same replacement registry;
- primary flip claims are reported on the clean-correct subset;
- weighted min-cut identity holds numerically and grouped MILP agrees on all
  small graphs solved to optimality;
- restricted-oracle, graph-fidelity, and registry-seed experiments are complete;
- latency and call counts include every attribution, editing, and verification
  stage.
