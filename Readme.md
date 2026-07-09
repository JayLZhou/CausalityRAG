# Repository Reboot Guide: RAG Answer Resilience

This document defines the new repository direction. The old pipeline was mainly
an evidence-attack benchmark: select an answer/fact, edit it, and report flip
rates. The new repository should instead be organized around a stronger object:

> A RAG answer's resilience is the minimum cost of type-valid counterfactual
> revisions to retrieved-context units required to make the reader stop
> preserving its original answer.

The core method is:

```text
clean RAG run
-> attribution/support graph over context tokens and answer tokens
-> editable token-unit proposal
-> projection from token flow to editable units
-> minimum evidence cut
-> type-valid counterfactual revision
-> rerun reader and validate answer change
```

The repository should be built around this spine, not around rule mining, not
around simple answer-string replacement, and not around architecture comparison
as the main claim.

## 1. Positioning

### One-sentence claim

We measure the resilience of a RAG answer by finding the minimum-cost set of
retrieved-context units whose type-valid counterfactual revision changes the
reader's answer.

### What this paper is

- A method for measuring RAG answer resilience.
- A graph/optimization formulation over retrieved context.
- A counterfactual evidence revision framework.
- A characterization of which answers, fact types, and question types have low
  or high resilience.

### What this paper is not

- Not a generic RAG attack paper.
- Not a rule-mining paper.
- Not a claim that we invented attribution graphs.
- Not a pure E&A paper that only reports flip ratios.
- Not a proof that the model's internal knowledge state is known.

### Recommended title direction

```text
Minimum Evidence Cuts for Measuring RAG Answer Resilience
```

Alternative:

```text
How Resilient Are RAG Answers?
Counterfactual Evidence Cuts in Retrieval-Augmented Generation
```

## 2. Problem Setup

For each example, we have:

```text
q       question
C       retrieved context
R       reader model / RAG pipeline
a       clean answer, a = R(q, C)
```

The object of study is the instance:

```text
(q, C, a)
```

We ask:

> What is the minimum evidence revision needed to make `R` stop preserving
> answer `a` under the same question `q`?

Gold answers can be used for evaluation and filtering, but the core method
should be able to operate from `q`, `C`, and `a`.

## 3. Core Definitions

### 3.1 Retrieved Context Tokens

Let the retrieved context be tokenized by the reader tokenizer:

```text
C = (t_1, t_2, ..., t_n)
```

The attribution/support graph is initially token-level because transformer
decoding and attribution are token-level.

### 3.2 Editable Token Unit

Do not force the intervention unit to be a full entity span. Also do not use
arbitrary raw subword tokens as final edit units.

Define an editable token unit `u` as:

> The smallest contiguous segment of retrieved-context tokens for which at
> least one valid counterfactual replacement exists in local context.

So `u` may be:

- a single token, e.g. `Paris -> London`;
- a token inside a larger entity, e.g. `New York -> New Jersey` by editing
  `York -> Jersey`;
- a multi-token entity or value, e.g. `San Diego -> Boston`;
- a numeric/date expression, e.g. `1947 -> 1952`;
- a relation cue, e.g. `was born in -> died in`;
- a short factual phrase when token/value replacement is insufficient.

Each unit has:

```text
text(u)       original text
tokens(u)     token interval
type(u)       PERSON / GPE / DATE / NUMBER / RELATION_CUE / PHRASE / ...
Omega(u)      type-valid replacement domain
cost(u)       edit cost
```

Important principle:

```text
token = attribution unit
editable unit = intervention unit
```

Most editable units will be one token. Multi-token units are used only when a
valid edit requires merging adjacent tokens.

### 3.3 Replacement Domain

For each editable unit `u`, define a replacement domain:

```text
Omega(u) = { valid counterfactual replacements for u }
```

Examples:

```text
DATE          same-granularity dates
PERSON        plausible person names
GPE/LOC       plausible locations
MONEY         same-currency / same-scale amounts
PERCENT       alternative percentages
CARDINAL      same-role numbers
RELATION_CUE  compatible relation cues
PHRASE        locally fluent factual phrase rewrites
```

The domain does not need to be enumerated fully. In implementation, it can be a
sampler:

```text
sample_replacements(u, m)
```

### 3.4 Answer Preservation Score

Define:

```text
K(q, C', a) in [0, 1]
```

as the degree to which the reader preserves the original answer `a` when run on
modified context `C'`.

Possible implementations:

```text
binary:
  K = 1 if R(q, C') is semantically equivalent to a
  K = 0 otherwise

continuous judge:
  K = semantic-equivalence score between R(q, C') and a

logprob-based:
  K = probability/log-likelihood assigned to answer a
```

For the first implementation, use a binary semantic judge. Later, add a
continuous score if stable.

### 3.5 Expected Answer Preservation

For a set of editable units `S`, a revision assignment is:

```text
rho: S -> Omega(S)
```

The revised context is:

```text
C_rho = C[S := rho(S)]
```

Define expected preservation:

```text
Pres(S) = E_{rho ~ P_S} [ K(q, C_rho, a) ]
```

where `P_S` is a replacement distribution over valid edits. In practice:

```text
Pres(S) ~= (1/m) * sum_j K(q, C_{rho_j}, a)
```

using `m` sampled type-valid revisions.

This expectation matters because a span can have many valid replacements. Some
may change the answer, and some may not. We should measure the average
preservation drop under plausible type-valid replacements, not cherry-pick one
adversarial value.

### 3.6 RAG Answer Resilience

Given threshold `tau`, define:

```text
Res_tau(q, C, a)
  = min_{S subset U(C)} cost(S)
    subject to Pres(S) <= tau
```

where:

```text
cost(S) = sum_{u in S} cost(u)
```

Interpretation:

- low `Res_tau`: the answer is fragile;
- high `Res_tau`: the answer is resilient;
- infinite/no solution: the allowed evidence edits cannot change the answer.

### 3.7 Generalized Evidence Responsibility

A single unit's responsibility should not be binary. Each unit has many possible
replacement values.

For unit `u` and contingency set `Gamma`, define:

```text
Resp(u, Gamma)
  =
  K(q, C_Gamma, a)
  - E_{y ~ P_u} K(q, C_{Gamma, u := y}, a)
  ------------------------------------------------
  1 + cost(Gamma)
```

Global responsibility:

```text
Resp(u) = max_Gamma Resp(u, Gamma)
```

Intuition:

- If editing `u` alone changes the answer, responsibility is high.
- If `u` only matters after several other units are edited, responsibility is
  discounted.
- If no valid replacement of `u` changes preservation, responsibility is low.

This follows the spirit of generalized responsibility: variables are not binary,
and their contribution is measured over a replacement domain.

## 4. Attribution/Support Graph

Start from the graph style of the RAG attribution-graph paper:

```text
question/context tokens
-> intermediate activations / attribution nodes
-> generated answer tokens
```

Let:

```text
G_attr(q, C, a) = (V, E, w)
```

with:

```text
V_C  context token nodes
V_Q  question token nodes
V_H  intermediate attribution/activation nodes
V_A  answer token nodes
```

Edges carry attribution/support weights:

```text
w(e) >= 0
```

This graph is not our main contribution. It is an estimator of support flow.
The repository should allow several graph estimators:

```text
circuit_tracing         strong, open-weight only
gradient_x_input        cheaper open-weight baseline
attention_rollout       cheap but weak
span_occlusion          black-box but expensive
leave_one_out           black-box causal baseline
llm_support_judge       heuristic baseline
```

The contribution is:

```text
support graph -> editable unit projection -> minimum evidence cut
```

not the attribution estimator itself.

## 5. Projection From Tokens to Editable Units

Given token-level support flow, project it onto editable units:

```text
pi: context tokens -> editable units
```

For each unit:

```text
support(u) = total support flow from tokens(u) to answer tokens
```

Implementations:

```text
sum:
  support(u) = sum_{t in tokens(u)} flow(t -> answer)

length-normalized:
  support(u) = sum flow / sqrt(|tokens(u)|)

max:
  support(u) = max_{t in tokens(u)} flow(t -> answer)
```

Default:

```text
support(u) = sum flow / sqrt(|tokens(u)|)
```

to avoid always favoring longer units.

## 6. Editable Unit Proposal

The repository needs a deterministic unit proposer.

### Inputs

```text
raw context C
reader tokenizer
token offsets
NER model
regex value extractor
relation cue patterns
optional answer a
```

### Candidate sources

Generate candidate units from:

```text
single high-flow tokens
NER mentions
numeric/date/money regexes
proper noun chunks
relation cue patterns
answer/alias matches
short factual phrases for yes/no or relation-heavy cases
```

### Critical design choice

Do not only propose full NER spans. Include single-token candidates inside NER
mentions when local replacement can preserve a valid entity or phrase.

Example:

```text
New York -> New Jersey
```

can be represented as:

```text
u = "York"
y = "Jersey"
```

with a local validator confirming that `New Jersey` is a valid place phrase.

### Validation

A candidate edit `(u -> y)` is valid only if:

```text
1. replacement is type-compatible or locally phrase-compatible;
2. modified text remains fluent;
3. modified text is a factual/content edit, not punctuation/noise;
4. replacement is not an alias of the original answer;
5. prompt/question/instruction text is not modified;
6. local validator accepts the changed phrase/sentence.
```

### Output schema

```json
{
  "unit_id": "u17",
  "text": "York",
  "char_start": 120,
  "char_end": 124,
  "token_start": 31,
  "token_end": 32,
  "sentence_id": 2,
  "unit_type": "PLACE_COMPONENT",
  "sources": ["token", "inside_ner"],
  "edit_cost": 1.0,
  "replacement_policy": "place_component",
  "support": 0.73
}
```

## 7. Minimum Evidence Cut

The exact resilience objective is combinatorial:

```text
min cost(S) subject to Pres(S) <= tau
```

The graph estimator gives an approximate support-flow objective:

```text
min cost(S)
subject to remaining_support_flow(S) <= tau_graph
```

### Unit-level graph

Create a support graph over editable units:

```text
source -> editable units -> answer sink
```

or a richer graph if sentence/phrase/group structure is useful:

```text
source -> sentence groups -> editable units -> answer sink
```

For node cut, use node splitting:

```text
u_in -> u_out
```

with edge capacity:

```text
capacity(u_in, u_out) = cost(u)
```

Edges carrying support flow connect through `u_in` and `u_out`. A minimum cut
that cuts `u_in -> u_out` selects unit `u` for revision.

### Practical variants

Implement both:

```text
minimum cut:
  find smallest-cost set predicted to cut answer support

budgeted maximum support removal:
  given budget b, choose units maximizing removed support
```

The budgeted variant is often easier to debug first.

## 8. Counterfactual Revision

For selected set `S_hat`, generate `m` valid revision assignments:

```text
rho_1, ..., rho_m
```

Each assignment maps:

```text
u -> y, y in Omega(u)
```

Then produce revised contexts:

```text
C_rho_j = C[S_hat := rho_j(S_hat)]
```

Rerun:

```text
a'_j = R(q, C_rho_j)
```

Score:

```text
K_j = K(q, C_rho_j, a)
```

Estimate:

```text
Pres(S_hat) = mean_j K_j
```

## 9. Metrics

Primary:

```text
Res_tau                 minimum revision cost
Pres(S_hat)             answer preservation after predicted cut
answer_preservation_drop
cost_per_drop
success_rate            fraction Pres <= tau
```

Secondary:

```text
flip                    answer differs from clean answer
strict_override         answer changes and adopts fake value
compliance              fake value appears in output
answer_removed          original evidence value removed from served context
edit_validity           replacement passed validation
```

For loop/agentic pipelines:

```text
rounds_to_answer
non_termination
retrieval_trace_overlap
```

For method comparison:

```text
edit_count_to_flip
support_removed
cut_size
runtime
reader_calls
```

## 10. Baselines

Compare against:

```text
random editable unit
answer-string unit
highest attribution unit
LLM evidence judge selector
leave-one-out selector
full NER span selector
oracle/gold selector, evaluation only
```

Key hypothesis:

```text
minimum evidence cut finds smaller and more effective revision sets than
answer-string, random, or top-attribution single-unit baselines.
```

## 11. Experiments

### P0 gate

Run before writing the full paper:

```text
1 reader
2 datasets
100-200 examples each
naive RAG first
```

Compare:

```text
random
answer-string
top-attribution unit
LLM evidence judge
minimum evidence cut
leave-one-out oracle-ish selector
```

Report:

```text
edit_count_to_flip
Pres(S)
answer_preservation_drop
valid_edit_rate
runtime
case cards
```

P0 success condition:

```text
minimum evidence cut produces lower-cost and/or higher-drop revisions than
answer-string and top-attribution baselines.
```

### Main characterization

After P0:

```text
datasets x readers x RAG pipelines
fact/unit type x question type
corroboration vs cut size
single-unit vs multi-unit revisions
black-box vs open-weight support graph estimators
```

### Must-have table

Fact/unit type by question type:

```text
rows:    edited unit type
columns: question type
cells:   Res_tau / preservation drop / n
```

This table should show that resilience is workload-dependent.

## 12. Case Cards

Each case card should show:

```text
question
clean context snippet
clean answer
selected units
attribution/support scores
minimum cut cost
counterfactual edits
poisoned context snippet
new answer
K / flip / compliance
failure mode
```

Important case buckets:

```text
single-token fragile answer
multi-unit corroborated answer
answer-string baseline fails but min-cut succeeds
top-attribution fails because corroboration remains
numeric/comparison derived answer
yes/no label answer
parametric-memory-resistant answer
invalid edit filtered by validator
```

## 13. Suggested Repository Layout

```text
rag_resilience/
  data/
    loaders/
    schemas.py

  rag/
    retriever.py
    readers.py
    pipelines.py

  attribution/
    base.py
    attention_rollout.py
    gradients.py
    occlusion.py
    leave_one_out.py
    circuit_tracing.py

  units/
    propose.py
    validate.py
    replacements.py
    costs.py
    schema.py

  graph/
    project.py
    min_cut.py
    budgeted.py
    support_graph.py

  revision/
    apply.py
    score.py
    audit.py

  experiments/
    run_cell.py
    run_p0.py
    run_main_grid.py
    aggregate.py

  analysis/
    tables.py
    figures.py
    case_cards.py

  configs/
    readers.yaml
    datasets.yaml
    replacement_policies.yaml

  tests/
    test_unit_alignment.py
    test_replacement_validity.py
    test_min_cut.py
    test_revision_apply.py
```

## 14. Audit Log Schema

Every example should emit JSONL with:

```json
{
  "id": "...",
  "dataset": "...",
  "question": "...",
  "clean_answer": "...",
  "context_ids": ["..."],
  "units": [
    {
      "unit_id": "u17",
      "text": "York",
      "type": "PLACE_COMPONENT",
      "tokens": [31, 32],
      "support": 0.73,
      "cost": 1.0,
      "valid_replacements": ["Jersey", "Orleans"]
    }
  ],
  "selected_cut": ["u17"],
  "cut_cost": 1.0,
  "revisions": [
    {
      "assignment": {"u17": "Jersey"},
      "poisoned_answer": "...",
      "K": 0.0,
      "flip": true,
      "compliance": true,
      "valid_edit": true
    }
  ],
  "preservation": 0.0,
  "success": true
}
```

## 15. Claims To Avoid

Do not claim:

```text
we invented attribution graphs
min-cut guarantees LLM answer change
token attribution is true causality
closed-book correctness proves parametric knowledge
rule mining is the contribution
all RAG failures are evidence failures
```

Safe claim:

```text
We estimate a minimum evidence revision set from an attribution-derived support
graph and validate the prediction by counterfactually revising the retrieved
context and rerunning the RAG reader.
```

## 16. First Implementation Milestone

Build P0 in this order:

1. Basic RAG runner that stores clean context, answer, and token offsets.
2. Editable unit proposer with:
   - single-token units;
   - NER/value units;
   - relation cue patterns;
   - local edit validator.
3. One attribution estimator:
   - start with span/token occlusion or attention rollout;
   - add circuit tracing only after the pipeline works.
4. Unit projection and support scoring.
5. Budgeted support-removal selector.
6. Minimum cut selector.
7. Type-valid replacement sampler.
8. Rerun + answer preservation judge.
9. P0 comparison table and case cards.

If P0 does not beat answer-string and top-attribution baselines, keep the graph
selector as an ablation and return to the benchmark framing. If P0 works, the
new repository should be written as a method paper around RAG answer resilience.

