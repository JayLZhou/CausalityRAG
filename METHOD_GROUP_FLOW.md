# Grouped Contribution-Flow Method

This document is the normative method specification. Historical ratio,
hypergraph, contracted-token, and closed-background variants are ablations.

## 1. Exact Reader Problem

For a question `q`, top-k retrieved chunks `C`, frozen reader `R`, and clean
answer `a0 = R(q, C)`, let `V(a, a0)` be the frozen answer-equivalence
verifier. The intervention domain contains only surface-word tokens in `C`.
Query tokens, generated answer tokens, model parameters, and retrieval results
are immutable.

Before selection, an answer-blind registry fixes at most one legal replacement
`rho_i` for each editable token `i`. The registry uses contextual entity type,
POS, tag, and morphology constraints. Every selector receives the same
editable domain and the same replacements, but no selector sees replacement
text while scoring tokens.

For `S` in the editable domain, let `C[rho_S]` apply all replacements in `S`
simultaneously. The canonical-replacement resilience is

```text
r_R(q) = min |S|
         s.t. V(R(q, C[rho_S]), a0) = 0.
```

A successful tested set is therefore a verified upper bound on `r_R(q)`.

Equivalently, let `x_i in {0, 1}` indicate whether token `i` is replaced and
let `f_R(x)` be one exactly when the frozen reader no longer preserves `a0`.
The true optimization problem is the black-box 0/1 program

```text
min sum_i x_i
s.t. f_R(x) = 1.
```

The difficulty is not integrality by itself: `f_R` has no explicit linear
constraints and is available only through a full reader execution. The graph
formulation below is therefore an explicit surrogate, not a claimed exact
linearization of the reader.

### Why a surrogate is necessary

Without structural assumptions, the reader flip predicate is an arbitrary,
possibly non-monotone set function. Any algorithm that tests fewer than
`2^n - 1` subsets can be made to observe the same answers from two black-box
readers with different optima: choose an untested, unreturned set `T`, and let
one reader never flip while the other flips only on `T`. Adding edits can also
restore an LLM answer, so monotonicity cannot be assumed without measurement.

Consequently, no graph method honestly has an unconditional approximation
ratio to `r_R`. The method below has a theorem for an explicit graph surrogate
and always reports reader verification separately.

## 2. Contribution DAG

Run the clean reader once with eager attention and trace positive local
target-logit contributions through the original transformer:

```text
context input copies -> residual/attention/MLP writes -> clean answer target
```

Backward beam search proposes complete paths. Pruned or non-positive mass is
absorbed rather than reintroduced through synthetic background edges. The
result is a path-preserving absorbing DAG `D = (N, E, c)` with non-negative
capacities.

The supersource connects context-token input roots. Query-token copies may be
intermediate conditioning nodes, but they are never editable. The sink is the
clean answer-logit target. The clean answer defines the output being explained;
it is never matched lexically against chunk tokens.

Every editable layer-copy gate `g` has an input-token label `ell(g)`. Editing
token `i` removes every editable gate with `ell(g) = i`. Keeping copies at their
original layers is essential: contracting equal token positions across layers
creates paths that do not exist in the transformer.

For token set `S`, define residual contribution flow

```text
Phi_D(S) = maxflow after removing every gate labelled by a token in S.
```

## 3. Normalized Hybrid Flow

The path graph captures multi-stage bottlenecks but can miss support discarded
by sparse path extraction. ARC-JSD provides a complementary unary intervention
score `u_i >= 0`. Normalize both channels to unit total mass and use equal
weight by default:

```text
Phi_alpha(S)
  = alpha * Phi_D(S) / Phi_D(empty)
  + (1 - alpha) * sum_{i notin S} u_i / sum_i u_i,

alpha = 1/2.
```

This is represented as one augmented flow network: each unary score is an
independent source-to-sink path whose gate carries the same token label as its
layer copies. Thus the optimizer and theorem below are unchanged. `alpha` is
fixed on development data before test evaluation; alpha sweeps are an
ablation, not per-query tuning.

## 4. Grouped Flow Resilience

For a remaining-support threshold `B`, define

```text
r_G(B) = min |S|  s.t. Phi_alpha(S) <= B.
```

Without shared labels, this threshold objective is the Maximum Flow Blocker
Problem studied by Bentoumi et al. (EJOR 2025), whose published methods are
exact ILP and branch-and-cut formulations. With shared token labels, our
problem combines flow blocker with Minimum Label `s-t` Cut. Even the `B = 0`
case contains minimum label cut: subdivide each labelled edge by a vertex with
the same label. The shared cost of copies with one label is what prevents one
ordinary min-cut from solving the problem exactly.

The optimization objective is therefore not claimed as new by itself. The
paper contribution is its grouped, layer-preserving realization for RAG token
resilience; the bounded-rank bicriteria min-cut algorithm; and the empirical
connection to verified reader interventions.

## 5. K-Guess Copy Relaxation

Let `r_i` be the number of active gates carrying token label `i`, and let
`r = max_i r_i`. Give every copy of label `i` relaxation cost `1 / r_i`.
For every cardinality guess `k = 1, ..., K`:

1. Set `mu_k = k / (eta B)`.
2. Solve `min_Y c(Y) + mu_k Phi_copy(Y)` exactly by one weighted `s-t` min-cut.
3. Round copy cut `Y` to token set `S = {ell(g): g in Y}`.
4. Remove all copies of labels in `S` and measure `Phi_alpha(S)`.

Return the smallest candidate satisfying
`Phi_alpha(S) <= (1 + eta) B`. Then delete labels in increasing restoration-flow
order whenever feasibility is preserved. This threshold pruning returns an
inclusion-minimal set in that order and cannot weaken either bound below. The
core relaxation uses exactly `K` weighted min-cuts; pruning adds at most two
residual max-flow evaluations per label in its input set. Invalid registry
tokens remain in the DAG with an uncuttable gate, so excluding them does not
delete their information-flow paths.

### Bicriteria theorem

The guarantee needs only graph-side conditions:

1. all finite contribution capacities are non-negative;
2. every editable token label occurs on at most `r` active gates;
3. the strict graph optimum has size `k* <= K` and residual flow at most `B`;
4. `B > 0` and `eta > 0`.

It does not assume that the reader flip predicate is monotone or that the
contribution graph is a perfect reader model. Under these conditions, the
candidate generated for guess `k*` satisfies

```text
|S| <= r (1 + 1/eta) k*,
Phi_alpha(S) <= (1 + eta) B.
```

Proof sketch. Removing all copies of the optimal `k*` labels has copy cost
exactly `k*` and residual at most `B`. Optimality of the fixed-Lagrangian
min-cut gives

```text
c(Y) + mu Phi_copy(Y) <= k* + mu B.
```

With `mu = k* / (eta B)`, each non-negative term is bounded separately:

```text
c(Y) <= (1 + 1/eta) k*,
Phi_copy(Y) <= (1 + eta) B.
```

Every selected label contributes at least `1/r` to `c(Y)`, so rounding gives
`|S| <= r c(Y)`. Removing all copies of each rounded label can only decrease
flow, which proves the claim. Feasibility-preserving threshold pruning only
removes labels, so the same theorem holds for the returned pruned set.

The rank cap is a modeling/accuracy tradeoff, not hidden pruning: all DAG nodes
remain, while only the highest-throughput internal copies are editable gates.
Report the realized `r`, excluded-copy mass, and cap sweep.

### Exact optimization oracle

The graph surrogate has a compact exact MILP used only for evaluation. Let
`x_i` select token label `i`, let binary `z_v` indicate the source side of an
`s-t` cut, and let `y_e` indicate a finite contribution edge crossing that cut.
Fix context roots to the source side and the answer target to the sink side.
For every layer-copy gate `g`, impose

```text
z_g_in - z_g_out <= x_ell(g).
```

For every finite edge `e = (u, v)`, impose

```text
y_e >= z_u_out - z_v_in,
sum_e c_e y_e <= B.
```

Minimizing `sum_i x_i` is exactly `r_G(B)` by max-flow/min-cut duality. The
formulation is NP-hard because `x_i` is shared by every copy of label `i`; it
is therefore an oracle/solver baseline, not the proposed branch-free method.

## 6. Reader Verification

The threshold solution is the primary native output. Fixed budgets use a
graph-only reverse-delete chain for evaluation curves; that repair heuristic
does not inherit the theorem above.

For every returned set:

1. Fetch its replacements from the shared strict registry.
2. Modify only the selected chunk-token offsets, with no deletion.
3. Rerun the same frozen reader and decoding configuration.
4. Record normalized exact match, token F1, and semantic-equivalence audit.

The graph guarantee concerns `r_G(B)`. A verified flip bounds `r_R(q)` from
above. Relating `r_G` to `r_R` requires an empirically tested graph-faithfulness
assumption and is never presented as an unconditional theorem.

## 7. Required Evidence

The main paper must include:

- fixed-budget and native-threshold curves against JSD unary, MIRAGE,
  attention, random-valid, lexical relevance, and YVETTE;
- strict shared-registry replacements and at least three registry seeds;
- an exact grouped-flow MILP audit and a restricted exhaustive reader oracle
  for small token pools;
- cap/rank, absorbing-vs-closed, graph-only, unary-only, and alpha ablations;
- local-HF eager re-verification of every final answer;
- latency, forward/backward passes, editor calls, reader calls, and peak memory;
- at least two QA datasets and two reader families before a venue claim.

## 8. Optimization References

- Bentoumi et al., [Integer linear programming formulations for the maximum
  flow blocker problem](https://doi.org/10.1016/j.ejor.2025.02.013), EJOR 2025.
- Zhang and Tang, [Minimum Label s-t Cut has Large Integrality
  Gaps](https://arxiv.org/abs/1908.11491), Information and Computation 2020.
- Zhang, [Approximating the Weighted Minimum Label s-t Cut
  Problem](https://arxiv.org/abs/2011.06204), Theoretical Computer Science 2022.
- Chestnut and Zenklusen, [Hardness and Approximation for Network Flow
  Interdiction](https://arxiv.org/abs/1511.02486), Networks 2017.
