# Geometric Contribution-Flow Method: Normative Specification

This document is the normative theory specification for the current method.
It supersedes the earlier hazard, hypergraph, witness, tuple-lineage,
densest-subgraph, and layer-copy group-rounding attempts. Those routes may be
used only as historical baselines and are not part of the proposed algorithm.

## 1. True Token Resilience

For a query `q`, its top-k retrieved chunks `C`, a frozen reader `R`, and its
clean answer `a0 = R(q, C)`, fix one legal non-deleting replacement `rho_i` for
every editable chunk-token occurrence `i`. Query tokens, answer tokens,
retrieval results, and model parameters are immutable.

For a token set `S`, let `C[rho_S]` apply the fixed replacements in `S`. The
true fixed-replacement resilience is

```text
r_R(q) = min |S|
         s.t. R(q, C[rho_S]) is not equivalent to a0.
```

Equivalently, with `x_i in {0, 1}` and a deterministic flip predicate `f_q`,

```text
min sum_i x_i
s.t. f_q(x) = 1.
```

This is a black-box 0/1 optimization problem. Without assumptions on `f_q`, a
sub-exhaustive algorithm has no nontrivial approximation ratio to `r_R(q)`.
The reader can be non-monotone, and its constraint is not an explicit ILP
constraint. Every successful tested set is only a verified upper bound on the
true resilience.

## 2. Contribution Graph

Generate the clean answer once with the frozen vLLM reader, then trace
positive, answer-specific information flow through the same checkpoint with
Hugging Face SDPA execution. The second load is used only to expose gradients
and intermediate activations; it does not generate another reader answer.
Full attention weights are reconstructed exactly from each layer's Q/K states
one layer at a time after backward, rather than retained for every layer during
the forward pass. For
a source token state `u` and a receiver state `v`, the local signed
contribution is

```text
r(u, v) = <dJ_q / do_v, m(u, v)>,
```

where `J_q` is the mean clean-answer target logit and `m(u, v)` is the actual
attention/residual/MLP write from `u` to `v`. Positive contributions are routed
backward from the answer target and then projected to chunk-token occurrences.
No clean-answer string matching is used.

If the clean answer has `m` scored tokens, `J_q` is their mean logit and the
answer sink injects mass `1/m` at each corresponding predictor position. This
objective seed is defined directly by `J_q`; it is not estimated with
`gradient x final residual`, which is not a valid mass decomposition after the
model's final normalization.

Sparse backward tracing always preserves the strongest positive word-bearing
context source at the input layer in addition to the ordinary top-k beam. A
row is valid only when positive flow connects at least one such context source
to the answer sink. Empty graphs, missing answer terminals, and graphs without
a context-to-answer path are explicit failures, never zero-flow successes.

The retrieved chunks are rendered in full. No per-chunk or prompt-length
truncation is part of the method. If the exact rendered sequence exceeds the
checkpoint's real context window, construction fails explicitly rather than
clipping the evidence.

The projected network has:

- an artificial source `s`;
- one purchasable gate for every legally replaceable chunk token;
- non-negative contribution edges between projected token states;
- an answer target `t`.

The query conditions all contribution values but is not editable. A larger
edge capacity means more clean-answer-supporting contribution is routed across
that edge.

For token set `S`, define residual contribution flow

```text
Phi_q(S) = MaxFlow(G_q - S, s, t).
```

Closing a token gate removes all contribution routes using that token. Only
complete source-to-answer flow contributes to `Phi_q(S)`.

## 3. Graph Threshold Problem

For an internal flow threshold `B > 0`, define

```text
k_q*(B) = min |S|  s.t. Phi_q(S) <= B.
```

Positive-threshold flow interdiction is NP-hard. In contrast, for any fixed
token price `lambda >= 0`, the supported problem

```text
S_lambda = argmin_S lambda |S| + Phi_q(S)
```

is solved exactly by one token-gated weighted `s-t` min-cut: give each
purchasable token gate capacity `lambda` and retain every contribution edge at
its flow capacity. Cutting a token pays once; cutting remaining contribution
edges pays their residual flow.

## 4. Geometric Cardinality Scales

Let `n` be the number of purchasable token gates. Fix `eta > 0` and
`gamma > 0`. Construct scales

```text
K_0 = 1,
K_r = (1 + gamma)^r,
```

capping the final scale at `n`. For every scale `K_r`, set

```text
lambda_r = eta B / K_r
```

and solve

```text
S_r = argmin_S lambda_r |S| + Phi_q(S)
```

by one exact min-cut. Return the smallest generated set satisfying

```text
Phi_q(S_r) <= (1 + eta) B.
```

The number of min-cut calls is at most

```text
1 + ceil(log_(1 + gamma) n).
```

The dependency is one-way:

```text
(n, gamma) -> K_r,
(B, eta, K_r) -> lambda_r,
(G_q, lambda_r) -> S_r.
```

`K_r` is a predetermined numerical scale, not an estimated optimum and not a
cardinality constraint.

## 5. Bicriteria Theorem

Let `S*` be an optimum of the strict graph problem at threshold `B`, and let
`k* = |S*|`. Suppose `k* <= n`. There is a geometric scale `K_r` such that

```text
k* <= K_r < (1 + gamma) k*.
```

For `lambda_r = eta B / K_r`, exact min-cut optimality gives

```text
lambda_r |S_r| + Phi_q(S_r)
    <= lambda_r k* + Phi_q(S*)
    <= eta B + B.
```

Both terms on the left are non-negative. Therefore

```text
Phi_q(S_r) <= (1 + eta) B
```

Moreover, retaining the optimum's priced token term gives

```text
|S_r|
    <= k* + B / lambda_r
    =  k* + K_r / eta
    <  (1 + (1 + gamma)/eta) k*.
```

Hence the algorithm is a

```text
(1 + (1 + gamma)/eta, 1 + eta)
```

bicriteria approximation to the graph threshold problem.

With `gamma = 1` and `eta = 1`, the guarantee is

```text
|S| < 3 |S*|,
Phi_q(S) <= 2B.
```

It is a `(3, 2)` bicriteria guarantee, not a 2-approximation. Enumerating
every integer cardinality scale removes the geometric factor and yields
`(2, 2)` at `O(n)` min-cuts.

If the desired final residual threshold is `T`, run the theorem with

```text
B = T / (1 + eta).
```

For `eta = 1`, use `B = T/2`; then every theorem-covered result satisfies
`Phi_q(S) <= T`.

### Scope of the theorem

The theorem is deterministic and assumption-light, but it concerns only the
explicit contribution-flow surrogate. It does not imply that the reader flips,
and it does not compare `|S|` with the true black-box optimum `r_R(q)`.

## 6. Statistical Link to Reader Flips

The graph theorem can be connected to the real reader through held-out
calibration. This is a population guarantee, not a per-query causal theorem.

Fix every component except the normalized final threshold `beta`:

```text
T_q(beta) = beta Phi_q(empty).
```

Let `A_beta(q)` be the deterministic complete policy that runs the geometric
solver with internal threshold `T_q(beta)/(1 + eta)`, applies the frozen
replacement registry, and either returns an edited set or abstains. Define

```text
Z_beta(q) = 1{A_beta(q) returns a candidate},
Y_beta(q) = 1{the returned edit flips the reader answer}.
```

For the query population `Q`, define coverage and conditional failure risk

```text
kappa(beta) = P[Z_beta(Q) = 1],
risk(beta)  = P[Y_beta(Q) = 0 | Z_beta(Q) = 1].
```

Because replacements and greedy decoding are fixed, `Y_beta(q)` is 0/1 for a
fixed query. Probability is over new queries drawn from the deployment
population. If replacements are randomized, their randomness must be included
in both calibration and deployment in the same way.

### What the law of large numbers gives

For a fixed `beta` and i.i.d. calibration queries, let

```text
N_beta = sum_i Z_beta(q_i),
F_beta = sum_i Z_beta(q_i) (1 - Y_beta(q_i)).
```

When `kappa(beta) > 0`, the strong law of large numbers gives

```text
F_beta / N_beta -> risk(beta)
```

almost surely as the calibration size grows. This justifies estimating graph
faithfulness from interventions, but it is asymptotic and does not by itself
justify choosing `beta` on the same finite sample.

### Finite-sample selective guarantee

Predeclare a finite threshold grid `BETA` of size `M`. For each threshold,
compute a simultaneous one-sided upper confidence bound `U_beta` for the
binomial failure probability using confidence level `1 - delta/M`. An exact
Clopper-Pearson bound is preferred; a simpler Hoeffding bound is

```text
U_beta = F_beta / N_beta
         + sqrt(log(M/delta) / (2 N_beta)).
```

Select the highest-coverage threshold satisfying

```text
U_beta <= epsilon
```

and a predeclared minimum accepted count. The union bound makes all `M` risk
bounds valid simultaneously. Consequently, even though the threshold is
selected from calibration data, with probability at least `1 - delta` over
the calibration sample,

```text
P[Y_beta(Q) = 1 | Z_beta(Q) = 1] >= 1 - epsilon
```

for a fresh query from the same population.

This is a PAC-style selective flip guarantee. The method may abstain, so
coverage must always be reported. A simultaneous lower confidence bound
`kappa_L` on coverage gives

```text
P[Z_beta(Q) = 1 and Y_beta(Q) = 1]
    >= kappa_L (1 - epsilon).
```

For an additive risk-estimation tolerance `xi`, Hoeffding requires

```text
N_beta >= log(M/delta) / (2 xi^2).
```

Exact binomial bounds are substantially tighter for the sample sizes used in
the experiments.

## 7. Combined Guarantee

Under the graph conditions and the calibration conditions, the returned policy
has two separate guarantees:

1. For each theorem-covered query,

   ```text
   |S| < (1 + (1 + gamma)/eta) k_q*(B),
   Phi_q(S) <= (1 + eta) B.
   ```

2. With confidence at least `1 - delta` over calibration data, a fresh accepted
   query flips with conditional probability at least `1 - epsilon`.

These statements must not be collapsed into an approximation ratio to the
unknown black-box optimum. The optimization theorem controls graph cost; the
statistical theorem controls population-level reader fidelity.

## 8. Calibration Assumptions

The statistical guarantee requires:

1. Calibration and deployment queries are exchangeable, or i.i.d. from the
   same population.
2. Retrieval, reader, prompt, decoding, graph construction, `gamma`, `eta`,
   replacement policy, and answer verifier are frozen before calibration.
3. The finite family of threshold policies is predeclared. If other
   hyperparameters are selected on calibration data, they must be included in
   the union-bound family or tuned on a separate development split.
4. Every accepted calibration query is actually replaced and rerun through
   the same reader used at deployment.
5. There are enough accepted calibration examples to certify the requested
   risk. Distribution shift invalidates the guarantee.

No assumption that attention is causal, that residual flow is a true flip
probability, or that the reader flip predicate is monotone is needed for the
held-out statistical guarantee.

## 9. Required Calibration Experiment

Use three disjoint splits:

1. Development: choose graph construction, `gamma`, `eta`, registry policy,
   and threshold grid.
2. Calibration: run real interventions for every predeclared threshold policy,
   compute simultaneous one-sided risk and coverage bounds, and select `beta`.
3. Test: freeze `beta` and report conditional flip rate, coverage, overall flip
   rate, edit count, and confidence intervals without retuning.

For multiple datasets, either calibrate each dataset separately or predeclare a
pooled mixture distribution. A pooled guarantee does not imply a guarantee for
every constituent dataset; dataset-conditional bounds must be reported when
that distinction matters.
