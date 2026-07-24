# Exploratory global-beta sweep on 2WikiMultiHopQA

Date: 2026-07-24

Server: `yujia-server3`

Artifact directory:
`/data1/yujia/CausalityRAG/exp/2wiki/beta_sweep_v1`

## Status and interpretation

This is an exploratory sweep over the same 1,000 queries used by
`all1000-final-v3`. It tests whether the previously fixed global
`beta=0.1` was unnecessarily strict. It is not an independent validation/test
split and must not be presented as a calibrated generalization guarantee.

Every query still uses the same dataset-level beta. Query-specific absolute
flow thresholds are

```text
B_q = beta * Phi_q(empty)
```

and, with `eta=1`, the accepted normalized remaining-flow threshold is

```text
tau = (1 + eta) * beta = 2 * beta.
```

The contribution graph, frozen clean vLLM answer, retrieved top-5 chunks,
solver, and replacement contract are shared across the sweep. A cross-beta
registry closure produced 3,821 candidate tokens: 3,386 valid replacements
and 435 invalid replacements, with zero final candidate misses. Final solves
use the same frozen `allow-only` replacement domain.

## Full sweep

Each beta was evaluated once over all 1,000 queries with concurrent vLLM
reader calls. The 275 clean-exact rows below are filtered from that same
all-query execution.

| beta | tau | Clean candidates | Clean flips / 275 | Conditional flip | Mean edits | All candidates | All flips / 1,000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.10 | 197 | 145 (52.73%) | 73.60% | 2.812 | 808 | 596 (59.60%) |
| 0.075 | 0.15 | 207 | 151 (54.91%) | 72.95% | 2.531 | 831 | 602 (60.20%) |
| 0.100 | 0.20 | 212 | 151 (54.91%) | 71.23% | 2.250 | 847 | 603 (60.30%) |
| 0.125 | 0.25 | 219 | 154 (56.00%) | 70.32% | 2.196 | 861 | 606 (60.60%) |
| 0.150 | 0.30 | 226 | 159 (57.82%) | 70.35% | 1.991 | 887 | 615 (61.50%) |
| 0.200 | 0.40 | 237 | 161 (58.55%) | 67.93% | 1.785 | 913 | 623 (62.30%) |
| 0.225 | 0.45 | 242 | 167 (60.73%) | 69.01% | 1.715 | 925 | 629 (62.90%) |
| **0.250** | **0.50** | **246** | **170 (61.82%)** | **69.11%** | **1.642** | **935** | **634 (63.40%)** |
| 0.275 | 0.55 | 248 | 168 (61.09%) | 67.74% | 1.540 | 939 | 629 (62.90%) |
| 0.300 | 0.60 | 249 | 164 (59.64%) | 65.86% | 1.450 | 943 | 621 (62.10%) |
| 0.350 | 0.70 | 252 | 163 (59.27%) | 64.68% | 1.246 | 948 | 613 (61.30%) |
| 0.400 | 0.80 | 254 | 162 (58.91%) | 63.78% | 1.228 | 953 | 618 (61.80%) |

The within-sweep optimum for both scopes is `beta=0.25`, or `tau=0.50`.
Relative to the within-sweep `beta=0.1` result:

- clean-exact coverage rises from 212/275 to 246/275;
- clean-exact end-to-end flips rise from 151 to 170, +6.91 points;
- all-query flips rise from 603 to 634, +3.10 points;
- mean clean-exact edits fall from 2.250 to 1.642, a 27.0% reduction;
- conditional clean-exact flip rate falls from 71.23% to 69.11%.

Thus the gain comes from a better coverage/strength trade-off. The original
threshold rejected too many useful small interventions. The graph guarantee
also becomes weaker: `beta=0.25, eta=1` certifies remaining graph flow at most
`0.50 * Phi_q(empty)`, rather than the former `0.20 * Phi_q(empty)`.

On paired query outcomes, `beta=0.25` has 28 clean-exact wins unique to it
versus nine unique to `beta=0.1` (two-sided exact McNemar/binomial
`p=0.00256`). Across all queries the discordance is 67 versus 36
(`p=0.00293`). These are exploratory same-sample values and do not correct for
selecting beta after inspecting the grid.

## Submodular pilot simulation

`exp/analyze_beta_sweep.py` constructs the label-free flow profile

```text
[candidate exists, selected/editable units, remaining flow] per beta
```

and applies RBF facility-location greedy selection. Reader outcomes are
consulted only after pilot selection. Exact recovery of the full-data optimum
is not reliable for small pilots:

| Pilot queries | Direct centers: clean beta | Direct centers: all beta | Stratified-random exact-match rate, clean | Stratified-random exact-match rate, all |
|---:|---:|---:|---:|---:|
| 10 | 0.35 | 0.35 | 23.8% | 10.0% |
| 25 | 0.40 | 0.35 | 23.4% | 11.2% |
| 50 | 0.25 | 0.35 | 34.4% | 17.4% |
| 100 | 0.25 | 0.35 | 59.0% | 29.0% |

Direct facility centers select a near-optimal but not optimal `beta=0.35` in
the all-query scope, losing 2.10 end-to-end points relative to `beta=0.25`.
Using facility centers only as strata and then sampling randomly improves the
100-query clean-exact simulation, but it remains too unstable to support a
formal claim. A 1% pilot of ten queries is clearly insufficient.

## Decision

Global beta calibration is worth retaining as an experimental component:
the sweep found a materially better operating point with fewer edits.

The current submodular-center pilot should not be promoted to the final method.
For a defensible calibrated result:

1. obtain an independent 2Wiki validation pool;
2. propose beta on a pilot or stratified-random calibration subset;
3. freeze the selected beta before a separate random audit;
4. report a one-sided binomial/Clopper-Pearson lower bound;
5. run the frozen beta on an untouched test set.

The existing all-1000 formal result remains unchanged until that independent
calibration/test separation exists.
