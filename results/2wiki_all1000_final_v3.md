# 2WikiMultiHopQA all-1000 final contribution-flow run

Run date: 2026-07-24  
Server: `yujia-server3`, GPU 2 (A100)  
Run directory: `/data1/yujia/CausalityRAG/runs/2wiki/all1000-final-v3`  
Repository branch: `main`; the exact clean commit is recorded in `manifest.json`

## Frozen setup

- Input:
  `/data1/yujia/RAGData/2wiki-exp/results/retrieval_2wiki_top5_all1000.jsonl`
- Queries: 1,000
- Retrieved chunks per query: 5
- Reader: vLLM serving `Qwen2.5-7B-Instruct` as `qwen2.5-7b`
- Reader decoding: greedy, 16 concurrent workers
- Contribution graph: HF SDPA direct-activation target-logit attribution
- Context/prompt truncation: none; only the model context window applies
- Solver: `geometric-k-guessing`
- Projection: `layer-copy-token`
- Capacity mode: `raw`
- `beta=0.1`, `eta=1`, `gamma=1`
- Final selection rule: remaining contribution-flow fraction `<=0.2`
- Edit budget: none
- Replacement contract: answer-blind, non-deleting, one-token, strict
  contextual POS/tag/morphology
- Evaluated selector: contribution-flow only. No cardinality-matched,
  unary, ARC-JSD, or other diagnostic selector was generated or evaluated.

## Stage checks

| Stage | Result |
|---|---:|
| Frozen context rows | 1,000 |
| Context token units | 464,402 |
| Nonempty clean reader targets | 999 |
| Reader abstentions | 1 |
| Clean exact-match queries | 275 |
| Valid contribution graphs | 999 |
| Graph failures | 0 |
| Final registry candidate tokens | 2,993 |
| Valid strict replacements | 2,663 |
| Invalid/uneditable replacements | 330 |
| Registry fixed-point misses | 0 |
| Final flow rows with an editable network | 972 |
| Rows with no registry-editable units | 27 |
| Server unit tests | 89 passed |

The final registry reached its fixed point after four numbered registry
iterations. The final `allow-only` solve uses only tokens with a frozen valid
replacement.

## Final reader metrics

An exact-normalized answer change relative to the frozen clean vLLM answer is
counted as a flip.

### Threshold-constrained primary metric

| Metric | Clean-exact scope | All-query scope |
|---|---:|---:|
| Denominator | 275 | 1,000 |
| Clean-correct / clean-incorrect | 275 / 0 | 275 / 725 |
| Queries with a flow candidate | 212 | 847 |
| Candidate coverage | 77.09% | 84.70% |
| Valid evaluated candidates | 212 | 847 |
| Replacement failures | 0 | 0 |
| Flips | 152 | 603 |
| Conditional flip rate | 71.70% | 71.19% |
| Overall flip rate | 55.27% | 60.30% |
| Mean selected tokens | 2.250 | 2.259 |

`Conditional flip rate = flips / valid evaluated candidates`.

`Overall flip rate = flips / every query in the stated scope`; queries without
a candidate therefore count as non-flips.

The clean-exact scope is the formal attack-success report because every clean
answer in its denominator matches gold. The all-query scope is an
answer-change diagnostic, not an attack-success rate, because 725 clean
answers in that scope do not exactly match gold.

### Actual interventions for above-threshold candidates

The threshold-constrained metric does not send a reader request when no
candidate has remaining flow at most 0.2. To test the available interventions
rather than silently assigning all of them a failure, a second evaluation uses
the normal threshold candidate when present and otherwise edits the nonempty
contribution-flow candidate with the lowest remaining-flow fraction.

| Metric | Clean-exact scope | All-query scope |
|---|---:|---:|
| Denominator | 275 | 1,000 |
| Within-threshold candidates | 212 | 847 |
| Above-threshold candidates actually tested | 49 | 123 |
| Queries actually edited and evaluated | 261 | 970 |
| Queries with no executable candidate | 14 | 30 |
| Flips from within-threshold candidates | 152 | 603 |
| Flips from above-threshold candidates | 29 | 73 |
| Total flips | 181 | 676 |
| Flip rate among edited queries | 69.35% | 69.69% |
| End-to-end flip rate over the full scope | **65.82%** | **67.60%** |

The above-threshold candidates achieved 29/49 flips (59.18%) in the
clean-exact scope and 73/123 (59.35%) over all queries. These are real edited
vLLM reader calls, but they do not satisfy the stated flow `<=0.2` constraint
and therefore remain a separately labeled fallback result.

## No-candidate breakdown

| Reason | Clean-exact scope | All-query scope |
|---|---:|---:|
| Minimum available candidate remained above flow 0.2 | 49 | 123 |
| No registry-editable units | 13 | 27 |
| Editable network but no nonempty supported candidate | 1 | 2 |
| Empty clean reader target | 0 | 1 |
| Total without a selected candidate | 63 | 153 |

In the all-available evaluation, the 49/123 above-threshold rows in this table
are actually edited and tested. Only 14 clean-exact queries and 30 all-scope
queries remain without an executable candidate: 13/27 have no registry-valid
editable unit, 1/2 have no nonempty supported flow candidate, and the all-scope
set additionally contains one empty clean reader target.

Among evaluated candidates, selected set sizes were 1–8 tokens for the
clean-exact scope and 1–9 for all queries. The largest accepted remaining-flow
fractions were 0.198789 and 0.199448 respectively, both below 0.2.

## Artifacts

```text
/data1/yujia/CausalityRAG/runs/2wiki/all1000-final-v3/
  run.yaml
  01_context/context_units.jsonl
  02_reader/clean_targets.jsonl
  03_graph/contribution_graph.jsonl
  04_flow/
    initial.jsonl
    iteration_01.jsonl
    iteration_02.jsonl
    iteration_03.jsonl
    iteration_04.jsonl
    final.jsonl
    final.summary.json
  05_registry/
    iteration_01.jsonl
    iteration_02.jsonl
    iteration_03.jsonl
    iteration_04.jsonl
    final.jsonl
    final.summary.json
  06_evaluation/
    clean_exact.jsonl
    clean_exact.summary.json
    all_queries.jsonl
    all_queries.summary.json
    clean_exact_all_available.jsonl
    clean_exact_all_available.summary.json
    all_queries_all_available.jsonl
    all_queries_all_available.summary.json
  REPORT.md
  manifest.json
```

The context, clean-answer, and graph JSONL files were reused unchanged from
the completed `all1000-final-v2` run. Flow, registry, and both reader
evaluations were recomputed in `all1000-final-v3` after removing the unwanted
matched-selector diagnostic.
