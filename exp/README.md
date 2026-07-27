# ReFlow experiment runner

`run_contribution_aware_flow_contract_attack.py` is the retained ReFlow
experimental runner. It evaluates the Breakpoint Flow-Contract frontier and
verifies candidate interventions with the frozen reader. Its
`breakpoint` mode uses analytic endpoints and exact objective-line
intersections to recover the extreme supported frontier.

`arc_jsd.py` and `run_arc_jsd_sentence_lift_attack.py` implement the retained
ARC-JSD sentence-attribution baseline and its sentence-to-token lift. They are
kept only for external comparison.

## Gradient attribution baselines

`run_gradient_attribution_baselines.py` implements two fixed Top-5 external
baselines for the same frozen reader and replacement contract:

- `gradient_x_input`: the absolute input-embedding times gradient attribution
  of the mean teacher-forced clean-answer log-likelihood;
- `integrated_gradients`: the same objective integrated from a zero embedding
  baseline to the real context embeddings. Only context positions are
  interpolated; prompt and teacher-forced answer embeddings stay fixed.

The runner writes a selection JSONL. It never calls the reader during ranking;
use `scripts/evaluate_reader.py` with `--ignore-remaining-flow-threshold` to
perform the matched vLLM verification. Ranking covers all non-punctuation
context tokens and is independent of replacement availability. During
evaluation, missing counterfactual replacements are generated, validated, and
persisted in the same shared pool used by ReFlow and every other baseline.

Example (server-side GPU attribution):

```bash
python exp/run_gradient_attribution_baselines.py \
  --input RETRIEVAL.jsonl \
  --clean-reference CLEAN_TARGETS.jsonl \
  --context-units CONTEXT_UNITS.jsonl \
  --method gradient_x_input \
  --model-path /data1/yujia/models/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --out gradient_x_input_top5.jsonl
```

Matched evaluation:

```bash
python scripts/evaluate_reader.py \
  --input RETRIEVAL.jsonl \
  --gate gradient_x_input_top5.jsonl \
  --clean-reference CLEAN_TARGETS.jsonl \
  --context-units CONTEXT_UNITS.jsonl \
  --replacement-pool REPLACEMENTS.jsonl \
  --cf-pools CF_POOLS.json \
  --ignore-remaining-flow-threshold \
  --out gradient_x_input_top5_reader.jsonl
```
