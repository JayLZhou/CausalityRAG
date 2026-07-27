# ReFlow

ReFlow finds a small set of retrieved-context tokens whose counterfactual,
type-preserving replacements change a frozen RAG reader's response.

## Method

For a query, `ContributionGraphBuilder.build(...)` performs the complete graph
construction in one call: it runs the reader forward and backward, attributes
the Transformer messages, propagates closed answer-support flow, and contracts
all stage copies of the same retrieved token into one token node. Positive flow
between different token labels is aggregated, fixed prompt/query flow anchors
the source, and answer-predictor flow anchors the target.

The graph and optimizer always contain every non-punctuation retrieved token.
Token selection never depends on replacement availability. After a method
proposes a token set, ReFlow resolves the corresponding counterfactual
replacements on demand: it reuses validated entries from a shared pool, tries
the typed rule library, calls the local editor LLM for missing entries, validates
the result with spaCy, and atomically writes the result back to the pool.
Function words or other positions for which no meaningful counterfactual exists
are recorded and skipped only after selection.

For a contract price `t`, the inner solver computes the exact token-gated
minimum cut. ReFlow analytically constructs the minimum-cardinality full-cut
and empty-set endpoints. For two endpoint solutions, it evaluates the exact
price at which their cardinality-flow objective lines intersect. The resulting
min-cut either exposes a strict lower-hull vertex or certifies that the two
solutions are adjacent. This output-sensitive recursion enumerates every
extreme supported cardinality-flow point without a price grid, numerical price
probe, or recursion-depth cutoff. Candidates are ordered by cardinality and
directly verified by running the frozen reader after applying their fixed
counterfactual token replacements.

```text
retrieved chunks
  -> clean reader response
  -> complete all-token contribution graph

complete contribution graph
  -> breakpoint flow-contract frontier
  -> shared on-demand replacement pool
  -> counterfactual token replacements
  -> frozen-reader verification
```

The graph is a search surrogate. A returned intervention is valid only when
the reader verification changes the response under the same prompt, retrieval,
model, decoding configuration, and replacement pool.

## Main entry point

```bash
python exp/run_contribution_aware_flow_contract_attack.py \
  --input DATA.jsonl \
  --graphs CONTRIBUTION_GRAPH.jsonl \
  --units-cache CONTEXT_UNITS.jsonl \
  --replacement-pool REPLACEMENTS.jsonl \
  --cf-pools CF_POOLS.json \
  --out REFLOW.jsonl \
  --n 1000 \
  --frontier-mode breakpoint
```

Required preparation commands are:

```text
scripts/build_context_units.py
scripts/generate_reader_targets.py
scripts/build_contribution_graph.py
```

`scripts/build_contribution_graph.py` derives its token-node domain directly
from the retrieved chunks and therefore does not accept a replacement registry
or a filtered token-unit file.

If a compatible closed message-flow artifact already exists, its expensive
model forward/backward pass can be reused. The following command contracts it
into the final all-token Contribution Graph without loading the model:

```bash
python scripts/contract_contribution_graph.py \
  --input DATA.jsonl \
  --raw-graphs CLOSED_MESSAGE_FLOW.jsonl \
  --units-cache CONTEXT_UNITS.jsonl \
  --out CONTRIBUTION_GRAPH.jsonl \
  --summary-out CONTRIBUTION_GRAPH.summary.json
```

The converter never overwrites `--raw-graphs`, validates aligned query IDs and
the complete token domain, and atomically publishes the new graph only after
all rows succeed. Sharded ReFlow outputs can be validated and merged with
`scripts/merge_reflow_shards.py`.

The default reader endpoint is configured with `YVETTE_LLM_BASE_URL` and
`YVETTE_LLM_MODEL`. Graph construction requires a local Hugging Face model
because it records activations and gradients; reader verification is performed
through the OpenAI-compatible vLLM endpoint.

## Repository layout

```text
causalityrag/  contribution graph construction, max-flow, replacement, reader
scripts/       data preparation and evaluation entry points
exp/           Breakpoint Flow-Contract runner
tests/         shared pipeline tests
configs/       dataset configuration
```

The retained external baselines live under `exp/`. The ARC-JSD
sentence-to-token lift is implemented in `exp/run_arc_jsd_sentence_lift_attack.py`.
The gradient attribution runner in
`exp/run_gradient_attribution_baselines.py` provides Gradient×Input-Top5 and
Integrated-Gradients-Top5; both rank tokens independently of the ReFlow
contribution graph and use the same on-demand replacement pool during
evaluation. Replacement availability is never used to filter or rerank any
method's candidates.
