# ReFlow

ReFlow finds a small set of retrieved-context tokens whose counterfactual,
type-preserving replacements change a frozen RAG reader's response.

## Method

For a query, ReFlow constructs an answer-conditioned contribution graph from
the reader's Transformer messages. Positive contributions are projected onto a
token-gated network with a fixed query source and answer target. A token gate
is the only removable part of this network.

For a contract price `t`, the inner solver computes the exact token-gated
minimum cut. Breakpoint recursion enumerates the distinct supported cuts as
`t` changes. Candidates are ordered by cardinality and directly verified by
running the frozen reader after applying their fixed counterfactual token
replacements.

```text
retrieved chunks
  -> context-unit annotation
  -> clean reader response
  -> answer-conditioned contribution graph
  -> breakpoint flow-contract frontier
  -> counterfactual token replacements
  -> frozen-reader verification
```

The graph is a search surrogate. A returned intervention is valid only when
the reader verification changes the response under the same prompt, retrieval,
model, decoding configuration, and replacement registry.

## Main entry point

```bash
python exp/run_contribution_aware_flow_contract_attack.py \
  --input DATA.jsonl \
  --graphs CONTRIBUTION_GRAPH.jsonl \
  --units-cache CONTEXT_UNITS.jsonl \
  --replacement-registry REPLACEMENTS.jsonl \
  --out REFLOW.jsonl \
  --n 1000 \
  --frontier-mode breakpoint
```

Required preparation commands are:

```text
scripts/build_context_units.py
scripts/generate_reader_targets.py
scripts/build_contribution_graph.py
scripts/build_replacement_registry.py
```

The default reader endpoint is configured with `YVETTE_LLM_BASE_URL` and
`YVETTE_LLM_MODEL`. Graph construction requires a local Hugging Face model
because it records activations and gradients; reader verification is performed
through the OpenAI-compatible vLLM endpoint.

## Repository layout

```text
causalityrag/  contribution graph projection, max-flow, replacement, reader
scripts/       data preparation and evaluation entry points
exp/           Breakpoint Flow-Contract runner
tests/         shared pipeline tests
configs/       dataset configuration
```

Exploratory implementations are intentionally not versioned in this repository.
