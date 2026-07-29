# ReFlow

ReFlow finds a small set of retrieved-context tokens whose fixed
counterfactual replacements change a frozen RAG reader's response.

## Pipeline

```text
retrieved chunks
  -> clean response
  -> answer-conditioned contribution graph
  -> exact breakpoint flow-contract frontier
  -> frozen counterfactual replacements
  -> reader verification
```

The contribution graph contains every non-punctuation retrieved token.
Positive answer-support messages are propagated through the Transformer and
contracted by surface-token position. ReFlow then restricts optimization to a
method-independent editable domain fixed before any method runs. For each
contract price, the inner token-gated cut is solved exactly by max-flow. An
output-sensitive breakpoint recursion enumerates all extreme supported
cardinality-flow solutions without a price grid.

The graph is a search surrogate. A token set is a valid intervention only when
direct reader verification changes the response.

## Frozen Replacement Protocol

All methods use one position-level pool constructed over the largest retrieval
range in the experiment. For a study with `k in {1, 3, 5, 10}`, this is the
HotpotQA top-10 prefix. Pool construction is independent of ReFlow and all
baselines:

1. Tokenize every retrieved chunk and declare content tokens with POS in
   `{NOUN, PROPN, VERB, ADJ, ADV, NUM}` editable.
2. Group occurrences by surface form, entity type, POS, tag, morphology, and
   named-entity slot.
3. Ask the editor LLM for multiple same-type alternatives that express
   incompatible factual information.
4. Use spaCy to validate contextual grammar and morphology.
5. Use WordNet to reject synonyms and close lexical paraphrases.
6. Use a blind LLM semantic judge to reject type drift, paraphrases, and edits
   that do not express an incompatible factual alternative.
7. Expand validated typed candidates to position-level rows, sort them, freeze
   the file, and record its SHA-256 digest.

Evaluation is strictly read-only. Every runner checks the expected pool digest.
For a fixed seed, the replacement is a deterministic hash of the token's
global position ID. A method may not generate a private replacement, skip a
selected token, or backfill it with a lower-ranked token. An uncovered selected
position is a protocol violation.

Pool construction requires the NLTK WordNet corpus:

```bash
python -m nltk.downloader wordnet omw-1.4
```

### Build and freeze the pool

```bash
python scripts/build_shared_replacement_inventory.py \
  --input RETRIEVAL_TOP10.jsonl \
  --units-cache TOKEN_UNITS_TOP10.jsonl \
  --positions-out pool/positions.jsonl \
  --typed-keys-out pool/typed_keys.jsonl \
  --manifest-out pool/inventory.json \
  --n 1000 --k 10

python scripts/generate_shared_typed_replacement_pool.py \
  --typed-keys pool/typed_keys.jsonl \
  --seed pool/typed_candidates.jsonl \
  --out pool/typed_candidates.jsonl \
  --unresolved-out pool/unresolved.jsonl \
  --manifest-out pool/generation.json

python scripts/freeze_shared_replacement_pool.py \
  --positions pool/positions.jsonl \
  --typed-candidates pool/typed_candidates.jsonl \
  --out pool/shared_pool.jsonl \
  --manifest-out pool/shared_pool.manifest.json \
  --exclude-unresolved
```

When extending top-k, `scripts/seed_typed_pool_from_position_pool.py` can seed
typed candidates from an existing frozen position pool. Exclusions are common
to every method and represent positions for which no legal counterfactual was
found under the declared contract.

## ReFlow

Build token units, clean targets, and the contribution graph:

```text
scripts/build_context_units.py
scripts/generate_reader_targets.py
scripts/build_contribution_graph.py
```

If a compatible closed message-flow artifact already exists,
`scripts/contract_contribution_graph.py` contracts it into the final token
graph without another model forward/backward pass.

Generate the graph frontier:

```bash
python exp/run_reflow.py \
  --input RETRIEVAL.jsonl \
  --graphs CONTRIBUTION_GRAPH.jsonl \
  --units-cache TOKEN_UNITS.jsonl \
  --shared-pool pool/shared_pool.jsonl \
  --expected-pool-sha256 POOL_SHA256 \
  --out REFLOW_FRONTIER.jsonl \
  --summary-out REFLOW_FRONTIER.summary.json \
  --n 1000 --k 5
```

Verify frontier candidates:

```bash
python scripts/evaluate_reflow.py \
  --input RETRIEVAL.jsonl \
  --units-cache TOKEN_UNITS.jsonl \
  --frontier REFLOW_FRONTIER.jsonl \
  --shared-pool pool/shared_pool.jsonl \
  --expected-pool-sha256 POOL_SHA256 \
  --out REFLOW_RESULTS.jsonl \
  --summary-out REFLOW_RESULTS.summary.json \
  --n 1000 --k 5
```

## Baselines

The retained attribution baselines are ARC-JSD, Gradient x Input, and
Integrated Gradients. Their rankers never modify context. Use the common ReFlow
token budget and frozen pool for reader evaluation:

```bash
python scripts/evaluate_matched_budget_baselines.py \
  --input RETRIEVAL.jsonl \
  --units-cache TOKEN_UNITS.jsonl \
  --reflow-results REFLOW_RESULTS.jsonl \
  --scores arc_jsd=ARC_JSD_SCORES.jsonl \
  --scores gradient_x_input=GRADIENT_SCORES.jsonl \
  --scores integrated_gradients=IG_SCORES.jsonl \
  --shared-pool pool/shared_pool.jsonl \
  --expected-pool-sha256 POOL_SHA256 \
  --out BASELINE_RESULTS.jsonl \
  --summary-out BASELINE_RESULTS.summary.json
```

Finally, `scripts/audit_shared_replacement_protocol.py` verifies that every
executed edit exactly matches the deterministic candidate from the frozen pool.

Graph construction requires a local Hugging Face model because it records
activations and gradients. Reader verification uses the OpenAI-compatible
endpoint configured by `YVETTE_LLM_BASE_URL` and `YVETTE_LLM_MODEL`.

## Layout

```text
causalityrag/  graph, flow-contract, replacement-pool, and reader primitives
exp/           ReFlow and external attribution rankers
scripts/       reproducible preparation and evaluation entry points
tests/         unit and protocol tests
```
