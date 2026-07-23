# CausalityRAG

CausalityRAG measures the resilience of a retrieval-augmented answer: it finds
a small set of retrieved-context tokens whose fixed, type-valid replacement
changes a frozen reader's answer.

The final method optimizes an explicit contribution-flow surrogate and then
verifies the selected intervention on the reader:

```text
retrieval records
  -> cached token/linguistic units
  -> direct-activation absorbing contribution DAG
  -> projected token contribution network
  -> geometric weighted min-cuts
  -> fixed-point strict replacement registry
  -> reader intervention and local-HF verification
```

Query tokens, generated answer tokens, retrieval results, and model weights are
never edited. Context edits are non-deleting single-token replacements. A
successful reader rerun is a verified upper bound on the true black-box
resilience; the approximation guarantee applies only to the graph surrogate.

## Final method

The final `direct-activation` graph uses the reader's realized attention OV
writes, residual writes, and MLP output writes, contracted with the
clean-answer target-logit gradient. Sparse path mass is absorbed with
`--absorbing-flow`.

Let `Phi(S)` be the residual source-to-answer flow after closing the shared
token gates in set `S`. For an internal threshold `B`, the graph problem is

```text
min |S|  subject to  Phi(S) <= B.
```

For each internal geometric scale `K_r`, the implementation solves the supported
objective

```text
lambda_r |S| + Phi(S),  where lambda_r = eta B / K_r,
```

with one exact weighted min-cut. With scale ratio `1 + gamma`, the number of
min-cuts is logarithmic in the editable token count. If the strict graph
optimum has size `k`, the returned graph candidate satisfies

```text
|S| < (1 + (1 + gamma) / eta) k
Phi(S) <= (1 + eta) B.
```

For `gamma = eta = 1`, this is a `(3, 2)` bicriteria guarantee for the graph
surrogate, not a reader-level 2-approximation. The complete statement,
assumptions, calibration argument, and limitations are in
[METHOD_CONTRIBUTION_FLOW.md](METHOD_CONTRIBUTION_FLOW.md).

`K_r` is internal solver notation. It is unrelated to the command-line
variable `K` used below for the number of retrieved contexts.

## Repository layout

```text
causalityrag/       reusable graph, optimization, reader, and editing code
scripts/            stable entry points for the final pipeline
configs/            frozen/reference configurations
tests/              tests for the final and shared implementation
exp/                baselines, ablations, plots, and historical runners
results/            tracked compact result summaries
```

The `exp/` directory is intentionally outside the final pipeline. See
[exp/README.md](exp/README.md) before running historical scripts.
Use [configs/dataset_template.yaml](configs/dataset_template.yaml) as the
per-dataset configuration checklist. Dataset-specific configuration files may
coexist under `configs/`.

## Stable entry points

| Stage | Command | Output |
|---|---|---|
| Annotation service | `scripts/spacy_annotation_server.py` | localhost API |
| Token cache | `scripts/build_token_units_cache.py` | token/sentence JSONL |
| Contribution graph | `scripts/build_contribution_graph.py` | graph JSONL |
| Flow optimization | `scripts/solve_contribution_flow.py` | candidate JSONL |
| Replacement registry | `scripts/build_replacement_registry.py` | registry JSONL |
| Reader evaluation | `scripts/evaluate_reader.py` | intervention JSONL |
| Local-HF verification | `scripts/verify_hf_results.py` | verified JSONL |
| Artifact manifest | `scripts/build_artifact_manifest.py` | manifest JSON |
| Tests | `scripts/run_tests.py` | lightweight test report |

## Final pipeline contract

The final proposal is pure contribution graph. No ARC-JSD unary score,
standalone attention-baseline score, or replacement-derived score is added to
the optimization network.

| Stage | Required input | Produced artifact | Acceptance check |
|---|---|---|---|
| 1. Token cache | retrieval records, spaCy | `token_units.jsonl` | `queries=N`, nonzero units, context hashes stored |
| 2. Contribution graph | records, frozen clean reader answer, local model | `contribution_graph.jsonl` | every row is `ok`, method is direct activation, clean answer stored |
| 3. Graph optimization | graph, token cache | `contribution_flow.initial.jsonl` | projected-token network and geometric solver recorded |
| 4. Registry closure | flow candidates, replacement pools | registry plus re-solved flow | zero evaluated registry misses |
| 5. Reader evaluation | final flow, fixed registry, clean answer | budget and native evaluation JSONL | zero registry misses and replacement failures |
| 6. Local-HF verification | saved edits, same local model | `hf_verification.jsonl` | clean and edited variants regenerated with eager attention |
| 7. Manifest | all frozen artifacts | `manifest.json` | hashes, sizes, line counts, commit, and dirty state recorded |

The immutable dependency chain is:

```text
DATA + K
  -> TOKEN_UNITS
  -> CLEAN_TARGET + CONTRIBUTION_GRAPH
  -> PURE_GRAPH_FLOW_CANDIDATES
  -> FIXED_REPLACEMENT_REGISTRY
  -> FINAL_GRAPH_FLOW_CANDIDATES
  -> READER_EVALUATIONS
  -> LOCAL_HF_VERIFICATION
  -> MANIFEST
```

Changing `DATA`, `K`, reader weights, prompt, decoding, graph parameters,
registry contents, or solver parameters starts a new run and requires a new
run directory.

## Requirements

Python 3.10 or 3.11 is recommended for the GPU environment.

### GPU/scoring environment

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
```

This environment runs graph construction and local Hugging Face verification.

### spaCy environment

Keep spaCy in a separate environment when its binary dependencies conflict
with the GPU stack:

```bash
python -m venv .venv-spacy
source .venv-spacy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-spacy.txt
python -m spacy download en_core_web_lg
```

### Server-side development tests

Do not create these environments or run these tests on the local workstation.
Use a configured server environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

On a server without pytest, use:

```bash
python scripts/run_tests.py
```

## Input contract

The input may be JSONL, a JSON list, or a JSON object containing one of
`records`, `questions`, `data`, `examples`, or `items`.

A minimal record is:

```json
{
  "id": "query-1",
  "question": "Which film was released first?",
  "answer": "Film A",
  "retrieved": [
    {
      "chunk_id": "doc-1",
      "title": "Film A",
      "text": "Film A was released in 1975.",
      "rank": 1
    }
  ]
}
```

Accepted aliases include `_id`/`qid`/`question_id` for IDs,
`retrieved`/`contexts`/`ctxs`/`docs`/`passages`/`results` for retrieved
contexts, and `text`/`content`/`passage`/`body` for context text. Every stage
preserves the record ID; misaligned or missing IDs fail fast.

## Services and environment variables

Start the spaCy service in the spaCy environment:

```bash
export YVETTE_SPACY_MODEL=en_core_web_lg
python scripts/spacy_annotation_server.py --host 127.0.0.1 --port 8021
```

Graph targeting and reader evaluation use an OpenAI-compatible
chat-completions server:

```bash
export CAUSALITYRAG_SPACY_BASE_URL=http://127.0.0.1:8021
export CAUSALITYRAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
export CAUSALITYRAG_LLM_MODEL=qwen2.5-7b
```

The older `YVETTE_LLM_BASE_URL` and `YVETTE_LLM_MODEL` names remain supported
as fallbacks.

## Multi-dataset execution

Every dataset runs through the same seven stages independently. Normalize each
source dataset to the input contract above, then assign a unique `DATASET`,
`DATA`, `N`, `K`, and `RUN_DIR`. Token caches, graphs, registries, evaluations,
and manifests must never be shared across datasets.

Recommended directory layout:

```text
out/
  <dataset-a>/
    smoke10/
    final/
  <dataset-b>/
    smoke10/
    final/
```

The commands below run the complete pipeline for one dataset. Repeat them for
each dataset in the evaluation suite:

```bash
export DATASET=my_dataset
export DATA=/path/to/retrieval_records.jsonl
export MODEL=/data1/yujia/models/Qwen2.5-7B-Instruct
export CF_POOLS=/path/to/counterfactual_pools.json
export TYPE_RULES=/path/to/type_rules.yaml  # optional
export RUN_DIR="out/${DATASET}/final"
export N=1000  # replace with the number of records selected for this dataset
export K=5
export BETA=0.25
export ETA=1.0
export GAMMA=1.0
export MAX_K_GUESS=0
export MAX_BUDGET=5
export MAX_NATIVE_TOKENS=10
export RELAXED_FLOW_THRESHOLD=0.5
mkdir -p "$RUN_DIR"
```

`CF_POOLS` is a JSON object with `type_pool` and optional tab-delimited
`role_pool` keys. `TYPE_RULES` is optional. If no compatible rule metadata YAML
is available, leave it empty and remove the `--type-rules "$TYPE_RULES"` line
from stages 4 and 5.

For each dataset, first use `N=10` and
`RUN_DIR="out/${DATASET}/smoke10"` on the GPU server. After it passes, restore
the intended dataset-specific `N`, switch to
`RUN_DIR="out/${DATASET}/final"`, and start from an empty directory. Never mix
smoke and final artifacts.

### 1. Freeze token and linguistic units

```bash
python scripts/build_token_units_cache.py \
  --input "$DATA" \
  --out "$RUN_DIR/token_units.jsonl" \
  --summary-out "$RUN_DIR/token_units.summary.json" \
  --backend service \
  --workers 16 \
  --n "$N" \
  --k "$K"
```

Use `--backend local-process --spacy-model en_core_web_lg` when spaCy is
installed in the current environment.

### 2. Build the direct-activation absorbing contribution graph

```bash
python scripts/build_contribution_graph.py \
  --input "$DATA" \
  --out "$RUN_DIR/contribution_graph.jsonl" \
  --summary-out "$RUN_DIR/contribution_graph.summary.json" \
  --model-path "$MODEL" \
  --target reader \
  --device cuda \
  --dtype bfloat16 \
  --max-context-tokens 800 \
  --max-length 1024 \
  --edge-topk 6 \
  --max-receivers-per-layer 48 \
  --max-edges 5000 \
  --graph-method direct-activation \
  --absorbing-flow \
  --n "$N" \
  --k "$K"
```

Each row stores the frozen `clean_answer` used as its graph target.

The OpenAI-compatible reader selected by `CAUSALITYRAG_LLM_MODEL` must use the
same model weights, prompt, top-k contexts, and greedy decoding contract as
the local model at `MODEL`.

### 3. Produce an initial contribution-flow solution

```bash
python scripts/solve_contribution_flow.py \
  --input "$DATA" \
  --graphs "$RUN_DIR/contribution_graph.jsonl" \
  --units-cache "$RUN_DIR/token_units.jsonl" \
  --out "$RUN_DIR/contribution_flow.initial.jsonl" \
  --summary-out "$RUN_DIR/contribution_flow.initial.summary.json" \
  --solver geometric-k-guessing \
  --projection layer-copy-token \
  --capacity-mode raw \
  --beta "$BETA" \
  --eta "$ETA" \
  --gamma "$GAMMA" \
  --max-k-guess "$MAX_K_GUESS" \
  --n "$N" \
  --k "$K"
```

`--max-k-guess 0` means the geometric grid runs through the complete editable
token count. A positive value truncates the guarantee to instances whose graph
optimum is no larger than that value.

### 4. Close the strict replacement registry

Selection never sees replacement text. The registry fixes one answer-blind
replacement per candidate and marks invalid tokens uncuttable.

Build the first registry:

```bash
python scripts/build_replacement_registry.py \
  --input "$DATA" \
  --gates "$RUN_DIR/contribution_flow.initial.jsonl" \
  --units-cache "$RUN_DIR/token_units.jsonl" \
  --cf-pools "$CF_POOLS" \
  --type-rules "$TYPE_RULES" \
  --out "$RUN_DIR/replacement_registry.01.jsonl" \
  --summary-out "$RUN_DIR/replacement_registry.01.summary.json" \
  --max-budget "$MAX_BUDGET" \
  --max-native-tokens "$MAX_NATIVE_TOKENS" \
  --backend service \
  --workers 16 \
  --n "$N" \
  --k "$K"
```

Re-solve while excluding known-invalid tokens:

```bash
python scripts/solve_contribution_flow.py \
  --input "$DATA" \
  --graphs "$RUN_DIR/contribution_graph.jsonl" \
  --units-cache "$RUN_DIR/token_units.jsonl" \
  --replacement-registry "$RUN_DIR/replacement_registry.01.jsonl" \
  --replacement-registry-policy exclude-known-invalid \
  --out "$RUN_DIR/contribution_flow.01.jsonl" \
  --summary-out "$RUN_DIR/contribution_flow.01.summary.json" \
  --solver geometric-k-guessing \
  --projection layer-copy-token \
  --capacity-mode raw \
  --beta "$BETA" \
  --eta "$ETA" \
  --gamma "$GAMMA" \
  --max-k-guess "$MAX_K_GUESS" \
  --n "$N" \
  --k "$K"
```

If `evaluated_candidate_registry_misses` is nonzero, build the next registry
with the previous registry and the latest flow output:

```bash
python scripts/build_replacement_registry.py \
  --input "$DATA" \
  --gates "$RUN_DIR/contribution_flow.01.jsonl" \
  --existing-registry "$RUN_DIR/replacement_registry.01.jsonl" \
  --units-cache "$RUN_DIR/token_units.jsonl" \
  --cf-pools "$CF_POOLS" \
  --type-rules "$TYPE_RULES" \
  --out "$RUN_DIR/replacement_registry.02.jsonl" \
  --summary-out "$RUN_DIR/replacement_registry.02.summary.json" \
  --max-budget "$MAX_BUDGET" \
  --max-native-tokens "$MAX_NATIVE_TOKENS" \
  --backend service \
  --workers 16 \
  --n "$N" \
  --k "$K"
```

Use `replacement_registry.02.jsonl` in the next solve, increment the suffixes,
and repeat until the flow summary reports:

```json
{
  "registry_fixed_point": true,
  "evaluated_candidate_registry_misses": 0
}
```

For the frozen final artifact, run once more with the fixed registry and
`--replacement-registry-policy allow-only`. This makes the shared editable
domain explicit and immutable:

```bash
export REGISTRY="$RUN_DIR/replacement_registry.02.jsonl"

python scripts/solve_contribution_flow.py \
  --input "$DATA" \
  --graphs "$RUN_DIR/contribution_graph.jsonl" \
  --units-cache "$RUN_DIR/token_units.jsonl" \
  --replacement-registry "$REGISTRY" \
  --replacement-registry-policy allow-only \
  --out "$RUN_DIR/contribution_flow.FINAL.jsonl" \
  --summary-out "$RUN_DIR/contribution_flow.FINAL.summary.json" \
  --solver geometric-k-guessing \
  --projection layer-copy-token \
  --capacity-mode raw \
  --beta "$BETA" \
  --eta "$ETA" \
  --gamma "$GAMMA" \
  --max-k-guess "$MAX_K_GUESS" \
  --n "$N" \
  --k "$K"

export FLOW="$RUN_DIR/contribution_flow.FINAL.jsonl"
```

Keep `REGISTRY` pointed at whichever numbered registry reached the fixed point.

### 5. Evaluate saved selections with the served reader

First run the fixed-budget evaluations:

```bash
for BUDGET in 1 3 5; do
  python scripts/evaluate_reader.py \
    --input "$DATA" \
    --gate "$FLOW" \
    --clean-reference "$RUN_DIR/contribution_graph.jsonl" \
    --replacement-registry "$REGISTRY" \
    --units-cache "$RUN_DIR/token_units.jsonl" \
    --cf-pools "$CF_POOLS" \
    --type-rules "$TYPE_RULES" \
    --out "$RUN_DIR/evaluation.b${BUDGET}.jsonl" \
    --summary-out "$RUN_DIR/evaluation.b${BUDGET}.summary.json" \
    --selection-mode budget \
    --token-budget "$BUDGET" \
    --max-tokens "$BUDGET" \
    --clean-correct-policy exact \
    --strict-replacements \
    --k "$K"
done
```

Then run the native graph-threshold evaluation. With `BETA=0.25` and `ETA=1`,
the solver accepts candidates up to the relaxed residual-flow fraction
`(1 + ETA) * BETA = 0.50`:

```bash
python scripts/evaluate_reader.py \
  --input "$DATA" \
  --gate "$FLOW" \
  --clean-reference "$RUN_DIR/contribution_graph.jsonl" \
  --replacement-registry "$REGISTRY" \
  --units-cache "$RUN_DIR/token_units.jsonl" \
  --cf-pools "$CF_POOLS" \
  --type-rules "$TYPE_RULES" \
  --out "$RUN_DIR/evaluation.native.jsonl" \
  --summary-out "$RUN_DIR/evaluation.native.summary.json" \
  --selection-mode threshold \
  --remaining-flow-threshold "$RELAXED_FLOW_THRESHOLD" \
  --max-tokens "$MAX_NATIVE_TOKENS" \
  --clean-correct-policy exact \
  --strict-replacements \
  --k "$K"
```

The primary output is the contribution-flow selection. The artifact also
contains a cardinality-matched ranking by graph-local token support for
diagnostic comparison; it is not ARC-JSD and is not part of the proposed
selector.

### 6. Perform final local-HF eager verification

Served-reader results are diagnostic. Final answer-change metrics should be
recomputed with the same local model weights and eager attention backend used
for scoring:

```bash
python scripts/verify_hf_results.py \
  --input "$DATA" \
  --results \
    "$RUN_DIR/evaluation.b1.jsonl" \
    "$RUN_DIR/evaluation.b3.jsonl" \
    "$RUN_DIR/evaluation.b5.jsonl" \
    "$RUN_DIR/evaluation.native.jsonl" \
  --out "$RUN_DIR/hf_verification.jsonl" \
  --summary-out "$RUN_DIR/hf_verification.summary.json" \
  --model-path "$MODEL" \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 96 \
  --k "$K"
```

Use `--clean-targets` only when a separate JSONL explicitly freezes
`{"id", "target_answer"}` pairs.

### 7. Freeze an artifact manifest

```bash
python scripts/build_artifact_manifest.py \
  --repository . \
  --artifacts \
    "$RUN_DIR/token_units.jsonl" \
    "$RUN_DIR/contribution_graph.jsonl" \
    "$FLOW" \
    "$REGISTRY" \
    "$RUN_DIR/hf_verification.jsonl" \
  --metadata-json "{\"dataset\":\"$DATASET\",\"n\":$N,\"k\":$K}" \
  --out "$RUN_DIR/manifest.json"
```

The manifest records SHA-256 hashes, byte and line counts, the current commit,
branch, and dirty-worktree state.

## Expected final artifacts

```text
out/<dataset>/final/
  token_units.jsonl
  token_units.summary.json
  contribution_graph.jsonl
  contribution_graph.summary.json
  contribution_flow.initial.jsonl
  contribution_flow.initial.summary.json
  replacement_registry.01.jsonl
  replacement_registry.01.summary.json
  ... additional registry/flow closure iterations ...
  contribution_flow.FINAL.jsonl
  contribution_flow.FINAL.summary.json
  evaluation.b1.jsonl
  evaluation.b1.summary.json
  evaluation.b3.jsonl
  evaluation.b3.summary.json
  evaluation.b5.jsonl
  evaluation.b5.summary.json
  evaluation.native.jsonl
  evaluation.native.summary.json
  hf_verification.jsonl
  hf_verification.summary.json
  manifest.json
```

## Method review checklist

- The stable commands contain no `--unary-score-rows`, `--graph-weight`, or
  ARC-JSD artifact.
- The graph method is `direct-activation` with `--absorbing-flow`.
- The query, answer, and retrieval ranking remain fixed; only retrieved
  context-token gates are purchasable.
- `layer-copy-token` is the current CLI name for projecting all transformer
  layer copies of one context token into one token node. It is not the old
  grouped layer-copy rounding method.
- The final solver is `geometric-k-guessing` with explicit `BETA`, `ETA`,
  `GAMMA`, and `MAX_K_GUESS`.
- `MAX_K_GUESS=0` covers the full editable token domain.
- Invalid replacements remain in the flow graph but are made uncuttable.
- Registry closure ends only when
  `evaluated_candidate_registry_misses == 0`.
- Budget results use the same fixed registry and evaluate budgets 1, 3, and 5.
- Native evaluation uses the solver's relaxed threshold
  `(1 + ETA) * BETA`.
- Final reported reader results come from local-HF eager verification rather
  than the served-reader diagnostic.

## Output and reproducibility rules

- JSONL stage outputs are immutable inputs to later stages.
- Keep the same `id`, record order, top-k value, model weights, tokenizer,
  prompt, dtype, and eager-attention backend throughout a final run.
- Use the token cache to enforce context hashes and offsets.
- Do not regenerate replacements separately for competing selectors.
- Do not interpret vLLM/server disagreement as an intervention effect.
- Record the final registry fixed point and artifact manifest.

## Tests

Run tests only in the configured server environments, not on the local
workstation. Unit tests that do not load the model can run in the server spaCy
or development environment:

The final test suite covers input normalization, token offsets, contextual
replacement validity, reader metrics, direct contribution graphs, projected
token flow, exact weighted min-cuts, geometric scales, registry filtering, and
saved-intervention verification. ARC-JSD tests are retained under `exp/`.

```bash
python scripts/run_tests.py
```

or:

```bash
python -m pytest -q
```

GPU integration stages require the model and external data and are not part of
the lightweight unit suite. Run the 10-query smoke pipeline for every dataset
on the GPU server before starting its frozen full run.

## Multi-dataset reporting

Report every dataset separately before producing macro or micro aggregates.
Each dataset result must reference its own manifest, sample count, retrieval
depth, clean-correct coverage, budget metrics, native-threshold metrics, and
local-HF verification summary. Cross-dataset aggregation is valid only after
the per-dataset artifacts pass the same acceptance checks.

Historical dataset-specific baselines, ablations, plotting scripts, and
protocols are retained under `exp/` for provenance. They are not stable final
entry points and may use historical terminology or artifact schemas.
