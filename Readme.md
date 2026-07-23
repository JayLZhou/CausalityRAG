# CausalityRAG

CausalityRAG measures the resilience of a retrieval-augmented answer: it finds
a small set of retrieved-context tokens whose fixed, type-valid replacement
changes a frozen reader's answer.

The final method optimizes an explicit contribution-flow surrogate and then
verifies the selected intervention on the reader:

```text
retrieval records
  -> frozen context/linguistic units
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
| Context units | `scripts/build_context_units.py` | frozen context-unit JSONL |
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
| 1. Context units | retrieval records, spaCy | `context_units.jsonl` | `queries=N`, nonzero units, context hashes stored |
| 2. Contribution graph | records, frozen clean reader answer, local model | `contribution_graph.jsonl` | every row is `ok`, method is direct activation, clean answer stored |
| 3. Graph optimization | graph, context units | initial flow JSONL | projected-token network and geometric solver recorded |
| 4. Registry closure | flow candidates, replacement pools | registry plus re-solved flow | zero evaluated registry misses |
| 5. Reader evaluation | final flow, fixed registry, clean answer | native evaluation JSONL | zero registry misses and replacement failures |
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
`DATA`, `N`, `K`, and `RUN_ID`. Context units, graphs, registries, evaluations,
and manifests must never be shared across datasets or runs.

The canonical storage layout is:

```text
runs/<dataset>/<run-id>/
  run.yaml
  01_context/
  02_graph/
  03_flow/
  04_registry/
  05_evaluation/
  06_verification/
  manifest.json
```

The commands below run the complete pipeline for one dataset. Repeat them for
each dataset in the evaluation suite:

```bash
export DATASET=my_dataset
export RUN_ID=final-v1
export RUN_ROOT=/data1/yujia/CausalityRAG/runs
export CONFIG=configs/my_dataset.yaml
export DATA=/path/to/retrieval_records.jsonl
export MODEL=/data1/yujia/models/Qwen2.5-7B-Instruct
export CF_POOLS=/path/to/counterfactual_pools.json
export TYPE_RULES=/path/to/type_rules.yaml  # optional
export N=1000  # replace with the number of records selected for this dataset
export K=5
export BETA=0.1
export ETA=1.0
export GAMMA=1.0
export MAX_K_GUESS=0
export RELAXED_FLOW_THRESHOLD=0.2

export RUN_DIR="${RUN_ROOT}/${DATASET}/${RUN_ID}"
export CONTEXT_DIR="${RUN_DIR}/01_context"
export GRAPH_DIR="${RUN_DIR}/02_graph"
export FLOW_DIR="${RUN_DIR}/03_flow"
export REGISTRY_DIR="${RUN_DIR}/04_registry"
export EVALUATION_DIR="${RUN_DIR}/05_evaluation"
export VERIFICATION_DIR="${RUN_DIR}/06_verification"

export CONTEXT_UNITS="${CONTEXT_DIR}/context_units.jsonl"
export GRAPH="${GRAPH_DIR}/contribution_graph.jsonl"
export FLOW_INITIAL="${FLOW_DIR}/initial.jsonl"

mkdir -p \
  "$CONTEXT_DIR" \
  "$GRAPH_DIR" \
  "$FLOW_DIR" \
  "$REGISTRY_DIR" \
  "$EVALUATION_DIR" \
  "$VERIFICATION_DIR"

cp "$CONFIG" "$RUN_DIR/run.yaml"
```

`CF_POOLS` is a JSON object with `type_pool` and optional tab-delimited
`role_pool` keys. `TYPE_RULES` is optional. If no compatible rule metadata YAML
is available, leave it empty and remove the `--type-rules "$TYPE_RULES"` line
from stages 4 and 5. Create `CONFIG` by copying
`configs/dataset_template.yaml`, then fill in the same frozen values exported
above.

For each dataset, first use `N=10` and `RUN_ID=smoke10` on the GPU server.
After it passes, start a new shell, restore the intended dataset-specific `N`,
set a fresh final run ID such as `RUN_ID=final-v1`, and recreate the derived
path variables above. Never reuse a smoke directory for a final run.

### 1. Freeze token and linguistic units

```bash
python scripts/build_context_units.py \
  --input "$DATA" \
  --out "$CONTEXT_UNITS" \
  --summary-out "$CONTEXT_DIR/summary.json" \
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
  --out "$GRAPH" \
  --summary-out "$GRAPH_DIR/summary.json" \
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
  --graphs "$GRAPH" \
  --context-units "$CONTEXT_UNITS" \
  --out "$FLOW_INITIAL" \
  --summary-out "$FLOW_DIR/initial.summary.json" \
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

#### Registry implementation

For each query, `build_replacement_registry.py` performs the following steps:

1. Collect every token ID in the strict-threshold witness, the native
   relaxed-threshold candidate, and their size-matched graph diagnostics.
2. Reuse entries from the previous registry iteration; an existing valid or
   invalid decision is never regenerated.
3. Try a deterministic typed replacement first: same-slot entity pool,
   date/number shift, or relation-cue substitution.
4. If no typed replacement passes, ask the local editor for one replacement
   word with temperature zero. The editor receives the target token, its local
   passage, and linguistic hints, but not the question, clean answer, or gold
   answer.
5. Validate exact offsets, non-deletion, one-word surface form, changed value,
   contextual POS, strict tags for function/verb classes, and relevant
   `Number`, `Tense`, `VerbForm`, and `Person` morphology.
6. Store the accepted replacement under `replacements[unit_id]`; otherwise
   store the failure reason under `invalid[unit_id]`.

The registry is therefore an intervention-feasibility artifact, not an
optimization score. Re-solving with `exclude-known-invalid` prevents invalid
tokens from being purchased while keeping their contribution nodes and edges
in the graph. The closure loop continues until all newly evaluated candidates
already have a valid or invalid registry entry. The final `allow-only` solve
then permits only tokens with a frozen valid replacement. Reader evaluation
loads those exact values and never regenerates them.

Build the first registry:

```bash
python scripts/build_replacement_registry.py \
  --input "$DATA" \
  --gates "$FLOW_INITIAL" \
  --context-units "$CONTEXT_UNITS" \
  --cf-pools "$CF_POOLS" \
  --type-rules "$TYPE_RULES" \
  --out "$REGISTRY_DIR/iteration_01.jsonl" \
  --summary-out "$REGISTRY_DIR/iteration_01.summary.json" \
  --backend service \
  --workers 16 \
  --n "$N" \
  --k "$K"
```

Re-solve while excluding known-invalid tokens:

```bash
python scripts/solve_contribution_flow.py \
  --input "$DATA" \
  --graphs "$GRAPH" \
  --context-units "$CONTEXT_UNITS" \
  --replacement-registry "$REGISTRY_DIR/iteration_01.jsonl" \
  --replacement-registry-policy exclude-known-invalid \
  --out "$FLOW_DIR/iteration_01.jsonl" \
  --summary-out "$FLOW_DIR/iteration_01.summary.json" \
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
  --gates "$FLOW_DIR/iteration_01.jsonl" \
  --existing-registry "$REGISTRY_DIR/iteration_01.jsonl" \
  --context-units "$CONTEXT_UNITS" \
  --cf-pools "$CF_POOLS" \
  --type-rules "$TYPE_RULES" \
  --out "$REGISTRY_DIR/iteration_02.jsonl" \
  --summary-out "$REGISTRY_DIR/iteration_02.summary.json" \
  --backend service \
  --workers 16 \
  --n "$N" \
  --k "$K"
```

Use `iteration_02.jsonl` in the next solve, increment the suffixes,
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
export REGISTRY_FIXED_POINT="$REGISTRY_DIR/iteration_02.jsonl"
cp "$REGISTRY_FIXED_POINT" "$REGISTRY_DIR/final.jsonl"
export REGISTRY="$REGISTRY_DIR/final.jsonl"

python scripts/solve_contribution_flow.py \
  --input "$DATA" \
  --graphs "$GRAPH" \
  --context-units "$CONTEXT_UNITS" \
  --replacement-registry "$REGISTRY" \
  --replacement-registry-policy allow-only \
  --out "$FLOW_DIR/final.jsonl" \
  --summary-out "$FLOW_DIR/final.summary.json" \
  --solver geometric-k-guessing \
  --projection layer-copy-token \
  --capacity-mode raw \
  --beta "$BETA" \
  --eta "$ETA" \
  --gamma "$GAMMA" \
  --max-k-guess "$MAX_K_GUESS" \
  --n "$N" \
  --k "$K"

export FLOW="$FLOW_DIR/final.jsonl"
```

Keep `REGISTRY_FIXED_POINT` pointed at whichever numbered registry reached the
fixed point. The copy to `final.jsonl` gives every dataset the same downstream
path while preserving all numbered closure iterations.

### 5. Evaluate saved selections with the served reader

Run only the native graph-threshold evaluation. With `BETA=0.1` and `ETA=1`,
the solver accepts candidates up to the relaxed residual-flow fraction
`(1 + ETA) * BETA = 0.20`:

```bash
python scripts/evaluate_reader.py \
  --input "$DATA" \
  --gate "$FLOW" \
  --clean-reference "$GRAPH" \
  --replacement-registry "$REGISTRY" \
  --context-units "$CONTEXT_UNITS" \
  --cf-pools "$CF_POOLS" \
  --type-rules "$TYPE_RULES" \
  --out "$EVALUATION_DIR/native.jsonl" \
  --summary-out "$EVALUATION_DIR/native.summary.json" \
  --remaining-flow-threshold "$RELAXED_FLOW_THRESHOLD" \
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
  --results "$EVALUATION_DIR/native.jsonl" \
  --out "$VERIFICATION_DIR/local_hf.jsonl" \
  --summary-out "$VERIFICATION_DIR/local_hf.summary.json" \
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
    "$CONTEXT_UNITS" \
    "$GRAPH" \
    "$FLOW" \
    "$REGISTRY" \
    "$VERIFICATION_DIR/local_hf.jsonl" \
  --metadata-json "{\"dataset\":\"$DATASET\",\"run_id\":\"$RUN_ID\",\"n\":$N,\"k\":$K}" \
  --out "$RUN_DIR/manifest.json"
```

The manifest records SHA-256 hashes, byte and line counts, the current commit,
branch, and dirty-worktree state.

## Expected final artifacts

```text
runs/<dataset>/<run-id>/
  run.yaml
  01_context/
    context_units.jsonl
    summary.json
  02_graph/
    contribution_graph.jsonl
    summary.json
  03_flow/
    initial.jsonl
    initial.summary.json
    iteration_01.jsonl
    iteration_01.summary.json
    ... additional closure iterations ...
    final.jsonl
    final.summary.json
  04_registry/
    iteration_01.jsonl
    iteration_01.summary.json
    ... additional closure iterations ...
    final.jsonl
  05_evaluation/
    native.jsonl
    native.summary.json
  06_verification/
    local_hf.jsonl
    local_hf.summary.json
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
- Native evaluation uses the solver's relaxed threshold
  `(1 + ETA) * BETA`.
- Final reported reader results come from local-HF eager verification rather
  than the served-reader diagnostic.

## Output and reproducibility rules

- JSONL stage outputs are immutable inputs to later stages.
- Keep the same `id`, record order, top-k value, model weights, tokenizer,
  prompt, dtype, and eager-attention backend throughout a final run.
- Use the frozen context-unit artifact to enforce context hashes and offsets.
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
depth, clean-correct coverage, native-threshold metrics, and local-HF
verification summary. Cross-dataset aggregation is valid only after the
per-dataset artifacts pass the same acceptance checks.

Historical dataset-specific baselines, ablations, plotting scripts, and
protocols are retained under `exp/` for provenance. They are not stable final
entry points and may use historical terminology or artifact schemas.
