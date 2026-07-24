# CausalityRAG

CausalityRAG measures the resilience of a retrieval-augmented answer: it finds
a small set of retrieved-context tokens whose fixed, type-valid replacement
changes a frozen reader's answer.

The final method optimizes an explicit contribution-flow surrogate and then
verifies the selected intervention on the reader:

```text
retrieval records
  -> frozen context/linguistic units
  -> frozen clean answers generated concurrently by vLLM
  -> direct-activation absorbing contribution DAG
  -> projected token contribution network
  -> geometric weighted min-cuts
  -> fixed-point strict replacement registry
  -> concurrent vLLM reader intervention evaluation
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
| Clean reader targets | `scripts/generate_reader_targets.py` | vLLM answer JSONL |
| Contribution graph | `scripts/build_contribution_graph.py` | graph JSONL |
| Flow optimization | `scripts/solve_contribution_flow.py` | candidate JSONL |
| Replacement registry | `scripts/build_replacement_registry.py` | registry JSONL |
| Reader evaluation | `scripts/evaluate_reader.py` | intervention JSONL |
| Artifact manifest | `scripts/build_artifact_manifest.py` | manifest JSON |
| Tests | `scripts/run_tests.py` | lightweight test report |

## Final pipeline contract

The final proposal is pure contribution graph. No ARC-JSD unary score,
standalone attention-baseline score, or replacement-derived score is added to
the optimization network.

| Stage | Required input | Produced artifact | Acceptance check |
|---|---|---|---|
| 1. Context units | retrieval records, spaCy | `context_units.jsonl` | `queries=N`, nonzero units, context hashes stored |
| 2. Clean targets | records, vLLM | `clean_targets.jsonl` | every ID has a frozen answer or explicit reader-abstention status |
| 3. Contribution graph | records, clean targets, same checkpoint in HF SDPA | `contribution_graph.jsonl` | every non-abstention row is `ok`, has a positive context-to-answer path, and stores the clean answer |
| 4. Graph optimization | graph, context units | initial flow JSONL | projected-token network and geometric solver recorded |
| 5. Registry closure | flow candidates, replacement pools | registry plus re-solved flow | zero evaluated registry misses |
| 6. Reader evaluation | final flow, fixed registry, clean answer, vLLM | native evaluation JSONL | zero registry misses and replacement failures |
| 7. Manifest | all frozen artifacts | `manifest.json` | hashes, sizes, line counts, commit, and dirty state recorded |

The immutable dependency chain is:

```text
DATA + K
  -> TOKEN_UNITS
  -> VLLM_CLEAN_TARGETS
  -> HF_SDPA_CONTRIBUTION_GRAPH
  -> PURE_GRAPH_FLOW_CANDIDATES
  -> FIXED_REPLACEMENT_REGISTRY
  -> FINAL_GRAPH_FLOW_CANDIDATES
  -> VLLM_READER_EVALUATIONS
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

This environment runs graph construction. The local Hugging Face SDPA model is
used only because contribution tracing needs gradients and intermediate
activations; it does not generate a second set of reader metrics. Attention
weights are recomputed exactly one layer at a time after backward, avoiding the
all-layer eager-attention memory peak.

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

## Retrieval contract

This repository starts from frozen retrieval records; it does not silently
re-chunk, re-embed, rerank, or shorten them. `K` selects the first `K` complete
retrieved chunks and never means a token budget.

For comparable multi-dataset experiments, freeze the same upstream retrieval
protocol before running this pipeline: chunk unit and token size, overlap,
embedding checkpoint, embedded fields, query instruction, tokenizer context
handling, vector normalization, index and similarity metric, top-k, and any
reranker. Record all of these values in the dataset `run.yaml`. A legacy
artifact built with variable native paragraphs or an artificial embedding
length cap is not directly comparable to a fixed-chunk artifact and must be
regenerated before cross-dataset aggregation.

## Services and environment variables

Start the spaCy service in the spaCy environment:

```bash
export YVETTE_SPACY_MODEL=en_core_web_lg
python scripts/spacy_annotation_server.py --host 127.0.0.1 --port 8021
```

Clean-target generation, replacement editing, and final reader evaluation use
one vLLM OpenAI-compatible endpoint:

```bash
export CAUSALITYRAG_SPACY_BASE_URL=http://127.0.0.1:8021
export CAUSALITYRAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
export CAUSALITYRAG_LLM_MODEL=qwen2.5-7b
```

The older `YVETTE_LLM_BASE_URL` and `YVETTE_LLM_MODEL` names remain supported
as fallbacks.

The checkpoint behind that served model must be the same checkpoint passed to
`build_contribution_graph.py --model-path`. These are not two reader
evaluations. vLLM is the only answer-generation and answer-change backend. The
HF SDPA load is an attribution instrument used only during graph construction.
If both would use the same GPU, first generate and save `clean_targets.jsonl`,
stop vLLM, build the graphs, and then restart vLLM for registry closure and
final evaluation.

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
  02_reader/
  03_graph/
  04_flow/
  05_registry/
  06_evaluation/
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
export RELAXED_FLOW_THRESHOLD=0.2

export RUN_DIR="${RUN_ROOT}/${DATASET}/${RUN_ID}"
export CONTEXT_DIR="${RUN_DIR}/01_context"
export READER_DIR="${RUN_DIR}/02_reader"
export GRAPH_DIR="${RUN_DIR}/03_graph"
export FLOW_DIR="${RUN_DIR}/04_flow"
export REGISTRY_DIR="${RUN_DIR}/05_registry"
export EVALUATION_DIR="${RUN_DIR}/06_evaluation"

export CONTEXT_UNITS="${CONTEXT_DIR}/context_units.jsonl"
export CLEAN_TARGETS="${READER_DIR}/clean_targets.jsonl"
export GRAPH="${GRAPH_DIR}/contribution_graph.jsonl"
export FLOW_INITIAL="${FLOW_DIR}/initial.jsonl"

mkdir -p \
  "$CONTEXT_DIR" \
  "$READER_DIR" \
  "$GRAPH_DIR" \
  "$FLOW_DIR" \
  "$REGISTRY_DIR" \
  "$EVALUATION_DIR"

cp "$CONFIG" "$RUN_DIR/run.yaml"
```

`CF_POOLS` is a JSON object with `type_pool` and optional tab-delimited
`role_pool` keys. `TYPE_RULES` is optional. If no compatible rule metadata YAML
is available, leave it empty and remove the `--type-rules "$TYPE_RULES"` line
from stages 5 and 6. Create `CONFIG` by copying
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

### 2. Freeze clean reader targets with vLLM

Start the Qwen checkpoint with vLLM under the served name exported in
`CAUSALITYRAG_LLM_MODEL`, then submit clean-answer requests concurrently:

```bash
python scripts/generate_reader_targets.py \
  --input "$DATA" \
  --out "$CLEAN_TARGETS" \
  --summary-out "$READER_DIR/clean_targets.summary.json" \
  --workers 16 \
  --n "$N" \
  --k "$K"
```

This JSONL is the only clean-answer source used downstream. It is generated
once, saved in input order, and reused as the graph target and flip baseline.
If the reader genuinely returns an empty answer, the row is frozen as
`reader_abstention_empty_answer`; it is never converted to gold, `unknown`, or
an `ok` empty graph.

### 3. Build the direct-activation absorbing contribution graph

When vLLM and graph construction share a GPU, stop vLLM before this command.
The local HF SDPA load consumes the frozen vLLM answers; it does not generate
answers itself.

```bash
python scripts/build_contribution_graph.py \
  --input "$DATA" \
  --out "$GRAPH" \
  --summary-out "$GRAPH_DIR/summary.json" \
  --model-path "$MODEL" \
  --target results \
  --target-results "$CLEAN_TARGETS" \
  --device cuda \
  --dtype bfloat16 \
  --edge-topk 6 \
  --max-receivers-per-layer 48 \
  --max-edges 5000 \
  --absorbing-flow \
  --n "$N" \
  --k "$K"
```

Each row stores the frozen `clean_answer` used as its graph target.

No per-chunk or whole-prompt truncation is configured. All retrieved text is
rendered and tokenized exactly. The only applicable sequence boundary is the
checkpoint's real model context window; exceeding it is reported as
`sequence_exceeds_model_context` instead of silently clipping evidence.

The answer-objective sink is seeded uniformly across the predictors of the
mean clean-answer-token logit. A non-abstention graph is accepted only when it
has positive answer-terminal flow and a positive context-to-answer path. Empty
or unusable graphs receive an explicit failure status. Reader abstentions are
counted separately; any actual graph failure makes the command exit nonzero.

### 4. Produce an initial contribution-flow solution

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
  --n "$N" \
  --k "$K"
```

There is no edit-budget parameter. The geometric grid always covers the full
editable-token count; `GAMMA` controls only the spacing of its internal scales.

### 5. Close the strict replacement registry

Selection never sees replacement text. The registry fixes one answer-blind
replacement per candidate and marks invalid tokens uncuttable.

#### Registry implementation

For each query, `build_replacement_registry.py` performs the following steps:

1. Collect every token ID in the strict-threshold witness and the native
   relaxed-threshold contribution-flow candidate.
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
  --n "$N" \
  --k "$K"

export FLOW="$FLOW_DIR/final.jsonl"
```

Keep `REGISTRY_FIXED_POINT` pointed at whichever numbered registry reached the
fixed point. The copy to `final.jsonl` gives every dataset the same downstream
path while preserving all numbered closure iterations.

### 6. Evaluate saved selections with vLLM

Run the native graph-threshold evaluation under two reporting scopes. With
`BETA=0.1` and `ETA=1`, the solver accepts candidates up to the relaxed
residual-flow fraction `(1 + ETA) * BETA = 0.20`. Restart the same vLLM
endpoint used in stage 2 before running.

First report the formal attack metric over queries whose saved clean answer is
an exact match to gold:

```bash
python scripts/evaluate_reader.py \
  --input "$DATA" \
  --gate "$FLOW" \
  --clean-reference "$CLEAN_TARGETS" \
  --replacement-registry "$REGISTRY" \
  --context-units "$CONTEXT_UNITS" \
  --cf-pools "$CF_POOLS" \
  --type-rules "$TYPE_RULES" \
  --out "$EVALUATION_DIR/clean_exact.jsonl" \
  --summary-out "$EVALUATION_DIR/clean_exact.summary.json" \
  --remaining-flow-threshold "$RELAXED_FLOW_THRESHOLD" \
  --clean-correct-policy exact \
  --reader-workers 16 \
  --strict-replacements \
  --k "$K"
```

Then report answer-change behavior over every input query, including queries
whose clean answer is not exact:

```bash
python scripts/evaluate_reader.py \
  --input "$DATA" \
  --gate "$FLOW" \
  --clean-reference "$CLEAN_TARGETS" \
  --replacement-registry "$REGISTRY" \
  --context-units "$CONTEXT_UNITS" \
  --cf-pools "$CF_POOLS" \
  --type-rules "$TYPE_RULES" \
  --out "$EVALUATION_DIR/all_queries.jsonl" \
  --summary-out "$EVALUATION_DIR/all_queries.summary.json" \
  --remaining-flow-threshold "$RELAXED_FLOW_THRESHOLD" \
  --clean-correct-policy exact \
  --include-clean-incorrect \
  --reader-workers 16 \
  --strict-replacements \
  --k "$K"
```

All independent edited-context requests are submitted concurrently to vLLM.
The stored stage-2 clean answer remains the baseline, so clean inference is not
repeated here.

The only evaluated selection is the contribution-flow candidate. In each
summary, `flip_rate` is conditional on a valid flow candidate and
`overall_flip_rate` uses every query in that reporting scope as its
denominator. The all-query value is an answer-change rate, not a formal attack
success rate, because it includes clean-incorrect queries.

### 7. Freeze an artifact manifest

```bash
python scripts/build_artifact_manifest.py \
  --repository . \
  --artifacts \
    "$CONTEXT_UNITS" \
    "$CLEAN_TARGETS" \
    "$GRAPH" \
    "$FLOW" \
    "$REGISTRY" \
    "$EVALUATION_DIR/clean_exact.jsonl" \
    "$EVALUATION_DIR/all_queries.jsonl" \
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
  02_reader/
    clean_targets.jsonl
    clean_targets.summary.json
  03_graph/
    contribution_graph.jsonl
    summary.json
  04_flow/
    initial.jsonl
    initial.summary.json
    iteration_01.jsonl
    iteration_01.summary.json
    ... additional closure iterations ...
    final.jsonl
    final.summary.json
  05_registry/
    iteration_01.jsonl
    iteration_01.summary.json
    ... additional closure iterations ...
    final.jsonl
  06_evaluation/
    clean_exact.jsonl
    clean_exact.summary.json
    all_queries.jsonl
    all_queries.summary.json
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
- The final solver is `geometric-k-guessing` with explicit `BETA`, `ETA`, and
  `GAMMA`; it always covers the full editable-token domain and uses no edit
  budget.
- Invalid replacements remain in the flow graph but are made uncuttable.
- Registry closure ends only when
  `evaluated_candidate_registry_misses == 0`.
- Native evaluation uses the solver's relaxed threshold
  `(1 + ETA) * BETA`.
- vLLM is the sole clean/edited answer backend; HF SDPA is used only for
  contribution tracing.
- Contexts and prompts are not silently truncated. A real model-window
  overflow or a missing context-to-answer graph path fails explicitly.

## Output and reproducibility rules

- JSONL stage outputs are immutable inputs to later stages.
- Keep the same `id`, record order, top-k value, model weights, tokenizer,
  prompt, and greedy decoding contract throughout a final run.
- Record vLLM as the reader backend and HF SDPA as the attribution backend;
  do not report them as two reader evaluations.
- Use the frozen context-unit artifact to enforce context hashes and offsets.
- Do not regenerate replacements separately for competing selectors.
- Record the final registry fixed point and artifact manifest.

## Tests

Run tests only in the configured server environments, not on the local
workstation. Unit tests that do not load the model can run in the server spaCy
or development environment:

The final test suite covers input normalization, token offsets, contextual
replacement validity, reader metrics, direct contribution graphs, projected
token flow, exact weighted min-cuts, geometric scales, registry filtering, and
saved-intervention evaluation. ARC-JSD tests are retained under `exp/`.

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
depth, clean-correct coverage, and vLLM native-threshold summary.
Cross-dataset aggregation is valid only after the
per-dataset artifacts pass the same acceptance checks.

Historical dataset-specific baselines, ablations, plotting scripts, and
protocols are retained under `exp/` for provenance. They are not stable final
entry points and may use historical terminology or artifact schemas.
