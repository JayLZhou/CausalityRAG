# ReFlow: Token-Level Counterfactual Explanations for RAG

ReFlow identifies a small set of tokens in the retrieved context whose joint
factual replacement changes the response of a fixed retrieval-augmented
generation (RAG) system. A selected set is only a search candidate until the
same reader is run on the edited context: it counts as a successful
counterfactual intervention only after direct reader verification.

This guide is the canonical reproduction guide. It specifies the frozen
protocol, the complete main experiment, every additional experiment currently
used by the project, the expected artifacts, and the audits that must pass
before numbers are reported.

## Contents

- [Reproducibility scope](#reproducibility-scope)
- [Frozen protocol](#frozen-protocol)
- [Experiment map](#experiment-map)
- [Environment](#environment)
- [Data preparation](#data-preparation)
- [Complete main experiment](#complete-main-experiment)
- [Metric definitions](#metric-definitions)
- [Additional experiments](#additional-experiments)
- [Artifact and audit requirements](#artifact-and-audit-requirements)
- [Tests and troubleshooting](#tests-and-troubleshooting)

## Reproducibility scope

### Final benchmark suite

The final main-table suite contains exactly eight datasets and 1,000 frozen
queries per dataset:

| Key | Dataset | Answer handling |
|---|---|---|
| `hotpotqa` | HotpotQA | Short-answer normalization |
| `timeqa` | TimeQA | All frozen answer components are required |
| `finqa` | FinQA | Short-answer/numeric normalization |
| `musique` | MuSiQue | Short-answer normalization |
| `quartz` | QuaRTz | Binary choice protocol; exact selected choice |
| `triviaqa` | TriviaQA | Any frozen gold alias is accepted |
| `2wiki` | 2WikiMultiHopQA | Short-answer normalization |
| `popqa` | PopQA | Any frozen gold alias is accepted |

`PubMedQA`, `QASPER`, and `MedQA` occur in older or appendix scripts, but they
are not members of the final eight-dataset main table. Do not mix their outputs
with the final suite. `scripts/rescore_table3_unified.py` and
`scripts/audit_table3_final8.py` enforce the authoritative dataset list.

### What is stored here

The repository includes code, tests, aggregate result files, rendered figures,
and audit summaries. It does not redistribute all licensed datasets, model
weights, or all 8,000 per-query graph and reader artifacts. Reproduction from
scratch therefore requires the source datasets and model weights below.
Aggregate files under `artifacts/` and `figures/` check reported values but are
not substitutes for the frozen per-query inputs.

### Portable commands and historical schedulers

Python entry points in `causalityrag/`, `exp/`, and `scripts/` are the portable
interfaces. Shell files in `scripts/launchers/` record the
original absolute paths, ports, GPU mappings, and resume logic. They are
provenance, not portable launchers; several intentionally refuse to run on a
different hostname. Review and override every path before reusing one.

Do not copy an aggregate number into a paper table unless the row-alignment,
pool-hash, execution, and metric audits in this guide pass.

## Frozen protocol

```text
source questions and corpus
  -> frozen 1,000-query manifest
  -> 384-token chunks with 64-token overlap
  -> one dense top-10 retrieval result
  -> token units and a frozen top-10 replacement pool
  -> clean top-5 reader response
  -> answer-conditioned contribution graph
  -> exact supported flow-contract frontier
  -> factual replacement and direct reader verification
  -> matched-budget baselines
  -> meaning-preserving control at the same selected positions
  -> unified metrics and fail-closed audits
```

| Component | Frozen setting |
|---|---|
| Query population | First 1,000 valid unique rows in source order |
| Retrieval | Qwen3-Embedding-0.6B, L2 normalization, cosine similarity, no reranker |
| Chunking | 384 embedding-tokenizer tokens, 64-token overlap |
| Retrieval range | Build top-10 once; main evaluation uses its top-5 prefix |
| Reader/graph backbone | Qwen2.5-7B-Instruct |
| Reader decoding | Greedy (`temperature=0`), at most 128 new tokens in formal service runs |
| Graph target | Frozen clean response; mean answer-token logit |
| Message graph | Closed positive message flow, contracted to surface-token positions |
| Receiver beam | 48 |
| Capacities | Positive capacities normalized by their query-level mean |
| ReFlow search | Exact supported breakpoint flow-contract frontier |
| Replacement seed | 0 |
| Random seeds | 0, 1, 2, 3, 4 |
| Baseline budget | Per-query ReFlow terminal token budget |

`configs/dataset_template.yaml` is the configuration checklist. CLI flags are
authoritative. Record dataset, model, tokenizer, prompt, decoding, and artifact
hashes in a manifest for every formal run.

### Shared factual replacement pool

The pool is built before any method selects tokens. The query, clean response,
and gold answer are not inputs to replacement generation.

1. Mark content tokens with POS in `{NOUN, PROPN, VERB, ADJ, ADV, NUM}`.
2. Group positions by surface form, entity type, POS, tag, morphology, and
   entity slot.
3. Generate single-token, same-type factual replacements.
4. Check contextual syntax and morphology with spaCy.
5. Reject synonyms and lexical paraphrases with WordNet.
6. Use a blind semantic judgment to reject paraphrases, malformed edits, and
   semantic-type drift.
7. Expand typed candidates to positions, sort, freeze, and hash the JSONL.

Evaluation is read-only and `--expected-pool-sha256` is mandatory. A method
may not generate a private replacement, skip a selected token, or backfill it
with a later-ranked token. Candidate choice is a deterministic hash of global
token-position ID and replacement seed.

If a typed key is unresolved, continue generation. The optional
`--exclude-unresolved` flag is valid only for a declared common-domain
exclusion frozen before any method runs; it must exclude the same positions
from every method.

### Meaning-preserving control

Factual interventions are paired with controls at exactly the same selected
positions. The control pool uses validated synonyms/aliases and deterministic
information-preserving renderings. Numeric/date positions without a faithful
paraphrase remain unchanged and are recorded as ineligible. This pool is also
frozen and hash-checked.

## Experiment map

| Result family | Population | Entry points | Shipped aggregate evidence/status |
|---|---:|---|---|
| Main comparison | 8 datasets x 1,000 | `run_table3_rankers.py`, `run_table3_reader_stage.py`, `rescore_table3_unified.py` | Audited final-eight snapshot in `artifacts/table3/final8_20260811/` |
| Budget to match ReFlow | Metric-specific clean-correct queries | `evaluate_fixed_budget_acc.py`, `run_exact_budget_match.py` | Final-eight files in `artifacts/token_shapley_final8/exact_budget*/` |
| Token Shapley | 8 datasets x 1,000 | `run_token_leverage_shap.py`, `prepare_shapley_eval_shards.py` | Final-eight files in `artifacts/token_shapley_final8/` |
| Search analyses | Frozen main artifacts | `build_selection_protocol_frontiers.py`, `run_residual_flow_fidelity.py`, `run_small_domain_oracle.py` | Regenerate and audit against the current final-eight manifests |
| Retrieval depth | 8 x 1,000, k=1/3/5/10 | Standard ReFlow stages per prefix | Audited final-eight k=1/3/10 bundles in `artifacts/retrieval_depth_qwen25_v1/`; k=5 is the main run |
| Receiver beam | 8 x 1,000, fixed k=5 | `build_contribution_graph_beams.py` plus standard evaluation | Regenerate from frozen main inputs; no complete portable bundle is committed |
| Reader sensitivity | 4 readers x 8 x 1,000 | Standard pipeline per model, `render_cross_model_sensitivity.py` | The committed `artifacts/model_sensitivity_v1/` snapshot predates the PopQA final-suite update; do not cite it as final-eight evidence |
| Poisoning/defense | 100 clean-correct HotpotQA queries | `run_security_pareto_attacks.py`, `run_security_pareto_defenses.py` | Aligned summaries in `figures/security_case_study/` |
| Evidence provenance | Matched covered queries | `run_sentence_sufficiency_case_study.py` | The committed `artifacts/sentence_sufficiency/` bundle uses an older suite; regenerate the current final-eight result |

Dynamic-cache, evidence-debugging, and reliability-triage scripts are
additional case studies, not part of the final main table.

## Environment

### Hardware and software

A CUDA GPU is required for graph construction and model-internal rankers. The
original formal runs used A100-class GPUs. One GPU is sufficient for sequential
Qwen2.5-7B runs; `run_table3_rankers.py` expects two visible GPUs for two
500-query shards. Mistral-Nemo-12B may require `--device-map balanced`.

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -r requirements-dev.txt
python -m nltk.downloader wordnet omw-1.4
python -m spacy download en_core_web_lg
```

`requirements-gpu.txt` pins `transformers==4.52.4`. The original reader used
vLLM, installed separately because its wheel depends on CUDA/PyTorch. Record
the exact vLLM, CUDA, driver, PyTorch, tokenizer, and model revisions. Do not
claim bitwise reproduction when these differ.

Run tests before a formal job:

```bash
python -m pytest -q tests
```

### Services

The pipeline expects:

1. an OpenAI-compatible chat endpoint at `http://127.0.0.1:8000/v1`;
2. an OpenAI-compatible Qwen3-Embedding-0.6B endpoint at
   `http://127.0.0.1:8017/v1`;
3. the repository spaCy service at `http://127.0.0.1:8021`.

Start spaCy:

```bash
python scripts/spacy_annotation_server.py \
  --host 127.0.0.1 --port 8021 --model en_core_web_lg
```

One possible reader launch is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b \
  --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
  --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
  --disable-log-requests --port 8000
```

Any embedding service is acceptable only if it implements the
OpenAI-compatible `/v1/embeddings` response used by
`prepare_dataset_retrieval.py` and serves the frozen embedding model.

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8017/v1/models
curl --fail http://127.0.0.1:8021/health

export CAUSALITYRAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
export CAUSALITYRAG_LLM_MODEL=qwen2.5-7b
export CAUSALITYRAG_SPACY_BASE_URL=http://127.0.0.1:8021
export CAUSALITYRAG_READER_MAX_TOKENS=128
export CAUSALITYRAG_READER_MODE=short_answer
```

For QuaRTz, use `CAUSALITYRAG_READER_MODE=quartz` in every stage.

### Formal-run preflight

Do not start a 1,000-query run until all of the following are true:

1. `git status --porcelain` is empty and the intended commit is recorded.
2. Dataset/corpus snapshots, model weights, tokenizers, prompts, and the
   package environment have immutable version identifiers or SHA-256 hashes.
3. The reader, embedding, and spaCy health checks above pass from the same
   process environment that will launch the experiment.
4. A separate 10-query end-to-end smoke run completes with aligned IDs, no
   pool-hash mismatch, no failed replacement, and the intended reader mode.
5. All protocol-sensitive tests in the final section pass.
6. Sufficient GPU memory and disk space are available for every planned
   shard; the experiment must not change k, precision, beam, batch semantics,
   or model revision after a failure.
7. Output directories are empty or are verified resumptions of the same
   manifest. Never resume into artifacts created by a different commit,
   dataset snapshot, model revision, pool hash, or prompt.

## Data preparation

This repository consumes normalized question and corpus files; it does not
download or redistribute the upstream benchmarks. A reproducibility release
must therefore accompany each dataset with a manifest containing the upstream
project URL/DOI, release or commit, split, license, original file hashes,
normalization script/commit, normalized file hashes, and the frozen 1,000 IDs
in order. Server-local paths in historical schedulers are provenance only. If
these manifests and source-access instructions are not released, a third party
cannot reproduce retrieval from scratch and the result should be described as
artifact verification rather than full reproduction.

Question files may be JSON or JSONL. Each retained row needs a unique ID under
`id`, `_id`, `qid`, or `question_id`, plus nonempty `question` and `answer`.
Optional `answers` and `answer_aliases` are preserved. Corpus rows need `text`
and may include `title` and an ID.

```json
{"id":"q1","question":"...","answer":"...","answers":["..."],"answer_aliases":["..."]}
```

```json
{"id":"d1","title":"...","text":"..."}
```

Empty IDs/questions/answers are removed before freezing the first 1,000 valid
unique rows. Source order is part of the experiment and must not be shuffled.

Normalize PopQA aliases first:

```bash
python scripts/prepare_popqa_bundle.py \
  --questions /path/to/popqa/questions.jsonl \
  --corpus /path/to/popqa/corpus.jsonl \
  --out-dir /path/to/normalized/popqa \
  --manifest-out /path/to/normalized/popqa/manifest.json
```

Freeze retrieval:

```bash
python scripts/prepare_dataset_retrieval.py \
  --dataset DATASET_KEY \
  --questions /path/to/questions.jsonl \
  --corpus /path/to/corpus.jsonl \
  --out-root /path/to/out/DATASET_KEY \
  --tokenizer-path /path/to/Qwen3-Embedding-0.6B \
  --n 1000 --top-k 10 --chunk-size 384 --overlap 64 \
  --embedding-base-url http://127.0.0.1:8017/v1 \
  --embedding-model Qwen3-Embedding-0.6B \
  --embedding-batch-size 32
```

Expected outputs are `retrieval/queries_1000.jsonl`,
`corpus_chunks_384t_o64.jsonl`, `top10_1000.jsonl`, its summary, and
`index/chunk_embeddings.npy`. Require exactly 1,000 unique IDs and ten ordered
chunks per row. Gold titles are used only for post-hoc retrieval recall.

Run a 10-query smoke test in a separate root. The script keeps formal filenames
even with `--n 10`, so never point a smoke test at the formal directory.

## Complete main experiment

Repeat this pipeline for every final dataset key. Set paths once and keep them
fixed throughout the run:

```bash
export DATASET_KEY=hotpotqa
export EXP_ROOT=/path/to/out
export DATASET_ROOT="$EXP_ROOT/$DATASET_KEY"
export RETRIEVAL="$DATASET_ROOT/retrieval/top10_1000.jsonl"
export TOKEN_UNITS="$DATASET_ROOT/inputs/token_units_top10_1000.jsonl"
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export READER_URL=http://127.0.0.1:8000/v1
export READER_MODEL=qwen2.5-7b
export SPACY_URL=http://127.0.0.1:8021
export CAUSALITYRAG_READER_MAX_TOKENS=128
export CAUSALITYRAG_READER_MODE=short_answer
```

Set `CAUSALITYRAG_READER_MODE=quartz` for QuaRTz.

### 1. Build token units over top-10

```bash
mkdir -p "$DATASET_ROOT/inputs"
python scripts/build_context_units.py \
  --input "$RETRIEVAL" \
  --out "$TOKEN_UNITS" \
  --summary-out "$DATASET_ROOT/inputs/token_units_top10_1000.summary.json" \
  --start 0 --n 1000 --k 10 --workers 16 \
  --backend service --spacy-base-url "$SPACY_URL"
```

Top-10 tokenization fixes the global token-position IDs used by the main and
retrieval-depth experiments.

### 2. Generate and freeze the factual pool

```bash
export POOL_DIR="$DATASET_ROOT/replacements/shared_pool_top10_v1"
mkdir -p "$POOL_DIR"

python scripts/build_shared_replacement_inventory.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --positions-out "$POOL_DIR/positions.jsonl" \
  --typed-keys-out "$POOL_DIR/typed_keys.jsonl" \
  --manifest-out "$POOL_DIR/inventory.json" \
  --n 1000 --k 10 --examples-per-key 3

python scripts/generate_shared_typed_replacement_pool.py \
  --typed-keys "$POOL_DIR/typed_keys.jsonl" \
  --seed "$POOL_DIR/typed_candidates.jsonl" \
  --out "$POOL_DIR/typed_candidates.jsonl" \
  --unresolved-out "$POOL_DIR/unresolved.jsonl" \
  --manifest-out "$POOL_DIR/generation.json" \
  --workers 12 --batch-size 24 --max-candidates 5 \
  --generation-rounds 8 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL" \
  --spacy-base-url "$SPACY_URL"
```

The generator is resumable. Rerun with the same files and a larger
`--attempt-offset` until every typed key has a candidate. Inspect
`generation.json` and `unresolved.jsonl` after each pass.

```bash
python scripts/freeze_shared_replacement_pool.py \
  --positions "$POOL_DIR/positions.jsonl" \
  --typed-candidates "$POOL_DIR/typed_candidates.jsonl" \
  --out "$POOL_DIR/shared_pool.jsonl" \
  --manifest-out "$POOL_DIR/shared_pool.manifest.json"

export FACTUAL_POOL="$POOL_DIR/shared_pool.jsonl"
export FACTUAL_POOL_SHA=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["pool_sha256"])' \
  "$POOL_DIR/shared_pool.manifest.json")
```

Require `frozen=true`, `coverage=1.0`, and equality between the stored and
actual pool hashes before continuing.

### 3. Freeze clean top-5 answers

```bash
export CLEAN="$DATASET_ROOT/inputs/clean_reference_top5_1000.jsonl"
python scripts/generate_reader_targets.py \
  --input "$RETRIEVAL" --out "$CLEAN" \
  --summary-out "$DATASET_ROOT/inputs/clean_reference_top5_1000.summary.json" \
  --start 0 --n 1000 --k 5 --workers 16 --retries 3 \
  --base-url "$READER_URL" --served-model "$READER_MODEL"
```

The graph target must be this frozen response. `--target gold` is diagnostic
only and is not allowed for reported main results.

### 4. Build the contribution graph

```bash
mkdir -p "$DATASET_ROOT/graphs"
export GRAPH="$DATASET_ROOT/graphs/contribution_graph_top5_1000.jsonl"

CUDA_VISIBLE_DEVICES=0 python scripts/build_contribution_graph.py \
  --input "$RETRIEVAL" --out "$GRAPH" \
  --summary-out "$DATASET_ROOT/graphs/contribution_graph_top5_1000.summary.json" \
  --model-path "$MODEL_PATH" --start 0 --n 1000 --k 5 \
  --target results --target-results "$CLEAN" \
  --device cuda --dtype bfloat16 --max-receivers-per-layer 48
```

For two GPUs, run `--start 0 --n 500` and `--start 500 --n 500`, then use
`scripts/merge_contribution_graph_shards.py`. Never concatenate shards without
validating IDs and frozen retrieval order.

### 5. Produce baseline rankings

Rankers never edit context. Random rankings are produced by the common reader
evaluator in Step 7.

```bash
mkdir -p "$DATASET_ROOT/methods/baselines"

CUDA_VISIBLE_DEVICES=0 python exp/run_attention.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --out "$DATASET_ROOT/methods/baselines/attention_top5_1000.jsonl" \
  --summary-out "$DATASET_ROOT/methods/baselines/attention_top5_1000.summary.json" \
  --model-path "$MODEL_PATH" --start 0 --n 1000 --k 5 \
  --device cuda --dtype bfloat16

CUDA_VISIBLE_DEVICES=0 python exp/run_gradient_attribution_baselines.py \
  --input "$RETRIEVAL" --clean-reference "$CLEAN" \
  --context-units "$TOKEN_UNITS" \
  --out "$DATASET_ROOT/methods/baselines/gradient_x_input_top5_1000.jsonl" \
  --summary-out "$DATASET_ROOT/methods/baselines/gradient_x_input_top5_1000.summary.json" \
  --method gradient_x_input --model-path "$MODEL_PATH" \
  --start 0 --n 1000 --k 5 --top-k 32 \
  --device cuda --dtype bfloat16

CUDA_VISIBLE_DEVICES=0 python exp/run_gradient_attribution_baselines.py \
  --input "$RETRIEVAL" --clean-reference "$CLEAN" \
  --context-units "$TOKEN_UNITS" \
  --out "$DATASET_ROOT/methods/baselines/integrated_gradients_top5_1000.jsonl" \
  --summary-out "$DATASET_ROOT/methods/baselines/integrated_gradients_top5_1000.summary.json" \
  --method integrated_gradients --ig-steps 16 --model-path "$MODEL_PATH" \
  --start 0 --n 1000 --k 5 --top-k 32 \
  --device cuda --dtype bfloat16

CUDA_VISIBLE_DEVICES=0 python exp/run_mirage.py \
  --input "$RETRIEVAL" --clean-reference "$CLEAN" \
  --units-cache "$TOKEN_UNITS" \
  --out "$DATASET_ROOT/methods/baselines/mirage_top5_1000.jsonl" \
  --summary-out "$DATASET_ROOT/methods/baselines/mirage_top5_1000.summary.json" \
  --model-path "$MODEL_PATH" --start 0 --n 1000 --k 5 \
  --cti-std-threshold 1.0 --device cuda --dtype bfloat16

CUDA_VISIBLE_DEVICES=0 python exp/run_arc_jsd.py \
  --input "$RETRIEVAL" --clean-reference "$CLEAN" \
  --out "$DATASET_ROOT/methods/baselines/arc_jsd_top5_1000.jsonl" \
  --summary-out "$DATASET_ROOT/methods/baselines/arc_jsd_top5_1000.summary.json" \
  --model-path "$MODEL_PATH" --start 0 --n 1000 --k 5 \
  --batch-size 4 --feature-top-k 64 --routing-last-layers 8 \
  --spacy-base-url "$SPACY_URL" --device cuda --dtype bfloat16
```

The two-GPU resumable driver is:

```bash
python scripts/run_table3_rankers.py \
  --out-root "$EXP_ROOT" --model-path "$MODEL_PATH" \
  --datasets "$DATASET_KEY" --n 1000
```

### 6. Build and verify the ReFlow frontier

```bash
mkdir -p "$DATASET_ROOT/methods/reflow"
export FRONTIER="$DATASET_ROOT/methods/reflow/frontier_top5_1000.jsonl"
export REFLOW_RESULTS="$DATASET_ROOT/methods/reflow/results_top5_1000.jsonl"

python exp/run_reflow.py \
  --input "$RETRIEVAL" --graphs "$GRAPH" --units-cache "$TOKEN_UNITS" \
  --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$FRONTIER" \
  --summary-out "$DATASET_ROOT/methods/reflow/frontier_top5_1000.summary.json" \
  --start 0 --n 1000 --k 5

python scripts/evaluate_reflow.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --frontier "$FRONTIER" --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$REFLOW_RESULTS" \
  --summary-out "$DATASET_ROOT/methods/reflow/results_top5_1000.summary.json" \
  --n 1000 --k 5 --workers 32 --replacement-seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

`--record-empty-frontier` is only for a targeted repair workflow. A final run
must repair and re-evaluate structural failures rather than treating them as
successful empty explanations.

### 7. Evaluate matched-budget baselines

```bash
mkdir -p "$DATASET_ROOT/audits/final_top10pool_k5"
export BASELINE_RESULTS="$DATASET_ROOT/audits/final_top10pool_k5/baselines_1000.jsonl"

python scripts/evaluate_matched_budget_baselines.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --reflow-results "$REFLOW_RESULTS" \
  --scores "attention=$DATASET_ROOT/methods/baselines/attention_top5_1000.jsonl" \
  --scores "gradient_x_input=$DATASET_ROOT/methods/baselines/gradient_x_input_top5_1000.jsonl" \
  --scores "integrated_gradients=$DATASET_ROOT/methods/baselines/integrated_gradients_top5_1000.jsonl" \
  --scores "mirage=$DATASET_ROOT/methods/baselines/mirage_top5_1000.jsonl" \
  --scores "arc_jsd=$DATASET_ROOT/methods/baselines/arc_jsd_top5_1000.jsonl" \
  --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$BASELINE_RESULTS" \
  --summary-out "$DATASET_ROOT/audits/final_top10pool_k5/baselines_1000.summary.json" \
  --random-seeds 0,1,2,3,4 --replacement-seed 0 \
  --n 1000 --k 5 --workers 32 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

Every baseline must actually replace exactly the ReFlow terminal number of
unique tokens on each valid-clean query. An uncovered ranked token is a
protocol violation, not permission to select a later token.

### 8. Build and evaluate the control pool

```bash
mkdir -p "$DATASET_ROOT/controls"
export CONTROL_POOL="$DATASET_ROOT/controls/paraphrase_pool_top5_v2.jsonl"
export CONTROL_RESULTS="$DATASET_ROOT/controls/paraphrase_results_top5_1000_v2.jsonl"

python scripts/build_paraphrase_control_pool.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --reflow-results "$REFLOW_RESULTS" \
  --baseline-results "$BASELINE_RESULTS" \
  --out "$CONTROL_POOL" \
  --manifest-out "$DATASET_ROOT/controls/paraphrase_pool_top5_v2.manifest.json" \
  --unresolved-out "$DATASET_ROOT/controls/paraphrase_pool_top5_v2.unresolved.jsonl" \
  --n 1000 --batch-size 16 --workers 32 \
  --attempts 8 --max-candidates 3 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"

export CONTROL_POOL_SHA=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["pool_sha256"])' \
  "$DATASET_ROOT/controls/paraphrase_pool_top5_v2.manifest.json")

python scripts/evaluate_paraphrase_controls.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --reflow-results "$REFLOW_RESULTS" \
  --baseline-results "$BASELINE_RESULTS" \
  --paraphrase-pool "$CONTROL_POOL" \
  --expected-pool-sha256 "$CONTROL_POOL_SHA" \
  --out "$CONTROL_RESULTS" \
  --summary-out "$DATASET_ROOT/controls/paraphrase_results_top5_1000_v2.summary.json" \
  --n 1000 --k 5 --workers 32 --replacement-seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

If the manifest is incomplete, rerun the builder with
`--existing-pool "$CONTROL_POOL"` until all required positions are resolved or
explicitly declared ineligible under the common control contract.

### 9. Score, render, and audit all eight datasets

After graphs, frontiers, pools, and rankers exist, the convenience driver runs
the reader-heavy stages and preliminary summaries:

```bash
python scripts/run_table3_reader_stage.py \
  --out-root "$EXP_ROOT" \
  --datasets hotpotqa timeqa finqa musique quartz triviaqa 2wiki popqa \
  --workers 32 --llm-base-url "$READER_URL" \
  --llm-model "$READER_MODEL"
```

The unified scorer is the final metric authority:

```bash
python scripts/rescore_table3_unified.py \
  --root "$EXP_ROOT" --out "$EXP_ROOT/table3_unified.json"

python scripts/render_table3_rows.py \
  --metrics-root "$EXP_ROOT" \
  --unified-json "$EXP_ROOT/table3_unified.json" \
  --out "$EXP_ROOT/table3_rows.tex"

python scripts/audit_table3_final8.py \
  --root "$EXP_ROOT" --out "$EXP_ROOT/table3_audit.json"

python scripts/audit_table3_baseline_execution.py \
  --root "$EXP_ROOT" \
  --out "$EXP_ROOT/table3_baseline_execution_audit.json"

python scripts/audit_table3_token_counts.py \
  --root "$EXP_ROOT" --out "$EXP_ROOT/table3_token_count_audit.json"
```

The result is ready only when every audit exits successfully.

## Metric definitions

`scripts/rescore_table3_unified.py` recomputes all metrics from per-query
outputs:

- **Ans-FR:** normalized edited response differs from the clean response.
  Ans-FR alone cannot determine whether a correct response became incorrect.
- **F1-FR:** on queries with clean token F1=1, edited F1 is lower.
- **EM-FR:** on the clean exact-match population, edited exact match is lost.
- **Acc-FR:** on the benchmark-specific clean-correct population, the edited
  answer is incorrect.
- **Ans/F1/EM/Acc-CFR:** paired per-query factual flip minus the corresponding
  meaning-preserving-control flip, averaged on valid paired outputs.

TriviaQA and PopQA accept any frozen alias. TimeQA treats frozen answer
components conjunctively. QuaRTz uses exact binary-choice matching. Invalid
clean answers are excluded. Invalid factual/control outputs are excluded from
the applicable denominator; a legal but unexecuted intervention contributes a
zero flip. Always report stored numerators and denominators, never reconstruct
them from rounded percentages.

The ReFlow terminal budget is query-specific. A successful query uses the
verified-flip attempt's token count; a failed query uses the largest saved
terminal attempt. Invalid-clean queries are excluded. The token-count audit
enforces this contract.

## Additional experiments

Every experiment below must reuse frozen query IDs, retrieval rows, token IDs,
factual pool, replacement seed, prompts, and metric code unless that component
is explicitly varied.

### A. Fixed-budget and budget-to-match

This experiment asks how many top-ranked tokens a baseline needs to match
ReFlow's Acc-FR on the same clean-correct population.

```bash
python scripts/evaluate_fixed_budget_acc.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --frontier "$FRONTIER" \
  --scores "mirage=$DATASET_ROOT/methods/baselines/mirage_top5_1000.jsonl" \
  --scores "arc_jsd=$DATASET_ROOT/methods/baselines/arc_jsd_top5_1000.jsonl" \
  --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$DATASET_ROOT/audits/fixed_budget_acc/per_query_1000.jsonl" \
  --summary-out "$DATASET_ROOT/audits/fixed_budget_acc/summary_1000.json" \
  --budgets 1,2,4,8,16,32,64,128 \
  --replacement-seed 0 --n 1000 --k 5 --workers 32 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL" \
  --reader-max-tokens 128
```

Refine the first matching integer budget rather than interpolating:

```bash
python scripts/run_exact_budget_match.py \
  --datasets hotpotqa,timeqa,finqa,musique,quartz,triviaqa,2wiki,popqa \
  --release "$PWD" --out-root "$EXP_ROOT" \
  --endpoint "$READER_URL" --workers 32 \
  --max-budget 512 --budget-block 16
```

`render_figure8_final8.py` writes a provenance JSON and LaTeX coordinates. It
requires explicit QuaRTz and PopQA inputs because they use benchmark-aware
rescoring. Run its `--help` and pass all final summaries; do not reuse results
from an older dataset suite.

### B. Token Shapley baseline

The final comparison uses the attention-filtered TokenLeverage SHAP
implementation with 128 attention candidates, a model-evaluation budget of
three times the candidate count, coalition batch size 32, and seed 42.

```bash
CUDA_VISIBLE_DEVICES=0 python exp/run_token_leverage_shap.py \
  --input "$RETRIEVAL" --clean-reference "$CLEAN" \
  --context-units "$TOKEN_UNITS" --model-path "$MODEL_PATH" \
  --out "$DATASET_ROOT/methods/baselines/token_shapley_top5_1000.jsonl" \
  --summary-out "$DATASET_ROOT/methods/baselines/token_shapley_top5_1000.summary.json" \
  --start 0 --n 1000 --k 5 \
  --top-k 128 --attention-candidates 128 \
  --budget-multiplier 3 --coalition-batch-size 32 \
  --seed 42 --device cuda --dtype bfloat16 --resume
```

Evaluate this ranking at the per-query ReFlow terminal budget, then build a
control pool over the exact Shapley-selected positions:

```bash
export SHAPLEY_ROOT="$DATASET_ROOT/analysis/token_shapley"
mkdir -p "$SHAPLEY_ROOT"

python scripts/evaluate_matched_budget_baselines.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --reflow-results "$REFLOW_RESULTS" \
  --scores "token_leverage_shap=$DATASET_ROOT/methods/baselines/token_shapley_top5_1000.jsonl" \
  --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$SHAPLEY_ROOT/evaluation_1000.jsonl" \
  --summary-out "$SHAPLEY_ROOT/evaluation_1000.summary.json" \
  --random-seeds "" --replacement-seed 0 \
  --n 1000 --k 5 --workers 32 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"

python scripts/prepare_shapley_eval_shards.py selected \
  --output "$SHAPLEY_ROOT" --shards 1 --shard-size 1000

python scripts/build_paraphrase_pool_for_selected_units.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --selected-manifest "$SHAPLEY_ROOT/selected_0.jsonl" \
  --existing-pool "$CONTROL_POOL" \
  --out "$SHAPLEY_ROOT/paraphrase_pool.jsonl" \
  --manifest-out "$SHAPLEY_ROOT/paraphrase_pool.manifest.json" \
  --unresolved-out "$SHAPLEY_ROOT/paraphrase_pool.unresolved.jsonl" \
  --batch-size 16 --workers 24 --attempts 6 --max-candidates 3 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"

export SHAPLEY_CONTROL_SHA=$(python -c \
  'import sys; from causalityrag.shared_replacement_pool import file_sha256; print(file_sha256(sys.argv[1]))' \
  "$SHAPLEY_ROOT/paraphrase_pool.jsonl")

python scripts/evaluate_paraphrase_controls.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --reflow-results "$REFLOW_RESULTS" \
  --baseline-results "$SHAPLEY_ROOT/evaluation_1000.jsonl" \
  --paraphrase-pool "$SHAPLEY_ROOT/paraphrase_pool.jsonl" \
  --expected-pool-sha256 "$SHAPLEY_CONTROL_SHA" \
  --out "$SHAPLEY_ROOT/control_1000.jsonl" \
  --summary-out "$SHAPLEY_ROOT/control_1000.summary.json" \
  --skip-reflow --n 1000 --k 5 --workers 32 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

Require 1,000 aligned ranking/factual/control rows, exact positive-budget
equality with ReFlow, control coverage 1.0, no unresolved positions, and no
unreported reader failures. Compute all eight metrics with:

```bash
python -m scripts.summarize_shapley_eight_metrics \
  --dataset "$DATASET_KEY" --retrieval "$RETRIEVAL" \
  --factual "$SHAPLEY_ROOT/evaluation_1000.jsonl" \
  --control "$SHAPLEY_ROOT/control_1000.jsonl" \
  --out "$SHAPLEY_ROOT/eight_metrics.json"
```

Large runs use `scripts/prepare_shapley_eval_shards.py`. Exact original
schedulers are the `run_final8_token_shapley_rank_*` and
`run_final8_token_shapley_eval_*` scripts. Render paper inputs with
`render_shapley_paper_artifacts.py`. Shipped metrics, timing, and exact-budget
summaries are under `artifacts/token_shapley_final8/`.

`exp/run_token_shapley.py` and `exp/run_token_svarm.py` are alternative
estimators. Do not label them as the final TokenLeverage SHAP run unless their
distinct method and hyperparameters are explicitly reported.

### C. Search-protocol and algorithm analyses

#### One-shot price, fixed grid, and exact frontier

Materialize alternative frontiers without rebuilding the graph:

```bash
export ANALYSIS_ROOT="$DATASET_ROOT/analysis"
mkdir -p "$ANALYSIS_ROOT/protocols"

python scripts/build_selection_protocol_frontiers.py \
  --frontier "$FRONTIER" \
  --one-shot-out "$ANALYSIS_ROOT/protocols/one_shot_frontier.jsonl" \
  --sweep-out "$ANALYSIS_ROOT/protocols/fixed_grid_frontier.jsonl" \
  --one-shot-price 1.0 \
  --grid-min-exp -8 --grid-max-exp 8 --grid-step 0.5

python scripts/evaluate_reflow.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --frontier "$ANALYSIS_ROOT/protocols/one_shot_frontier.jsonl" \
  --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$ANALYSIS_ROOT/protocols/one_shot_results.jsonl" \
  --summary-out "$ANALYSIS_ROOT/protocols/one_shot_results.summary.json" \
  --n 1000 --k 5 --workers 32 --replacement-seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"

python scripts/evaluate_reflow.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --frontier "$ANALYSIS_ROOT/protocols/fixed_grid_frontier.jsonl" \
  --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$ANALYSIS_ROOT/protocols/fixed_grid_results.jsonl" \
  --summary-out "$ANALYSIS_ROOT/protocols/fixed_grid_results.summary.json" \
  --n 1000 --k 5 --workers 32 --replacement-seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

The three protocols must be evaluated on the same 1,000 rows. Do not compare
frontier-construction candidates directly: compare only saved reader-verified
results from one-shot, fixed-grid, and exact-frontier search.

#### Residual-flow fidelity

```bash
python scripts/run_residual_flow_fidelity.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --graphs "$GRAPH" --shared-pool "$FACTUAL_POOL" \
  --out "$DATASET_ROOT/analysis/fidelity/probes.jsonl" \
  --summary-out "$DATASET_ROOT/analysis/fidelity/summary.json" \
  --n-queries 100 --budgets 1 3 5 --trials 10 \
  --workers 16 --k 5 --seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

Report query count, budgets, trials, and seed together. Never compare one
method's best trial with another method's mean trial.

#### Exhaustive micro-domain oracle

```bash
python scripts/run_small_domain_oracle.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --graphs "$GRAPH" --shared-pool "$FACTUAL_POOL" \
  --out "$DATASET_ROOT/analysis/oracle/per_query.jsonl" \
  --summary-out "$DATASET_ROOT/analysis/oracle/summary.json" \
  --n-queries 100 --domain-size 12 --workers 16 --k 5 --seed 0 \
  --success-metric answer-change \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

Use `--prepare-only` first to freeze and report the eligible micro-domain
population. ReFlow and exhaustive search must use the same success predicate.

After the protocol, fidelity, and restricted-oracle files exist for the
designated analysis dataset, aggregate them together with scaling statistics
from all eight main runs. `summarize_reflow_analysis.py` reads
graph/frontier/result files from every dataset for the scaling portion. Its
HotpotQA loader accepts the historical publication names; for a fresh
canonical run, provide equivalent links to the audited canonical files before
invoking it:

```bash
ln -s "$EXP_ROOT/hotpotqa/graphs/contribution_graph_top5_1000.jsonl" \
  "$EXP_ROOT/hotpotqa/graphs/contribution_graph_token_label_1000.jsonl"
ln -s "$EXP_ROOT/hotpotqa/graphs/contribution_graph_top5_1000.jsonl" \
  "$EXP_ROOT/hotpotqa/graphs/contribution_graph_closed_beam48_1000.jsonl"
ln -s "$EXP_ROOT/hotpotqa/methods/reflow/frontier_top5_1000.jsonl" \
  "$EXP_ROOT/hotpotqa/audits/final_top10pool_k5/reflow_frontier_1000.jsonl"
ln -s "$EXP_ROOT/hotpotqa/methods/reflow/results_top5_1000.jsonl" \
  "$EXP_ROOT/hotpotqa/audits/final_top10pool_k5/reflow_1000_v2.jsonl"

python scripts/summarize_reflow_analysis.py \
  --out-root "$EXP_ROOT" \
  --one-shot "$ANALYSIS_ROOT/protocols/one_shot_results.jsonl" \
  --sweep "$ANALYSIS_ROOT/protocols/fixed_grid_results.jsonl" \
  --frontier "$REFLOW_RESULTS" \
  --fidelity-summary "$ANALYSIS_ROOT/fidelity/summary.json" \
  --oracle-summary "$ANALYSIS_ROOT/oracle/summary.json" \
  --out "$EXP_ROOT/reflow_analysis.json"
```

Create the links only if those historical names do not already exist and only
after checking that the canonical graph is the one used by the final
frontier. The aggregator reports mean graph nodes, editable tokens, positive
edges, frontier candidates, max-flow calls, graph/frontier/reader time, and
reader calls. These are saved-stage times; they exclude service startup,
queueing, and data download.

#### Nested-prefix exhaustive comparison

`run_prefix_exhaustive_comparison.py` and
`evaluate_prefix_answer_frontier.py` compare exhaustive search and ReFlow over
nested prefixes of the ten highest positive incident-score tokens. First
freeze `COHORT_RESULTS` as a JSONL containing only preselected clean-F1-correct
query IDs that have ten positive incident-score tokens eligible under both the
factual and control contracts. Eligibility and cohort selection must be
computed before either search is run. Record the selection rule, requested
size, realized size, exclusions by reason, and SHA-256. Do not silently replace
an ineligible query after observing either method's result.

```bash
export PREFIX_ROOT="$ANALYSIS_ROOT/prefix_n1_10"
export COHORT_RESULTS=/path/to/frozen_clean_f1_cohort.jsonl
mkdir -p "$PREFIX_ROOT"

python scripts/run_prefix_exhaustive_comparison.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --graphs "$GRAPH" --shared-pool "$FACTUAL_POOL" \
  --cohort-results "$COHORT_RESULTS" \
  --required-units-out "$PREFIX_ROOT/required_units.jsonl" \
  --out "$PREFIX_ROOT/results.jsonl" \
  --summary-out "$PREFIX_ROOT/summary.json" \
  --max-n 10 --workers 32 --k 5 --seed 0 --prepare-only

python scripts/build_paraphrase_pool_for_selected_units.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --selected-manifest "$PREFIX_ROOT/required_units.jsonl" \
  --existing-pool "$CONTROL_POOL" \
  --out "$PREFIX_ROOT/paraphrase_pool.jsonl" \
  --manifest-out "$PREFIX_ROOT/paraphrase_pool.manifest.json" \
  --unresolved-out "$PREFIX_ROOT/paraphrase_pool.unresolved.jsonl" \
  --batch-size 16 --workers 24 --attempts 6 --max-candidates 3 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"

python scripts/run_prefix_exhaustive_comparison.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --graphs "$GRAPH" --shared-pool "$FACTUAL_POOL" \
  --cohort-results "$COHORT_RESULTS" \
  --paraphrase-pool "$PREFIX_ROOT/paraphrase_pool.jsonl" \
  --required-units-out "$PREFIX_ROOT/required_units.jsonl" \
  --out "$PREFIX_ROOT/results.jsonl" \
  --summary-out "$PREFIX_ROOT/summary.json" \
  --max-n 10 --workers 32 --k 5 --seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"

python scripts/evaluate_prefix_answer_frontier.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --graphs "$GRAPH" --shared-pool "$FACTUAL_POOL" \
  --prefix-results "$PREFIX_ROOT/results.jsonl" \
  --paraphrase-pool "$PREFIX_ROOT/paraphrase_pool.jsonl" \
  --out "$PREFIX_ROOT/answer_results.jsonl" \
  --summary-out "$PREFIX_ROOT/answer_summary.json" \
  --max-n 10 --workers 32 --k 5 --seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

Require control-pool coverage 1.0 and an empty unresolved file before either
search runs. The exhaustive search is cardinality ordered; ReFlow is evaluated
on the same prefix and replacement candidates. Reader-call accounting must be
reported as independent logical verifications, without credit from cache
sharing between methods.

After all eight datasets finish, validate F1 results and aggregate the answer
frontier (repeat the two dataset arguments for every final key):

```bash
python scripts/validate_prefix_exhaustive_results.py \
  --dataset "hotpotqa=$EXP_ROOT/hotpotqa/analysis/prefix_n1_10=COUNT" \
  --dataset "popqa=$EXP_ROOT/popqa/analysis/prefix_n1_10=COUNT" \
  --out "$EXP_ROOT/prefix_exhaustive_validation.json"

python scripts/aggregate_prefix_answer_frontier.py \
  --summary "hotpotqa=$EXP_ROOT/hotpotqa/analysis/prefix_n1_10/answer_summary.json" \
  --summary "popqa=$EXP_ROOT/popqa/analysis/prefix_n1_10/answer_summary.json" \
  --out "$EXP_ROOT/prefix_answer_frontier_final8.json"

python scripts/plot_prefix_answer_frontier.py \
  --input "$EXP_ROOT/prefix_answer_frontier_final8.json" \
  --out "$EXP_ROOT/prefix_answer_frontier_final8.pdf"
```

`figures/prefix_f1_frontier/` is an older-suite figure and includes QASPER and
PubMedQA. It is not evidence for the current final-eight result.

### D. Retrieval-depth sensitivity

Test `k in {1,3,5,10}` as nested prefixes of the same frozen top-10 rows. For
each k, rebuild the clean response, token domain, graph, frontier, and reader
verification. Do not truncate a top-5 graph to simulate another k.

```bash
python scripts/materialize_retrieval_prefix.py \
  --input "$RETRIEVAL" \
  --units-cache "$TOKEN_UNITS" \
  --shared-pool "$FACTUAL_POOL" \
  --expected-source-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out-dir /path/to/retrieval_depth/kK/DATASET_KEY/inputs \
  --k K --n 1000 --reuse-existing
```

This command writes `retrieval.jsonl`, `units.jsonl`, `shared_pool.jsonl`, and
`manifest.json` into the requested directory. The four k settings must be
materialized from the same top-10 retrieval file, token cache, and frozen
factual pool. Check each prefix manifest before starting model inference; it
records both source and output hashes and rejects a mismatched source-pool
hash.

For each prefix, run only the clean-target, graph, ReFlow-frontier, repair, and
factual-reader stages with `--k K`. Store them under
`kK/DATASET_KEY/{clean,graph,frontier,final,factual}`; this is the layout
consumed by the audited summarizer. Retrieval-depth sensitivity does not use
matched-budget baselines or controls. The exact portable stage arguments are
the corresponding Steps 3, 4, and 6 above with prefix-local retrieval, units,
and pool paths. `run_retrieval_depth_qwen25_server3.sh` shows the complete
repair/resume topology, but its service and machine-management code is
host-specific.

After all eight datasets for one k pass row, ID, pool-hash, and no-frontier
audits, summarize:

```bash
python scripts/summarize_retrieval_depth.py \
  --repo "$PWD" --run-root /path/to/retrieval_depth_run \
  --k K --out-dir /path/to/retrieval_depth_summary/kK
```

The original scheduler `run_retrieval_depth_qwen25_server3.sh` contains
machine paths and a protected spaCy PID; do not run it unchanged. Checked k=1,
k=3, and k=10 bundles and SHA files are under
`artifacts/retrieval_depth_qwen25_v1/`. Reuse k=5 only if every frozen input
and configuration hash matches the audited main run.

### E. Receiver-beam sensitivity

At fixed k=5, build all beam graphs in one model load:

```bash
python scripts/build_contribution_graph_beams.py \
  --input "$RETRIEVAL" \
  --out-template "$DATASET_ROOT/beam/graph_d{beam}.jsonl" \
  --summary-template "$DATASET_ROOT/beam/graph_d{beam}.summary.json" \
  --model-path "$MODEL_PATH" --start 0 --n 1000 --k 5 \
  --target-results "$CLEAN" \
  --receiver-beams 16 32 48 64 \
  --device cuda --dtype bfloat16 --resume
```

Run ReFlow and reader verification independently for each graph (repeat for
`BEAM=16,32,48,64`):

```bash
export BEAM=16
export BEAM_GRAPH="$DATASET_ROOT/beam/graph_d${BEAM}.jsonl"
export BEAM_FRONTIER="$DATASET_ROOT/beam/frontier_d${BEAM}.jsonl"
export BEAM_RESULTS="$DATASET_ROOT/beam/results_d${BEAM}.jsonl"

python exp/run_reflow.py \
  --input "$RETRIEVAL" --graphs "$BEAM_GRAPH" \
  --units-cache "$TOKEN_UNITS" --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$BEAM_FRONTIER" \
  --summary-out "$DATASET_ROOT/beam/frontier_d${BEAM}.summary.json" \
  --start 0 --n 1000 --k 5

python scripts/evaluate_reflow.py \
  --input "$RETRIEVAL" --units-cache "$TOKEN_UNITS" \
  --frontier "$BEAM_FRONTIER" --shared-pool "$FACTUAL_POOL" \
  --expected-pool-sha256 "$FACTUAL_POOL_SHA" \
  --out "$BEAM_RESULTS" \
  --summary-out "$DATASET_ROOT/beam/results_d${BEAM}.summary.json" \
  --n 1000 --k 5 --workers 32 --replacement-seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"
```

The four runs must have identical query IDs, retrieval, clean targets, token
domain, factual pool, replacement seed, and reader settings; receiver beam is
the only changed factor. Report retained contribution mass, nodes, edges,
graph time, coverage, flip metrics, mean terminal edits, reader calls, and
end-to-end seconds/query. The historical scheduler varies 16/32/64 around the
beam-48 default; if the paper uses another grid, keep the script, manifest,
README, and caption consistent.

### F. Reader-model sensitivity

The evaluated readers are Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507,
Llama-3.1-8B-Instruct, and Mistral-Nemo-Instruct-2407. Retrieval and the
method-independent factual pool stay frozen. For each reader, rebuild its
clean responses, answer-conditioned graphs, frontiers, factual outputs, and
same-position control pool/results. Never reuse another reader's clean target,
graph, selected positions, or controls.

The original seven-dataset scheduler is `run_cross_model_reflow_server3.sh`;
PopQA was added by `run_popqa_cross_model_server3.sh`. They are machine-specific
provenance. Portable aggregation is:

```bash
python scripts/render_cross_model_sensitivity.py \
  --model 'qwen2.5-7b|Qwen2.5-7B|Qwen2.5|default|/path/to/qwen25/root' \
  --model 'qwen3-4b|Qwen3-4B|Qwen3|sweep|/path/to/qwen3/root' \
  --model 'llama3.1-8b|Llama-3.1-8B|Llama3.1|sweep|/path/to/llama/root' \
  --model 'mistral-nemo-12b|Mistral-Nemo-12B|Mistral|sweep|/path/to/mistral/root' \
  --quartz-retrieval /path/to/quartz/retrieval/top10_1000.jsonl \
  --out-json /path/to/cross_model_sensitivity.json \
  --out-values-tex /path/to/cross_model_sensitivity_values.tex \
  --out-appendix-tex /path/to/cross_model_sensitivity_rows.tex
```

Inspect coverage, invalid-answer counts, missing-control counts, and every
factual/control status histogram; never report only successful queries. The
committed `artifacts/model_sensitivity_v1/cross_model_validation.json` is a
legacy snapshot whose eighth dataset is PubMedQA. It is useful for schema and
hash validation only, not as evidence for the current final-eight paper. A
current result must contain PopQA, exclude PubMedQA, contain 1,000 aligned
factual and control rows for every model/dataset pair, and have zero
`missing_paraphrase` statuses before rendering.

### G. RAG poisoning and defense

The security cohort is 100 frozen HotpotQA queries answered correctly by the
clean reader. ReFlow, PoisonedRAG, and TPARAG use the same cohort, retriever,
reader, top-5 context, and target answers.

```bash
python scripts/run_security_pareto_attacks.py \
  --retrieval "$RETRIEVAL" --reflow "$REFLOW_RESULTS" \
  --out /path/to/security/attacks_100.jsonl \
  --summary-out /path/to/security/attacks_100.summary.json \
  --n 100 --workers 16 --seed 0 \
  --tparag-iterations 5 --retrieved-k 5 \
  --injected-count 5 --attack-candidate-count 20 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL" \
  --embed-base-url http://127.0.0.1:8017/v1 \
  --embed-model Qwen3-Embedding-0.6B

python scripts/run_security_pareto_defenses.py \
  --attacks /path/to/security/attacks_100.jsonl \
  --units "$TOKEN_UNITS" --reflow "$REFLOW_RESULTS" \
  --out /path/to/security/defenses_100.jsonl \
  --summary-out /path/to/security/defenses_100.summary.json \
  --workers 16 --llm-base-url "$READER_URL" \
  --llm-model "$READER_MODEL"

python scripts/plot_security_pareto.py \
  --attack-summary /path/to/security/attacks_100.summary.json \
  --attack-rows /path/to/security/attacks_100.jsonl \
  --reflow "$REFLOW_RESULTS" \
  --defense-summary /path/to/security/defenses_100.summary.json \
  --out-prefix /path/to/security/security_pareto \
  --data-out /path/to/security/security_pareto.data.json
```

Attack footprint is replaced tokens for ReFlow and all inserted passage tokens
for PoisonedRAG/TPARAG. Defense footprint is tokens verified, protected, or
isolated before defended generation. Report untargeted and targeted ASR,
attacked accuracy, exact footprints, and bootstrap intervals. The aligned
shipped summaries are under `figures/security_case_study/`.

Official external PoisonedRAG and TPARAG repositories are not vendored.
`run_security_pareto_attacks.py` is the aligned local reproduction. Record any
different external revision and do not present it as the shipped run.

### H. Evidence/provenance sufficiency

This experiment asks whether sentences containing selected tokens preserve
enough evidence for the same reader to answer correctly:

- `Full` uses all top-5 chunks.
- `ReFlow` uses distinct sentences containing final ReFlow tokens.
- `MIRAGE` and `ARC-JSD` take ranked sentences up to the same per-query
  distinct-sentence budget.
- Sentences return to original chunk/sentence order before reading.
- Metrics are EM and token F1 with 10,000 query-level bootstrap resamples.
- Empty ReFlow sets and baseline coverage shortfalls are excluded with counts,
  never silently imputed.

```bash
python scripts/run_sentence_sufficiency_case_study.py \
  --retrieval "$RETRIEVAL" --units "$TOKEN_UNITS" \
  --reflow "$REFLOW_RESULTS" \
  --mirage "$DATASET_ROOT/methods/baselines/mirage_top5_1000.jsonl" \
  --arc-jsd "$DATASET_ROOT/methods/baselines/arc_jsd_top5_1000.jsonl" \
  --out /path/to/sentence_sufficiency/DATASET_KEY/per_query.jsonl \
  --summary-out /path/to/sentence_sufficiency/DATASET_KEY/summary.json \
  --workers 16 --max-tokens 64 \
  --bootstrap-samples 10000 --bootstrap-seed 0 \
  --llm-base-url "$READER_URL" --llm-model "$READER_MODEL"

python scripts/plot_sentence_sufficiency.py \
  --summary "DATASET_LABEL=/path/to/sentence_sufficiency/DATASET_KEY/summary.json" \
  --datasets-per-row 1 \
  --out-prefix /path/to/sentence_sufficiency/figure
```

Run `--prepare-only` first to freeze the common covered cohort. The exact
shipped protocol is in `artifacts/sentence_sufficiency/README.md`. Its appendix
uses an older dataset suite, so do not call it the final eight-dataset suite.

### I. Additional non-main case studies

- `run_dynamic_answer_cache_case_study.py` replays cache policies over frozen
  request/update traces.
- `summarize_cache_invalidation_case_study.py` compares ReFlow invalidation
  with MIRAGE, ARC-JSD, IG, and Random.
- `run_evidence_debugging_case_study.py` prepares/summarizes a blinded study;
  invoke it as `python -m scripts.run_evidence_debugging_case_study ...`.
- `run_reliability_triage_case_study.py` has `prepare`, `evaluate`, and
  `summarize` stages for intervention size as a fragility signal.

These are not automatically included in final tables. If used, freeze a new
manifest, state the cohort and seeds, and report them separately.

## Artifact and audit requirements

### Canonical layout

```text
out/DATASET_KEY/
  retrieval/
    queries_1000.jsonl
    corpus_chunks_384t_o64.jsonl
    top10_1000.jsonl
    top10_1000.summary.json
    index/chunk_embeddings.npy
  inputs/
    token_units_top10_1000.jsonl
    clean_reference_top5_1000.jsonl
  replacements/shared_pool_top10_v1/
    positions.jsonl
    typed_keys.jsonl
    typed_candidates.jsonl
    unresolved.jsonl
    shared_pool.jsonl
    shared_pool.manifest.json
  graphs/
    contribution_graph_top5_1000.jsonl
  methods/reflow/
    frontier_top5_1000.jsonl
    results_top5_1000.jsonl
  methods/baselines/
    attention_top5_1000.jsonl
    gradient_x_input_top5_1000.jsonl
    integrated_gradients_top5_1000.jsonl
    mirage_top5_1000.jsonl
    arc_jsd_top5_1000.jsonl
  controls/
    paraphrase_pool_top5_v2.jsonl
    paraphrase_pool_top5_v2.manifest.json
    paraphrase_results_top5_1000_v2.jsonl
    paraphrase_results_top5_1000_v2.summary.json
  audits/final_top10pool_k5/
    baselines_1000.jsonl
    factual_metrics_1000.json
    table3_metrics_1000_v2.json
  logs/
  manifest.json
```

### Mandatory invariants

Before reporting a result, verify:

1. Every formal JSONL has exactly 1,000 nonempty rows and unique IDs.
2. IDs/order match retrieval, units, clean responses, graphs, frontiers,
   ReFlow results, baselines, and controls.
3. Query, retrieval, model, tokenizer, prompt, decoding, and pool hashes are
   recorded.
4. Actual pool hashes equal the expected hashes passed to evaluators.
5. Factual replacement uses only the shared pool and seed 0.
6. Every method uses the same editable domain; no method backfills uncovered
   selections.
7. Each valid-clean baseline has exactly the ReFlow terminal number of unique
   selected and modified tokens and a reader call.
8. Random uses seeds 0--4; average only after computing per-seed metrics.
9. Factual and control evaluations use identical selected positions.
10. Metric denominators and invalid-output counts are stored explicitly.
11. No-frontier graphs are repaired or reported as coverage failures, never
    relabeled as successful empty explanations.
12. Tables/plots are rendered from saved artifacts, never manually edited.

Audit executed replacements:

```bash
python scripts/audit_shared_replacement_protocol.py \
  --pool "$FACTUAL_POOL" \
  --reflow-results "$REFLOW_RESULTS" \
  --baseline-results "$BASELINE_RESULTS" \
  --seed 0
```

After a dataset run passes all audits, create a checksum manifest for the
exact files used to render the reported results. List files explicitly rather
than passing a directory, so temporary logs and partially written shards
cannot enter the publication bundle unnoticed:

```bash
export RUN_METADATA_JSON='{
  "dataset":"hotpotqa",
  "queries":1000,
  "retrieval_depth":5,
  "retrieval_model":"Qwen3-Embedding-0.6B",
  "reader_model":"Qwen2.5-7B-Instruct",
  "reader_revision":"REPLACE_WITH_EXACT_REVISION",
  "prompt_sha256":"REPLACE_WITH_SHA256",
  "replacement_seed":0,
  "random_seeds":[0,1,2,3,4]
}'

python scripts/build_artifact_manifest.py \
  --artifacts \
    "$RETRIEVAL" "$TOKEN_UNITS" "$CLEAN" \
    "$FACTUAL_POOL" "$GRAPH" "$FRONTIER" \
    "$REFLOW_RESULTS" "$BASELINE_RESULTS" \
    "$CONTROL_POOL" "$CONTROL_RESULTS" \
    "$DATASET_ROOT/audits/final_top10pool_k5/table3_metrics_1000_v2.json" \
  --metadata-json "$RUN_METADATA_JSON" \
  --repository "$PWD" \
  --out "$DATASET_ROOT/manifest.json"
```

Replace every placeholder in `RUN_METADATA_JSON`. Extend it with the source
dataset version and license, query-cohort ID/hash, chunking configuration,
tokenizer revision, decoding settings, package-lock hashes, CUDA, driver, and
GPU versions. The manifest tool records the exact git commit, branch, dirty
state, SHA-256, byte size, and line count. Formal results must be generated
from a clean commit (`dirty=false`). Keep the manifest with the results and
rebuild it whenever any listed artifact changes.

The checked aggregate snapshot
`artifacts/table3/final8_20260811/audit.json` records final dataset order,
valid/invalid clean cohorts, method sets, status histograms, and source hashes.
Its absolute paths are historical provenance and will not exist elsewhere.

### Shipped aggregate integrity

Retrieval-depth bundles include portable SHA files:

```bash
cd artifacts/retrieval_depth_qwen25_v1/k1
sha256sum --check SHA256SUMS
```

On macOS use `shasum -a 256 -c SHA256SUMS`. Treat aggregate files under
`artifacts/` and `figures/` as read-only publication records.

## Tests and troubleshooting

Run the complete suite:

```bash
python -m pytest -q tests
```

For protocol-sensitive changes, at minimum run:

```bash
pytest -q \
  tests/test_shared_replacement_pool.py \
  tests/test_replacement_pool_generation.py \
  tests/test_reflow_evaluation.py \
  tests/test_evaluation_metrics.py \
  tests/test_correctness_conditioned_flips.py \
  tests/test_fixed_budget_acc.py \
  tests/test_artifact_manifest.py
```

### Common failures

**Pool hash mismatch.** Stop. Confirm manifest path, actual SHA-256,
replacement seed, and whether the pool changed after rankings were produced.
Never bypass `--expected-pool-sha256`.

**Unresolved replacement keys.** Resume typed generation with a larger attempt
offset and inspect rejection reasons. Do not freeze a partial formal pool
unless unresolved positions are a declared method-independent exclusion.

**Missing or misordered IDs.** Use shard merge utilities, not shell
concatenation. Every merge must restore frozen retrieval order.

**Invalid reader output.** Check reader mode and prompt. QuaRTz requires its
binary-choice protocol. Preserve the failure status; do not manually extract a
convenient answer from free text.

**CUDA out of memory.** Run shards sequentially, lower a ranker batch size, or
use `--device-map balanced` for Mistral-Nemo. Do not silently change k, beam,
precision, or graph objective.

**No frontier candidate.** Inspect graph coverage and pool eligibility. Use
`repair_disconnected_contribution_graphs.py` and
`prepare_no_frontier_resume.py`, preserving hashes of original and repaired
artifacts.

**Reader intermittency.** Resume by query ID, not row position. Verify the
endpoint and rerun alignment audits after merging.

## Repository layout

```text
causalityrag/   graph, flow, token, pool, reader, and metric primitives
configs/        frozen protocol checklists
exp/            ReFlow and attribution/Shapley rankers
scripts/        preparation, evaluation, aggregation, audit, and render tools
scripts/launchers/  original server launch scripts, kept as provenance
revision/       experiments added during the paper revision
tests/          unit, integration, metric, and protocol tests
artifacts/      checked aggregate result and audit records
figures/        rendered outputs and their data files
```

[`EXPERIMENT_RUNBOOK.md`](EXPERIMENT_RUNBOOK.md) preserves historical execution
order and source-path notes. If it conflicts with this guide on the final suite,
the final-eight scorers, audits, and this guide take precedence.
