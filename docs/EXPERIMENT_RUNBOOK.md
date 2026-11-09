# Eight-Dataset Evaluation Runbook

This runbook freezes the protocol for HotpotQA, TimeQA, FinQA, MuSiQue,
QuaRTz, TriviaQA, 2WikiMultiHopQA, and PubMedQA. HotpotQA is complete; the same contract
is used for the remaining seven datasets.

## Fixed Protocol

- Primary population: 1,000 query IDs per dataset.
- Retrieval: Qwen3-Embedding-0.6B dense retrieval with cosine similarity and
  no reranker.
- Retrieval range: build top-10 once and use nested prefixes for
  `k in {1, 3, 5, 10}`.
- Main table: top-5 retrieved chunks.
- Corpus unit: 384 Qwen3 embedding-model tokens with 64-token overlap.
  Documents shorter than 384 tokens remain unchanged.
- Reader and graph model: Qwen2.5-7B-Instruct.
- Reader decoding: greedy, stopping at EOS or the model context window.
- Graph: closed positive message flow, receiver beam 48, then contraction by
  surface-token position.
- ReFlow: exact supported breakpoint flow-contract frontier with
  mean-normalized positive contribution capacities.
- Replacement domain: the frozen top-10 position pool. Main-table evaluation
  reads this pool with `k=5`.
- Matched budget: for query `q`, every baseline modifies exactly the number of
  positions actually modified by ReFlow on `q`.
- Baselines: five-seed Random, answer-position Attention, Gradient x Input,
  Integrated Gradients, MIRAGE, and ARC-JSD.
- Metrics: intent-to-treat Answer-, F1-, EM-, and Acc-Flip plus the paired
  factual-minus-paraphrase counterparts.

The query and gold answer are never used to generate factual replacements.
Gold supporting passages are used only to compute post-hoc retrieval recall.

## Frozen Query Sources

Use source order after removing rows with an empty query ID, question, or gold
answer. Freeze the first 1,000 remaining rows and write their IDs to the
dataset manifest.

| Dataset | Question source | Corpus source | Preflight note |
|---|---|---|---|
| TimeQA | `/data1/yujia/RAGData/timeqa/questions/timeqa.json` | `/data1/yujia/RAGData/timeqa/corpus/timeqa_corpus.json` | 2,613 valid source rows |
| FinQA | `/data1/yujia/RAGData/finqa/questions/finqa.json` | `/data1/yujia/RAGData/finqa/corpus/finqa_corpus.json` | Skip 13 empty-gold rows occurring in the first 1,000 raw rows |
| MuSiQue | `/data1/yujia/RAGData/musique/questions/musique.json` | `/data1/yujia/RAGData/musique/corpus/musique_corpus.json` | All 1,000 rows are answerable |
| QuaRTz | `/data1/yujia/RAGData/quartz/questions/quartz.json` | `/data1/yujia/RAGData/quartz/corpus/quartz_corpus.json` | Preserve the benchmark's binary comparison labels |
| TriviaQA | `/data1/yujia/RAGData/triviaqa/prepared/questions_1000.jsonl` | `/data1/yujia/RAGData/triviaqa/prepared/corpus.jsonl` | Frozen Wikipedia-dev subset: 1,000 queries and 1,537 linked evidence documents |
| 2Wiki | `/data1/yujia/RAGData/2wiki/questions/2wikimultihopqa.json` | `/data1/yujia/RAGData/2wiki/corpus/2wikimultihopqa_corpus.json` | Rebuild top-10; the existing artifact is top-5 only |
| PubMedQA | `/data1/yujia/RAGData/pubmedqa/questions/pubmedqa.json` | `/data1/yujia/RAGData/pubmedqa/corpus/pubmedqa_corpus.json` | Preserve yes/no/maybe labels verbatim |

Do not use `/data1/yujia/RAGData/universal/*/questions.jsonl` for the primary
table: those files contain only 500 queries per dataset.

## Artifact Layout

Every dataset uses the same directory topology:

```text
/data1/yujia/CausalityRAG/out/<dataset>/
  manifest.json
  retrieval/
    corpus_chunks_384t_o64.jsonl
    queries_1000.jsonl
    top10_1000.jsonl
    top10_1000.summary.json
    index/
  inputs/
    token_units_top10_1000.jsonl
    clean_reference_top5_1000.jsonl
  replacements/
    shared_pool_top10_v1/
      positions.jsonl
      typed_keys.jsonl
      typed_candidates.jsonl
      shared_pool.jsonl
      shared_pool.manifest.json
  graphs/
    contribution_graph_top5_1000.jsonl
  methods/
    reflow/
      frontier_top5_1000.jsonl
      results_top5_1000.jsonl
    baselines/
      attention_top5_1000.jsonl
      gradient_x_input_top5_1000.jsonl
      integrated_gradients_top5_1000.jsonl
      mirage_top5_1000.jsonl
      arc_jsd_top5_1000.jsonl
  controls/
    paraphrase_pool_top5_v1.jsonl
    paraphrase_results_top5_1000.jsonl
  audits/
    final_top10pool_k5/
      baselines_1000.jsonl
      factual_metrics_1000.json
      protocol_audit.json
  logs/
```

All JSONL files must have exactly 1,000 aligned query rows unless the manifest
explicitly declares a corpus-level artifact. Every frozen pool and retrieval
file records a SHA-256 digest in `manifest.json`.

## Execution Order

For each dataset:

1. Freeze 1,000 query IDs and build the canonical 384/64 corpus chunks.
2. Retrieve top-10 and audit row count, unique IDs, nested-prefix ordering, and
   post-hoc gold-title recall.
3. Build top-10 token units.
4. Generate, validate, freeze, and hash the method-independent top-10 factual
   replacement pool.
5. Generate one frozen clean response from the top-5 prefix.
6. Build the final top-5 contribution graph in one end-to-end execution.
7. Produce the six baseline rankings. Random is generated in the shared
   evaluator; the five model-based rankers never perform replacements.
8. Run the exact ReFlow breakpoint frontier and verify candidates with the
   frozen reader.
9. Evaluate every baseline with ReFlow's per-query actual edit count and the
   same factual pool.
10. Build the meaning-preserving control pool on the union of selected
    positions. Numeric and date positions are excluded from paraphrasing and
    remain unchanged in the control context; the artifact records their count.
11. Compute the eight main-table metrics (six for PubMedQA, where the two Acc
    rows are omitted), five-seed Random statistics,
    protocol audit, coverage failures, average tokens, and reader calls.
12. Freeze the complete dataset manifest before copying values into the paper.

## Dataset Order

Run a 10-query end-to-end smoke test before each 1,000-query job.

1. 2Wiki: closest multi-hop transfer from HotpotQA and already has a top-5
   retrieval artifact for cross-checking.
2. PubMedQA: short corpus and label answers; catches answer-normalization bugs.
3. TimeQA: temporal values and multi-answer strings.
4. MuSiQue: multi-hop transfer with exactly 1,000 answerable rows.
5. FinQA: numeric answers and explicit empty-gold filtering.
6. TriviaQA: open-domain Wikipedia evidence and alias-rich answers.
7. QuaRTz: short scientific comparisons and binary answer normalization.

## Mandatory Preflight

Do not start a formal job unless all checks pass:

- source checkout is an immutable release directory, not the dirty development
  checkout;
- all unit/protocol tests pass;
- retrieval, pool, model, tokenizer, prompt, and query-manifest hashes are
  recorded;
- ports 8001, 8002, and 8003 are healthy behind port 8000 for reader-heavy
  stages;
- all three GPUs accept a new CUDA process before local graph/ranker stages;
- spaCy `en_core_web_lg` is healthy;
- at least 100 GB of free disk remains;
- the 10-query smoke test has zero alignment or replacement-pool violations.

The current server must be rebooted or its GPU driver reset before graph and
model-internal baseline stages: GPU0 reports an NVML error, ports 8001 and 8002
are down, and new CUDA processes cannot initialize.

The retrieval and shared-pool stages may run before that reset. They use the
embedding service on port 8017, spaCy on port 8021, and the existing Qwen
service through port 8000; they do not create a new CUDA context. Launch the
seven-dataset pool stage from an immutable release as follows:

```bash
screen -dmS reflow_7datasets_pools bash -lc '
  cd /data1/yujia/CausalityRAG_release_<commit>
  export PYTHONUNBUFFERED=1
  /data1/yujia/envs/graphrag/bin/python \
    scripts/run_seven_dataset_pools.py \
    2>&1 | tee -a \
    /data1/yujia/CausalityRAG/out/seven_dataset_pools.log
'
```

The runner performs a fail-closed 10-query pool smoke test before freezing each
1,000-query pool. It writes rejected candidates to `unresolved.jsonl` and
never freezes a pool while a semantic token remains uncovered.

## Unattended Top-5 Graph Runner

After the shared services are healthy, the remaining formal top-5 graphs can
be built with the resumable runner:

```bash
screen -dmS reflow_all_top5_auto bash -lc '
  cd /data1/yujia/CausalityRAG_release_<commit>
  export PYTHONUNBUFFERED=1
  while true; do
    /data1/yujia/envs/graphrag/bin/python \
      scripts/run_remaining_top5_graphs.py \
      --out-root /data1/yujia/CausalityRAG/out \
      --status-out \
        /data1/yujia/CausalityRAG/out/top5_graph_automation_status.json \
      >> /data1/yujia/CausalityRAG/out/top5_graph_automation.log 2>&1 \
      && break
    sleep 60
  done
'
```

The runner validates and resumes retrieval, token typing, the frozen top-10
replacement pool, top-5 clean targets, and two-GPU graph shards in dataset
order. It stops the vLLM replicas before graph construction so that both GPUs
are available, merges the two aligned 500-query shards only after both pass
validation, and restarts reader services for the next dataset. Completed
artifacts are never recomputed. The JSON status file records the active
dataset and stage for progress checks.
