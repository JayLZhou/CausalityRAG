#!/usr/bin/env bash
# verify_pool_a5.sh TAG DATASET : TS, ContextCite and the pool's own MiCE-RAG under an editor pool,
# reusing that pool's ReFlow results so budgets match the A7(b) run.
set -euo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; P=$R/pool_sensitivity; N=$R/baselines_new
TAG=$1 DS=$2
mkdir -p $P/${TAG}_a5/$DS
cp -n $P/$TAG/$DS/reflow_1000.jsonl $P/$TAG/$DS/reflow_1000.summary.json $P/${TAG}_a5/$DS/
export READER_TIMEOUT=1800 EXTRA_SCORES="ts=$N/ts/$DS/scores_1000.jsonl contextcite=$N/$DS/contextcite_1000.jsonl mice_rag=$N/$TAG/$DS/mice_rag_1000.jsonl"
bash $P/scripts/run_pool_sensitivity.sh $DS ${TAG}_a5 $R/editor_pools/$TAG/$DS/shared_pool.jsonl $PORT
