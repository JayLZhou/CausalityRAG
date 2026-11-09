#!/usr/bin/env bash
# verify_pool_rage.sh TAG DATASET : the pool's own RAGE rankings under that editor pool (budgets from its A7b run).
set -euo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; P=$R/pool_sensitivity; N=$R/baselines_new
TAG=$1 DS=$2
mkdir -p $P/${TAG}_rage/$DS
cp -n $P/$TAG/$DS/reflow_1000.jsonl $P/$TAG/$DS/reflow_1000.summary.json $P/${TAG}_rage/$DS/
export EXTRA_SCORES="rage=$N/$TAG/$DS/rage_1000.jsonl"
bash $P/scripts/run_pool_sensitivity.sh $DS ${TAG}_rage $R/editor_pools/$TAG/$DS/shared_pool.jsonl $PORT
