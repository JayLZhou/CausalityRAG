#!/usr/bin/env bash
# miceflip_pool_job.sh EDITOR_TAG DATASET : first-flip MiCE-RAG under an A7(b) editor pool,
# then factual metrics, controls and CFR, with the budgets of that pool's ReFlow run.
set -euo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; P=$R/pool_sensitivity
TAG=$1 DS=$2; O=$P/${TAG}_miceflip/$DS; POOL=$R/editor_pools/$TAG/$DS/shared_pool.jsonl
mkdir -p $O
[[ -s $O/baselines_1000.summary.json ]] || /data1/yujia/envs/graphrag/bin/python $R/queue/mice_flip.py $DS $PORT $TAG
cp -n $P/$TAG/$DS/reflow_1000.jsonl $P/$TAG/$DS/reflow_1000.summary.json $O/
bash $P/scripts/run_pool_sensitivity.sh $DS ${TAG}_miceflip $POOL $PORT
