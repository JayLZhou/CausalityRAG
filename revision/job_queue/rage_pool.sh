#!/usr/bin/env bash
# rage_pool.sh TAG DATASET : RAGE (adapted) search with another editor's pool, at that pool's ReFlow budget.
set -euo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; S=/data1/yujia/CausalityRAG
TAG=$1 DS=$2; base=$S/out/$DS
attn=$base/methods/baselines/attention_top5_1000.jsonl
[[ $DS == hotpotqa ]] && attn=$base/methods/baselines/attention_prompt_last_1000.jsonl
mkdir -p $R/baselines_new/$TAG/$DS
export CAUSALITYRAG_READER_MODE=short_answer PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=$GPU /data1/yujia/envs/graphrag/bin/python $R/baselines_new/scripts/run_new_baselines.py --method rage   --input $base/retrieval/top10_1000.jsonl --units-cache $base/inputs/token_units_top10_1000.jsonl   --reflow-results $R/pool_sensitivity/$TAG/$DS/reflow_1000.jsonl --prompt-attention $attn   --shared-pool $R/editor_pools/$TAG/$DS/shared_pool.jsonl --out $R/baselines_new/$TAG/$DS/rage_1000.jsonl --n 1000
