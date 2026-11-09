#!/usr/bin/env bash
# a7c.sh DATASET SEED : original Qwen pool, another valid alternative per token (A7c).
set -euo pipefail
S=/data1/yujia/CausalityRAG; N=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/baselines_new
P=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity
case $1 in
  hotpotqa) pool=$S/out/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl ;;
  popqa)    pool=$S/out_table3_final8_20260811/popqa/replacements/shared_pool_top10_v1/shared_pool.jsonl ;;
  *)        pool=$S/out/$1/replacements/shared_pool_top10_v1/shared_pool.jsonl ;;
esac
# Rankings that do not depend on the replacement (TS, ContextCite) are re-verified too.
export REPL_SEED=$2 EXTRA_SCORES="ts=$N/ts/$1/scores_1000.jsonl contextcite=$N/$1/contextcite_1000.jsonl"
bash $P/scripts/run_pool_sensitivity.sh "$1" "qwen_seed$2" "$pool" "$PORT"
