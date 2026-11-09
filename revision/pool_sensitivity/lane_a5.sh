#!/usr/bin/env bash
# Matched-budget verification of the revision baselines (TS recomputed, ContextCite,
# MiCE-RAG, RAGE adapted) next to the five Table 3 baselines, original Qwen pool.
set -uo pipefail
for ds in hotpotqa musique popqa; do
  case $ds in
    hotpotqa) pool=/data1/yujia/CausalityRAG/out/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl ;;
    popqa)    pool=/data1/yujia/CausalityRAG/out_table3_final8_20260811/popqa/replacements/shared_pool_top10_v1/shared_pool.jsonl ;;
    *)        pool=/data1/yujia/CausalityRAG/out/$ds/replacements/shared_pool_top10_v1/shared_pool.jsonl ;;
  esac
  export EXTRA_SCORES="ts=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/baselines_new/ts/$ds/scores_1000.jsonl contextcite=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/baselines_new/$ds/contextcite_1000.jsonl mice_rag=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/baselines_new/$ds/mice_rag_1000.jsonl rage=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/baselines_new/$ds/rage_1000.jsonl"
  echo "START $ds $(date -Is)"
  bash /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/scripts/run_pool_sensitivity.sh $ds qwen_a5 $pool 8065 > /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/logs/qwen_a5_$ds.log 2>&1 && echo "DONE $ds $(date -Is)" || echo "FAILED $ds $(date -Is)"
done
echo "LANE FINISHED $(date -Is)"
