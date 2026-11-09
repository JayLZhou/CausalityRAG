#!/usr/bin/env bash
# Re-invoke jobs that started with the old interpreter; completed stages are skipped.
set -uo pipefail
case "$1" in
  A) PORT=8064; JOBS=("musique qwen_original /data1/yujia/CausalityRAG/out/musique/replacements/shared_pool_top10_v1/shared_pool.jsonl" "musique llama31_8b /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/editor_pools/llama31_8b/musique/shared_pool.jsonl" "hotpotqa gpt56_sol /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/editor_pools/gpt56_sol/hotpotqa/shared_pool.jsonl") ;;
  B) PORT=8065; JOBS=("hotpotqa llama31_8b /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/editor_pools/llama31_8b/hotpotqa/shared_pool.jsonl" "finqa llama31_8b /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/editor_pools/llama31_8b/finqa/shared_pool.jsonl") ;;
esac
sleep 5
exec 9>/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/lane_$1.lock; flock 9
for job in "${JOBS[@]}"; do
  set -- $job
  [[ -s /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/$2/$1/table3_metrics_1000.json ]] && { echo "ALREADY $1 $2"; continue; }
  echo "START $1 $2 $(date -Is)"
  bash /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/scripts/run_pool_sensitivity.sh $1 $2 $3 $PORT > /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/logs/$2_$1.redo.log 2>&1 && echo "DONE $1 $2 $(date -Is)" || echo "FAILED $1 $2 $(date -Is)"
done
echo "LANE FINISHED $(date -Is)"
