#!/usr/bin/env bash
# Runs after the matching Llama lane releases its lock (same reader port).
set -uo pipefail
case "$1" in
  A) PORT=8064; JOBS=("hotpotqa gpt56_sol /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/editor_pools/gpt56_sol/hotpotqa/shared_pool.jsonl" "finqa gpt56_sol /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/editor_pools/gpt56_sol/finqa/shared_pool.jsonl") ;;
  B) PORT=8065; JOBS=("musique gpt56_sol /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/editor_pools/gpt56_sol/musique/shared_pool.jsonl") ;;
esac
exec 9>/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/lane_$1.lock; flock 9
for job in "${JOBS[@]}"; do
  set -- $job
  echo "START $1 $2 $(date -Is)"
  bash /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/scripts/run_pool_sensitivity.sh $1 $2 $3 $PORT > /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/logs/$2_$1.log 2>&1 && echo "DONE $1 $2 $(date -Is)" || echo "FAILED $1 $2 $(date -Is)"
done
echo "LANE FINISHED $(date -Is)"
