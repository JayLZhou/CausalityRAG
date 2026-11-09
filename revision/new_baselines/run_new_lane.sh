#!/usr/bin/env bash
# ContextCite, MiCE-RAG, RAGE (adapted + native) on the main-table setting (Qwen pool).
set -uo pipefail
[[ "$(hostname)" == "3haoji-A100" ]] || exit 1
GPU=$1; shift
B=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/baselines_new
S=/data1/yujia/CausalityRAG
export CAUSALITYRAG_READER_MODE=short_answer PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
for ds in "$@"; do
  case $ds in
    hotpotqa) base=$S/out/hotpotqa; main=$S/out_no_frontier_repair_20260810/hotpotqa/results_1000.jsonl
              attn=$base/methods/baselines/attention_prompt_last_1000.jsonl
              pool=$base/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl ;;
    popqa)    base=$S/out_popqa_20260810/popqa; main=$base/methods/reflow/results_top5_1000.jsonl
              attn=$base/methods/baselines/attention_top5_1000.jsonl
              pool=$base/replacements/shared_pool_top10_v1/shared_pool.jsonl ;;
    *)        base=$S/out/$ds; main=$S/out_no_frontier_repair_20260810/$ds/results_1000.jsonl
              attn=$base/methods/baselines/attention_top5_1000.jsonl
              pool=$base/replacements/shared_pool_top10_v1/shared_pool.jsonl ;;
  esac
  mkdir -p $B/$ds
  for m in contextcite mice_rag rage rage_native; do
    echo "START $ds $m $(date -Is)"
    CUDA_VISIBLE_DEVICES=$GPU /data1/yujia/envs/graphrag/bin/python $B/scripts/run_new_baselines.py --method $m \
      --input $base/retrieval/top10_1000.jsonl --units-cache $base/inputs/token_units_top10_1000.jsonl \
      --reflow-results $main --prompt-attention $attn \
      --gxi-scores $base/methods/baselines/gradient_x_input_top5_1000.jsonl --shared-pool $pool \
      --out $B/$ds/${m}_1000.jsonl --n 1000 > $B/logs/${m}_$ds.log 2>&1 \
      && echo "DONE $ds $m $(date -Is)" || echo "FAILED $ds $m $(date -Is)"
  done
done
echo "LANE FINISHED $(date -Is)"
