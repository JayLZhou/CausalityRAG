#!/usr/bin/env bash
# Recompute Token Shapley (TS) rankings with the paper settings used for 2Wiki/TriviaQA:
# attention top-128 candidates, LeverageSHAP paired sampling, budget multiplier 3, seed 42.
set -uo pipefail
[[ "$(hostname)" == "3haoji-A100" ]] || exit 1
GPU=$1; shift
B=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/baselines_new
TS=/data1/yujia/CausalityRAG_token_shapley_smoke
S=/data1/yujia/CausalityRAG
export PYTHONPATH=$TS CAUSALITYRAG_READER_MODE=short_answer PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
for ds in "$@"; do
  case $ds in
    hotpotqa) base=$S/out/hotpotqa; clean=$base/inputs/clean_reference_1000.jsonl ;;
    popqa)    base=$S/out_popqa_20260810/popqa; clean=$base/inputs/clean_targets_top5_1000.jsonl ;;
    *)        base=$S/out/$ds; clean=$base/inputs/clean_targets_top5_1000.jsonl ;;
  esac
  mkdir -p $B/ts/$ds
  echo "START $ds $(date -Is)"
  CUDA_VISIBLE_DEVICES=$GPU /data1/yujia/envs/qwen3_vllm_0102/bin/python $TS/exp/run_token_leverage_shap.py \
    --input $base/retrieval/top10_1000.jsonl --clean-reference $clean \
    --context-units $base/inputs/token_units_top10_1000.jsonl \
    --model-path /data1/yujia/models/Qwen2.5-7B-Instruct \
    --out $B/ts/$ds/scores_1000.jsonl --summary-out $B/ts/$ds/scores_1000.summary.json \
    --start 0 --n 1000 --k 5 --top-k 128 --attention-candidates 128 --budget-multiplier 3 \
    --coalition-batch-size 32 --seed 42 --resume > $B/logs/ts_$ds.log 2>&1 \
    && echo "DONE $ds $(date -Is)" || echo "FAILED $ds $(date -Is)"
done
echo "LANE FINISHED $(date -Is)"
