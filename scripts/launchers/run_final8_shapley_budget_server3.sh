#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "3haoji-A100" ]] || exit 1

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
DATA="$ROOT/shapley_rank_bundle"
EXTRA="$ROOT/shapley_eval_bundle"
EVAL_ROOT="$ROOT/out/token_leverage_shap_final8_m128_3m_eval"
COARSE="$ROOT/out/token_leverage_shap_final8_m128_3m_budget_sweep"
EXACT="$ROOT/out/token_leverage_shap_final8_m128_3m_budget_exact"
PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python
PACKAGES="$ROOT/third_party"
MODEL=/data1/yujia/models/Qwen2.5-7B-Instruct
DATASETS=(triviaqa 2wiki)

export PYTHONPATH="$PACKAGES:$ROOT"
export NLTK_DATA="$ROOT/nltk_data"
cd "$ROOT"

run_budget_shards() {
  local dataset="$1"
  local budgets="$2"
  local out="$3"
  local seed_root="${4:-}"
  local pool="$EXTRA/$dataset/factual_pool.jsonl"
  local pool_sha
  pool_sha="$(sha256sum "$pool" | cut -d ' ' -f 1)"
  mkdir -p "$out/logs"
  local -a pids=()
  local gpu seed_args
  for gpu in 0 1; do
    seed_args=()
    if [[ -n "$seed_root" ]]; then
      seed_args=(--seed-results "$seed_root/shard_${gpu}.jsonl")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
    CAUSALITYRAG_READER_MODE=short_answer \
    "$PYTHON" scripts/evaluate_fixed_budget_acc.py \
      --input "$EVAL_ROOT/$dataset/input_${gpu}.jsonl" \
      --units-cache "$EVAL_ROOT/$dataset/units_${gpu}.jsonl" \
      --frontier "$EVAL_ROOT/$dataset/reflow_${gpu}.jsonl" \
      --scores "token_leverage_shap=$EVAL_ROOT/$dataset/scores_${gpu}.jsonl" \
      --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
      "${seed_args[@]}" \
      --out "$out/shard_${gpu}.jsonl" \
      --summary-out "$out/shard_${gpu}.summary.json" \
      --budgets "$budgets" --skip-reflow \
      --replacement-seed 0 --n 500 --k 5 --workers 1 \
      --model-path "$MODEL" --device cuda --dtype bfloat16 \
      --reader-mode short_answer \
      >"$out/logs/shard_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0 pid
  for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
  done
  (( status == 0 ))
}

for dataset in "${DATASETS[@]}"; do
  out="$COARSE/$dataset"
  run_budget_shards "$dataset" "1,2,4,8,16,32,64,128" "$out"
  "$PYTHON" scripts/summarize_shapley_budget_sweep.py \
    --dataset "$dataset" --retrieval "$DATA/$dataset/input.jsonl" \
    --reflow "$EXTRA/$dataset/reflow.jsonl" \
    --shards "$out/shard_0.jsonl" "$out/shard_1.jsonl" \
    --out-rows "$out/results_1000.jsonl" --out-summary "$out/summary.json" \
    >"$out/logs/summary.log" 2>&1
  touch "$out/COMPLETE"
done
touch "$COARSE/SERVER3_COARSE"

for dataset in "${DATASETS[@]}"; do
  out="$EXACT/$dataset"
  mkdir -p "$out/logs"
  coarse_summary="$COARSE/$dataset/summary.json"
  first="$($PYTHON -c 'import json,sys; x=json.load(open(sys.argv[1])).get("first_matching_tested_budget"); print("none" if x is None else int(x))' "$coarse_summary")"
  if [[ "$first" == none || "$first" == 1 ]]; then
    "$PYTHON" scripts/merge_shapley_budget_refinement.py \
      --coarse "$coarse_summary" --out "$out/summary.json"
    touch "$out/COMPLETE"
    continue
  fi
  lower=$(( first / 2 ))
  budgets="$(seq -s, $((lower + 1)) "$first")"
  run_budget_shards "$dataset" "$budgets" "$out" "$COARSE/$dataset"
  "$PYTHON" scripts/summarize_shapley_budget_sweep.py \
    --dataset "$dataset" --retrieval "$DATA/$dataset/input.jsonl" \
    --reflow "$EXTRA/$dataset/reflow.jsonl" \
    --shards "$out/shard_0.jsonl" "$out/shard_1.jsonl" \
    --out-rows "$out/results_1000.jsonl" \
    --out-summary "$out/fine_summary.json" \
    >"$out/logs/summary.log" 2>&1
  "$PYTHON" scripts/merge_shapley_budget_refinement.py \
    --coarse "$coarse_summary" --fine "$out/fine_summary.json" \
    --out "$out/summary.json"
  touch "$out/COMPLETE"
done
touch "$EXACT/SERVER3_EXACT"

