#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "2haoji-A100" ]] || exit 1

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
DATA="$ROOT/shapley_rank_bundle"
EXTRA="$ROOT/shapley_eval_bundle"
EVAL_ROOT="$ROOT/out/token_leverage_shap_final8_m128_3m_eval"
HOT_EVAL="$ROOT/out/attention_filtered_leverage_shap_hotpot_m128_3m_eval"
COARSE="$ROOT/out/token_leverage_shap_final8_m128_3m_budget_sweep"
OUT_ROOT="$ROOT/out/token_leverage_shap_final8_m128_3m_budget_exact"
PYTHON=/data1/yujia/envs/token_shapley_smoke/bin/python
PACKAGES=/data1/yujia/envs/token_shapley_smoke_pkgs
MODEL=/data1/szgao/resources/pretrain/Qwen2.5-7B-Instruct
DATASETS=(hotpotqa timeqa finqa musique quartz popqa triviaqa 2wiki)

export PYTHONPATH="$PACKAGES:$ROOT"
cd "$ROOT"
while [[ ! -f "$COARSE/SERVER2_COARSE" ]]; do
  sleep 60
done

for dataset in "${DATASETS[@]}"; do
  out="$OUT_ROOT/$dataset"
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
  reader_mode=short_answer
  if [[ "$dataset" == hotpotqa ]]; then
    eval_dir="$HOT_EVAL"
    retrieval="$ROOT/hotpot_top10_1000.jsonl"
    reflow="$ROOT/hotpot_reflow_1000.jsonl"
    pool="$ROOT/hotpot_shared_pool_top10_final.jsonl"
  else
    eval_dir="$EVAL_ROOT/$dataset"
    retrieval="$DATA/$dataset/input.jsonl"
    reflow="$EXTRA/$dataset/reflow.jsonl"
    pool="$EXTRA/$dataset/factual_pool.jsonl"
  fi
  pool_sha="$(sha256sum "$pool" | cut -d ' ' -f 1)"

  pids=()
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" \
    CAUSALITYRAG_READER_MODE="$reader_mode" \
    "$PYTHON" scripts/evaluate_fixed_budget_acc.py \
      --input "$eval_dir/input_${gpu}.jsonl" \
      --units-cache "$eval_dir/units_${gpu}.jsonl" \
      --frontier "$eval_dir/reflow_${gpu}.jsonl" \
      --scores "token_leverage_shap=$eval_dir/scores_${gpu}.jsonl" \
      --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
      --seed-results "$COARSE/$dataset/shard_${gpu}.jsonl" \
      --out "$out/shard_${gpu}.jsonl" \
      --summary-out "$out/shard_${gpu}.summary.json" \
      --budgets "$budgets" --skip-reflow \
      --replacement-seed 0 --n 250 --k 5 --workers 1 \
      --model-path "$MODEL" --device cuda --dtype bfloat16 \
      --reader-mode "$reader_mode" \
      >"$out/logs/shard_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
  done
  (( status == 0 ))

  "$PYTHON" scripts/summarize_shapley_budget_sweep.py \
    --dataset "$dataset" --retrieval "$retrieval" --reflow "$reflow" \
    --shards "$out/shard_0.jsonl" "$out/shard_1.jsonl" \
      "$out/shard_2.jsonl" "$out/shard_3.jsonl" \
    --out-rows "$out/results_1000.jsonl" \
    --out-summary "$out/fine_summary.json" \
    >"$out/logs/summary.log" 2>&1
  "$PYTHON" scripts/merge_shapley_budget_refinement.py \
    --coarse "$coarse_summary" --fine "$out/fine_summary.json" \
    --out "$out/summary.json"
  touch "$out/COMPLETE"
done

touch "$OUT_ROOT/SERVER2_EXACT" "$OUT_ROOT/SERVER3_BUDGET_IMPORTED"
touch "$OUT_ROOT/EXACT_COMPLETE"
