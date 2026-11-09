#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "3haoji-A100" ]] || exit 1

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
DATA="$ROOT/shapley_rank_bundle"
EXTRA="$ROOT/shapley_eval_bundle"
RANK_ROOT="$ROOT/out/token_leverage_shap_final8_m128_3m"
OUT_ROOT="$ROOT/out/token_leverage_shap_final8_m128_3m_eval"
PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python
PACKAGES="$ROOT/third_party"
MODEL=/data1/yujia/models/Qwen2.5-7B-Instruct
DATASETS=(triviaqa 2wiki)

export PYTHONPATH="$PACKAGES:$ROOT"
export NLTK_DATA="$ROOT/nltk_data"
cd "$ROOT"

run_parallel() {
  local -a pids=()
  local status=0
  for gpu in 0 1; do
    "$@" "$gpu" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
  done
  (( status == 0 ))
}

for dataset in "${DATASETS[@]}"; do
  out="$OUT_ROOT/$dataset"
  rank="$RANK_ROOT/$dataset"
  mkdir -p "$out/logs"
  export CAUSALITYRAG_READER_MODE=short_answer
  for index in 0 1; do
    [[ "$(wc -l < "$rank/shard_${index}.jsonl")" -eq 500 ]]
  done

  "$PYTHON" scripts/prepare_shapley_eval_shards.py scores \
    --score-root "$rank" --output "$out" --shards 2 --shard-size 500
  "$PYTHON" scripts/prepare_shapley_eval_shards.py prepare \
    --input "$DATA/$dataset/input.jsonl" \
    --units "$DATA/$dataset/units.jsonl" \
    --reflow "$EXTRA/$dataset/reflow.jsonl" \
    --scores "$out/scores_1000.jsonl" \
    --output "$out" --shards 2 --shard-size 500

  pool="$EXTRA/$dataset/factual_pool.jsonl"
  pool_sha="$(cut -d ' ' -f 1 < "$EXTRA/$dataset/factual_pool.sha256")"
  factual_shard() {
    local gpu="$1"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      scripts/evaluate_matched_budget_baselines.py \
      --input "$out/input_${gpu}.jsonl" \
      --units-cache "$out/units_${gpu}.jsonl" \
      --reflow-results "$out/reflow_${gpu}.jsonl" \
      --scores "token_leverage_shap=$out/scores_${gpu}.jsonl" \
      --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
      --out "$out/evaluation_${gpu}.jsonl" \
      --summary-out "$out/evaluation_${gpu}.summary.json" \
      --random-seeds "" --n 500 --k 5 --workers 1 \
      --model-path "$MODEL" --device cuda --dtype bfloat16 \
      >"$out/logs/evaluation_${gpu}.log" 2>&1
  }
  run_parallel factual_shard
  "$PYTHON" scripts/prepare_shapley_eval_shards.py merge \
    --output "$out" --shards 2 --shard-size 500 \
    >"$out/logs/merge.log" 2>&1

  "$PYTHON" scripts/prepare_shapley_eval_shards.py selected \
    --output "$out" --shards 2 --shard-size 500
  existing_pool="$EXTRA/$dataset/paraphrase_pool.jsonl"
  pool_shard() {
    local gpu="$1"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      scripts/build_paraphrase_pool_for_selected_units.py \
      --input "$out/input_${gpu}.jsonl" \
      --units-cache "$out/units_${gpu}.jsonl" \
      --selected-manifest "$out/selected_${gpu}.jsonl" \
      --existing-pool "$existing_pool" \
      --out "$out/paraphrase_pool_${gpu}.jsonl" \
      --manifest-out "$out/paraphrase_pool_${gpu}.manifest.json" \
      --unresolved-out "$out/paraphrase_pool_${gpu}.unresolved.jsonl" \
      --batch-size 16 --workers 1 --attempts 4 --max-candidates 3 \
      --model-path "$MODEL" --device cuda --dtype bfloat16 \
      >"$out/logs/paraphrase_pool_${gpu}.log" 2>&1
  }
  run_parallel pool_shard
  "$PYTHON" scripts/prepare_shapley_eval_shards.py merge-pools \
    --existing-pool "$existing_pool" --output "$out" --shards 2 \
    >"$out/logs/merge_paraphrase_pool.log" 2>&1
  control_pool="$out/paraphrase_pool_shapley.jsonl"
  control_sha="$(tr -d '[:space:]' < "$out/paraphrase_pool_shapley.sha256")"

  control_shard() {
    local gpu="$1"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      scripts/evaluate_paraphrase_controls.py \
      --input "$out/input_${gpu}.jsonl" \
      --units-cache "$out/units_${gpu}.jsonl" \
      --reflow-results "$out/reflow_${gpu}.jsonl" \
      --baseline-results "$out/evaluation_${gpu}.jsonl" \
      --paraphrase-pool "$control_pool" \
      --expected-pool-sha256 "$control_sha" \
      --out "$out/control_${gpu}.jsonl" \
      --summary-out "$out/control_${gpu}.summary.json" \
      --n 500 --k 5 --workers 1 --skip-reflow \
      --model-path "$MODEL" --device cuda --dtype bfloat16 \
      >"$out/logs/control_${gpu}.log" 2>&1
  }
  run_parallel control_shard
  "$PYTHON" scripts/prepare_shapley_eval_shards.py merge-controls \
    --output "$out" --shards 2 --shard-size 500
  "$PYTHON" scripts/summarize_shapley_eight_metrics.py \
    --dataset "$dataset" --retrieval "$DATA/$dataset/input.jsonl" \
    --factual "$out/evaluation_1000.jsonl" \
    --control "$out/control_1000.jsonl" \
    --out "$out/eight_metrics.json" \
    >"$out/logs/eight_metrics.log" 2>&1
  touch "$out/COMPLETE"
done

touch "$OUT_ROOT/SERVER3_COMPLETE"

