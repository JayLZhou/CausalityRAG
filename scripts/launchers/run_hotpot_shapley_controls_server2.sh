#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "2haoji-A100" ]] || {
  echo "This runner is restricted to server2." >&2
  exit 1
}

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
PYTHON=/data1/yujia/envs/token_shapley_smoke/bin/python
PACKAGES=/data1/yujia/envs/token_shapley_smoke_pkgs
MODEL=/data1/szgao/resources/pretrain/Qwen2.5-7B-Instruct
OUT="$ROOT/out/attention_filtered_leverage_shap_hotpot_m128_3m_eval"
EXISTING_POOL="$ROOT/hotpot_paraphrase_pool_v3.jsonl"
export PYTHONPATH="$PACKAGES:$ROOT"
export NLTK_DATA="$ROOT/nltk_data"
export CAUSALITYRAG_READER_MODE=short_answer

"$PYTHON" scripts/prepare_shapley_eval_shards.py selected --output "$OUT"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$index" "$PYTHON" \
    scripts/build_paraphrase_pool_for_selected_units.py \
    --input "$OUT/input_${index}.jsonl" \
    --units-cache "$OUT/units_${index}.jsonl" \
    --selected-manifest "$OUT/selected_${index}.jsonl" \
    --existing-pool "$EXISTING_POOL" \
    --out "$OUT/paraphrase_pool_${index}.jsonl" \
    --manifest-out "$OUT/paraphrase_pool_${index}.manifest.json" \
    --unresolved-out "$OUT/paraphrase_pool_${index}.unresolved.jsonl" \
    --batch-size 16 --workers 1 --attempts 4 --max-candidates 3 \
    --model-path "$MODEL" --device cuda --dtype bfloat16 \
    >"$OUT/logs/paraphrase_pool_${index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
(( status == 0 ))
pids=()

"$PYTHON" scripts/prepare_shapley_eval_shards.py merge-pools \
  --existing-pool "$EXISTING_POOL" --output "$OUT" \
  >"$OUT/logs/merge_paraphrase_pool.log" 2>&1
CONTROL_POOL="$OUT/paraphrase_pool_shapley.jsonl"
CONTROL_SHA="$(tr -d '[:space:]' < "$OUT/paraphrase_pool_shapley.sha256")"

for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$index" "$PYTHON" \
    scripts/evaluate_paraphrase_controls.py \
    --input "$OUT/input_${index}.jsonl" \
    --units-cache "$OUT/units_${index}.jsonl" \
    --reflow-results "$OUT/reflow_${index}.jsonl" \
    --baseline-results "$OUT/evaluation_${index}.jsonl" \
    --paraphrase-pool "$CONTROL_POOL" \
    --expected-pool-sha256 "$CONTROL_SHA" \
    --out "$OUT/control_${index}.jsonl" \
    --summary-out "$OUT/control_${index}.summary.json" \
    --n 250 --k 5 --workers 1 --skip-reflow \
    --model-path "$MODEL" --device cuda --dtype bfloat16 \
    >"$OUT/logs/control_${index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
(( status == 0 ))
pids=()

"$PYTHON" scripts/prepare_shapley_eval_shards.py merge-controls --output "$OUT"
"$PYTHON" scripts/summarize_shapley_eight_metrics.py \
  --dataset hotpotqa \
  --retrieval "$ROOT/hotpot_top10_1000.jsonl" \
  --factual "$OUT/evaluation_1000.jsonl" \
  --control "$OUT/control_1000.jsonl" \
  --out "$OUT/eight_metrics.json" \
  >"$OUT/logs/eight_metrics.log" 2>&1
touch "$OUT/CONTROLS_COMPLETE"
trap - INT TERM EXIT
