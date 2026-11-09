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
POOL="$ROOT/hotpot_shared_pool_top10_final.jsonl"
POOL_SHA=5f161ed2405becbda7cd39517d1d1291c562574a404c975e42f2d91ad8b31bef
RANK_ROOT="$ROOT/out/attention_filtered_leverage_shap_hotpot_m128_3m"
OUT="$ROOT/out/attention_filtered_leverage_shap_hotpot_m128_3m_eval"
mkdir -p "$OUT/logs"

for index in 0 1 2 3; do
  [[ "$(wc -l < "$RANK_ROOT/shard_${index}.jsonl")" -eq 250 ]]
done
[[ "$(sha256sum "$POOL" | cut -d ' ' -f 1)" == "$POOL_SHA" ]]

"$PYTHON" scripts/prepare_shapley_eval_shards.py scores \
  --score-root "$RANK_ROOT" --output "$OUT"
"$PYTHON" scripts/prepare_shapley_eval_shards.py prepare \
  --input hotpot_top10_1000.jsonl \
  --units hotpot_units_top5_1000.jsonl \
  --reflow hotpot_reflow_1000.jsonl \
  --scores "$OUT/scores_1000.jsonl" \
  --output "$OUT"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$index" \
  PYTHONPATH="$PACKAGES:$ROOT" \
  CAUSALITYRAG_READER_MODE=short_answer \
  "$PYTHON" scripts/evaluate_matched_budget_baselines.py \
    --input "$OUT/input_${index}.jsonl" \
    --units-cache "$OUT/units_${index}.jsonl" \
    --reflow-results "$OUT/reflow_${index}.jsonl" \
    --scores "token_leverage_shap=$OUT/scores_${index}.jsonl" \
    --shared-pool "$POOL" \
    --expected-pool-sha256 "$POOL_SHA" \
    --out "$OUT/evaluation_${index}.jsonl" \
    --summary-out "$OUT/evaluation_${index}.summary.json" \
    --random-seeds "" \
    --n 250 --k 5 --workers 1 \
    --model-path "$MODEL" --device cuda --dtype bfloat16 \
    >"$OUT/logs/evaluation_${index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
(( status == 0 ))

"$PYTHON" scripts/prepare_shapley_eval_shards.py merge --output "$OUT" \
  >"$OUT/logs/merge.log" 2>&1
touch "$OUT/COMPLETE"
pids=()
trap - INT TERM EXIT
