#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "2haoji-A100" ]] || {
  echo "This runner is restricted to server2." >&2
  exit 1
}

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
DATA="$ROOT/shapley_rank_bundle"
PYTHON=/data1/yujia/envs/token_shapley_smoke/bin/python
PACKAGES=/data1/yujia/envs/token_shapley_smoke_pkgs
MODEL=/data1/szgao/resources/pretrain/Qwen2.5-7B-Instruct
OUT="$ROOT/out/token_leverage_shap_final8_m128_3m"
DATASETS=(timeqa finqa musique quartz popqa)

mkdir -p "$OUT/logs"
export PYTHONPATH="$PACKAGES:$ROOT"
export CAUSALITYRAG_READER_MODE=short_answer

run_worker() {
  local gpu="$1"
  local dataset output summary log
  for dataset in "${DATASETS[@]}"; do
    output="$OUT/$dataset/shard_${gpu}.jsonl"
    summary="$OUT/$dataset/shard_${gpu}.summary.json"
    log="$OUT/logs/${dataset}_shard_${gpu}.log"
    mkdir -p "$OUT/$dataset"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      "$ROOT/exp/run_token_leverage_shap.py" \
      --input "$DATA/$dataset/input.jsonl" \
      --clean-reference "$DATA/$dataset/clean.jsonl" \
      --context-units "$DATA/$dataset/units.jsonl" \
      --model-path "$MODEL" \
      --out "$output" \
      --summary-out "$summary" \
      --start "$((gpu * 250))" \
      --n 250 \
      --k 5 \
      --top-k 128 \
      --attention-candidates 128 \
      --budget-multiplier 3 \
      --coalition-batch-size 32 \
      --seed 42 \
      --resume \
      >"$log" 2>&1
  done
}

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

for gpu in 0 1 2 3; do
  run_worker "$gpu" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
(( status == 0 ))

touch "$OUT/RANKING_COMPLETE"
pids=()
trap - INT TERM EXIT
