#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "3haoji-A100" ]] || {
  echo "This runner is restricted to server3." >&2
  exit 1
}

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
SOURCE=/data1/yujia/CausalityRAG
PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python
MODEL=/data1/yujia/models/Qwen2.5-7B-Instruct
OUT="$ROOT/out/token_leverage_shap_final8_m128_3m"
DATASETS=(triviaqa)

mkdir -p "$OUT/logs"
export PYTHONPATH="$ROOT"
export CAUSALITYRAG_READER_MODE=short_answer

input_path() {
  printf '%s/out_table3_final8_20260811/%s/retrieval/top10_1000.jsonl' \
    "$SOURCE" "$1"
}

clean_path() {
  case "$1" in
    triviaqa) printf '%s/out/triviaqa/inputs/clean_targets_top5_1000.jsonl' "$SOURCE" ;;
    2wiki) printf '%s/out/2wiki/inputs/clean_reference_top5_1000.jsonl' "$SOURCE" ;;
  esac
}

units_path() {
  case "$1" in
    triviaqa) printf '%s/out/triviaqa/inputs/context_units_top5_1000.jsonl' "$SOURCE" ;;
    2wiki) printf '%s/out/2wiki/analysis_v1/inputs/token_units_top5_1000.jsonl' "$SOURCE" ;;
  esac
}

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
      --input "$(input_path "$dataset")" \
      --clean-reference "$(clean_path "$dataset")" \
      --context-units "$(units_path "$dataset")" \
      --model-path "$MODEL" \
      --out "$output" \
      --summary-out "$summary" \
      --start "$((gpu * 500))" \
      --n 500 \
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

for gpu in 0 1; do
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
