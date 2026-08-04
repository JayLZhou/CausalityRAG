#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/yujia/CausalityRAG/out
RELEASE=/data1/yujia/CausalityRAG_release_prefix_answer_frontier_20260805
PY=/data1/yujia/envs/graphrag/bin/python
MODEL=/data1/yujia/models/Qwen2.5-7B-Instruct
LOG_DIR="$ROOT/prefix_answer_frontier_logs"
mkdir -p "$LOG_DIR"

wait_for_gpus() {
  while true; do
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ ${#used[@]} -ge 2 && ${used[0]} -lt 2000 && ${used[1]} -lt 2000 ]]; then
      sleep 60
      mapfile -t confirmed < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      if [[ ${confirmed[0]} -lt 2000 && ${confirmed[1]} -lt 2000 ]]; then
        return
      fi
    fi
    printf '[wait-gpu] %s gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Is)" "${used[0]:-unknown}" "${used[1]:-unknown}"
    sleep 60
  done
}

start_server() {
  local gpu=$1
  local port=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name qwen2.5-7b \
    --port "$port" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.82 \
    --max-num-seqs 128 \
    --enable-chunked-prefill \
    --trust-remote-code \
    --disable-log-requests \
    >"$LOG_DIR/vllm_${port}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port=$1
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    sleep 5
  done
}

dataset_paths() {
  local dataset=$1
  INPUT="$ROOT/$dataset/retrieval/top10_1000.jsonl"
  UNITS="$ROOT/$dataset/inputs/token_units_top10_1000.jsonl"
  GRAPH="$ROOT/$dataset/graphs/contribution_graph_top5_1000.jsonl"
  POOL="$ROOT/$dataset/replacements/shared_pool_top10_v1/shared_pool.jsonl"
  if [[ $dataset == hotpotqa ]]; then
    UNITS="$ROOT/hotpotqa/inputs/token_units_top5_1000.jsonl"
    GRAPH="$ROOT/hotpotqa/graphs/contribution_graph_token_label_1000.jsonl"
    POOL="$ROOT/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl"
  fi
}

run_dataset() {
  local dataset=$1
  local port=$2
  dataset_paths "$dataset"
  local old="$ROOT/$dataset/analysis_v1/prefix_n1_10"
  local out="$ROOT/$dataset/analysis_v1/prefix_n1_10_answer_cfr"
  mkdir -p "$out"
  printf '[dataset-start] %s %s\n' "$(date -Is)" "$dataset"
  "$PY" "$RELEASE/scripts/evaluate_prefix_answer_frontier.py" \
    --input "$INPUT" \
    --units-cache "$UNITS" \
    --graphs "$GRAPH" \
    --shared-pool "$POOL" \
    --prefix-results "$old/results.jsonl" \
    --paraphrase-pool "$old/paraphrase_pool.jsonl" \
    --out "$out/results.jsonl" \
    --summary-out "$out/summary.json" \
    --workers 48 \
    --k 5 \
    --seed 0 \
    --llm-base-url "http://127.0.0.1:${port}/v1" \
    --llm-model qwen2.5-7b \
    >"$LOG_DIR/${dataset}.log" 2>&1
  printf '[dataset-done] %s %s\n' "$(date -Is)" "$dataset"
}

run_lane() {
  local port=$1
  shift
  local dataset
  for dataset in "$@"; do
    run_dataset "$dataset" "$port"
  done
}

wait_for_gpus
SERVER_ONE=$(start_server 0 8011)
SERVER_TWO=$(start_server 1 8012)
cleanup() {
  kill "$SERVER_ONE" "$SERVER_TWO" 2>/dev/null || true
}
trap cleanup EXIT
wait_for_server 8011
wait_for_server 8012

run_lane 8011 hotpotqa timeqa finqa qasper &
LANE_ONE=$!
run_lane 8012 2wiki musique pubmedqa quartz &
LANE_TWO=$!
wait "$LANE_ONE"
wait "$LANE_TWO"

SUMMARY_ARGS=()
for dataset in hotpotqa 2wiki timeqa musique finqa pubmedqa qasper quartz; do
  SUMMARY_ARGS+=(
    --summary
    "$dataset=$ROOT/$dataset/analysis_v1/prefix_n1_10_answer_cfr/summary.json"
  )
done
"$PY" "$RELEASE/scripts/aggregate_prefix_answer_frontier.py" \
  "${SUMMARY_ARGS[@]}" \
  --out "$ROOT/prefix_answer_frontier_all8.json" \
  >"$LOG_DIR/aggregate.log" 2>&1
printf '[all-done] %s\n' "$(date -Is)"
