#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU WAIT_PID START COUNT NAME" >&2
  exit 2
fi

gpu=$1
wait_pid=$2
start=$3
count=$4
name=$5

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
PACKAGES=/data1/yujia/envs/token_shapley_smoke_pkgs
MODEL=/data1/szgao/resources/pretrain/Qwen2.5-7B-Instruct
OUT="$ROOT/out/token_shapley_hotpot_perm8"

while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="$gpu"
export CAUSALITYRAG_READER_MODE=short_answer
export PYTHONPATH="$PACKAGES:$ROOT"
cd "$ROOT"

exec python3 exp/run_token_shapley.py \
  --input hotpot_top10_1000.jsonl \
  --clean-reference hotpot_clean_reference_1000.jsonl \
  --context-units hotpot_units_top5_1000.jsonl \
  --model-path "$MODEL" \
  --out "$OUT/${name}.jsonl" \
  --summary-out "$OUT/${name}.summary.json" \
  --start "$start" \
  --n "$count" \
  --k 5 \
  --top-k 128 \
  --permutations 8 \
  --coalition-batch-size 8 \
  --seed 42
