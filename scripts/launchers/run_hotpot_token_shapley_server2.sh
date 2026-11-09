#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
PACKAGES=/data1/yujia/envs/token_shapley_smoke_pkgs
MODEL=/data1/szgao/resources/pretrain/Qwen2.5-7B-Instruct
OUT="$ROOT/out/token_shapley_hotpot_perm8"

mkdir -p "$OUT/logs"

for shard in 0 1 2 3; do
  start=$((shard * 250))
  output="$OUT/shard_${shard}.jsonl"
  summary="$OUT/shard_${shard}.summary.json"
  log="$OUT/logs/shard_${shard}.log"
  pidfile="$OUT/shard_${shard}.pid"

  nohup bash -lc "
    cd '$ROOT'
    export CUDA_VISIBLE_DEVICES='$shard'
    export CAUSALITYRAG_READER_MODE=short_answer
    export PYTHONPATH='$PACKAGES:$ROOT'
    exec python3 exp/run_token_shapley.py \
      --input hotpot_top10_1000.jsonl \
      --clean-reference hotpot_clean_reference_1000.jsonl \
      --context-units hotpot_units_top5_1000.jsonl \
      --model-path '$MODEL' \
      --out '$output' \
      --summary-out '$summary' \
      --start '$start' \
      --n 250 \
      --k 5 \
      --top-k 128 \
      --permutations 8 \
      --coalition-batch-size 8 \
      --seed 42 \
      --resume
  " >"$log" 2>&1 < /dev/null &
  echo $! >"$pidfile"
done

for pidfile in "$OUT"/*.pid; do
  printf '%s ' "$pidfile"
  cat "$pidfile"
done
