#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
PACKAGES=/data1/yujia/envs/token_shapley_smoke_pkgs
MODEL=/data1/szgao/resources/pretrain/Qwen2.5-7B-Instruct
SHAPLEY_PID_FILE="$ROOT/out/attention_filtered_leverage_shap_hotpot_m128_3m/shard_2.pid"
OUT="$ROOT/out/popqa_000137_singletons.json"

while [[ -f "$SHAPLEY_PID_FILE" ]] && kill -0 "$(cat "$SHAPLEY_PID_FILE")" 2>/dev/null; do
  sleep 10
done

cd "$ROOT"
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH="$PACKAGES:$ROOT" \
  python3 scripts/validate_two_token_singletons.py \
    --input popqa_top10_1000.jsonl \
    --reflow popqa_reflow_1000.jsonl \
    --id popqa-000137 \
    --model-path "$MODEL" \
    --out "$OUT" \
    --reader-mode short_answer \
    --device cuda \
    --dtype bfloat16
