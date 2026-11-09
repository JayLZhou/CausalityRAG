#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
PACKAGES=/data1/yujia/envs/token_shapley_smoke_pkgs
MODEL=/data1/szgao/resources/pretrain/Qwen2.5-7B-Instruct
OUT="$ROOT/out/attention_filtered_leverage_shap_hotpot_pilot"

mkdir -p "$OUT/logs"

run_one() {
  local gpu="$1"
  local index="$2"
  local seed="$3"
  local tag="index${index}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CAUSALITYRAG_READER_MODE=short_answer \
  PYTHONPATH="$PACKAGES:$ROOT" \
    python3 "$ROOT/exp/run_token_leverage_shap.py" \
      --input "$ROOT/hotpot_top10_1000.jsonl" \
      --clean-reference "$ROOT/hotpot_clean_reference_1000.jsonl" \
      --context-units "$ROOT/hotpot_units_top5_1000.jsonl" \
      --model-path "$MODEL" \
      --out "$OUT/${tag}.jsonl" \
      --summary-out "$OUT/${tag}.summary.json" \
      --start "$index" \
      --n 1 \
      --k 5 \
      --top-k 128 \
      --attention-candidates 128 \
      --budget-multiplier 3 \
      --coalition-batch-size 8 \
      --seed "$seed" \
      >"$OUT/logs/${tag}.log" 2>&1
}

run_one 0 0 42 &
pid_0=$!
run_one 1 250 42 &
pid_1=$!
run_one 2 500 42 &
pid_2=$!
run_one 3 750 42 &
pid_3=$!
wait "$pid_0" "$pid_1" "$pid_2" "$pid_3"

echo "$OUT"
