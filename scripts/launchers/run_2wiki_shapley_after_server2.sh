#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "2haoji-A100" ]] || exit 1

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
DATA="$ROOT/shapley_rank_bundle/2wiki"
PYTHON=/data1/yujia/envs/token_shapley_smoke/bin/python
PACKAGES=/data1/yujia/envs/token_shapley_smoke_pkgs
MODEL=/data1/szgao/resources/pretrain/Qwen2.5-7B-Instruct
OUT="$ROOT/out/token_leverage_shap_final8_m128_3m"

main_pid="$(cat "$OUT/master.pid")"
while kill -0 "$main_pid" 2>/dev/null; do
  sleep 60
done
[[ -f "$OUT/RANKING_COMPLETE" ]]

mkdir -p "$OUT/2wiki" "$OUT/logs"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$PACKAGES:$ROOT" \
  CAUSALITYRAG_READER_MODE=short_answer \
  "$PYTHON" "$ROOT/exp/run_token_leverage_shap.py" \
    --input "$DATA/input.jsonl" \
    --clean-reference "$DATA/clean.jsonl" \
    --context-units "$DATA/units.jsonl" \
    --model-path "$MODEL" \
    --out "$OUT/2wiki/shard_${gpu}.jsonl" \
    --summary-out "$OUT/2wiki/shard_${gpu}.summary.json" \
    --start "$((gpu * 250))" --n 250 --k 5 --top-k 128 \
    --attention-candidates 128 --budget-multiplier 3 \
    --coalition-batch-size 32 --seed 42 --resume \
    >"$OUT/logs/2wiki_shard_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
(( status == 0 ))
touch "$OUT/2WIKI_COMPLETE"
