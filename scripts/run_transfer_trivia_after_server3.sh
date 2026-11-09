#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "3haoji-A100" ]] || exit 1

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
OUT="$ROOT/out/token_leverage_shap_final8_m128_3m"
DATA="$ROOT/shapley_rank_bundle/2wiki"
PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python
MODEL=/data1/yujia/models/Qwen2.5-7B-Instruct
ARCHIVE=/data1/yujia/token_leverage_shap_server3_final.tgz
SERVER2=yujia@10.249.42.56

main_pid="$(cat "$OUT/master.pid")"
while kill -0 "$main_pid" 2>/dev/null; do
  sleep 60
done
[[ -f "$OUT/RANKING_COMPLETE" ]]

mkdir -p "$OUT/2wiki" "$OUT/logs"
pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$ROOT" \
  CAUSALITYRAG_READER_MODE=short_answer \
  "$PYTHON" "$ROOT/exp/run_token_leverage_shap.py" \
    --input "$DATA/input.jsonl" \
    --clean-reference "$DATA/clean.jsonl" \
    --context-units "$DATA/units.jsonl" \
    --model-path "$MODEL" \
    --out "$OUT/2wiki/shard_${gpu}.jsonl" \
    --summary-out "$OUT/2wiki/shard_${gpu}.summary.json" \
    --start "$((gpu * 500))" --n 500 --k 5 --top-k 128 \
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

mkdir -p "$ROOT/out/token_leverage_shap_final8_m128_3m_eval"
"$ROOT/scripts/run_final8_token_shapley_eval_server3.sh" \
  >"$ROOT/out/token_leverage_shap_final8_m128_3m_eval/server3_master.log" 2>&1

tar -czf "$ARCHIVE" -C "$ROOT/out" \
  token_leverage_shap_final8_m128_3m/triviaqa \
  token_leverage_shap_final8_m128_3m/2wiki \
  token_leverage_shap_final8_m128_3m_eval/triviaqa \
  token_leverage_shap_final8_m128_3m_eval/2wiki
scp -q -o BatchMode=yes "$ARCHIVE" "$SERVER2:$ARCHIVE"
ssh -o BatchMode=yes "$SERVER2" \
  "tar -xzf '$ARCHIVE' -C '$ROOT/out' && touch '$OUT/TRIVIA_IMPORTED' '$OUT/2WIKI_COMPLETE' '$ROOT/out/token_leverage_shap_final8_m128_3m_eval/SERVER3_IMPORTED'"

"$ROOT/scripts/run_final8_shapley_budget_server3.sh" \
  >"$ROOT/out/token_leverage_shap_final8_m128_3m_budget_sweep/server3_master.log" 2>&1
tar -czf "$ARCHIVE" -C "$ROOT/out" \
  token_leverage_shap_final8_m128_3m_budget_sweep/triviaqa \
  token_leverage_shap_final8_m128_3m_budget_sweep/2wiki \
  token_leverage_shap_final8_m128_3m_budget_exact/triviaqa \
  token_leverage_shap_final8_m128_3m_budget_exact/2wiki
scp -q -o BatchMode=yes "$ARCHIVE" "$SERVER2:$ARCHIVE"
ssh -o BatchMode=yes "$SERVER2" \
  "tar -xzf '$ARCHIVE' -C '$ROOT/out' && touch '$ROOT/out/token_leverage_shap_final8_m128_3m_budget_sweep/SERVER3_BUDGET_IMPORTED' '$ROOT/out/token_leverage_shap_final8_m128_3m_budget_exact/SERVER3_BUDGET_IMPORTED'"
