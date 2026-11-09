#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
RANK="$ROOT/out/token_leverage_shap_final8_m128_3m"
EVAL="$ROOT/out/token_leverage_shap_final8_m128_3m_eval"

while [[ ! -f "$RANK/RANKING_COMPLETE" ]]; do
  sleep 60
done

mkdir -p "$EVAL"
"$ROOT/scripts/run_final8_token_shapley_eval_server2.sh" \
  >"$EVAL/master.log" 2>&1

while [[ ! -f "$EVAL/SERVER3_IMPORTED" ]]; do
  sleep 60
done
touch "$EVAL/FINAL8_COMPLETE"
