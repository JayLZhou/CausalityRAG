#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
REPO=${REPO:-/data1/yujia/CausalityRAG_release_model_sweep_20260807}
RUNNER="$REPO/scripts/launchers/run_cross_model_reflow_server3.sh"
OUT_ROOT=/data1/yujia/CausalityRAG/out/model_sensitivity_v1
LLAMA_LOG="$OUT_ROOT/logs/llama_direct.log"
MISTRAL_LOG="$OUT_ROOT/logs/mistral_direct.log"

if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "this queue is restricted to server 3 ($EXPECTED_HOST)" >&2
  exit 1
fi

while screen -ls 2>/dev/null | grep -qE '[.]model_sweep_llama[[:space:]]'; do
  sleep 60
done

if ! grep -qF \
  '[cross-model] complete model=llama3.1-8b n=1000' "$LLAMA_LOG"; then
  echo "Llama run ended without the n=1000 completion marker; Mistral not started" >&2
  exit 1
fi

cd "$REPO"
"$RUNNER" \
  mistral-nemo-12b \
  /data1/yujia/models/Mistral-Nemo-Instruct-2407 \
  mistral-nemo-instruct-2407 \
  10 >>"$MISTRAL_LOG" 2>&1
"$RUNNER" \
  mistral-nemo-12b \
  /data1/yujia/models/Mistral-Nemo-Instruct-2407 \
  mistral-nemo-instruct-2407 \
  1000 >>"$MISTRAL_LOG" 2>&1
