#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
REPO=${REPO:-/data1/yujia/CausalityRAG_release_model_sweep_20260807}
RUNNER="$REPO/scripts/run_cross_model_reflow_qwen3_server3.sh"
OUT_ROOT=/data1/yujia/CausalityRAG/out/model_sensitivity_v1
MISTRAL_LOG="$OUT_ROOT/logs/mistral_direct.log"
QWEN_LOG="$OUT_ROOT/logs/qwen3_direct.log"
ENV_LOG="$OUT_ROOT/logs/qwen3_env_setup.log"
QWEN_READER_PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python

if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "this queue is restricted to server 3 ($EXPECTED_HOST)" >&2
  exit 1
fi

while screen -ls 2>/dev/null | grep -qE '[.]model_sweep_mistral_queue[[:space:]]'; do
  sleep 60
done

if ! grep -qF \
  '[cross-model] complete model=mistral-nemo-12b n=1000' "$MISTRAL_LOG"; then
  echo "Mistral ended without the n=1000 completion marker; Qwen3 not started" >&2
  exit 1
fi
if ! grep -qF 'READY ' "$ENV_LOG"; then
  echo "Qwen3 reader environment is not ready" >&2
  exit 1
fi

cd "$REPO"
READER_PYTHON="$QWEN_READER_PYTHON" "$RUNNER" \
  qwen3-4b \
  /data1/yujia/models/Qwen3-4B-Instruct-2507 \
  qwen3-4b-instruct-2507 \
  10 >>"$QWEN_LOG" 2>&1
READER_PYTHON="$QWEN_READER_PYTHON" "$RUNNER" \
  qwen3-4b \
  /data1/yujia/models/Qwen3-4B-Instruct-2507 \
  qwen3-4b-instruct-2507 \
  1000 >>"$QWEN_LOG" 2>&1
