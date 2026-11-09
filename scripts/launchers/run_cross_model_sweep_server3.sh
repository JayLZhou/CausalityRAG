#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
REPO=${REPO:-/data1/yujia/CausalityRAG_release_model_sweep_20260807}
RUNNER="$REPO/scripts/launchers/run_cross_model_reflow_server3.sh"
LOG_ROOT=/data1/yujia/CausalityRAG/out/model_sensitivity_v1/logs

if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "this runner is restricted to server 3 ($EXPECTED_HOST)" >&2
  exit 1
fi
mkdir -p "$LOG_ROOT"

wait_for_model() {
  local name=$1
  local path=$2
  local pattern=$3
  while find "$path" -name '*.incomplete' -print -quit | grep -q .; do
    if ! pgrep -f "$pattern" >/dev/null; then
      echo "$name download stopped with incomplete files" >&2
      exit 1
    fi
    echo "[model-sweep] waiting for $name download"
    sleep 60
  done
  [[ -s "$path/config.json" ]] || { echo "$name config missing" >&2; exit 1; }
  echo "[model-sweep] model ready: $name"
}

run_one() {
  local key=$1
  local path=$2
  local served=$3
  echo "[model-sweep] smoke: $key"
  "$RUNNER" "$key" "$path" "$served" 10
  echo "[model-sweep] full: $key"
  "$RUNNER" "$key" "$path" "$served" 1000
}

QWEN3=/data1/yujia/models/Qwen3-4B-Instruct-2507
LLAMA=/data1/yujia/models/Llama-3.1-8B-Instruct
MISTRAL=/data1/yujia/models/Mistral-Nemo-Instruct-2407

wait_for_model qwen3-4b "$QWEN3" \
  'huggingface-cli download Qwen/Qwen3-4B-Instruct-2507'

# The larger downloads were paused only to give the replacement backbone the
# mirror bandwidth first. Resume them before GPU evaluation begins.
pkill -CONT -f '[h]uggingface-cli download NousResearch/Meta-Llama-3.1-8B-Instruct' 2>/dev/null || true
pkill -CONT -f '[h]uggingface-cli download mistralai/Mistral-Nemo-Instruct-2407' 2>/dev/null || true

run_one qwen3-4b "$QWEN3" qwen3-4b-instruct-2507

wait_for_model llama3.1-8b "$LLAMA" \
  'huggingface-cli download NousResearch/Meta-Llama-3.1-8B-Instruct'
run_one llama3.1-8b "$LLAMA" llama-3.1-8b-instruct

wait_for_model mistral-nemo-12b "$MISTRAL" \
  'huggingface-cli download mistralai/Mistral-Nemo-Instruct-2407'
run_one mistral-nemo-12b "$MISTRAL" mistral-nemo-instruct-2407

echo "[model-sweep] all three models complete"
