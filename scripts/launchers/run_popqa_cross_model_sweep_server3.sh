#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_popqa_models_20260811}
RUNNER=$RELEASE/scripts/launchers/run_popqa_cross_model_server3.sh
ANALYSIS_COMPLETE=${ANALYSIS_COMPLETE:-/data1/yujia/CausalityRAG/out_popqa_figures_20260811/popqa/figure8/REFINE_COMPLETE}
LOG_ROOT=${LOG_ROOT:-/data1/yujia/CausalityRAG/out/model_sensitivity_popqa_v1/logs}
mkdir -p "$LOG_ROOT"

while [[ ! -f "$ANALYSIS_COMPLETE" ]]; do
  echo "[popqa-model-sweep] waiting for Qwen2.5 PopQA analyses"
  sleep 30
done

run_one() {
  local key=$1 path=$2 served=$3
  [[ -s "$path/config.json" ]] || {
    echo "missing model config: $path" >&2
    exit 1
  }
  if find "$path" -path '*/quarantine_*' -prune -o \
    -name '*.incomplete' -print -quit | grep -q .; then
    echo "model has a non-quarantined incomplete file: $path" >&2
    exit 1
  fi
  echo "[popqa-model-sweep] start model=$key"
  "$RUNNER" "$key" "$path" "$served" \
    >"$LOG_ROOT/${key}.log" 2>&1
  echo "[popqa-model-sweep] complete model=$key"
}

run_one qwen3-4b \
  /data1/yujia/models/Qwen3-4B-Instruct-2507 \
  qwen3-4b-instruct-2507
run_one llama3.1-8b \
  /data1/yujia/models/Llama-3.1-8B-Instruct \
  llama-3.1-8b-instruct
run_one mistral-nemo-12b \
  /data1/yujia/models/Mistral-Nemo-Instruct-2407 \
  mistral-nemo-instruct-2407

date -Is >/data1/yujia/CausalityRAG/out/model_sensitivity_popqa_v1/COMPLETE
echo '[popqa-model-sweep] all added models complete'
