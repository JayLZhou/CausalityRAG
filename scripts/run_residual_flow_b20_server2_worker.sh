#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 GPU PORT DATASET [DATASET ...]" >&2
  exit 2
fi

GPU_INDEX="$1"
PORT="$2"
shift 2
DATASETS=("$@")

RUN_ROOT="/data1/yujia/CausalityRAG_rmf_b20_20260827"
CODE_ROOT="${RUN_ROOT}/code"
SOURCE_ROOT="${RUN_ROOT}/source/k5"
PYTHON_BIN="/data1/yujia/envs/graphrag/bin/python"
MODEL_DIR="/data1/public/models/Qwen2.5-7B-Instruct"
SERVED_MODEL="qwen2.5-7b"
BASE_URL="http://127.0.0.1:${PORT}/v1"
WORKER_LOG="${RUN_ROOT}/logs/worker_gpu${GPU_INDEX}.log"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/out"
exec > >(tee -a "${WORKER_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] [gpu%s] %s\n' "$(timestamp)" "${GPU_INDEX}" "$*"
}

wait_for_file() {
  local path="$1"
  while [[ ! -s "${path}" ]]; do
    sleep 10
  done
}

log "waiting for the verified environment and frozen inputs"
wait_for_file "${PYTHON_BIN}"
wait_for_file "${MODEL_DIR}/config.json"
wait_for_file "${CODE_ROOT}/scripts/run_residual_flow_fidelity.py"
for dataset in "${DATASETS[@]}"; do
  wait_for_file "${SOURCE_ROOT}/${dataset}/inputs/retrieval.jsonl"
  wait_for_file "${SOURCE_ROOT}/${dataset}/inputs/units.jsonl"
  wait_for_file "${SOURCE_ROOT}/${dataset}/inputs/shared_pool.jsonl"
  wait_for_file "${SOURCE_ROOT}/${dataset}/graph/contribution_graph.jsonl"
done

"${PYTHON_BIN}" -c 'import torch, transformers, vllm'

gpu_pids="$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ "${gpu_pids}" =~ [0-9] ]]; then
  log "GPU already has compute process(es): ${gpu_pids}; refusing to interfere"
  exit 1
fi
if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  log "port ${PORT} already has a health endpoint; refusing to attach"
  exit 1
fi

VLLM_LOG="${RUN_ROOT}/logs/vllm_gpu${GPU_INDEX}.log"
log "starting Qwen2.5-7B reader on port ${PORT}"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" setsid "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.88 \
  --disable-log-requests \
  --port "${PORT}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "stopping owned vLLM process group ${VLLM_PID}"
    kill -- "-${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 240); do
  if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "vLLM exited before readiness; inspect ${VLLM_LOG}"
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  log "vLLM readiness timed out"
  exit 1
fi

export CAUSALITYRAG_READER_MAX_TOKENS=64
for dataset in "${DATASETS[@]}"; do
  out_dir="${RUN_ROOT}/out/${dataset}/fidelity_b20"
  summary="${out_dir}/summary.json"
  mkdir -p "${out_dir}"
  if [[ -s "${summary}" ]]; then
    log "${dataset}: complete summary exists; skipping"
    continue
  fi
  log "${dataset}: starting 100 queries x 10 trials"
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/run_residual_flow_fidelity.py" \
    --input "${SOURCE_ROOT}/${dataset}/inputs/retrieval.jsonl" \
    --units-cache "${SOURCE_ROOT}/${dataset}/inputs/units.jsonl" \
    --graphs "${SOURCE_ROOT}/${dataset}/graph/contribution_graph.jsonl" \
    --shared-pool "${SOURCE_ROOT}/${dataset}/inputs/shared_pool.jsonl" \
    --out "${out_dir}/probes_1000.jsonl" \
    --summary-out "${summary}" \
    --n-queries 100 \
    --budgets 20 \
    --trials 10 \
    --workers 32 \
    --k 5 \
    --seed 0 \
    --llm-base-url "${BASE_URL}" \
    --llm-model "${SERVED_MODEL}" \
    >"${RUN_ROOT}/logs/${dataset}_b20.log" 2>&1
  touch "${out_dir}/COMPLETE"
  log "${dataset}: complete"
done

touch "${RUN_ROOT}/WORKER_GPU${GPU_INDEX}_COMPLETE"
log "worker complete"

