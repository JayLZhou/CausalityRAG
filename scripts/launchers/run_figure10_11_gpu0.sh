#!/usr/bin/env bash
set -euo pipefail

# Run the missing Figure 10/11 reader evaluations on machine 14, GPU 0 only.
# The destination root is dedicated to this run. Existing completed summaries
# are reused, fidelity JSONL files are resumable, and partial protocol outputs
# are never overwritten automatically.

CODE_ROOT="/data2/yujia/CausalityRAG_fig8_fig9_20260804"
RUN_ROOT="/data2/yujia/CausalityRAG_fig10_20260804"
SENTENCE_ROOT="${CODE_ROOT}/sentence_sufficiency/inputs"
PYTHON_BIN="/data2/yujia/envs/graphrag/bin/python"
MODEL_DIR="/data2/yujia/models/Qwen2.5-7B-Instruct"
SERVED_MODEL="qwen2.5-7b"
PORT="8102"
BASE_URL="http://127.0.0.1:${PORT}/v1"
GPU_INDEX="0"

DATASETS=(hotpotqa timeqa finqa musique quartz qasper 2wiki pubmedqa)
MISSING_FIDELITY=(hotpotqa timeqa finqa musique quartz qasper pubmedqa)
MISSING_PROTOCOLS=(timeqa finqa musique quartz qasper 2wiki pubmedqa)

declare -A POOL_SHA=(
  [hotpotqa]="5f161ed2405becbda7cd39517d1d1291c562574a404c975e42f2d91ad8b31bef"
  [timeqa]="01f19f09fe2842006b036823ca820d649ce14fd23d41a74793f7f21f62fe8b44"
  [finqa]="e2276863d19c5c4f9148016eef4550128a4516baaa039e1e36e62db03ccdadb6"
  [musique]="335bc1027e62c72973c97f57f4b20fc769601b16698411b42be2dd3319eac31e"
  [quartz]="724115a67b2b8af90f74cfaeea47270e0412f7250b75a66d421f232d6cd148e6"
  [qasper]="810179ebf125c8c92b2773619cebe736bbeee71725c0aa12069ba1b0a56004c1"
  [2wiki]="968488f69f032fbea66405b385fe8c227ce92c018632812d6d6486ddfe2be19d"
  [pubmedqa]="08e69f1c60106def06f66e2d3c20551628e8a7478f47e2ba98b9a6015cd0e655"
)

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

units_path() {
  local dataset="$1"
  local top5="${SENTENCE_ROOT}/${dataset}/token_units_top5_1000.jsonl"
  local top10="${SENTENCE_ROOT}/${dataset}/token_units_top10_1000.jsonl"
  if [[ -s "${top5}" ]]; then
    printf '%s\n' "${top5}"
  elif [[ -s "${top10}" ]]; then
    printf '%s\n' "${top10}"
  else
    log "missing unit cache for ${dataset}" >&2
    return 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    log "required file is missing or empty: ${path}" >&2
    exit 1
  fi
}

mkdir -p "${RUN_ROOT}/out" "${RUN_ROOT}/logs"
require_file "${PYTHON_BIN}"
require_file "${MODEL_DIR}/config.json"
require_file "${CODE_ROOT}/scripts/run_residual_flow_fidelity.py"
require_file "${CODE_ROOT}/scripts/build_selection_protocol_frontiers.py"
require_file "${CODE_ROOT}/scripts/evaluate_reflow.py"

if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  log "port ${PORT} already serves a health endpoint; refusing to attach to an unknown service" >&2
  exit 1
fi

gpu_pids="$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ "${gpu_pids}" =~ [0-9] ]]; then
  log "GPU ${GPU_INDEX} already has compute process(es): ${gpu_pids}" >&2
  exit 1
fi

log "preflight: validating frozen inputs and pool hashes"
for dataset in "${DATASETS[@]}"; do
  require_file "${RUN_ROOT}/inputs/${dataset}/graph.jsonl"
  require_file "${RUN_ROOT}/inputs/${dataset}/frontier.jsonl"
  require_file "${RUN_ROOT}/inputs/${dataset}/shared_pool.jsonl"
  require_file "${SENTENCE_ROOT}/${dataset}/retrieval_top10_1000.jsonl"
  units_path "${dataset}" >/dev/null
  actual_sha="$(sha256sum "${RUN_ROOT}/inputs/${dataset}/shared_pool.jsonl" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${POOL_SHA[${dataset}]}" ]]; then
    log "pool SHA mismatch for ${dataset}: ${actual_sha}" >&2
    exit 1
  fi
done

log "building frozen One-shot and Grid protocol frontiers on CPU"
for dataset in "${MISSING_PROTOCOLS[@]}"; do
  protocol_dir="${RUN_ROOT}/out/${dataset}/protocols"
  mkdir -p "${protocol_dir}"
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/build_selection_protocol_frontiers.py" \
    --frontier "${RUN_ROOT}/inputs/${dataset}/frontier.jsonl" \
    --one-shot-out "${protocol_dir}/one_shot_frontier.jsonl" \
    --sweep-out "${protocol_dir}/fixed_grid_frontier.jsonl" \
    >"${RUN_ROOT}/logs/${dataset}_build_protocols.log" 2>&1
done

VLLM_LOG="${RUN_ROOT}/logs/vllm_gpu0.log"
log "starting vLLM on GPU ${GPU_INDEX}, port ${PORT}"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" setsid "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.82 \
  --disable-log-requests \
  --port "${PORT}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "stopping vLLM process group ${VLLM_PID}"
    kill -- "-${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "vLLM exited before becoming ready; inspect ${VLLM_LOG}" >&2
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  log "vLLM did not become ready within 15 minutes" >&2
  exit 1
fi
log "vLLM is ready"

run_fidelity() {
  local dataset="$1"
  local out_dir="${RUN_ROOT}/out/${dataset}/fidelity"
  local summary="${out_dir}/summary.json"
  if [[ -s "${summary}" ]]; then
    log "fidelity ${dataset}: complete summary exists; skipping"
    return 0
  fi
  mkdir -p "${out_dir}"
  log "fidelity ${dataset}: starting"
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/run_residual_flow_fidelity.py" \
    --input "${SENTENCE_ROOT}/${dataset}/retrieval_top10_1000.jsonl" \
    --units-cache "$(units_path "${dataset}")" \
    --graphs "${RUN_ROOT}/inputs/${dataset}/graph.jsonl" \
    --shared-pool "${RUN_ROOT}/inputs/${dataset}/shared_pool.jsonl" \
    --out "${out_dir}/probes.jsonl" \
    --summary-out "${summary}" \
    --n-queries 100 \
    --budgets 1 3 5 \
    --trials 10 \
    --workers 8 \
    --k 5 \
    --seed 0 \
    --llm-base-url "${BASE_URL}" \
    --llm-model "${SERVED_MODEL}" \
    >"${RUN_ROOT}/logs/${dataset}_fidelity.log" 2>&1
  log "fidelity ${dataset}: complete"
}

run_protocol() {
  local dataset="$1"
  local protocol="$2"
  local out_dir="${RUN_ROOT}/out/${dataset}/protocols"
  local frontier results summary
  if [[ "${protocol}" == "one_shot" ]]; then
    frontier="${out_dir}/one_shot_frontier.jsonl"
  else
    frontier="${out_dir}/fixed_grid_frontier.jsonl"
  fi
  results="${out_dir}/${protocol}_results.jsonl"
  summary="${out_dir}/${protocol}_summary.json"
  if [[ -s "${summary}" ]]; then
    log "${protocol} ${dataset}: complete summary exists; skipping"
    return 0
  fi
  if [[ -e "${results}" ]]; then
    log "${protocol} ${dataset}: partial results exist without summary; refusing to overwrite ${results}" >&2
    return 1
  fi
  log "${protocol} ${dataset}: starting"
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/evaluate_reflow.py" \
    --input "${SENTENCE_ROOT}/${dataset}/retrieval_top10_1000.jsonl" \
    --units-cache "$(units_path "${dataset}")" \
    --frontier "${frontier}" \
    --shared-pool "${RUN_ROOT}/inputs/${dataset}/shared_pool.jsonl" \
    --expected-pool-sha256 "${POOL_SHA[${dataset}]}" \
    --out "${results}" \
    --summary-out "${summary}" \
    --n 1000 \
    --k 5 \
    --workers 8 \
    --replacement-seed 0 \
    --llm-base-url "${BASE_URL}" \
    --llm-model "${SERVED_MODEL}" \
    >"${RUN_ROOT}/logs/${dataset}_${protocol}.log" 2>&1
  log "${protocol} ${dataset}: complete"
}

run_parallel_group() {
  local label="$1"
  shift
  local -a pids=()
  local -a names=()
  local item dataset protocol
  log "${label}: launching $# dataset jobs"
  for item in "$@"; do
    dataset="${item%%:*}"
    protocol="${item#*:}"
    if [[ "${protocol}" == "fidelity" ]]; then
      run_fidelity "${dataset}" &
    else
      run_protocol "${dataset}" "${protocol}" &
    fi
    pids+=("$!")
    names+=("${item}")
  done
  local failed=0 index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
      log "${label}: FAILED ${names[${index}]}" >&2
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    return 1
  fi
  log "${label}: all jobs complete"
}

fidelity_jobs=()
for dataset in "${MISSING_FIDELITY[@]}"; do
  fidelity_jobs+=("${dataset}:fidelity")
done
run_parallel_group "fidelity" "${fidelity_jobs[@]}"

one_shot_jobs=()
for dataset in "${MISSING_PROTOCOLS[@]}"; do
  one_shot_jobs+=("${dataset}:one_shot")
done
run_parallel_group "one-shot" "${one_shot_jobs[@]}"

grid_jobs=()
for dataset in "${MISSING_PROTOCOLS[@]}"; do
  grid_jobs+=("${dataset}:fixed_grid")
done
run_parallel_group "grid" "${grid_jobs[@]}"

log "ALL FIGURE 10/11 GPU0 JOBS COMPLETE"
