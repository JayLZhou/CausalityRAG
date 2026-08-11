#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-/data1/yujia/CausalityRAG_release_popqa_20260810/scripts/compat/transformers_452}
MODEL_PATH=${MODEL_PATH:-/data1/yujia/models/Qwen2.5-7B-Instruct}
MODEL=${MODEL:-qwen2.5-7b}
RUN_ROOT=${RUN_ROOT:-/data1/yujia/CausalityRAG/out_popqa_20260810/popqa}
REPAIR_ROOT=${REPAIR_ROOT:-/data1/yujia/CausalityRAG/out_popqa_no_frontier_repair_20260811}
DATASET_ROOT=$REPAIR_ROOT/popqa
RESUME_ROOT=$DATASET_ROOT/resume
LOG_ROOT=$DATASET_ROOT/logs
PID_ROOT=$DATASET_ROOT/pids
POOL=$RUN_ROOT/replacements/shared_pool_top10_v1/shared_pool.jsonl
POOL_SHA=42dd41818cdefac31ce788a1bf13402b07353ed032806c9a874f5fb8b19313e7
TARGETS=6

mkdir -p "$RESUME_ROOT" "$LOG_ROOT" "$PID_ROOT"

port_is_free() {
  ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$1$"
}

require_free_ports() {
  local port
  for port in "$@"; do
    if ! port_is_free "$port"; then
      echo "refusing to interrupt unowned service on port $port" >&2
      exit 1
    fi
  done
}

require_spacy_service() {
  if port_is_free 8021; then
    echo "required existing spaCy service is not listening on 127.0.0.1:8021" >&2
    exit 1
  fi
}

wait_endpoint() {
  local url=$1 deadline=$((SECONDS + ${2:-900}))
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "endpoint did not become ready: $url" >&2
      exit 1
    fi
    sleep 5
  done
}

stop_owned_group() {
  local pid_file=$1 expected=$2
  [[ -f "$pid_file" ]] || return 0
  local pid cmd
  pid=$(cat "$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]]; then
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ "$cmd" == *"$expected"* ]]; then
      kill -- "-$pid" 2>/dev/null || true
      local deadline=$((SECONDS + 30))
      while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do
        sleep 1
      done
      if kill -0 -- "-$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || true
      fi
    fi
  fi
  rm -f "$pid_file"
}

stop_readers() {
  stop_owned_group "$PID_ROOT/reader_8002.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/reader_8003.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/proxy_8000.pid" lb_proxy_two_gpu.py
}

cleanup() {
  stop_readers
}
trap cleanup EXIT

start_readers() {
  require_free_ports 8000 8002 8003
  local spec gpu port
  for spec in '0 8002' '1 8003'; do
    read -r gpu port <<<"$spec"
    PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
      -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$MODEL" \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 --port "$port" \
      >"$LOG_ROOT/vllm_${port}.log" 2>&1 < /dev/null &
    echo $! >"$PID_ROOT/reader_${port}.pid"
  done
  wait_endpoint http://127.0.0.1:8002/v1/models 900
  wait_endpoint http://127.0.0.1:8003/v1/models 900
  setsid "$PYTHON" "$RELEASE/scripts/lb_proxy_two_gpu.py" \
    >"$LOG_ROOT/proxy_8000.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models 120
}

if [[ -e "$REPAIR_ROOT/COMPLETE" ]]; then
  echo "repair already complete: $REPAIR_ROOT" >&2
  exit 0
fi
if [[ -e "$RESUME_ROOT/results.jsonl" ]]; then
  echo "refusing to overwrite partial repaired reader output" >&2
  exit 1
fi
require_spacy_service

echo '[popqa-repair] stage=repair-graphs'
"$PYTHON" "$RELEASE/scripts/repair_disconnected_contribution_graphs.py" \
  --graphs "$RUN_ROOT/graphs/contribution_graph_top5_1000.jsonl" \
  --shared-pool "$POOL" \
  --target-results "$RUN_ROOT/methods/reflow/results_top5_1000.jsonl" \
  --expected-pool-sha256 "$POOL_SHA" \
  --out "$DATASET_ROOT/graphs.jsonl" \
  --summary-out "$DATASET_ROOT/graphs.summary.json" \
  --n 1000 >"$LOG_ROOT/repair_graphs.log" 2>&1

echo '[popqa-repair] stage=prepare-selected'
"$PYTHON" "$RELEASE/scripts/prepare_no_frontier_resume.py" prepare \
  --input "$RUN_ROOT/retrieval/top10_1000.jsonl" \
  --units-cache "$RUN_ROOT/inputs/token_units_top10_1000.jsonl" \
  --repaired-graphs "$DATASET_ROOT/graphs.jsonl" \
  --old-frontier "$RUN_ROOT/methods/reflow/frontier_top5_1000.jsonl" \
  --old-results "$RUN_ROOT/methods/reflow/results_top5_1000.jsonl" \
  --out-dir "$RESUME_ROOT" --n 1000 \
  >"$LOG_ROOT/prepare.log" 2>&1

echo '[popqa-repair] stage=frontier'
"$PYTHON" "$RELEASE/exp/run_reflow.py" \
  --input "$RESUME_ROOT/input.jsonl" \
  --graphs "$RESUME_ROOT/graphs.jsonl" \
  --units-cache "$RESUME_ROOT/units.jsonl" \
  --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
  --out "$RESUME_ROOT/frontier.jsonl" \
  --summary-out "$RESUME_ROOT/frontier.summary.json" \
  --n "$TARGETS" --k 5 >"$LOG_ROOT/frontier.log" 2>&1
"$PYTHON" "$RELEASE/scripts/prepare_no_frontier_resume.py" merge-frontier \
  --manifest "$RESUME_ROOT/manifest.json" \
  --old-frontier "$RUN_ROOT/methods/reflow/frontier_top5_1000.jsonl" \
  --resumed-frontier "$RESUME_ROOT/frontier.jsonl" \
  --out "$DATASET_ROOT/frontier_1000.jsonl" \
  --summary-out "$DATASET_ROOT/frontier_1000.summary.json" \
  >"$LOG_ROOT/merge_frontier.log" 2>&1

echo '[popqa-repair] stage=reader'
start_readers
CAUSALITYRAG_READER_MODE=short_answer "$PYTHON" \
  "$RELEASE/scripts/evaluate_reflow.py" \
  --input "$RESUME_ROOT/input.jsonl" \
  --units-cache "$RESUME_ROOT/units.jsonl" \
  --frontier "$RESUME_ROOT/frontier.jsonl" \
  --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
  --out "$RESUME_ROOT/results.jsonl" \
  --summary-out "$RESUME_ROOT/results.summary.json" \
  --n "$TARGETS" --k 5 --workers 6 --replacement-seed 0 \
  --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$MODEL" \
  >"$LOG_ROOT/reader.log" 2>&1
stop_readers

echo '[popqa-repair] stage=merge-audit'
"$PYTHON" "$RELEASE/scripts/prepare_no_frontier_resume.py" merge-results \
  --manifest "$RESUME_ROOT/manifest.json" \
  --old-results "$RUN_ROOT/methods/reflow/results_top5_1000.jsonl" \
  --resumed-results "$RESUME_ROOT/results.jsonl" \
  --out "$DATASET_ROOT/results_1000.jsonl" \
  --summary-out "$DATASET_ROOT/results_1000.summary.json" \
  >"$LOG_ROOT/merge_results.log" 2>&1
"$PYTHON" "$RELEASE/scripts/audit_no_frontier_repair.py" \
  --root "$REPAIR_ROOT" --datasets popqa \
  --out "$REPAIR_ROOT/audit.json" >"$LOG_ROOT/audit.log" 2>&1
"$PYTHON" "$RELEASE/scripts/audit_popqa_alias_metrics.py" \
  --results "$DATASET_ROOT/results_1000.jsonl" \
  --out "$DATASET_ROOT/alias_metric_audit.json" --expected-rows 1000 \
  >"$LOG_ROOT/alias_metric_audit.log" 2>&1

date -Is >"$REPAIR_ROOT/COMPLETE"
echo "[popqa-repair] complete root=$REPAIR_ROOT"
