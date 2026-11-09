#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_popqa_figures_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-/data1/yujia/CausalityRAG_release_popqa_20260810/scripts/compat/transformers_452}
MODEL_PATH=/data1/yujia/models/Qwen2.5-7B-Instruct
MODEL=qwen2.5-7b
SOURCE=/data1/yujia/CausalityRAG/out_popqa_20260810/popqa
REPAIRED=/data1/yujia/CausalityRAG/out_popqa_no_frontier_repair_20260811/popqa
OUT=/data1/yujia/CausalityRAG/out_popqa_figures_20260811/popqa
LOG_ROOT=$OUT/logs
PID_ROOT=$OUT/figure8/pids
POOL=$SOURCE/replacements/shared_pool_top10_v1/shared_pool.jsonl
POOL_SHA=42dd41818cdefac31ce788a1bf13402b07353ed032806c9a874f5fb8b19313e7
mkdir -p "$LOG_ROOT" "$PID_ROOT"
cd "$RELEASE"

port_is_free() {
  ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$1$"
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
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
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
trap stop_readers EXIT

for port in 8000 8002 8003; do
  if ! port_is_free "$port"; then
    echo "refusing to interrupt unowned service on port $port" >&2
    exit 1
  fi
done
if port_is_free 8021; then
  echo "required existing spaCy service is not listening on 8021" >&2
  exit 1
fi
for spec in '0 8002' '1 8003'; do
  read -r gpu port <<<"$spec"
  PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$MODEL" \
    --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
    --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
    --disable-log-requests --gpu-memory-utilization 0.90 --port "$port" \
    >"$LOG_ROOT/figure8_vllm_${port}.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/reader_${port}.pid"
done
wait_endpoint http://127.0.0.1:8002/v1/models 900
wait_endpoint http://127.0.0.1:8003/v1/models 900
setsid "$PYTHON" scripts/lb_proxy_two_gpu.py \
  >"$LOG_ROOT/figure8_proxy.log" 2>&1 < /dev/null &
echo $! >"$PID_ROOT/proxy_8000.pid"
wait_endpoint http://127.0.0.1:8000/v1/models 120

CAUSALITYRAG_READER_MODE=short_answer \
  CAUSALITYRAG_READER_MAX_TOKENS=128 \
  "$PYTHON" scripts/evaluate_fixed_budget_acc.py \
  --input "$SOURCE/retrieval/top10_1000.jsonl" \
  --units-cache "$SOURCE/inputs/token_units_top10_1000.jsonl" \
  --frontier "$REPAIRED/frontier_1000.jsonl" \
  --scores "mirage=$SOURCE/methods/baselines/mirage_top5_1000.jsonl" \
  --scores "arc_jsd=$SOURCE/methods/baselines/arc_jsd_top5_1000.jsonl" \
  --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
  --out "$OUT/figure8/refine_k33_64_1000.jsonl" \
  --summary-out "$OUT/figure8/refine_k33_64_1000.summary.json" \
  --budgets "$(seq -s, 33 64)" \
  --seed-results "$OUT/figure8/coarse_1000.jsonl" --skip-reflow \
  --replacement-seed 0 --n 1000 --k 5 --workers 48 \
  --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$MODEL" \
  --reader-max-tokens 128 >"$LOG_ROOT/figure8_refine.log" 2>&1

stop_readers
date -Is >"$OUT/figure8/REFINE_COMPLETE"
echo '[popqa-figure8] refinement complete'
