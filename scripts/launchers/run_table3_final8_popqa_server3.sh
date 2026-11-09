#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/data1/yujia/CausalityRAG/out_popqa_20260810}
SOURCE_BASE=$SOURCE_ROOT/popqa
REPAIR_ROOT=${REPAIR_ROOT:-/data1/yujia/CausalityRAG/out_popqa_no_frontier_repair_20260811}
REPAIR_BASE=$REPAIR_ROOT/popqa
SEVEN_ROOT=${SEVEN_ROOT:-/data1/yujia/CausalityRAG/out_no_frontier_repair_20260810}
BASE_ROOT=${BASE_ROOT:-/data1/yujia/CausalityRAG/out}
FINAL_ROOT=${FINAL_ROOT:-/data1/yujia/CausalityRAG/out_table3_final8_20260811}
FINAL_BASE=$FINAL_ROOT/popqa
QWEN25=${QWEN25:-/data1/yujia/models/Qwen2.5-7B-Instruct}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-/data1/yujia/CausalityRAG_release_popqa_20260810/scripts/compat/transformers_452}
BASE_URL=http://127.0.0.1:8000/v1
MODEL=qwen2.5-7b
PID_ROOT=$FINAL_ROOT/pids
LOG_ROOT=$FINAL_ROOT/logs
mkdir -p "$PID_ROOT" "$LOG_ROOT" "$FINAL_BASE"
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
  local port
  for port in 8000 8002 8003; do
    if ! port_is_free "$port"; then
      echo "refusing to interrupt unowned service on port $port" >&2
      exit 1
    fi
  done
  if port_is_free 8021; then
    echo "required pre-existing spaCy service is not listening on 8021" >&2
    exit 1
  fi
  local spec gpu reader_port
  for spec in '0 8002' '1 8003'; do
    read -r gpu reader_port <<<"$spec"
    PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
      -m vllm.entrypoints.openai.api_server \
      --model "$QWEN25" --served-model-name "$MODEL" \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 \
      --port "$reader_port" \
      >"$LOG_ROOT/vllm_${reader_port}.log" 2>&1 < /dev/null &
    echo $! >"$PID_ROOT/reader_${reader_port}.pid"
  done
  wait_endpoint http://127.0.0.1:8002/v1/models 900
  wait_endpoint http://127.0.0.1:8003/v1/models 900
  setsid "$PYTHON" scripts/lb_proxy_two_gpu.py \
    >"$LOG_ROOT/proxy_8000.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models 120
}

safe_link() {
  local source=$1 target=$2
  if [[ -L "$target" ]]; then
    [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]] || {
      echo "refusing to replace mismatched symlink: $target" >&2
      exit 1
    }
  elif [[ -e "$target" ]]; then
    echo "refusing to replace existing path: $target" >&2
    exit 1
  else
    ln -s "$source" "$target"
  fi
}

while [[ ! -f "$SEVEN_ROOT/TABLE3_COMPLETE" ]]; do
  echo "[table3-final8] waiting for seven-dataset targeted baseline/control stage"
  sleep 60
done
reader_deadline=$((SECONDS + 180))
until port_is_free 8000 && port_is_free 8002 && port_is_free 8003; do
  if (( SECONDS >= reader_deadline )); then
    echo "stage 1 completed but reader ports remain occupied; refusing to interrupt" >&2
    exit 1
  fi
  echo "[table3-final8] waiting for stage-1 readers to release ports"
  sleep 5
done

echo '[table3-final8] stage=popqa-rankers'
"$PYTHON" scripts/run_table3_rankers.py \
  --out-root "$SOURCE_ROOT" --datasets popqa --n 1000 \
  >"$LOG_ROOT/popqa_rankers.log" 2>&1

echo '[table3-final8] stage=prepare-isolated-popqa-root'
safe_link "$SOURCE_BASE/retrieval" "$FINAL_BASE/retrieval"
safe_link "$SOURCE_BASE/inputs" "$FINAL_BASE/inputs"
safe_link "$SOURCE_BASE/replacements" "$FINAL_BASE/replacements"
mkdir -p "$FINAL_BASE/methods" "$FINAL_BASE/methods/reflow"
safe_link "$SOURCE_BASE/methods/baselines" "$FINAL_BASE/methods/baselines"
safe_link "$REPAIR_BASE/results_1000.jsonl" \
  "$FINAL_BASE/methods/reflow/results_top5_1000.jsonl"
safe_link "$REPAIR_BASE/results_1000.summary.json" \
  "$FINAL_BASE/methods/reflow/results_top5_1000.summary.json"
safe_link "$REPAIR_BASE/frontier_1000.jsonl" \
  "$FINAL_BASE/methods/reflow/frontier_top5_1000.jsonl"
safe_link "$REPAIR_BASE/frontier_1000.summary.json" \
  "$FINAL_BASE/methods/reflow/frontier_top5_1000.summary.json"

echo '[table3-final8] stage=popqa-readers-controls'
start_readers
"$PYTHON" scripts/run_table3_reader_stage.py \
  --out-root "$FINAL_ROOT" --datasets popqa --python "$PYTHON" \
  --workers 48 --max-paraphrase-passes 20 \
  --llm-base-url "$BASE_URL" --llm-model "$MODEL" \
  >"$LOG_ROOT/popqa_reader_stage.log" 2>&1
stop_readers

date -Is >"$FINAL_ROOT/POPQA_COMPLETE"

echo '[table3-final8] stage=assemble-seven-datasets'
for dataset in hotpotqa timeqa finqa musique quartz triviaqa 2wiki; do
  source="$SEVEN_ROOT/$dataset"
  target="$FINAL_ROOT/$dataset"
  mkdir -p \
    "$target/methods/reflow" \
    "$target/audits/final_top10pool_k5" \
    "$target/controls"
  safe_link "$BASE_ROOT/$dataset/retrieval" "$target/retrieval"
  safe_link "$source/results_1000.jsonl" \
    "$target/methods/reflow/results_top5_1000.jsonl"
  safe_link "$source/results_1000.summary.json" \
    "$target/methods/reflow/results_top5_1000.summary.json"
  safe_link "$source/table3_resume/baselines_1000.jsonl" \
    "$target/audits/final_top10pool_k5/baselines_1000.jsonl"
  safe_link "$source/table3_resume/controls_1000.jsonl" \
    "$target/controls/paraphrase_results_top5_1000_v2.jsonl"
  safe_link "$source/table3_resume/factual_metrics_1000.json" \
    "$target/audits/final_top10pool_k5/factual_metrics_1000.json"
  safe_link "$source/table3_resume/controls_1000.summary.json" \
    "$target/controls/paraphrase_results_top5_1000_v2.summary.json"
  safe_link "$source/table3_resume/table3_metrics_1000.json" \
    "$target/audits/final_top10pool_k5/table3_metrics_1000_v2.json"
done

echo '[table3-final8] stage=render-audit'
"$PYTHON" scripts/render_table3_rows.py \
  --metrics-root "$FINAL_ROOT" --out "$FINAL_ROOT/table3_rows.tex"
"$PYTHON" scripts/audit_table3_final8.py \
  --root "$FINAL_ROOT" --out "$FINAL_ROOT/audit.json"
"$PYTHON" scripts/audit_table3_token_counts.py \
  --root "$FINAL_ROOT" --out "$FINAL_ROOT/table3_token_count_audit.json"
"$PYTHON" scripts/render_table3_mean_edited_tokens.py \
  --root "$FINAL_ROOT" \
  --out-json "$FINAL_ROOT/table3_mean_edited_tokens.json" \
  --out-tex "$FINAL_ROOT/table3_mean_edited_tokens.tex"
date -Is >"$FINAL_ROOT/FINAL8_COMPLETE"
echo "[table3-final8] complete root=$FINAL_ROOT"
