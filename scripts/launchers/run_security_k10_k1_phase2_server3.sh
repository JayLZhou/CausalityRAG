#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

REPO=${REPO:-/data1/yujia/CausalityRAG_release_security_k10k1_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL=${MODEL:-/data1/yujia/models/Qwen2.5-7B-Instruct}
RUN_ROOT=${RUN_ROOT:-/data1/yujia/CausalityRAG/out/hotpotqa/security_k10_k1_v2}
LOG_ROOT=$RUN_ROOT/logs
PID_ROOT=$RUN_ROOT/pids
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-$REPO/scripts/compat/transformers_452}
mkdir -p "$LOG_ROOT" "$PID_ROOT" "$RUN_ROOT/graph" "$RUN_ROOT/frontier"
cd "$REPO"

line_count() {
  if [[ -f "$1" ]]; then awk 'NF {n++} END {print n+0}' "$1"; else echo 0; fi
}

complete_jsonl() {
  [[ "$(line_count "$1")" == "$2" ]]
}

port_is_free() {
  ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$1$"
}

wait_endpoint() {
  local url=$1 pid=$2 deadline=$((SECONDS + ${3:-900}))
  until curl -fsS "$url" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "service exited before becoming ready: $url" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "service did not become ready: $url" >&2
      exit 1
    fi
    sleep 5
  done
}

stop_reader() {
  local pid_file=$PID_ROOT/reader.pid
  [[ -f "$pid_file" ]] || return 0
  local pid cmd deadline
  pid=$(cat "$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]]; then
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    if [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* ]]; then
      kill -- "-$pid" 2>/dev/null || true
      deadline=$((SECONDS + 30))
      while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do
        sleep 1
      done
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
}
trap stop_reader EXIT

for name in poisonedrag tparag; do
  retrieval=$RUN_ROOT/inputs/${name}_retrieval.jsonl
  targets=$RUN_ROOT/inputs/${name}_targets.jsonl
  [[ "$(line_count "$retrieval")" == 100 ]] || { echo "invalid $retrieval" >&2; exit 1; }
  [[ "$(line_count "$targets")" == 100 ]] || { echo "invalid $targets" >&2; exit 1; }
  if ! complete_jsonl "$RUN_ROOT/inputs/${name}_units.jsonl" 100; then
    "$PYTHON" scripts/build_context_units.py \
      --input "$retrieval" --out "$RUN_ROOT/inputs/${name}_units.jsonl" \
      --summary-out "$RUN_ROOT/inputs/${name}_units.summary.json" \
      --n 100 --k 10 --workers 16 --backend service \
      --spacy-base-url http://127.0.0.1:8021 \
      >"$LOG_ROOT/${name}_units.log" 2>&1
  fi
done

run_graph() {
  local name=$1 gpu=$2
  if complete_jsonl "$RUN_ROOT/graph/${name}.jsonl" 100; then return 0; fi
  CAUSALITYRAG_READER_MODE=short_answer CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON" scripts/build_contribution_graph.py \
    --input "$RUN_ROOT/inputs/${name}_retrieval.jsonl" \
    --out "$RUN_ROOT/graph/${name}.jsonl" \
    --summary-out "$RUN_ROOT/graph/${name}.summary.json" \
    --model-path "$MODEL" --n 100 --k 10 --target results \
    --target-results "$RUN_ROOT/inputs/${name}_targets.jsonl" \
    --device cuda --dtype bfloat16 --max-receivers-per-layer 48 \
    >"$LOG_ROOT/${name}_graph.log" 2>&1
}

run_graph poisonedrag 0 & left=$!
run_graph tparag 1 & right=$!
left_status=0
right_status=0
wait "$left" || left_status=$?
wait "$right" || right_status=$?
if (( left_status != 0 || right_status != 0 )); then
  echo "graph construction failed: poisonedrag=$left_status tparag=$right_status" >&2
  exit 1
fi

for name in poisonedrag tparag; do
  "$PYTHON" scripts/build_security_flow_frontiers.py \
    --input "$RUN_ROOT/inputs/${name}_retrieval.jsonl" \
    --graphs "$RUN_ROOT/graph/${name}.jsonl" \
    --units "$RUN_ROOT/inputs/${name}_units.jsonl" \
    --out "$RUN_ROOT/frontier/${name}.jsonl" \
    --summary-out "$RUN_ROOT/frontier/${name}.summary.json" --k 10 \
    >"$LOG_ROOT/${name}_frontier.log" 2>&1
done

if ! port_is_free 8000; then
  echo "refusing to interrupt unowned service on port 8000" >&2
  exit 1
fi
PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=0 setsid "$PYTHON" \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen2.5-7b \
  --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
  --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
  --disable-log-requests --gpu-memory-utilization 0.90 --port 8000 \
  >"$LOG_ROOT/defense_vllm_8000.log" 2>&1 < /dev/null &
echo $! >"$PID_ROOT/reader.pid"
wait_endpoint http://127.0.0.1:8000/v1/models "$(cat "$PID_ROOT/reader.pid")" 900

CAUSALITYRAG_READER_MODE=short_answer CAUSALITYRAG_READER_MAX_TOKENS=128 \
  "$PYTHON" scripts/run_security_guard_keyword.py \
  --attacks "$RUN_ROOT/attacks_100.jsonl" \
  --poisonedrag-units "$RUN_ROOT/inputs/poisonedrag_units.jsonl" \
  --poisonedrag-frontier "$RUN_ROOT/frontier/poisonedrag.jsonl" \
  --tparag-units "$RUN_ROOT/inputs/tparag_units.jsonl" \
  --tparag-frontier "$RUN_ROOT/frontier/tparag.jsonl" \
  --out "$RUN_ROOT/guard_keyword_100.jsonl" \
  --summary-out "$RUN_ROOT/guard_keyword_100.summary.json" \
  --workers 12 --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model qwen2.5-7b --spacy-base-url http://127.0.0.1:8021 \
  --alpha 0.2 --beta 3 \
  >"$LOG_ROOT/guard_keyword.log" 2>&1 & guard_pid=$!

PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/run_robustrag_secure_decoding.py \
  --attacks "$RUN_ROOT/attacks_100.jsonl" --model-path "$MODEL" \
  --out "$RUN_ROOT/secure_decoding_100.jsonl" \
  --summary-out "$RUN_ROOT/secure_decoding_100.summary.json" \
  --device cuda --max-new-tokens 128 \
  >"$LOG_ROOT/secure_decoding.log" 2>&1 & decoding_pid=$!

guard_status=0
decoding_status=0
wait "$guard_pid" || guard_status=$?
wait "$decoding_pid" || decoding_status=$?
if (( guard_status != 0 || decoding_status != 0 )); then
  echo "defense evaluation failed: guard=$guard_status decoding=$decoding_status" >&2
  exit 1
fi
stop_reader

"$PYTHON" scripts/merge_security_defense_results.py \
  --guard-keyword "$RUN_ROOT/guard_keyword_100.jsonl" \
  --decoding "$RUN_ROOT/secure_decoding_100.jsonl" \
  --out "$RUN_ROOT/defenses_100.jsonl" \
  --summary-out "$RUN_ROOT/defenses_100.summary.json" \
  >"$LOG_ROOT/merge.log" 2>&1

date -Is >"$RUN_ROOT/COMPLETE"
echo "security top10/k1 defense experiment complete"
