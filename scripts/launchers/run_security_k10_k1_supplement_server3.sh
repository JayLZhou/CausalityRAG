#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

REPO=${REPO:-/data1/yujia/CausalityRAG_release_security_k10k1_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL=${MODEL:-/data1/yujia/models/Qwen2.5-7B-Instruct}
EMBED_MODEL=${EMBED_MODEL:-/data1/yujia/models/Qwen3-Embedding-0.6B}
INPUT=${INPUT:-/data1/yujia/CausalityRAG/out/hotpotqa/retrieval/top10_1000.jsonl}
CLEAN_TARGETS=${CLEAN_TARGETS:-/data1/yujia/CausalityRAG/out/hotpotqa/inputs/clean_reference_top10_1000.jsonl}
CANDIDATE_ATTACKS=${CANDIDATE_ATTACKS:-/data1/yujia/CausalityRAG/out/hotpotqa/security_k10_k1_v2/attacks_100.jsonl}
RUN_ROOT=${RUN_ROOT:-/data1/yujia/CausalityRAG/out/hotpotqa/security_k10_k1_v3}
LOG_ROOT=$RUN_ROOT/logs
PID_ROOT=$RUN_ROOT/pids
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-$REPO/scripts/compat/transformers_452}
mkdir -p "$LOG_ROOT" "$PID_ROOT" "$RUN_ROOT/inputs" "$RUN_ROOT/population"
cd "$REPO"

line_count() {
  if [[ -f "$1" ]]; then awk 'NF {n++} END {print n+0}' "$1"; else echo 0; fi
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

stop_owned_group() {
  local pid_file=$1 expected=$2
  [[ -f "$pid_file" ]] || return 0
  local pid cmd deadline
  pid=$(cat "$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]]; then
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    if [[ "$cmd" == *"$expected"* ]]; then
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

cleanup() {
  stop_owned_group "$PID_ROOT/reader.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/embed.pid" local_embedding_server.py
}
trap cleanup EXIT

"$PYTHON" scripts/prepare_security_top10_population.py \
  --retrieval "$INPUT" --clean-targets "$CLEAN_TARGETS" \
  --candidate-attacks "$CANDIDATE_ATTACKS" \
  --retained-out "$RUN_ROOT/population/retained.jsonl" \
  --supplement-reflow-out "$RUN_ROOT/population/supplement_reflow.jsonl" \
  --manifest-out "$RUN_ROOT/population/manifest.json" --n 100 \
  >"$LOG_ROOT/population_prepare.log" 2>&1

supplement_n=$(line_count "$RUN_ROOT/population/supplement_reflow.jsonl")
if (( supplement_n > 0 )); then
  for port in 8000 8017; do
    if ! port_is_free "$port"; then
      echo "refusing to interrupt unowned service on port $port" >&2
      exit 1
    fi
  done
  PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES=0 setsid "$PYTHON" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen2.5-7b \
    --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
    --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
    --disable-log-requests --gpu-memory-utilization 0.90 --port 8000 \
    >"$LOG_ROOT/vllm_8000.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/reader.pid"

  CUDA_VISIBLE_DEVICES=1 setsid "$PYTHON" \
    /data1/yujia/YVETTE/scripts/local_embedding_server.py \
    --model-path "$EMBED_MODEL" --served-model-name Qwen3-Embedding-0.6B \
    --port 8017 --device cuda --max-seq-len 512 --encode-batch-size 16 \
    >"$LOG_ROOT/embedding_8017.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/embed.pid"

  wait_endpoint http://127.0.0.1:8000/v1/models "$(cat "$PID_ROOT/reader.pid")" 900
  wait_endpoint http://127.0.0.1:8017/v1/models "$(cat "$PID_ROOT/embed.pid")" 600

  CAUSALITYRAG_READER_MODE=short_answer CAUSALITYRAG_READER_MAX_TOKENS=128 \
    "$PYTHON" scripts/run_security_pareto_attacks.py \
    --retrieval "$INPUT" \
    --reflow "$RUN_ROOT/population/supplement_reflow.jsonl" \
    --out "$RUN_ROOT/population/supplemental_attacks.jsonl" \
    --summary-out "$RUN_ROOT/population/supplemental_attacks.summary.json" \
    --n "$supplement_n" --workers 6 --tparag-iterations 5 \
    --retrieved-k 10 --injected-count 1 --attack-candidate-count 5 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model qwen2.5-7b \
    --embed-base-url http://127.0.0.1:8017/v1 \
    --embed-model Qwen3-Embedding-0.6B \
    >"$LOG_ROOT/supplemental_attacks.log" 2>&1
  cleanup
else
  : >"$RUN_ROOT/population/supplemental_attacks.jsonl"
fi

"$PYTHON" scripts/finalize_security_top10_population.py \
  --retained "$RUN_ROOT/population/retained.jsonl" \
  --supplemental "$RUN_ROOT/population/supplemental_attacks.jsonl" \
  --manifest "$RUN_ROOT/population/manifest.json" \
  --out "$RUN_ROOT/attacks_100.jsonl" \
  >"$LOG_ROOT/population_finalize.log" 2>&1

"$PYTHON" scripts/materialize_security_attacked_inputs.py \
  --attacks "$RUN_ROOT/attacks_100.jsonl" --out-dir "$RUN_ROOT/inputs" \
  >"$LOG_ROOT/materialize.log" 2>&1

date -Is >"$RUN_ROOT/ATTACKS_COMPLETE"
echo "security top10/k1 clean-correct population complete"
