#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

REPO=${REPO:-/data1/yujia/CausalityRAG_release_security_k10k1_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL=${MODEL:-/data1/yujia/models/Qwen2.5-7B-Instruct}
ATTACKS=${ATTACKS:-/data1/yujia/CausalityRAG/out/hotpotqa/security_k10_k1_v3/attacks_100.jsonl}
CLEAN_REFLOW=${CLEAN_REFLOW:-/data1/yujia/CausalityRAG/out/hotpotqa/security_simple_v1/reflow_results_top5_security100.jsonl}
CLEAN_UNITS=${CLEAN_UNITS:-/data1/yujia/CausalityRAG/out/hotpotqa/security_simple_v1/token_units_top10_security100.jsonl}
KEYWORD=${KEYWORD:-/data1/yujia/CausalityRAG/out/hotpotqa/security_k10_k1_v3/guard_keyword_100.jsonl}
DECODING=${DECODING:-/data1/yujia/CausalityRAG/out/hotpotqa/security_k10_k1_v3/secure_decoding_100.jsonl}
RUN_ROOT=${RUN_ROOT:-/data1/yujia/CausalityRAG/out/hotpotqa/security_provenance_v1}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-$REPO/scripts/compat/transformers_452}
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pids"
cd "$REPO"

for path in "$ATTACKS" "$CLEAN_REFLOW" "$CLEAN_UNITS" "$KEYWORD" "$DECODING" "$MODEL/config.json"; do
  [[ -s "$path" ]] || { echo "missing prerequisite: $path" >&2; exit 1; }
done

if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)8000$'; then
  echo "refusing to interrupt unowned service on port 8000" >&2
  exit 1
fi

stop_reader() {
  [[ -f "$RUN_ROOT/pids/reader.pid" ]] || return 0
  local pid
  pid=$(cat "$RUN_ROOT/pids/reader.pid")
  kill -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  rm -f "$RUN_ROOT/pids/reader.pid"
}
trap stop_reader EXIT

PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=0 setsid "$PYTHON" \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen2.5-7b \
  --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
  --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
  --disable-log-requests --gpu-memory-utilization 0.90 --port 8000 \
  >"$RUN_ROOT/logs/vllm.log" 2>&1 < /dev/null &
echo $! >"$RUN_ROOT/pids/reader.pid"

deadline=$((SECONDS + 900))
until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
  kill -0 "$(cat "$RUN_ROOT/pids/reader.pid")" 2>/dev/null || {
    echo "reader exited before becoming ready" >&2
    exit 1
  }
  (( SECONDS < deadline )) || { echo "reader startup timed out" >&2; exit 1; }
  sleep 5
done

rm -f "$RUN_ROOT/provenance_defense_100.jsonl" "$RUN_ROOT/provenance_defense_100.summary.json"
CAUSALITYRAG_READER_MODE=short_answer CAUSALITYRAG_READER_MAX_TOKENS=128 \
  "$PYTHON" scripts/run_security_provenance_defense.py \
  --attacks "$ATTACKS" \
  --clean-reflow "$CLEAN_REFLOW" \
  --clean-units "$CLEAN_UNITS" \
  --native-keyword "$KEYWORD" \
  --native-decoding "$DECODING" \
  --base-url http://127.0.0.1:8000/v1 --model qwen2.5-7b \
  --out "$RUN_ROOT/provenance_defense_100.jsonl" \
  --summary-out "$RUN_ROOT/provenance_defense_100.summary.json" \
  --workers 12 >"$RUN_ROOT/logs/provenance_defense.log" 2>&1

cat "$RUN_ROOT/provenance_defense_100.summary.json"
