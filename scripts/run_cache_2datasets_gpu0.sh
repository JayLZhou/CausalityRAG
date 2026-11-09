#!/usr/bin/env bash
set -euo pipefail

ROOT="${CACHE_EXPERIMENT_ROOT:-/data2/yujia/CausalityRAG_fig8_fig9_20260804}"
PY="${CACHE_EXPERIMENT_PYTHON:-/data2/yujia/envs/graphrag/bin/python}"
MODEL="${CACHE_EXPERIMENT_MODEL:-/data2/yujia/models/Qwen2.5-7B-Instruct}"
PORT="${CACHE_EXPERIMENT_PORT:-8101}"
INPUT_ROOT="$ROOT/cache_2datasets/inputs"
OUT_ROOT="$ROOT/cache_2datasets/out"
LOG_ROOT="$ROOT/cache_2datasets/logs"

mkdir -p "$OUT_ROOT/2wiki" "$OUT_ROOT/timeqa" "$LOG_ROOT"

for path in \
  "$ROOT/scripts/run_cache_update_workload.py" \
  "$ROOT/scripts/run_dynamic_answer_cache_case_study.py" \
  "$INPUT_ROOT/2wiki/stress_inputs.jsonl" \
  "$INPUT_ROOT/2wiki/token_units_top5_1000.jsonl" \
  "$INPUT_ROOT/2wiki/frontier_top5_1000.jsonl" \
  "$INPUT_ROOT/timeqa/stress_inputs.jsonl" \
  "$INPUT_ROOT/timeqa/token_units_top5_1000.jsonl" \
  "$INPUT_ROOT/timeqa/frontier_top5_1000.jsonl"; do
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -x "$PY" ]]; then
  echo "[error] Python environment is not executable: $PY" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "[error] model directory is unavailable: $MODEL" >&2
  exit 1
fi
if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits \
  2>/dev/null | grep -Eq '[0-9]'; then
  echo "[error] GPU0 already has a compute process" >&2
  exit 1
fi
if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[error] port $PORT is already serving a model" >&2
  exit 1
fi

echo "[reader] $(date -Is) starting one Qwen2.5-7B replica on physical GPU0"
CUDA_VISIBLE_DEVICES=0 setsid "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen2.5-7b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.82 \
  --disable-log-requests \
  --port "$PORT" \
  >"$LOG_ROOT/vllm_gpu0_${PORT}.log" 2>&1 < /dev/null &
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

healthy=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[error] vLLM exited before becoming healthy" >&2
    tail -100 "$LOG_ROOT/vllm_gpu0_${PORT}.log" >&2 || true
    exit 1
  fi
  sleep 5
done
if [[ "$healthy" -ne 1 ]]; then
  echo "[error] vLLM did not become healthy within 15 minutes" >&2
  exit 1
fi
echo "[reader] $(date -Is) GPU0 replica healthy on port $PORT"

"$PY" "$ROOT/scripts/run_cache_update_workload.py" evaluate \
  --input "$INPUT_ROOT/2wiki/stress_inputs.jsonl" \
  --out "$OUT_ROOT/2wiki/stress_results.jsonl" \
  --workers 24 \
  --llm-base-url "http://127.0.0.1:${PORT}/v1" \
  --llm-model qwen2.5-7b \
  >"$LOG_ROOT/evaluate_2wiki.log" 2>&1 &
EVAL_2WIKI_PID=$!

"$PY" "$ROOT/scripts/run_cache_update_workload.py" evaluate \
  --input "$INPUT_ROOT/timeqa/stress_inputs.jsonl" \
  --out "$OUT_ROOT/timeqa/stress_results.jsonl" \
  --workers 24 \
  --llm-base-url "http://127.0.0.1:${PORT}/v1" \
  --llm-model qwen2.5-7b \
  >"$LOG_ROOT/evaluate_timeqa.log" 2>&1 &
EVAL_TIMEQA_PID=$!

evaluation_status=0
wait "$EVAL_2WIKI_PID" || evaluation_status=1
wait "$EVAL_TIMEQA_PID" || evaluation_status=1
if [[ "$evaluation_status" -ne 0 ]]; then
  echo "[error] at least one reader evaluation failed" >&2
  exit 1
fi
echo "[reader] $(date -Is) both 1,460-row reader evaluations complete"

"$PY" "$ROOT/scripts/run_dynamic_answer_cache_case_study.py" \
  --stress-results "$OUT_ROOT/2wiki/stress_results.jsonl" \
  --units "$INPUT_ROOT/2wiki/token_units_top5_1000.jsonl" \
  --reflow-frontier "$INPUT_ROOT/2wiki/frontier_top5_1000.jsonl" \
  --out "$OUT_ROOT/2wiki/summary.json" \
  --requests 20000 \
  --updates-per-query 5 \
  --capacities 73 \
  --budgets 1 3 5 \
  --ttls 25 50 100 250 500 1000 \
  >"$LOG_ROOT/simulate_2wiki.log" 2>&1

"$PY" "$ROOT/scripts/run_dynamic_answer_cache_case_study.py" \
  --stress-results "$OUT_ROOT/timeqa/stress_results.jsonl" \
  --units "$INPUT_ROOT/timeqa/token_units_top5_1000.jsonl" \
  --reflow-frontier "$INPUT_ROOT/timeqa/frontier_top5_1000.jsonl" \
  --out "$OUT_ROOT/timeqa/summary.json" \
  --requests 20000 \
  --updates-per-query 5 \
  --capacities 73 \
  --budgets 1 3 5 \
  --ttls 25 50 100 250 500 1000 \
  >"$LOG_ROOT/simulate_timeqa.log" 2>&1

echo "[done] $(date -Is) LRU/LFU/ReFlow/TTL cache experiments complete"
