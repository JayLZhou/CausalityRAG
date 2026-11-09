#!/usr/bin/env bash
set -euo pipefail

ROOT="${SENTENCE_EXPERIMENT_ROOT:-/data2/yujia/CausalityRAG_fig8_fig9_20260804}"
PY="${SENTENCE_EXPERIMENT_PYTHON:-/data2/yujia/envs/graphrag/bin/python}"
MODEL="${SENTENCE_EXPERIMENT_MODEL:-/data2/yujia/models/Qwen2.5-7B-Instruct}"
PORT="${SENTENCE_EXPERIMENT_PORT:-8101}"
INPUT_ROOT="$ROOT/sentence_sufficiency/inputs"
OUT_ROOT="$ROOT/sentence_sufficiency/out"
LOG_ROOT="$ROOT/sentence_sufficiency/logs"
FIGURE_ROOT="$ROOT/sentence_sufficiency/figures"
DATASETS=(hotpotqa finqa musique quartz qasper pubmedqa)

mkdir -p "$LOG_ROOT" "$FIGURE_ROOT"
for dataset in "${DATASETS[@]}"; do
  mkdir -p "$OUT_ROOT/$dataset"
  for filename in \
    retrieval_top10_1000.jsonl \
    token_units_top10_1000.jsonl \
    reflow_results_top5_1000.jsonl \
    mirage_top5_1000.jsonl \
    arc_jsd_top5_1000.jsonl; do
    path="$INPUT_ROOT/$dataset/$filename"
    if [[ ! -f "$path" ]]; then
      echo "[error] missing required file: $path" >&2
      exit 1
    fi
  done
done

for path in \
  "$ROOT/scripts/run_sentence_sufficiency_case_study.py" \
  "$ROOT/scripts/plot_sentence_sufficiency.py"; do
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

echo "[reader] $(date -Is) starting Qwen2.5-7B on physical GPU0"
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
  >"$LOG_ROOT/vllm_gpu0_${PORT}_appendix.log" 2>&1 < /dev/null &
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
    tail -100 "$LOG_ROOT/vllm_gpu0_${PORT}_appendix.log" >&2 || true
    exit 1
  fi
  sleep 5
done
if [[ "$healthy" -ne 1 ]]; then
  echo "[error] vLLM did not become healthy within 15 minutes" >&2
  exit 1
fi
echo "[reader] $(date -Is) GPU0 reader healthy on port $PORT"

run_dataset() {
  local dataset="$1"
  "$PY" "$ROOT/scripts/run_sentence_sufficiency_case_study.py" \
    --retrieval "$INPUT_ROOT/$dataset/retrieval_top10_1000.jsonl" \
    --units "$INPUT_ROOT/$dataset/token_units_top10_1000.jsonl" \
    --reflow "$INPUT_ROOT/$dataset/reflow_results_top5_1000.jsonl" \
    --mirage "$INPUT_ROOT/$dataset/mirage_top5_1000.jsonl" \
    --arc-jsd "$INPUT_ROOT/$dataset/arc_jsd_top5_1000.jsonl" \
    --out "$OUT_ROOT/$dataset/per_query.jsonl" \
    --summary-out "$OUT_ROOT/$dataset/summary.json" \
    --workers 8 \
    --max-tokens 64 \
    --llm-base-url "http://127.0.0.1:${PORT}/v1" \
    --llm-model qwen2.5-7b
}

EVAL_PIDS=()
for dataset in "${DATASETS[@]}"; do
  run_dataset "$dataset" >"$LOG_ROOT/evaluate_${dataset}.log" 2>&1 &
  EVAL_PIDS+=("$!")
done

evaluation_status=0
for pid in "${EVAL_PIDS[@]}"; do
  wait "$pid" || evaluation_status=1
done
if [[ "$evaluation_status" -ne 0 ]]; then
  echo "[error] at least one appendix sentence-sufficiency evaluation failed" >&2
  for dataset in "${DATASETS[@]}"; do
    echo "[error-log] $dataset" >&2
    tail -40 "$LOG_ROOT/evaluate_${dataset}.log" >&2 || true
  done
  exit 1
fi
echo "[reader] $(date -Is) all six appendix evaluations complete"

"$PY" "$ROOT/scripts/plot_sentence_sufficiency.py" \
  --summary "HotpotQA=$OUT_ROOT/hotpotqa/summary.json" \
  --summary "FinQA=$OUT_ROOT/finqa/summary.json" \
  --summary "MuSiQue=$OUT_ROOT/musique/summary.json" \
  --summary "QuARTz=$OUT_ROOT/quartz/summary.json" \
  --summary "QASPER=$OUT_ROOT/qasper/summary.json" \
  --summary "PubMedQA=$OUT_ROOT/pubmedqa/summary.json" \
  --datasets-per-row 2 \
  --out-prefix "$FIGURE_ROOT/sentence_sufficiency_appendix"

echo "[done] $(date -Is) appendix 3-row sentence-sufficiency figure complete"
