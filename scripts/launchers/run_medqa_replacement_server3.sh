#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

REPO=${REPO:-/data1/yujia/CausalityRAG_release_medqa_20260809}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
SPACY_PYTHON=${SPACY_PYTHON:-/data1/yujia/envs/spacyner/bin/python}
OUT_ROOT=${OUT_ROOT:-/data1/yujia/CausalityRAG/out}
SENSITIVITY_ROOT=${SENSITIVITY_ROOT:-$OUT_ROOT/model_sensitivity_v1}
DATA_ROOT=${DATA_ROOT:-/data1/yujia/RAGData/paper_datasets/medqa}
SOURCE_ROOT=${SOURCE_ROOT:-$DATA_ROOT/source/medrag_textbooks}
QUESTIONS=${QUESTIONS:-$DATA_ROOT/questions/medqa.json}
CORPUS=${CORPUS:-$DATA_ROOT/corpus/medqa_textbooks.jsonl}
CORPUS_MANIFEST=${CORPUS_MANIFEST:-$DATA_ROOT/corpus/medqa_textbooks.manifest.json}
EMBED_MODEL=${EMBED_MODEL:-/data1/yujia/models/Qwen3-Embedding-0.6B}
QWEN25=${QWEN25:-/data1/yujia/models/Qwen2.5-7B-Instruct}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-$REPO/scripts/compat/transformers_452}
RUN_ROOT=${RUN_ROOT:-$OUT_ROOT/medqa/medqa_replacement_run_20260809}
RESUME_AFTER_RETRIEVAL=${RESUME_AFTER_RETRIEVAL:-0}
START_AT_TABLE3=${START_AT_TABLE3:-0}
ONLY_QWEN25=${ONLY_QWEN25:-0}
LOG_ROOT=$RUN_ROOT/logs
PID_ROOT=$RUN_ROOT/pids
mkdir -p "$LOG_ROOT" "$PID_ROOT"
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

require_free_ports() {
  local port
  for port in "$@"; do
    if ! port_is_free "$port"; then
      echo "refusing to interrupt unowned service on port $port" >&2
      exit 1
    fi
  done
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
      while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
      if kill -0 -- "-$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
    fi
  fi
  rm -f "$pid_file"
}

stop_embedding() {
  stop_owned_group "$PID_ROOT/embedding_8017.pid" local_embedding_server.py
}

stop_readers() {
  stop_owned_group "$PID_ROOT/reader_8002.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/reader_8003.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/proxy_8000.pid" lb_proxy_two_gpu.py
}

cleanup() {
  stop_embedding
  stop_readers
}
trap cleanup EXIT

start_embedding() {
  require_free_ports 8017
  CUDA_VISIBLE_DEVICES=1 setsid "$PYTHON" \
    /data1/yujia/YVETTE/scripts/local_embedding_server.py \
    --model-path "$EMBED_MODEL" --served-model-name Qwen3-Embedding-0.6B \
    --port 8017 --device cuda --max-seq-len 512 --encode-batch-size 16 \
    >"$LOG_ROOT/embedding_8017.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/embedding_8017.pid"
  wait_endpoint http://127.0.0.1:8017/v1/models 600
}

start_qwen25_readers() {
  require_free_ports 8000 8002 8003
  local spec gpu port
  for spec in '0 8002' '1 8003'; do
    read -r gpu port <<<"$spec"
    PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
      -m vllm.entrypoints.openai.api_server \
      --model "$QWEN25" --served-model-name qwen2.5-7b \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 --port "$port" \
      >"$LOG_ROOT/vllm_${port}.log" 2>&1 < /dev/null &
    echo $! >"$PID_ROOT/reader_${port}.pid"
  done
  wait_endpoint http://127.0.0.1:8002/v1/models 900
  wait_endpoint http://127.0.0.1:8003/v1/models 900
  setsid "$PYTHON" "$REPO/scripts/lb_proxy_two_gpu.py" \
    >"$LOG_ROOT/proxy_8000.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models 120
}

run_graph_shard() {
  local gpu=$1 start=$2 count=$3
  local shard_dir=$OUT_ROOT/medqa/graphs/shards
  local shard=$shard_dir/contribution_graph_top5_${start}_$((start + count - 1)).jsonl
  local summary=${shard%.jsonl}.summary.json
  mkdir -p "$shard_dir"
  if complete_jsonl "$shard" "$count" && [[ -s "$summary" ]]; then return; fi
  CAUSALITYRAG_READER_MODE=medqa CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    scripts/build_contribution_graph.py \
    --input "$OUT_ROOT/medqa/retrieval/top10_1000.jsonl" \
    --out "$shard" --summary-out "$summary" --model-path "$QWEN25" \
    --start "$start" --n "$count" --k 5 --target results \
    --target-results "$OUT_ROOT/medqa/inputs/clean_targets_top5_1000.jsonl" \
    --device cuda --dtype bfloat16 --max-receivers-per-layer 48 \
    >"$LOG_ROOT/graph_qwen25_${start}_$((start + count - 1)).log" 2>&1
}

if [[ "$START_AT_TABLE3" == 1 ]]; then
  echo '[medqa] stage=resume-at-table3'
  complete_jsonl "$OUT_ROOT/medqa/inputs/clean_targets_top5_1000.jsonl" 1000 || {
    echo 'refusing to resume: MedQA clean targets are not exactly 1000 rows' >&2
    exit 1
  }
  complete_jsonl "$OUT_ROOT/medqa/graphs/contribution_graph_top5_1000.jsonl" 1000 || {
    echo 'refusing to resume: MedQA contribution graph is not exactly 1000 rows' >&2
    exit 1
  }
  [[ -s "$OUT_ROOT/medqa/graphs/contribution_graph_top5_1000.summary.json" ]] || {
    echo 'refusing to resume: MedQA contribution graph summary is missing' >&2
    exit 1
  }
  [[ -s "$OUT_ROOT/medqa/replacements/shared_pool_top10_v1/shared_pool.jsonl" ]] || {
    echo 'refusing to resume: MedQA frozen replacement pool is missing' >&2
    exit 1
  }
  [[ -s "$OUT_ROOT/medqa/replacements/shared_pool_top10_v1/shared_pool.manifest.json" ]] || {
    echo 'refusing to resume: MedQA frozen replacement manifest is missing' >&2
    exit 1
  }
else
if [[ "$RESUME_AFTER_RETRIEVAL" == 1 ]]; then
  echo '[medqa] stage=resume-after-retrieval'
  complete_jsonl "$OUT_ROOT/medqa/retrieval/top10_1000.jsonl" 1000 || {
    echo 'refusing to resume: MedQA retrieval is not exactly 1000 rows' >&2
    exit 1
  }
  [[ -s "$OUT_ROOT/medqa/retrieval/top10_1000.summary.json" ]] || {
    echo 'refusing to resume: MedQA retrieval summary is missing' >&2
    exit 1
  }
else
  echo '[medqa] stage=freeze-corpus'
  revision=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
  "$PYTHON" scripts/prepare_medqa_textbook_corpus.py \
    --chunk-dir "$SOURCE_ROOT/chunk" --out "$CORPUS" \
    --manifest-out "$CORPUS_MANIFEST" --source-revision "$revision" \
    >"$LOG_ROOT/freeze_corpus.log" 2>&1

  echo '[medqa] stage=retrieval'
  start_embedding
  "$PYTHON" scripts/prepare_dataset_retrieval.py \
    --dataset medqa --questions "$QUESTIONS" --corpus "$CORPUS" \
    --out-root "$OUT_ROOT/medqa" --tokenizer-path "$EMBED_MODEL" \
    --embedding-base-url http://127.0.0.1:8017/v1 \
    --embedding-model Qwen3-Embedding-0.6B --embedding-batch-size 16 \
    >"$LOG_ROOT/retrieval.log" 2>&1
  stop_embedding
fi

echo '[medqa] stage=pool-and-clean-targets'
start_qwen25_readers
CAUSALITYRAG_READER_MODE=medqa "$PYTHON" scripts/generate_reader_targets.py \
  --input "$OUT_ROOT/medqa/retrieval/top10_1000.jsonl" \
  --out "$RUN_ROOT/clean_protocol_smoke_1.jsonl" \
  --summary-out "$RUN_ROOT/clean_protocol_smoke_1.summary.json" \
  --n 1 --k 5 --workers 1 --base-url http://127.0.0.1:8000/v1 \
  --served-model qwen2.5-7b >"$LOG_ROOT/clean_protocol_smoke.log" 2>&1
if ! grep -Eq '"clean_answer": "[ABCD]"' "$RUN_ROOT/clean_protocol_smoke_1.jsonl"; then
  echo 'MedQA prompt-only choice smoke failed closed' >&2
  exit 1
fi
"$PYTHON" scripts/run_seven_dataset_pools.py \
  --out-root "$OUT_ROOT" --datasets medqa \
  --max-generation-passes "${POOL_MAX_GENERATION_PASSES:-100}" \
  --exclude-unresolved-after-max-passes \
  >"$LOG_ROOT/pool.log" 2>&1
CAUSALITYRAG_READER_MODE=medqa "$PYTHON" scripts/generate_reader_targets.py \
  --input "$OUT_ROOT/medqa/retrieval/top10_1000.jsonl" \
  --out "$OUT_ROOT/medqa/inputs/clean_targets_top5_1000.jsonl" \
  --summary-out "$OUT_ROOT/medqa/inputs/clean_targets_top5_1000.summary.json" \
  --n 1000 --k 5 --workers 48 --base-url http://127.0.0.1:8000/v1 \
  --served-model qwen2.5-7b >"$LOG_ROOT/clean_qwen25.log" 2>&1
stop_readers

echo '[medqa] stage=qwen25-graph'
run_graph_shard 0 0 500 & left=$!
run_graph_shard 1 500 500 & right=$!
wait "$left"
wait "$right"
"$PYTHON" scripts/merge_contribution_graph_shards.py \
  --retrieval "$OUT_ROOT/medqa/retrieval/top10_1000.jsonl" \
  --shard "$OUT_ROOT/medqa/graphs/shards/contribution_graph_top5_0_499.jsonl" \
  --shard "$OUT_ROOT/medqa/graphs/shards/contribution_graph_top5_500_999.jsonl" \
  --out "$OUT_ROOT/medqa/graphs/contribution_graph_top5_1000.jsonl" \
  --summary-out "$OUT_ROOT/medqa/graphs/contribution_graph_top5_1000.summary.json" \
  --n 1000 >"$LOG_ROOT/merge_qwen25_graph.log" 2>&1
fi

echo '[medqa] stage=qwen25-table3-rankers-frontier'
CAUSALITYRAG_READER_MODE=medqa "$PYTHON" scripts/run_table3_rankers.py \
  --out-root "$OUT_ROOT" --datasets medqa --model-path "$QWEN25" \
  >"$LOG_ROOT/table3_rankers.log" 2>&1
"$PYTHON" scripts/run_table3_frontiers.py \
  --out-root "$OUT_ROOT" --datasets medqa \
  >"$LOG_ROOT/table3_frontier.log" 2>&1

echo '[medqa] stage=qwen25-table3-reader-control'
start_qwen25_readers
CAUSALITYRAG_READER_MODE=medqa "$PYTHON" scripts/run_table3_reader_stage.py \
  --out-root "$OUT_ROOT" --datasets medqa \
  --llm-base-url http://127.0.0.1:8000/v1 --llm-model qwen2.5-7b \
  >"$LOG_ROOT/table3_reader.log" 2>&1
stop_readers

if [[ "$ONLY_QWEN25" == 1 ]]; then
  echo '[medqa] qwen2.5 complete'
  date -Is >"$RUN_ROOT/QWEN25_COMPLETE"
  exit 0
fi

echo '[medqa] stage=cross-model-qwen3'
REPO="$REPO" FROZEN_ROOT="$OUT_ROOT" OUTPUT_ROOT="$SENSITIVITY_ROOT" \
  scripts/launchers/run_cross_model_reflow_server3.sh qwen3-4b \
  /data1/yujia/models/Qwen3-4B-Instruct-2507 qwen3-4b 1000 \
  >"$LOG_ROOT/cross_qwen3.log" 2>&1

echo '[medqa] stage=cross-model-llama'
REPO="$REPO" FROZEN_ROOT="$OUT_ROOT" OUTPUT_ROOT="$SENSITIVITY_ROOT" \
  scripts/launchers/run_cross_model_reflow_server3.sh llama3.1-8b \
  /data1/yujia/models/Llama-3.1-8B-Instruct llama3.1-8b 1000 \
  >"$LOG_ROOT/cross_llama.log" 2>&1

echo '[medqa] stage=cross-model-mistral'
REPO="$REPO" FROZEN_ROOT="$OUT_ROOT" OUTPUT_ROOT="$SENSITIVITY_ROOT" \
  scripts/launchers/run_cross_model_reflow_server3.sh mistral-nemo-12b \
  /data1/yujia/models/Mistral-Nemo-Instruct-2407 mistral-nemo-12b 1000 \
  >"$LOG_ROOT/cross_mistral.log" 2>&1

echo '[medqa] complete'
date -Is >"$RUN_ROOT/COMPLETE"
