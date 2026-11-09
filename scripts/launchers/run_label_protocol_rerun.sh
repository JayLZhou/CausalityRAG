#!/usr/bin/env bash
set -euo pipefail

# Rebuild only the answer protocol, graph, and evaluations for constrained-answer datasets.
REPO=${REPO:-/data1/yujia/CausalityRAG_release_paraphrase_v2d_20260805}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL=${MODEL:-/data1/yujia/models/Qwen2.5-7B-Instruct}
OLD_OUT=${OLD_OUT:-/data1/yujia/CausalityRAG/out}
OUT=${OUT:-/data1/yujia/CausalityRAG/out_label_protocol_20260805}
LOGDIR=${LOGDIR:-/data1/yujia/vllm_logs/label_protocol_20260805}
DATASETS=(pubmedqa quartz)

mkdir -p "$LOGDIR"

for dataset in "${DATASETS[@]}"; do
  mkdir -p "$OUT/$dataset"/{retrieval,inputs,replacements/shared_pool_top10_v1}
  ln -sfn "$OLD_OUT/$dataset/retrieval/top10_1000.jsonl" \
    "$OUT/$dataset/retrieval/top10_1000.jsonl"
  ln -sfn "$OLD_OUT/$dataset/inputs/token_units_top10_1000.jsonl" \
    "$OUT/$dataset/inputs/token_units_top10_1000.jsonl"
  ln -sfn "$OLD_OUT/$dataset/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
    "$OUT/$dataset/replacements/shared_pool_top10_v1/shared_pool.jsonl"
  ln -sfn "$OLD_OUT/$dataset/replacements/shared_pool_top10_v1/shared_pool.manifest.json" \
    "$OUT/$dataset/replacements/shared_pool_top10_v1/shared_pool.manifest.json"
done

stop_readers() {
  fuser -k 8000/tcp 8002/tcp 8003/tcp 2>/dev/null || true
  pkill -f "[v]llm.entrypoints.openai.api_server.*--port 8002" 2>/dev/null || true
  pkill -f "[v]llm.entrypoints.openai.api_server.*--port 8003" 2>/dev/null || true
  pkill -f "[l]b_proxy_two_gpu.py" 2>/dev/null || true
}

start_readers() {
  stop_readers
  CUDA_VISIBLE_DEVICES=0 setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen2.5-7b --dtype bfloat16 \
    --trust-remote-code --max-model-len 32768 --enable-chunked-prefill \
    --max-num-seqs 128 --enforce-eager --disable-log-requests \
    --gpu-memory-utilization 0.90 --port 8002 >"$LOGDIR/vllm_8002.log" 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen2.5-7b --dtype bfloat16 \
    --trust-remote-code --max-model-len 32768 --enable-chunked-prefill \
    --max-num-seqs 128 --enforce-eager --disable-log-requests \
    --gpu-memory-utilization 0.90 --port 8003 >"$LOGDIR/vllm_8003.log" 2>&1 < /dev/null &
  until curl -fsS http://127.0.0.1:8002/v1/models >/dev/null && \
        curl -fsS http://127.0.0.1:8003/v1/models >/dev/null; do
    sleep 10
  done
  setsid "$PYTHON" "$REPO/scripts/lb_proxy_two_gpu.py" \
    >"$LOGDIR/proxy_8000.log" 2>&1 < /dev/null &
  until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; do sleep 2; done
}

cd "$REPO"
start_readers
for dataset in "${DATASETS[@]}"; do
  if [[ "${SKIP_EXISTING_TARGETS:-0}" == "1" && \
        -s "$OUT/$dataset/inputs/clean_targets_top5_1000.jsonl" ]]; then
    echo "[label-protocol] reusing existing targets for $dataset"
  else
    CAUSALITYRAG_READER_MODE="$dataset" "$PYTHON" scripts/generate_reader_targets.py \
      --input "$OUT/$dataset/retrieval/top10_1000.jsonl" \
      --out "$OUT/$dataset/inputs/clean_targets_top5_1000.jsonl" \
      --summary-out "$OUT/$dataset/inputs/clean_targets_top5_1000.summary.json" \
      --n 1000 --k 5 --workers 64 --base-url http://127.0.0.1:8000/v1 \
      --served-model qwen2.5-7b
  fi
done

stop_readers

for index in 0 1; do
  dataset=${DATASETS[$index]}
  CUDA_VISIBLE_DEVICES="$index" CAUSALITYRAG_READER_MODE="$dataset" \
    "$PYTHON" scripts/build_contribution_graph.py \
    --input "$OUT/$dataset/retrieval/top10_1000.jsonl" \
    --out "$OUT/$dataset/graphs/contribution_graph_top5_1000.jsonl" \
    --summary-out "$OUT/$dataset/graphs/contribution_graph_top5_1000.summary.json" \
    --model-path "$MODEL" --n 1000 --k 5 --target results \
    --target-results "$OUT/$dataset/inputs/clean_targets_top5_1000.jsonl" \
    --device cuda --dtype bfloat16 \
    >"$LOGDIR/graph_${dataset}.log" 2>&1 &
done
wait

"$PYTHON" scripts/run_table3_rankers.py --out-root "$OUT" \
  --datasets pubmedqa quartz --model-path "$MODEL" --n 1000
"$PYTHON" scripts/run_table3_frontiers.py --out-root "$OUT" \
  --datasets pubmedqa quartz --n 1000 --k 5

start_readers
for dataset in "${DATASETS[@]}"; do
  CAUSALITYRAG_READER_MODE="$dataset" "$PYTHON" scripts/run_table3_reader_stage.py \
    --out-root "$OUT" --datasets "$dataset" --workers 64 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model qwen2.5-7b
done
echo "[label-protocol] complete: $OUT"
