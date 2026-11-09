#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

REPO=${REPO:-/data1/yujia/CausalityRAG_release_medqa_20260809}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
OUT_ROOT=${OUT_ROOT:-/data1/yujia/CausalityRAG/out}
QWEN25=${QWEN25:-/data1/yujia/models/Qwen2.5-7B-Instruct}
RUN_ROOT=${RUN_ROOT:-$OUT_ROOT/medqa/medqa_replacement_run_20260809}
SHARD_DIR=$OUT_ROOT/medqa/graphs/shards
LEFT=$SHARD_DIR/contribution_graph_top5_0_499.jsonl
PARTIAL=$SHARD_DIR/contribution_graph_top5_500_993.partial.jsonl
TAIL=$SHARD_DIR/contribution_graph_top5_994_999.jsonl
TAIL_SUMMARY=${TAIL%.jsonl}.summary.json
GRAPH=$OUT_ROOT/medqa/graphs/contribution_graph_top5_1000.jsonl
GRAPH_SUMMARY=${GRAPH%.jsonl}.summary.json

line_count() {
  awk 'NF {n++} END {print n+0}' "$1"
}

cd "$REPO"
[[ "$(line_count "$LEFT")" == 500 ]] || {
  echo 'left graph shard is not exactly 500 rows' >&2
  exit 1
}
[[ "$(line_count "$SHARD_DIR/contribution_graph_top5_500_999.jsonl")" == 494 ]] || {
  echo 'right partial graph shard is not exactly 494 rows' >&2
  exit 1
}
cp "$SHARD_DIR/contribution_graph_top5_500_999.jsonl" "$PARTIAL"

if ! CAUSALITYRAG_READER_MODE=medqa CUDA_VISIBLE_DEVICES=1 "$PYTHON" \
  scripts/build_contribution_graph.py \
  --input "$OUT_ROOT/medqa/retrieval/top10_1000.jsonl" \
  --out "$TAIL" --summary-out "$TAIL_SUMMARY" --model-path "$QWEN25" \
  --start 994 --n 6 --k 5 --target results \
  --target-results "$OUT_ROOT/medqa/inputs/clean_targets_top5_1000.jsonl" \
  --device cuda --dtype bfloat16 --max-receivers-per-layer 48 \
  >"$RUN_ROOT/logs/graph_qwen25_994_999.log" 2>&1; then
  echo '[medqa] graph tail reported excluded rows; validating artifacts'
fi
[[ "$(line_count "$TAIL")" == 6 && -s "$TAIL_SUMMARY" ]] || {
  echo 'graph tail did not produce six auditable rows' >&2
  exit 1
}

"$PYTHON" scripts/merge_contribution_graph_shards.py \
  --retrieval "$OUT_ROOT/medqa/retrieval/top10_1000.jsonl" \
  --shard "$LEFT" --shard "$PARTIAL" --shard "$TAIL" \
  --out "$GRAPH" --summary-out "$GRAPH_SUMMARY" --n 1000 \
  >"$RUN_ROOT/logs/merge_qwen25_graph.log" 2>&1
[[ "$(line_count "$GRAPH")" == 1000 && -s "$GRAPH_SUMMARY" ]] || {
  echo 'merged MedQA contribution graph failed its 1000-row audit' >&2
  exit 1
}

exec env START_AT_TABLE3=1 ONLY_QWEN25=1 \
  ./scripts/launchers/run_medqa_replacement_server3.sh
