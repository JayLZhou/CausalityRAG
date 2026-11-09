#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL_KEY MODEL_PATH SERVED_MODEL" >&2
  exit 2
fi
if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

MODEL_KEY=$1
MODEL_PATH=$2
SERVED_MODEL=$3
RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_popqa_models_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
READER_PYTHON=${READER_PYTHON:-$PYTHON}
if [[ "$MODEL_KEY" == "qwen3-4b" && "$READER_PYTHON" == "$PYTHON" ]]; then
  READER_PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python
fi
FROZEN=${FROZEN:-/data1/yujia/CausalityRAG/out_popqa_20260810/popqa}
FINAL8=${FINAL8:-/data1/yujia/CausalityRAG/out_table3_final8_20260811/popqa}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data1/yujia/CausalityRAG/out/model_sensitivity_popqa_v1}
RUN_ROOT=$OUTPUT_ROOT/$MODEL_KEY
BASE=$RUN_ROOT/popqa
LOG_ROOT=$RUN_ROOT/logs
PID_ROOT=$RUN_ROOT/pids
INPUT=$FROZEN/retrieval/top10_1000.jsonl
UNITS=$FROZEN/inputs/token_units_top10_1000.jsonl
FACTUAL_POOL=$FROZEN/replacements/shared_pool_top10_v1/shared_pool.jsonl
CONTROL_POOL=$FINAL8/controls/paraphrase_pool_top5_v2.jsonl
FACTUAL_SHA=42dd41818cdefac31ce788a1bf13402b07353ed032806c9a874f5fb8b19313e7
CONTROL_SHA=77cdadb8a1e51b77ab6209d7e020288086a8f9f5ca85781af1c8dcf6a4734838

mkdir -p "$LOG_ROOT" "$PID_ROOT" "$BASE/clean" "$BASE/graph/shards" \
  "$BASE/frontier" "$BASE/factual" "$BASE/final" "$BASE/synonym"
cd "$RELEASE"

line_count() {
  if [[ -f "$1" ]]; then
    awk 'NF {count++} END {print count+0}' "$1"
  else
    echo 0
  fi
}

complete_jsonl() {
  [[ "$(line_count "$1")" == "$2" ]]
}

port_is_free() {
  ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$1$"
}

wait_endpoint() {
  local url=$1 timeout=${2:-900} pid=${3:-}
  local deadline=$((SECONDS + timeout))
  until curl -fsS "$url" >/dev/null 2>&1; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "endpoint process exited before becoming ready: $url pid=$pid" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "endpoint did not become ready: $url" >&2
      exit 1
    fi
    sleep 5
  done
}

wait_ports_quiescent() {
  local stable=0 port
  while (( stable < 5 )); do
    for port in 8000 8002 8003; do
      if ! port_is_free "$port"; then
        echo "refusing to interrupt unowned service on port $port" >&2
        exit 1
      fi
    done
    stable=$((stable + 1))
    sleep 2
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

cleanup() {
  stop_readers
}
trap cleanup EXIT

start_readers() {
  local port
  wait_ports_quiescent
  if port_is_free 8021; then
    echo "required existing spaCy service is not listening on 8021" >&2
    exit 1
  fi
  local spec gpu reader_port
  for spec in '0 8002' '1 8003'; do
    read -r gpu reader_port <<<"$spec"
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$READER_PYTHON" \
      -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 \
      --port "$reader_port" \
      >"$LOG_ROOT/vllm_${reader_port}.log" 2>&1 < /dev/null &
    echo $! >"$PID_ROOT/reader_${reader_port}.pid"
  done
  wait_endpoint http://127.0.0.1:8002/v1/models 900 "$(cat "$PID_ROOT/reader_8002.pid")"
  wait_endpoint http://127.0.0.1:8003/v1/models 900 "$(cat "$PID_ROOT/reader_8003.pid")"
  setsid "$PYTHON" scripts/lb_proxy_two_gpu.py \
    >"$LOG_ROOT/proxy_8000.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models 120 "$(cat "$PID_ROOT/proxy_8000.pid")"
}

write_empty_baselines() {
  "$PYTHON" - "$INPUT" "$BASE/synonym/empty_baselines.jsonl" <<'PY'
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
rows = []
with source.open(encoding='utf-8') as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            rows.append({'id': str(row.get('id', '')), 'methods': {}})
target.write_text(
    ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8'
)
PY
}

for path in "$INPUT" "$UNITS" "$FACTUAL_POOL" "$CONTROL_POOL" \
  "$MODEL_PATH/config.json"; do
  [[ -s "$path" ]] || { echo "missing required artifact: $path" >&2; exit 1; }
done
[[ "$(sha256sum "$FACTUAL_POOL" | awk '{print $1}')" == "$FACTUAL_SHA" ]]
[[ "$(sha256sum "$CONTROL_POOL" | awk '{print $1}')" == "$CONTROL_SHA" ]]
[[ "$(line_count "$INPUT")" == 1000 ]]
[[ "$(line_count "$UNITS")" == 1000 ]]
if [[ -f "$RUN_ROOT/COMPLETE" ]]; then
  echo "cross-model PopQA already complete: $MODEL_KEY"
  exit 0
fi

echo "[popqa-model] model=$MODEL_KEY stage=clean"
start_readers
if ! complete_jsonl "$BASE/clean/clean_targets.jsonl" 1000; then
  CAUSALITYRAG_READER_MODE=short_answer \
    CAUSALITYRAG_READER_MAX_TOKENS=128 \
    "$PYTHON" scripts/generate_reader_targets.py \
    --input "$INPUT" --out "$BASE/clean/clean_targets.jsonl" \
    --summary-out "$BASE/clean/summary.json" --n 1000 --k 5 \
    --workers 48 --base-url http://127.0.0.1:8000/v1 \
    --served-model "$SERVED_MODEL" >"$LOG_ROOT/clean.log" 2>&1
fi
complete_jsonl "$BASE/clean/clean_targets.jsonl" 1000
stop_readers

echo "[popqa-model] model=$MODEL_KEY stage=graph"
if ! complete_jsonl "$BASE/graph/contribution_graph.jsonl" 1000; then
  if [[ "$MODEL_KEY" == "mistral-nemo-12b" ]]; then
    CAUSALITYRAG_READER_MODE=short_answer CUDA_VISIBLE_DEVICES=0,1 \
      "$PYTHON" scripts/build_contribution_graph.py \
      --input "$INPUT" --out "$BASE/graph/contribution_graph.jsonl" \
      --summary-out "$BASE/graph/summary.json" --model-path "$MODEL_PATH" \
      --start 0 --n 1000 --k 5 --target results \
      --target-results "$BASE/clean/clean_targets.jsonl" \
      --device cuda --device-map balanced --dtype bfloat16 \
      --max-receivers-per-layer 48 >"$LOG_ROOT/graph.log" 2>&1
  else
    CAUSALITYRAG_READER_MODE=short_answer CUDA_VISIBLE_DEVICES=0 \
      "$PYTHON" scripts/build_contribution_graph.py \
      --input "$INPUT" --out "$BASE/graph/shards/part0.jsonl" \
      --summary-out "$BASE/graph/shards/part0.summary.json" \
      --model-path "$MODEL_PATH" --start 0 --n 500 --k 5 \
      --target results --target-results "$BASE/clean/clean_targets.jsonl" \
      --device cuda --dtype bfloat16 --max-receivers-per-layer 48 \
      >"$LOG_ROOT/graph_part0.log" 2>&1 &
    left=$!
    CAUSALITYRAG_READER_MODE=short_answer CUDA_VISIBLE_DEVICES=1 \
      "$PYTHON" scripts/build_contribution_graph.py \
      --input "$INPUT" --out "$BASE/graph/shards/part1.jsonl" \
      --summary-out "$BASE/graph/shards/part1.summary.json" \
      --model-path "$MODEL_PATH" --start 500 --n 500 --k 5 \
      --target results --target-results "$BASE/clean/clean_targets.jsonl" \
      --device cuda --dtype bfloat16 --max-receivers-per-layer 48 \
      >"$LOG_ROOT/graph_part1.log" 2>&1 &
    right=$!
    wait "$left"
    wait "$right"
    "$PYTHON" scripts/merge_contribution_graph_shards.py \
      --retrieval "$INPUT" --shard "$BASE/graph/shards/part0.jsonl" \
      --shard "$BASE/graph/shards/part1.jsonl" \
      --out "$BASE/graph/contribution_graph.jsonl" \
      --summary-out "$BASE/graph/summary.json" --n 1000 \
      >"$LOG_ROOT/graph_merge.log" 2>&1
  fi
fi
complete_jsonl "$BASE/graph/contribution_graph.jsonl" 1000

echo "[popqa-model] model=$MODEL_KEY stage=frontier"
if ! complete_jsonl "$BASE/frontier/initial.jsonl" 1000; then
  "$PYTHON" exp/run_reflow.py \
    --input "$INPUT" --graphs "$BASE/graph/contribution_graph.jsonl" \
    --units-cache "$UNITS" --shared-pool "$FACTUAL_POOL" \
    --expected-pool-sha256 "$FACTUAL_SHA" \
    --out "$BASE/frontier/initial.jsonl" \
    --summary-out "$BASE/frontier/initial.summary.json" --n 1000 --k 5 \
    --record-empty-frontier \
    >"$LOG_ROOT/frontier.log" 2>&1
fi
complete_jsonl "$BASE/frontier/initial.jsonl" 1000

echo "[popqa-model] model=$MODEL_KEY stage=factual-initial"
start_readers
if ! complete_jsonl "$BASE/factual/initial_results.jsonl" 1000; then
  CAUSALITYRAG_READER_MODE=short_answer \
    CAUSALITYRAG_READER_MAX_TOKENS=128 \
    "$PYTHON" scripts/evaluate_reflow.py \
    --input "$INPUT" --units-cache "$UNITS" \
    --frontier "$BASE/frontier/initial.jsonl" \
    --shared-pool "$FACTUAL_POOL" --expected-pool-sha256 "$FACTUAL_SHA" \
    --out "$BASE/factual/initial_results.jsonl" \
    --summary-out "$BASE/factual/initial_results.summary.json" \
    --n 1000 --k 5 --workers 48 --replacement-seed 0 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
    >"$LOG_ROOT/factual_initial.log" 2>&1
fi
complete_jsonl "$BASE/factual/initial_results.jsonl" 1000

no_frontier=$("$PYTHON" - "$BASE/factual/initial_results.jsonl" <<'PY'
import json, sys
print(sum(
    json.loads(line).get('evaluation_status') == 'no_frontier_candidate'
    for line in open(sys.argv[1], encoding='utf-8') if line.strip()
))
PY
)
echo "[popqa-model] model=$MODEL_KEY no_frontier=$no_frontier"
if (( no_frontier > 0 )); then
  "$PYTHON" scripts/repair_disconnected_contribution_graphs.py \
    --graphs "$BASE/graph/contribution_graph.jsonl" \
    --shared-pool "$FACTUAL_POOL" \
    --target-results "$BASE/factual/initial_results.jsonl" \
    --expected-pool-sha256 "$FACTUAL_SHA" \
    --out "$BASE/final/graphs.jsonl" \
    --summary-out "$BASE/final/graphs.summary.json" --n 1000 \
    >"$LOG_ROOT/repair_graphs.log" 2>&1
  "$PYTHON" scripts/prepare_no_frontier_resume.py prepare \
    --input "$INPUT" --units-cache "$UNITS" \
    --repaired-graphs "$BASE/final/graphs.jsonl" \
    --old-frontier "$BASE/frontier/initial.jsonl" \
    --old-results "$BASE/factual/initial_results.jsonl" \
    --out-dir "$BASE/final/resume" --n 1000 \
    >"$LOG_ROOT/repair_prepare.log" 2>&1
  targets=$(line_count "$BASE/final/resume/input.jsonl")
  [[ "$targets" == "$no_frontier" ]]
  "$PYTHON" exp/run_reflow.py \
    --input "$BASE/final/resume/input.jsonl" \
    --graphs "$BASE/final/resume/graphs.jsonl" \
    --units-cache "$BASE/final/resume/units.jsonl" \
    --shared-pool "$FACTUAL_POOL" --expected-pool-sha256 "$FACTUAL_SHA" \
    --out "$BASE/final/resume/frontier.jsonl" \
    --summary-out "$BASE/final/resume/frontier.summary.json" \
    --n "$targets" --k 5 >"$LOG_ROOT/repair_frontier.log" 2>&1
  CAUSALITYRAG_READER_MODE=short_answer \
    CAUSALITYRAG_READER_MAX_TOKENS=128 \
    "$PYTHON" scripts/evaluate_reflow.py \
    --input "$BASE/final/resume/input.jsonl" \
    --units-cache "$BASE/final/resume/units.jsonl" \
    --frontier "$BASE/final/resume/frontier.jsonl" \
    --shared-pool "$FACTUAL_POOL" --expected-pool-sha256 "$FACTUAL_SHA" \
    --out "$BASE/final/resume/results.jsonl" \
    --summary-out "$BASE/final/resume/results.summary.json" \
    --n "$targets" --k 5 --workers 48 --replacement-seed 0 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
    >"$LOG_ROOT/repair_reader.log" 2>&1
  "$PYTHON" scripts/prepare_no_frontier_resume.py merge-frontier \
    --manifest "$BASE/final/resume/manifest.json" \
    --old-frontier "$BASE/frontier/initial.jsonl" \
    --resumed-frontier "$BASE/final/resume/frontier.jsonl" \
    --out "$BASE/final/frontier.jsonl" \
    --summary-out "$BASE/final/frontier.summary.json" \
    >"$LOG_ROOT/repair_merge_frontier.log" 2>&1
  "$PYTHON" scripts/prepare_no_frontier_resume.py merge-results \
    --manifest "$BASE/final/resume/manifest.json" \
    --old-results "$BASE/factual/initial_results.jsonl" \
    --resumed-results "$BASE/final/resume/results.jsonl" \
    --out "$BASE/factual/results.jsonl" \
    --summary-out "$BASE/factual/summary.json" \
    >"$LOG_ROOT/repair_merge_results.log" 2>&1
else
  cp "$BASE/graph/contribution_graph.jsonl" "$BASE/final/graphs.jsonl"
  cp "$BASE/graph/summary.json" "$BASE/final/graphs.summary.json"
  cp "$BASE/frontier/initial.jsonl" "$BASE/final/frontier.jsonl"
  cp "$BASE/frontier/initial.summary.json" "$BASE/final/frontier.summary.json"
  cp "$BASE/factual/initial_results.jsonl" "$BASE/factual/results.jsonl"
  cp "$BASE/factual/initial_results.summary.json" "$BASE/factual/summary.json"
fi
complete_jsonl "$BASE/final/graphs.jsonl" 1000
complete_jsonl "$BASE/final/frontier.jsonl" 1000
complete_jsonl "$BASE/factual/results.jsonl" 1000

echo "[popqa-model] model=$MODEL_KEY stage=control"
write_empty_baselines
if ! complete_jsonl "$BASE/synonym/results.jsonl" 1000; then
  CAUSALITYRAG_READER_MODE=short_answer \
    CAUSALITYRAG_READER_MAX_TOKENS=128 \
    "$PYTHON" scripts/evaluate_paraphrase_controls.py \
    --input "$INPUT" --units-cache "$UNITS" \
    --reflow-results "$BASE/factual/results.jsonl" \
    --baseline-results "$BASE/synonym/empty_baselines.jsonl" \
    --paraphrase-pool "$CONTROL_POOL" --expected-pool-sha256 "$CONTROL_SHA" \
    --out "$BASE/synonym/results.jsonl" \
    --summary-out "$BASE/synonym/summary.json" --n 1000 --k 5 \
    --workers 48 --replacement-seed 0 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
    >"$LOG_ROOT/control.log" 2>&1
fi
complete_jsonl "$BASE/synonym/results.jsonl" 1000
stop_readers

echo "[popqa-model] model=$MODEL_KEY stage=audit"
"$PYTHON" - "$RUN_ROOT" "$MODEL_KEY" "$MODEL_PATH" "$SERVED_MODEL" \
  "$INPUT" "$UNITS" "$FACTUAL_POOL" "$CONTROL_POOL" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
key, model_path, served = sys.argv[2:5]
sources = list(map(Path, sys.argv[5:]))

def rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

base = root / 'popqa'
artifacts = {
    'clean': base / 'clean/clean_targets.jsonl',
    'graph': base / 'final/graphs.jsonl',
    'frontier': base / 'final/frontier.jsonl',
    'factual': base / 'factual/results.jsonl',
    'control': base / 'synonym/results.jsonl',
}
loaded = {name: rows(path) for name, path in artifacts.items()}
ids = [[str(row.get('id', '')) for row in value] for value in loaded.values()]
if any(len(value) != 1000 or len(set(value)) != 1000 for value in ids):
    raise ValueError('artifact row/ID cardinality failure')
if any(value != ids[0] for value in ids[1:]):
    raise ValueError('artifact ID alignment failure')
factual = loaded['factual']
report = {
    'schema': 'causalityrag.popqa_cross_model.v1',
    'server': '3haoji-A100',
    'model_key': key,
    'model_path': str(Path(model_path).resolve()),
    'served_model': served,
    'queries': 1000,
    'reader_mode': 'short_answer',
    'guided_decoding': False,
    'replacement_seed': 0,
    'receiver_beam': 48,
    'factual_status': dict(Counter(
        str(row.get('evaluation_status', '')) for row in factual
    )),
    'remaining_no_frontier': sum(
        row.get('evaluation_status') == 'no_frontier_candidate'
        for row in factual
    ),
    'artifact_sha256': {name: sha(path) for name, path in artifacts.items()},
    'frozen_sources': {
        path.name: {'path': str(path.resolve()), 'sha256': sha(path)}
        for path in sources
    },
}
(root / 'audit.json').write_text(
    json.dumps(report, indent=2) + '\n', encoding='utf-8'
)
if report['remaining_no_frontier']:
    raise ValueError('no-frontier repair is incomplete')
PY
date -Is >"$RUN_ROOT/COMPLETE"
echo "[popqa-model] complete model=$MODEL_KEY root=$RUN_ROOT"
