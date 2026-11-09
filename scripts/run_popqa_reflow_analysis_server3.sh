#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_popqa_figures_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-/data1/yujia/CausalityRAG_release_popqa_20260810/scripts/compat/transformers_452}
MODEL_PATH=${MODEL_PATH:-/data1/yujia/models/Qwen2.5-7B-Instruct}
MODEL=${MODEL:-qwen2.5-7b}
SOURCE=${SOURCE:-/data1/yujia/CausalityRAG/out_popqa_20260810/popqa}
REPAIRED=${REPAIRED:-/data1/yujia/CausalityRAG/out_popqa_no_frontier_repair_20260811/popqa}
OUT=${OUT:-/data1/yujia/CausalityRAG/out_popqa_figures_20260811/popqa}
LOG_ROOT=$OUT/logs
PID_ROOT=$OUT/pids
INPUT=$SOURCE/retrieval/top10_1000.jsonl
UNITS=$SOURCE/inputs/token_units_top10_1000.jsonl
POOL=$SOURCE/replacements/shared_pool_top10_v1/shared_pool.jsonl
GRAPHS=$REPAIRED/graphs.jsonl
FRONTIER=$REPAIRED/frontier_1000.jsonl
POOL_SHA=42dd41818cdefac31ce788a1bf13402b07353ed032806c9a874f5fb8b19313e7
GRAPH_SHA=b64d23aa1784927339ceb492196a7b4bc4938c9d3f3a9b85d1213e00dac625e9
FRONTIER_SHA=1953dfa22aeacb60ac38d3f9d379ad791fad22419a4bd6477cebdd361a55d2e5

mkdir -p "$LOG_ROOT" "$PID_ROOT" "$OUT/fidelity" "$OUT/protocol" \
  "$OUT/figure8"
cd "$RELEASE"

line_count() {
  if [[ -f "$1" ]]; then
    awk 'NF {count++} END {print count+0}' "$1"
  else
    echo 0
  fi
}

require_file() {
  [[ -s "$1" ]] || { echo "missing required artifact: $1" >&2; exit 1; }
}

require_sha() {
  local actual
  actual=$(sha256sum "$1" | awk '{print $1}')
  [[ "$actual" == "$2" ]] || {
    echo "artifact fingerprint mismatch: $1 $actual != $2" >&2
    exit 1
  }
}

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
  for port in 8000 8002 8003; do
    if ! port_is_free "$port"; then
      echo "refusing to interrupt unowned service on port $port" >&2
      exit 1
    fi
  done
  if port_is_free 8021; then
    echo "required existing spaCy service is not listening on 8021" >&2
    exit 1
  fi
  local spec gpu reader_port
  for spec in '0 8002' '1 8003'; do
    read -r gpu reader_port <<<"$spec"
    PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
      -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$MODEL" \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 \
      --port "$reader_port" \
      >"$LOG_ROOT/vllm_${reader_port}.log" 2>&1 < /dev/null &
    echo $! >"$PID_ROOT/reader_${reader_port}.pid"
  done
  wait_endpoint http://127.0.0.1:8002/v1/models 900
  wait_endpoint http://127.0.0.1:8003/v1/models 900
  setsid "$PYTHON" "$RELEASE/scripts/lb_proxy_two_gpu.py" \
    >"$LOG_ROOT/proxy_8000.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models 120
}

run_protocol() {
  local name=$1 frontier=$2
  local rows=$OUT/protocol/${name}_1000.jsonl
  local summary=$OUT/protocol/${name}_1000.summary.json
  if [[ "$(line_count "$rows")" == 1000 ]] && [[ -s "$summary" ]]; then
    echo "[popqa-analysis] reuse protocol=$name"
    return
  fi
  CAUSALITYRAG_READER_MODE=short_answer \
    CAUSALITYRAG_READER_MAX_TOKENS=128 \
    "$PYTHON" scripts/evaluate_reflow.py \
    --input "$INPUT" --units-cache "$UNITS" --frontier "$frontier" \
    --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
    --out "$rows" --summary-out "$summary" --n 1000 --k 5 \
    --workers 48 --replacement-seed 0 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$MODEL" \
    >"$LOG_ROOT/protocol_${name}.log" 2>&1
}

for path in "$INPUT" "$UNITS" "$POOL" "$GRAPHS" "$FRONTIER"; do
  require_file "$path"
done
require_sha "$POOL" "$POOL_SHA"
require_sha "$GRAPHS" "$GRAPH_SHA"
require_sha "$FRONTIER" "$FRONTIER_SHA"
[[ "$(line_count "$INPUT")" == 1000 ]]
[[ "$(line_count "$UNITS")" == 1000 ]]
[[ "$(line_count "$GRAPHS")" == 1000 ]]
[[ "$(line_count "$FRONTIER")" == 1000 ]]

if [[ -f "$OUT/COMPLETE" ]]; then
  echo "PopQA analyses already complete: $OUT"
  exit 0
fi

start_readers

echo '[popqa-analysis] stage=fidelity'
CAUSALITYRAG_READER_MODE=short_answer \
  CAUSALITYRAG_READER_MAX_TOKENS=128 \
  "$PYTHON" scripts/run_residual_flow_fidelity.py \
  --input "$INPUT" --units-cache "$UNITS" --graphs "$GRAPHS" \
  --shared-pool "$POOL" --out "$OUT/fidelity/probes_3000.jsonl" \
  --summary-out "$OUT/fidelity/summary.json" --n-queries 100 \
  --budgets 1 3 5 --trials 10 --workers 48 --k 5 --seed 0 \
  --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$MODEL" \
  >"$LOG_ROOT/fidelity.log" 2>&1
"$PYTHON" scripts/audit_residual_flow_fidelity_seed.py \
  --input "$INPUT" --units-cache "$UNITS" --graphs "$GRAPHS" \
  --shared-pool "$POOL" --probes "$OUT/fidelity/probes_3000.jsonl" \
  --n-queries 100 --budgets 1 3 5 --trials 10 --k 5 --seed 0 \
  --out "$OUT/fidelity/seed_audit.json" \
  >"$LOG_ROOT/fidelity_seed_audit.log" 2>&1

echo '[popqa-analysis] stage=selection-protocols'
"$PYTHON" scripts/build_selection_protocol_frontiers.py \
  --frontier "$FRONTIER" \
  --one-shot-out "$OUT/protocol/one_shot_frontier_1000.jsonl" \
  --sweep-out "$OUT/protocol/grid_frontier_1000.jsonl" \
  >"$LOG_ROOT/build_protocols.log" 2>&1
run_protocol exact "$FRONTIER"
run_protocol one_shot "$OUT/protocol/one_shot_frontier_1000.jsonl"
run_protocol fixed_grid "$OUT/protocol/grid_frontier_1000.jsonl"

echo '[popqa-analysis] stage=figure8-coarse'
CAUSALITYRAG_READER_MODE=short_answer \
  CAUSALITYRAG_READER_MAX_TOKENS=128 \
  "$PYTHON" scripts/evaluate_fixed_budget_acc.py \
  --input "$INPUT" --units-cache "$UNITS" --frontier "$FRONTIER" \
  --scores "mirage=$SOURCE/methods/baselines/mirage_top5_1000.jsonl" \
  --scores "arc_jsd=$SOURCE/methods/baselines/arc_jsd_top5_1000.jsonl" \
  --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
  --out "$OUT/figure8/coarse_1000.jsonl" \
  --summary-out "$OUT/figure8/coarse_1000.summary.json" \
  --budgets 1,2,4,8,16,32,64,128 --skip-reflow \
  --replacement-seed 0 --n 1000 --k 5 --workers 48 \
  --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$MODEL" \
  --reader-max-tokens 128 >"$LOG_ROOT/figure8_coarse.log" 2>&1

stop_readers

echo '[popqa-analysis] stage=manifest'
"$PYTHON" - "$OUT" "$INPUT" "$UNITS" "$POOL" "$GRAPHS" \
  "$FRONTIER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, *sources = map(Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

manifest = {
    'schema': 'causalityrag.popqa_reflow_analysis.v1',
    'server': '3haoji-A100',
    'model': 'Qwen2.5-7B-Instruct',
    'queries': 1000,
    'reader_mode': 'short_answer',
    'guided_decoding': False,
    'replacement_seed': 0,
    'fidelity_seed': 0,
    'sources': {
        path.name: {'path': str(path.resolve()), 'sha256': sha256(path)}
        for path in sources
    },
}
(out / 'manifest.json').write_text(
    json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
)
PY
date -Is >"$OUT/COMPLETE"
echo "[popqa-analysis] complete root=$OUT"
