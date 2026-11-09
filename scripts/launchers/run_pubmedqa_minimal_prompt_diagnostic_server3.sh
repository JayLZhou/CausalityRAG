#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

REPO=${REPO:-/data1/yujia/CausalityRAG_release_pubmedqa_minimal_prompt_20260810}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL=${MODEL:-/data1/yujia/models/Qwen2.5-7B-Instruct}
SERVED_MODEL=${SERVED_MODEL:-qwen2.5-7b}
SOURCE=/data1/yujia/CausalityRAG/out_label_protocol_20260805/pubmedqa
OLD_RESULTS=/data1/yujia/CausalityRAG/out_pubmedqa_readerfix_20260809/pubmedqa/methods/reflow/results_top5_1000.jsonl
RUN_ROOT=${RUN_ROOT:-/data1/yujia/CausalityRAG/out_pubmedqa_minimal_prompt_20260810/n50}
N=50
BEAM=48

INPUT=$SOURCE/retrieval/top10_1000.jsonl
UNITS=$SOURCE/inputs/token_units_top10_1000.jsonl
POOL=$SOURCE/replacements/shared_pool_top10_v1/shared_pool.jsonl
POOL_SHA=08e69f1c60106def06f66e2d3c20551628e8a7478f47e2ba98b9a6015cd0e655
CLEAN=$RUN_ROOT/clean_targets.jsonl
GRAPH=$RUN_ROOT/graph.jsonl
FRONTIER=$RUN_ROOT/frontier.jsonl
RESULTS=$RUN_ROOT/factual.jsonl
LOG_DIR=$RUN_ROOT/logs
PID_DIR=$RUN_ROOT/pids

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$PID_DIR"
cd "$REPO"
export CAUSALITYRAG_READER_MAX_TOKENS=16

line_count() {
  if [[ -f "$1" ]]; then
    awk 'NF {n++} END {print n+0}' "$1"
  else
    echo 0
  fi
}

archive_partial() {
  local path=$1
  if [[ -e "$path" ]]; then
    mv "$path" "$path.partial.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
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
      if kill -0 -- "-$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || true
      fi
    fi
  fi
  rm -f "$pid_file"
}

stop_readers() {
  stop_owned_group "$PID_DIR/reader_8002.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_DIR/reader_8003.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_DIR/proxy_8000.pid" lb_proxy_two_gpu.py
}

cleanup() {
  stop_readers
}
trap cleanup EXIT

start_readers() {
  require_free_ports 8000 8002 8003
  local spec gpu port
  for spec in '0 8002' '1 8003'; do
    read -r gpu port <<<"$spec"
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
      -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name "$SERVED_MODEL" \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 --port "$port" \
      >"$LOG_DIR/vllm_${port}.log" 2>&1 </dev/null &
    echo $! >"$PID_DIR/reader_${port}.pid"
  done
  wait_endpoint http://127.0.0.1:8002/v1/models 900
  wait_endpoint http://127.0.0.1:8003/v1/models 900
  setsid "$PYTHON" "$REPO/scripts/lb_proxy_two_gpu.py" \
    >"$LOG_DIR/proxy_8000.log" 2>&1 </dev/null &
  echo $! >"$PID_DIR/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models 120
}

for required in "$INPUT" "$UNITS" "$POOL" "$OLD_RESULTS" "$MODEL/config.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "$POOL" | awk '{print $1}')" != "$POOL_SHA" ]]; then
  echo "frozen replacement pool SHA mismatch" >&2
  exit 1
fi

if [[ "$(line_count "$CLEAN")" != "$N" ]]; then
  archive_partial "$CLEAN"
  archive_partial "$RUN_ROOT/clean_targets.summary.json"
  start_readers
  echo "[pubmedqa-minimal] clean n=$N"
  CAUSALITYRAG_READER_MODE=pubmedqa "$PYTHON" scripts/generate_reader_targets.py \
    --input "$INPUT" --out "$CLEAN" \
    --summary-out "$RUN_ROOT/clean_targets.summary.json" \
    --n "$N" --k 5 --workers 48 \
    --base-url http://127.0.0.1:8000/v1 --served-model "$SERVED_MODEL" \
    >"$LOG_DIR/clean.log" 2>&1
  stop_readers
fi

if [[ "$(line_count "$GRAPH")" != "$N" ]]; then
  archive_partial "$GRAPH"
  archive_partial "$RUN_ROOT/graph.summary.json"
  echo "[pubmedqa-minimal] graph n=$N beam=$BEAM"
  CUDA_VISIBLE_DEVICES=0 CAUSALITYRAG_READER_MODE=pubmedqa \
    "$PYTHON" scripts/build_contribution_graph.py \
    --input "$INPUT" --out "$GRAPH" \
    --summary-out "$RUN_ROOT/graph.summary.json" \
    --model-path "$MODEL" --n "$N" --k 5 \
    --target results --target-results "$CLEAN" \
    --device cuda --dtype bfloat16 --max-receivers-per-layer "$BEAM" \
    >"$LOG_DIR/graph.log" 2>&1
fi

if [[ "$(line_count "$FRONTIER")" != "$N" ]]; then
  archive_partial "$FRONTIER"
  archive_partial "$RUN_ROOT/frontier.summary.json"
  echo "[pubmedqa-minimal] frontier n=$N"
  "$PYTHON" exp/run_reflow.py \
    --input "$INPUT" --graphs "$GRAPH" --units-cache "$UNITS" \
    --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
    --out "$FRONTIER" --summary-out "$RUN_ROOT/frontier.summary.json" \
    --n "$N" --k 5 >"$LOG_DIR/frontier.log" 2>&1
fi

if [[ "$(line_count "$RESULTS")" != "$N" ]]; then
  archive_partial "$RESULTS"
  archive_partial "$RUN_ROOT/factual.summary.json"
  start_readers
  echo "[pubmedqa-minimal] factual n=$N"
  CAUSALITYRAG_READER_MODE=pubmedqa "$PYTHON" scripts/evaluate_reflow.py \
    --input "$INPUT" --units-cache "$UNITS" --frontier "$FRONTIER" \
    --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
    --out "$RESULTS" --summary-out "$RUN_ROOT/factual.summary.json" \
    --n "$N" --k 5 --workers 48 --replacement-seed 0 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
    >"$LOG_DIR/factual.log" 2>&1
  stop_readers
fi

CAUSALITYRAG_READER_MODE=pubmedqa "$PYTHON" - \
  "$INPUT" "$CLEAN" "$GRAPH" "$FRONTIER" "$RESULTS" "$OLD_RESULTS" \
  "$RUN_ROOT/audit.json" "$POOL_SHA" <<'PY'
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from causalityrag.io import record_id
from causalityrag.reader import reader_completion_text, reader_prompt

input_path, clean_path, graph_path, frontier_path, results_path, old_path, audit_path, pool_sha = sys.argv[1:]
labels = {"yes", "no", "maybe"}

def rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

source = rows(input_path)[:50]
clean = rows(clean_path)
graphs = rows(graph_path)
frontiers = rows(frontier_path)
results = rows(results_path)
old = rows(old_path)[:50]
expected = [record_id(row) for row in source]
for name, values in {
    "clean": clean,
    "graphs": graphs,
    "frontiers": frontiers,
    "results": results,
    "old": old,
}.items():
    if len(values) != 50 or [str(row.get("id", "")) for row in values] != expected:
        raise SystemExit(f"{name} is not aligned to the first 50 frozen IDs")

invalid = []
for row in clean:
    if str(row.get("clean_answer", "")).casefold() not in labels:
        invalid.append((row.get("id"), "clean", row.get("clean_answer")))
for row in graphs:
    if str(row.get("target_answer", "")).casefold() not in labels:
        invalid.append((row.get("id"), "graph", row.get("target_answer")))
for row in results:
    for field in ("clean_answer", "gold_answer", "edited_answer"):
        if str(row.get(field, "")).casefold() not in labels:
            invalid.append((row.get("id"), field, row.get(field)))
if invalid:
    raise SystemExit(f"invalid labels: {invalid[:5]}")

system, user = reader_prompt(
    "Does the intervention help?",
    [{"chunk_id": "c0", "text": "Evidence."}],
    reader_mode="pubmedqa",
)
if "JSON" in system + user or reader_completion_text("yes", reader_mode="pubmedqa") != "yes":
    raise SystemExit("minimal PubMedQA prompt contract is not active")

def summarize(values):
    flips = [row for row in values if row.get("verified_flip")]
    clean_correct = [row for row in values if row.get("clean_correct")]
    return {
        "queries": len(values),
        "clean_correct": len(clean_correct),
        "verified_flips": len(flips),
        "answer_flip_rate": len(flips) / len(values),
        "clean_correct_flips": sum(bool(row.get("verified_flip")) for row in clean_correct),
        "clean_correct_flip_rate": sum(bool(row.get("verified_flip")) for row in clean_correct) / max(1, len(clean_correct)),
        "mean_modified_tokens_all": statistics.fmean(int(row.get("n_modified_tokens", 0)) for row in values),
        "mean_modified_tokens_flips": statistics.fmean(int(row.get("n_modified_tokens", 0)) for row in flips) if flips else 0.0,
        "reader_calls": sum(int(row.get("reader_calls", 0)) for row in values),
        "status_histogram": dict(Counter(str(row.get("evaluation_status", "")) for row in values)),
    }

audit = {
    "schema": "causalityrag.pubmedqa_minimal_prompt_diagnostic.v1",
    "protocol": {
        "system": system,
        "user_suffix": "Answer with exactly one label: yes, no, or maybe. Do not explain.",
        "teacher_forced_completion": "plain canonical label",
        "guided_decoding": False,
    },
    "minimal_prompt": summarize(results),
    "old_mixed_protocol": summarize(old),
    "clean_label_agreement_with_old": sum(
        str(left.get("clean_answer", "")).casefold()
        == str(right.get("clean_answer", "")).casefold()
        for left, right in zip(results, old)
    ),
    "clean_label_histogram": dict(Counter(str(row.get("clean_answer", "")).casefold() for row in results)),
    "edited_label_histogram": dict(Counter(str(row.get("edited_answer", "")).casefold() for row in results)),
    "graph_status_histogram": dict(Counter(str(row.get("status", "")) for row in graphs)),
    "frontier_status_histogram": dict(Counter(str(row.get("frontier_status", "")) for row in frontiers)),
    "queries_with_frontier": sum(bool(row.get("frontier_candidates")) for row in frontiers),
    "mean_frontier_size": statistics.fmean(int(row.get("n_frontier", 0)) for row in frontiers),
    "sha256": {
        "clean": sha(clean_path),
        "graph": sha(graph_path),
        "frontier": sha(frontier_path),
        "factual": sha(results_path),
        "shared_pool": pool_sha,
    },
}
Path(audit_path).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2))
PY

date -Is >"$RUN_ROOT/COMPLETE"
echo "[pubmedqa-minimal] complete root=$RUN_ROOT"
