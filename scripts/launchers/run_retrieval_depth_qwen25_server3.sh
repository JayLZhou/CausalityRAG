#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-7B retrieval-depth sensitivity on nested top-10 prefixes.

REPO=${REPO:-/data1/yujia/CausalityRAG_release_retrieval_depth_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL_PATH=${MODEL_PATH:-/data1/yujia/models/Qwen2.5-7B-Instruct}
SERVED_MODEL=${SERVED_MODEL:-qwen2.5-7b}
OUT_ROOT=${OUT_ROOT:-/data1/yujia/CausalityRAG/out/retrieval_depth_qwen25_v1}
N=${N:-1000}
SPACY_PID=${SPACY_PID:-1483747}
read -r -a KS <<< "${RETRIEVAL_DEPTH_KS:-1 3 5 10}"
DATASETS=(hotpotqa timeqa finqa musique quartz triviaqa 2wiki popqa)
LOG_ROOT=$OUT_ROOT/logs

mkdir -p "$LOG_ROOT" "$OUT_ROOT/pids"
cd "$REPO"
export CAUSALITYRAG_READER_MAX_TOKENS=${CAUSALITYRAG_READER_MAX_TOKENS:-128}

line_count() {
  if [[ ! -f "$1" ]]; then
    echo 0
  else
    awk 'NF {count++} END {print count+0}' "$1"
  fi
}

complete_jsonl() {
  [[ "$(line_count "$1")" == "$2" ]]
}

source_root() {
  if [[ "$1" == popqa ]]; then
    echo /data1/yujia/CausalityRAG/out_popqa_20260810/popqa
  else
    echo "/data1/yujia/CausalityRAG/out/$1"
  fi
}

source_pool() {
  local root
  root=$(source_root "$1")
  case "$1" in
    hotpotqa|triviaqa)
      echo "$root/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl"
      ;;
    *)
      echo "$root/replacements/shared_pool_top10_v1/shared_pool.jsonl"
      ;;
  esac
}

reader_mode() {
  case "$1" in
    quartz) echo quartz ;;
    *) echo short_answer ;;
  esac
}

prefix_root() {
  echo "$OUT_ROOT/k$1/$2"
}

check_protected_spacy() {
  local cmd
  [[ -r "/proc/$SPACY_PID/cmdline" ]] || {
    echo "protected spaCy PID is not running: $SPACY_PID" >&2
    exit 1
  }
  cmd=$(tr '\0' ' ' <"/proc/$SPACY_PID/cmdline")
  [[ "$cmd" == *"spacy_annotation_server.py --port 8021"* ]] || {
    echo "PID $SPACY_PID is not the protected spaCy service" >&2
    exit 1
  }
  fuser 8021/tcp 2>/dev/null | grep -qw "$SPACY_PID" || {
    echo "protected spaCy service is not listening on 8021" >&2
    exit 1
  }
}

stop_owned_group() {
  local pid_file=$1
  local expected=$2
  [[ -f "$pid_file" ]] || return 0
  local pid cmd
  pid=$(cat "$pid_file")
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || [[ ! -r "/proc/$pid/cmdline" ]]; then
    rm -f "$pid_file"
    return 0
  fi
  cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
  [[ "$cmd" == *"$expected"* ]] || {
    echo "refusing to stop unowned PID $pid from $pid_file" >&2
    exit 1
  }
  kill -- "-$pid" 2>/dev/null || true
  local deadline=$((SECONDS + 30))
  while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

stop_readers() {
  stop_owned_group "$OUT_ROOT/pids/reader_8002.pid" "vllm.entrypoints.openai.api_server"
  stop_owned_group "$OUT_ROOT/pids/reader_8003.pid" "vllm.entrypoints.openai.api_server"
  stop_owned_group "$OUT_ROOT/pids/proxy_8000.pid" "lb_proxy_two_gpu.py"
}

require_port_free() {
  if fuser "$1/tcp" >/dev/null 2>&1; then
    echo "port $1 is occupied; refusing to interrupt its service" >&2
    exit 1
  fi
}

wait_endpoint() {
  local url=$1
  local deadline=$((SECONDS + 900))
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "endpoint did not become ready: $url" >&2
      exit 1
    fi
    sleep 5
  done
}

start_readers() {
  stop_readers
  require_port_free 8000
  require_port_free 8002
  require_port_free 8003
  for spec in '0 8002' '1 8003'; do
    read -r gpu port <<<"$spec"
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
      -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 --port "$port" \
      >"$LOG_ROOT/vllm_${port}.log" 2>&1 </dev/null &
    echo $! >"$OUT_ROOT/pids/reader_${port}.pid"
  done
  wait_endpoint http://127.0.0.1:8002/v1/models
  wait_endpoint http://127.0.0.1:8003/v1/models
  setsid "$PYTHON" scripts/lb_proxy_two_gpu.py \
    >"$LOG_ROOT/proxy_8000.log" 2>&1 </dev/null &
  echo $! >"$OUT_ROOT/pids/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models
}

write_run_manifest() {
  "$PYTHON" - "$OUT_ROOT/manifest.json" "$REPO" "$MODEL_PATH" "$N" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

out, repo, model, n = sys.argv[1:]
critical = [
    "causalityrag/contribution_graph.py",
    "causalityrag/io.py",
    "causalityrag/reader.py",
    "causalityrag/token_units.py",
    "exp/run_reflow.py",
    "scripts/build_contribution_graph.py",
    "scripts/evaluate_reflow.py",
    "scripts/generate_reader_targets.py",
    "scripts/materialize_retrieval_prefix.py",
    "scripts/prepare_no_frontier_resume.py",
    "scripts/repair_disconnected_contribution_graphs.py",
    "scripts/launchers/run_retrieval_depth_qwen25_server3.sh",
]
repo_path = Path(repo)
hashes = {}
for relative in critical:
    path = repo_path / relative
    hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
try:
    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    commit = "unknown"
value = {
    "schema": "causalityrag.retrieval_depth_sensitivity.v1",
    "server": "3haoji-A100",
    "model": "Qwen2.5-7B-Instruct",
    "model_path": str(Path(model).resolve()),
    "queries_per_dataset": int(n),
    "datasets": [
        "hotpotqa", "timeqa", "finqa", "musique",
        "quartz", "triviaqa", "2wiki", "popqa",
    ],
    "retrieval_depths": [1, 3, 5, 10],
    "nested_source_depth": 10,
    "receiver_beam": 48,
    "replacement_seed": 0,
    "reader_decoding": "temperature_0_no_guided_decoding",
    "quartz_protocol": "prompt_only_choice_text_with_strict_postprocessing",
    "code_commit": commit,
    "critical_file_sha256": hashes,
}
path = Path(out)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
PY
}

materialize_prefixes() {
  for k in "${KS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      local source base input units pool pool_sha
      source=$(source_root "$dataset")
      base=$(prefix_root "$k" "$dataset")
      input="$source/retrieval/top10_1000.jsonl"
      units="$source/inputs/token_units_top10_1000.jsonl"
      pool=$(source_pool "$dataset")
      pool_sha=$(sha256sum "$pool" | awk '{print $1}')
      mkdir -p "$base/inputs"
      "$PYTHON" scripts/materialize_retrieval_prefix.py \
        --input "$input" --units-cache "$units" --shared-pool "$pool" \
        --expected-source-pool-sha256 "$pool_sha" \
        --out-dir "$base/inputs" --n "$N" --k "$k" --reuse-existing \
        >"$LOG_ROOT/materialize_${dataset}_k${k}.log" 2>&1
      complete_jsonl "$base/inputs/retrieval.jsonl" "$N"
      complete_jsonl "$base/inputs/units.jsonl" "$N"
      [[ -s "$base/inputs/shared_pool.jsonl" ]]
      [[ -s "$base/inputs/manifest.json" ]]
    done
  done
}

generate_clean_targets() {
  for k in "${KS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      local base out
      base=$(prefix_root "$k" "$dataset")
      out="$base/clean/targets.jsonl"
      mkdir -p "$base/clean"
      if complete_jsonl "$out" "$N"; then
        echo "[retrieval-depth] reuse clean dataset=$dataset k=$k"
        continue
      fi
      CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
        "$PYTHON" scripts/generate_reader_targets.py \
        --input "$base/inputs/retrieval.jsonl" --out "$out" \
        --summary-out "$base/clean/summary.json" --n "$N" --k "$k" \
        --workers 48 --base-url http://127.0.0.1:8000/v1 \
        --served-model "$SERVED_MODEL" \
        >"$LOG_ROOT/clean_${dataset}_k${k}.log" 2>&1
      complete_jsonl "$out" "$N" || {
        echo "incomplete clean targets: dataset=$dataset k=$k" >&2
        exit 1
      }
    done
  done
}

run_graph_and_frontier() {
  local gpu=$1
  local k=$2
  shift 2
  for dataset in "$@"; do
    local base graph frontier pool pool_sha
    base=$(prefix_root "$k" "$dataset")
    graph="$base/graph/contribution_graph.jsonl"
    frontier="$base/frontier/initial.jsonl"
    pool="$base/inputs/shared_pool.jsonl"
    pool_sha=$(sha256sum "$pool" | awk '{print $1}')
    mkdir -p "$base/graph" "$base/frontier" "$base/final" "$base/factual"
    if ! complete_jsonl "$graph" "$N"; then
      CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
        scripts/build_contribution_graph.py \
        --input "$base/inputs/retrieval.jsonl" --out "$graph" \
        --summary-out "$base/graph/summary.json" --model-path "$MODEL_PATH" \
        --n "$N" --k "$k" --target results \
        --target-results "$base/clean/targets.jsonl" \
        --device cuda --dtype bfloat16 --max-receivers-per-layer 48 \
        >"$LOG_ROOT/graph_${dataset}_k${k}.log" 2>&1
    fi
    complete_jsonl "$graph" "$N" || {
      echo "incomplete graph: dataset=$dataset k=$k" >&2
      exit 1
    }
    if ! complete_jsonl "$frontier" "$N"; then
      "$PYTHON" exp/run_reflow.py \
        --input "$base/inputs/retrieval.jsonl" --graphs "$graph" \
        --units-cache "$base/inputs/units.jsonl" --shared-pool "$pool" \
        --expected-pool-sha256 "$pool_sha" --out "$frontier" \
        --summary-out "$base/frontier/initial.summary.json" \
        --n "$N" --k "$k" --record-empty-frontier \
        >"$LOG_ROOT/frontier_${dataset}_k${k}.log" 2>&1
    fi
    complete_jsonl "$frontier" "$N" || {
      echo "incomplete frontier: dataset=$dataset k=$k" >&2
      exit 1
    }
  done
}

repair_no_frontier() {
  local dataset=$1
  local k=$2
  local base pool pool_sha initial graph no_frontier targets excluded
  base=$(prefix_root "$k" "$dataset")
  pool="$base/inputs/shared_pool.jsonl"
  pool_sha=$(sha256sum "$pool" | awk '{print $1}')
  initial="$base/factual/initial_results.jsonl"
  graph="$base/graph/contribution_graph.jsonl"
  no_frontier=$("$PYTHON" - "$initial" <<'PY'
import json
import sys
print(sum(
    json.loads(line).get("evaluation_status") == "no_frontier_candidate"
    for line in open(sys.argv[1], encoding="utf-8") if line.strip()
))
PY
)
  echo "[retrieval-depth] dataset=$dataset k=$k no_frontier=$no_frontier"
  if (( no_frontier == 0 )); then
    cp "$graph" "$base/final/graphs.jsonl"
    cp "$base/graph/summary.json" "$base/final/graphs.summary.json"
    cp "$base/frontier/initial.jsonl" "$base/final/frontier.jsonl"
    cp "$base/frontier/initial.summary.json" "$base/final/frontier.summary.json"
    cp "$initial" "$base/factual/results.jsonl"
    cp "$base/factual/initial_results.summary.json" "$base/factual/summary.json"
    return
  fi
  "$PYTHON" scripts/repair_disconnected_contribution_graphs.py \
    --graphs "$graph" --shared-pool "$pool" --target-results "$initial" \
    --expected-pool-sha256 "$pool_sha" --out "$base/final/graphs.jsonl" \
    --summary-out "$base/final/graphs.summary.json" --n "$N" \
    >"$LOG_ROOT/repair_graph_${dataset}_k${k}.log" 2>&1
  "$PYTHON" scripts/prepare_no_frontier_resume.py prepare \
    --input "$base/inputs/retrieval.jsonl" \
    --units-cache "$base/inputs/units.jsonl" \
    --repaired-graphs "$base/final/graphs.jsonl" \
    --old-frontier "$base/frontier/initial.jsonl" \
    --old-results "$initial" --out-dir "$base/final/resume" --n "$N" \
    >"$LOG_ROOT/repair_prepare_${dataset}_k${k}.log" 2>&1
  targets=$(line_count "$base/final/resume/input.jsonl")
  excluded=$("$PYTHON" - "$base/final/resume/manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get(
    "excluded_no_executable_queries", 0
))
PY
)
  (( targets + excluded == no_frontier ))
  "$PYTHON" exp/run_reflow.py \
    --input "$base/final/resume/input.jsonl" \
    --graphs "$base/final/resume/graphs.jsonl" \
    --units-cache "$base/final/resume/units.jsonl" \
    --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
    --out "$base/final/resume/frontier.jsonl" \
    --summary-out "$base/final/resume/frontier.summary.json" \
    --n "$targets" --k "$k" \
    >"$LOG_ROOT/repair_frontier_${dataset}_k${k}.log" 2>&1
  CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
    "$PYTHON" scripts/evaluate_reflow.py \
    --input "$base/final/resume/input.jsonl" \
    --units-cache "$base/final/resume/units.jsonl" \
    --frontier "$base/final/resume/frontier.jsonl" \
    --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
    --out "$base/final/resume/results.jsonl" \
    --summary-out "$base/final/resume/results.summary.json" \
    --n "$targets" --k "$k" --workers 48 --replacement-seed 0 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
    >"$LOG_ROOT/repair_reader_${dataset}_k${k}.log" 2>&1
  "$PYTHON" scripts/prepare_no_frontier_resume.py merge-frontier \
    --manifest "$base/final/resume/manifest.json" \
    --old-frontier "$base/frontier/initial.jsonl" \
    --resumed-frontier "$base/final/resume/frontier.jsonl" \
    --out "$base/final/frontier.jsonl" \
    --summary-out "$base/final/frontier.summary.json" \
    >"$LOG_ROOT/repair_merge_frontier_${dataset}_k${k}.log" 2>&1
  "$PYTHON" scripts/prepare_no_frontier_resume.py merge-results \
    --manifest "$base/final/resume/manifest.json" \
    --old-results "$initial" \
    --resumed-results "$base/final/resume/results.jsonl" \
    --out "$base/factual/results.jsonl" \
    --summary-out "$base/factual/summary.json" \
    >"$LOG_ROOT/repair_merge_results_${dataset}_k${k}.log" 2>&1
}

run_factual_evaluations() {
  for k in "${KS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      local base pool pool_sha initial
      base=$(prefix_root "$k" "$dataset")
      pool="$base/inputs/shared_pool.jsonl"
      pool_sha=$(sha256sum "$pool" | awk '{print $1}')
      initial="$base/factual/initial_results.jsonl"
      if ! complete_jsonl "$initial" "$N"; then
        CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
          "$PYTHON" scripts/evaluate_reflow.py \
          --input "$base/inputs/retrieval.jsonl" \
          --units-cache "$base/inputs/units.jsonl" \
          --frontier "$base/frontier/initial.jsonl" \
          --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
          --out "$initial" \
          --summary-out "$base/factual/initial_results.summary.json" \
          --n "$N" --k "$k" --workers 48 --replacement-seed 0 \
          --llm-base-url http://127.0.0.1:8000/v1 \
          --llm-model "$SERVED_MODEL" \
          >"$LOG_ROOT/factual_${dataset}_k${k}.log" 2>&1
      fi
      complete_jsonl "$initial" "$N" || {
        echo "incomplete factual results: dataset=$dataset k=$k" >&2
        exit 1
      }
      if ! complete_jsonl "$base/factual/results.jsonl" "$N"; then
        repair_no_frontier "$dataset" "$k"
      fi
      complete_jsonl "$base/factual/results.jsonl" "$N"
      complete_jsonl "$base/final/frontier.jsonl" "$N"
    done
  done
}

check_protected_spacy
[[ -s "$MODEL_PATH/config.json" ]]
write_run_manifest
materialize_prefixes
trap stop_readers EXIT

echo "[retrieval-depth] stage=clean"
start_readers
generate_clean_targets
stop_readers

echo "[retrieval-depth] stage=graph-frontier"
for k in "${KS[@]}"; do
  echo "[retrieval-depth] graph k=$k"
  run_graph_and_frontier 0 "$k" hotpotqa finqa quartz 2wiki &
  left_pid=$!
  run_graph_and_frontier 1 "$k" timeqa musique triviaqa popqa &
  right_pid=$!
  wait "$left_pid"
  wait "$right_pid"
done

echo "[retrieval-depth] stage=reader-verification"
start_readers
run_factual_evaluations
stop_readers
check_protected_spacy
date -Is >"$OUT_ROOT/COMPLETE"
echo "[retrieval-depth] complete root=$OUT_ROOT"
