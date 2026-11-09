#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-7B receiver-beam sensitivity at the frozen default retrieval depth k=5.

REPO=${REPO:-/data1/yujia/CausalityRAG_release_retrieval_depth_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL_PATH=${MODEL_PATH:-/data1/yujia/models/Qwen2.5-7B-Instruct}
SERVED_MODEL=${SERVED_MODEL:-qwen2.5-7b}
OUT_ROOT=${OUT_ROOT:-/data1/yujia/CausalityRAG/out/receiver_beam_qwen25_v1}
N=${N:-1000}
K=5
SPACY_PID=${SPACY_PID:-1483747}
BEAMS=(16 32 64)
DATASETS=(hotpotqa timeqa finqa musique quartz triviaqa 2wiki popqa)
LOG_ROOT=$OUT_ROOT/logs

mkdir -p "$LOG_ROOT" "$OUT_ROOT/pids" "$OUT_ROOT/shared"
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
  case "$1" in
    quartz) echo /data1/yujia/CausalityRAG/out_label_protocol_20260805/quartz ;;
    popqa) echo /data1/yujia/CausalityRAG/out_popqa_20260810/popqa ;;
    *) echo "/data1/yujia/CausalityRAG/out/$1" ;;
  esac
}

retrieval_source_root() {
  if [[ "$1" == popqa ]]; then
    echo /data1/yujia/CausalityRAG/out_popqa_20260810/popqa
  else
    echo "/data1/yujia/CausalityRAG/out/$1"
  fi
}

source_pool() {
  local root
  root=$(retrieval_source_root "$1")
  case "$1" in
    hotpotqa|triviaqa)
      echo "$root/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl"
      ;;
    *)
      echo "$root/replacements/shared_pool_top10_v1/shared_pool.jsonl"
      ;;
  esac
}

frozen_result() {
  case "$1" in
    hotpotqa)
      echo /data1/yujia/CausalityRAG/out/hotpotqa/audits/final_top10pool_k5/reflow_1000_v2.jsonl
      ;;
    popqa)
      echo /data1/yujia/CausalityRAG/out_popqa_figures_20260811/popqa/protocol/exact_1000.jsonl
      ;;
    *)
      echo "$(source_root "$1")/methods/reflow/results_top5_1000.jsonl"
      ;;
  esac
}

frozen_graph() {
  case "$1" in
    hotpotqa)
      echo /data1/yujia/CausalityRAG/out/hotpotqa/graphs/contribution_graph_closed_beam48_1000.jsonl
      ;;
    *) echo "$(source_root "$1")/graphs/contribution_graph_top5_1000.jsonl" ;;
  esac
}

frozen_frontier() {
  case "$1" in
    hotpotqa)
      echo /data1/yujia/CausalityRAG/out/hotpotqa/audits/final_top10pool_k5/reflow_frontier_1000.jsonl
      ;;
    *) echo "$(source_root "$1")/methods/reflow/frontier_top5_1000.jsonl" ;;
  esac
}

reader_mode() {
  case "$1" in
    quartz) echo quartz ;;
    *) echo short_answer ;;
  esac
}

beam_root() {
  echo "$OUT_ROOT/d$1/$2"
}

shared_root() {
  echo "$OUT_ROOT/shared/$1"
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

copy_frozen() {
  local source=$1
  local destination=$2
  [[ -s "$source" ]] || {
    echo "missing frozen source: $source" >&2
    exit 1
  }
  mkdir -p "$(dirname "$destination")"
  if [[ -e "$destination" ]]; then
    [[ "$(sha256sum "$source" | awk '{print $1}')" == \
       "$(sha256sum "$destination" | awk '{print $1}')" ]] || {
      echo "frozen destination SHA mismatch: $destination" >&2
      exit 1
    }
  else
    cp "$source" "$destination"
  fi
}

materialize_frozen_inputs() {
  for dataset in "${DATASETS[@]}"; do
    local source shared input units pool clean graph frontier
    source=$(retrieval_source_root "$dataset")
    shared=$(shared_root "$dataset")
    input="$source/retrieval/top10_1000.jsonl"
    units="$source/inputs/token_units_top10_1000.jsonl"
    pool=$(source_pool "$dataset")
    clean=$(frozen_result "$dataset")
    graph=$(frozen_graph "$dataset")
    frontier=$(frozen_frontier "$dataset")
    complete_jsonl "$input" "$N"
    complete_jsonl "$units" "$N"
    complete_jsonl "$clean" "$N"
    complete_jsonl "$graph" "$N"
    complete_jsonl "$frontier" "$N"
    copy_frozen "$input" "$shared/retrieval.jsonl"
    copy_frozen "$units" "$shared/units.jsonl"
    copy_frozen "$pool" "$shared/shared_pool.jsonl"
    copy_frozen "$clean" "$shared/clean_targets.jsonl"
    "$PYTHON" - "$shared/manifest.json" "$dataset" "$input" "$units" \
      "$pool" "$clean" "$graph" "$frontier" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, dataset, *paths = sys.argv[1:]
labels = ("retrieval", "units", "pool", "clean", "default_d48_graph", "default_d48_frontier")
manifest = {"dataset": dataset, "retrieval_depth": 5, "default_receiver_beam": 48}
for label, raw_path in zip(labels, paths):
    path = Path(raw_path)
    manifest[label] = {
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
Path(out).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
  done
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
    "causalityrag/message_flow.py",
    "exp/run_reflow.py",
    "scripts/build_contribution_graph_beams.py",
    "scripts/evaluate_reflow.py",
    "scripts/launchers/run_receiver_beam_qwen25_server3.sh",
]
repo_path = Path(repo)
hashes = {
    relative: hashlib.sha256((repo_path / relative).read_bytes()).hexdigest()
    for relative in critical
}
try:
    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    commit = "unknown"
value = {
    "schema": "causalityrag.receiver_beam_sensitivity.v1",
    "server": "3haoji-A100",
    "model": "Qwen2.5-7B-Instruct",
    "model_path": str(Path(model).resolve()),
    "queries_per_dataset": int(n),
    "datasets": [
        "hotpotqa", "timeqa", "finqa", "musique",
        "quartz", "triviaqa", "2wiki", "popqa",
    ],
    "retrieval_depth": 5,
    "receiver_beams": [16, 32, 48, 64],
    "generated_receiver_beams": [16, 32, 64],
    "reused_default_receiver_beam": 48,
    "new_graph_passes_per_query": 1,
    "multi_beam_shared_trace": True,
    "replacement_seed": 0,
    "code_commit": commit,
    "critical_file_sha256": hashes,
}
Path(out).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
PY
}

run_graph_and_frontier() {
  local gpu=$1
  shift
  for dataset in "$@"; do
    local shared graph_template summary_template
    shared=$(shared_root "$dataset")
    graph_template="$OUT_ROOT/d{beam}/$dataset/graph/contribution_graph.jsonl"
    summary_template="$OUT_ROOT/d{beam}/$dataset/graph/summary.json"
    local need_graph=0
    for beam in "${BEAMS[@]}"; do
      local base graph
      base=$(beam_root "$beam" "$dataset")
      graph="$base/graph/contribution_graph.jsonl"
      mkdir -p "$base/graph" "$base/frontier" "$base/factual" "$base/final"
      complete_jsonl "$graph" "$N" || need_graph=1
    done
    if (( need_graph )); then
      CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
        scripts/build_contribution_graph_beams.py \
        --input "$shared/retrieval.jsonl" --out-template "$graph_template" \
        --summary-template "$summary_template" --model-path "$MODEL_PATH" \
        --n "$N" --k "$K" --target-results "$shared/clean_targets.jsonl" \
        --receiver-beams "${BEAMS[@]}" --device cuda --dtype bfloat16 --resume \
        --allow-unusable \
        >"$LOG_ROOT/graph_${dataset}.log" 2>&1
    fi
    for beam in "${BEAMS[@]}"; do
      local base graph frontier pool pool_sha
      base=$(beam_root "$beam" "$dataset")
      graph="$base/graph/contribution_graph.jsonl"
      frontier="$base/frontier/initial.jsonl"
      pool="$shared/shared_pool.jsonl"
      pool_sha=$(sha256sum "$pool" | awk '{print $1}')
      complete_jsonl "$graph" "$N" || {
        echo "incomplete graph: dataset=$dataset d=$beam" >&2
        exit 1
      }
      if ! complete_jsonl "$frontier" "$N"; then
        "$PYTHON" exp/run_reflow.py \
          --input "$shared/retrieval.jsonl" --graphs "$graph" \
          --units-cache "$shared/units.jsonl" --shared-pool "$pool" \
          --expected-pool-sha256 "$pool_sha" --out "$frontier" \
          --summary-out "$base/frontier/initial.summary.json" \
          --n "$N" --k "$K" --record-empty-frontier \
          >"$LOG_ROOT/frontier_${dataset}_d${beam}.log" 2>&1
      fi
      complete_jsonl "$frontier" "$N" || {
        echo "incomplete frontier: dataset=$dataset d=$beam" >&2
        exit 1
      }
    done
  done
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

repair_no_frontier() {
  local dataset=$1
  local beam=$2
  local base shared pool pool_sha initial graph no_frontier targets excluded
  base=$(beam_root "$beam" "$dataset")
  shared=$(shared_root "$dataset")
  pool="$shared/shared_pool.jsonl"
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
  echo "[receiver-beam] dataset=$dataset d=$beam no_frontier=$no_frontier"
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
    >"$LOG_ROOT/repair_graph_${dataset}_d${beam}.log" 2>&1
  "$PYTHON" scripts/prepare_no_frontier_resume.py prepare \
    --input "$shared/retrieval.jsonl" --units-cache "$shared/units.jsonl" \
    --repaired-graphs "$base/final/graphs.jsonl" \
    --old-frontier "$base/frontier/initial.jsonl" --old-results "$initial" \
    --out-dir "$base/final/resume" --n "$N" \
    >"$LOG_ROOT/repair_prepare_${dataset}_d${beam}.log" 2>&1
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
  if (( targets > 0 )); then
    "$PYTHON" exp/run_reflow.py \
      --input "$base/final/resume/input.jsonl" \
      --graphs "$base/final/resume/graphs.jsonl" \
      --units-cache "$base/final/resume/units.jsonl" \
      --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
      --out "$base/final/resume/frontier.jsonl" \
      --summary-out "$base/final/resume/frontier.summary.json" \
      --n "$targets" --k "$K" \
      >"$LOG_ROOT/repair_frontier_${dataset}_d${beam}.log" 2>&1
    CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
      "$PYTHON" scripts/evaluate_reflow.py \
      --input "$base/final/resume/input.jsonl" \
      --units-cache "$base/final/resume/units.jsonl" \
      --frontier "$base/final/resume/frontier.jsonl" \
      --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
      --out "$base/final/resume/results.jsonl" \
      --summary-out "$base/final/resume/results.summary.json" \
      --n "$targets" --k "$K" --workers 48 --replacement-seed 0 \
      --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
      >"$LOG_ROOT/repair_reader_${dataset}_d${beam}.log" 2>&1
  fi
  "$PYTHON" scripts/prepare_no_frontier_resume.py merge-frontier \
    --manifest "$base/final/resume/manifest.json" \
    --old-frontier "$base/frontier/initial.jsonl" \
    --resumed-frontier "$base/final/resume/frontier.jsonl" \
    --out "$base/final/frontier.jsonl" \
    --summary-out "$base/final/frontier.summary.json" \
    >"$LOG_ROOT/repair_merge_frontier_${dataset}_d${beam}.log" 2>&1
  "$PYTHON" scripts/prepare_no_frontier_resume.py merge-results \
    --manifest "$base/final/resume/manifest.json" --old-results "$initial" \
    --resumed-results "$base/final/resume/results.jsonl" \
    --out "$base/factual/results.jsonl" \
    --summary-out "$base/factual/summary.json" \
    >"$LOG_ROOT/repair_merge_results_${dataset}_d${beam}.log" 2>&1
}

run_factual_evaluations() {
  for beam in "${BEAMS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      local base shared pool pool_sha initial
      base=$(beam_root "$beam" "$dataset")
      shared=$(shared_root "$dataset")
      pool="$shared/shared_pool.jsonl"
      pool_sha=$(sha256sum "$pool" | awk '{print $1}')
      initial="$base/factual/initial_results.jsonl"
      if ! complete_jsonl "$initial" "$N"; then
        CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
          "$PYTHON" scripts/evaluate_reflow.py \
          --input "$shared/retrieval.jsonl" --units-cache "$shared/units.jsonl" \
          --frontier "$base/frontier/initial.jsonl" --shared-pool "$pool" \
          --expected-pool-sha256 "$pool_sha" --out "$initial" \
          --summary-out "$base/factual/initial_results.summary.json" \
          --n "$N" --k "$K" --workers 48 --replacement-seed 0 \
          --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
          >"$LOG_ROOT/factual_${dataset}_d${beam}.log" 2>&1
      fi
      complete_jsonl "$initial" "$N" || {
        echo "incomplete factual results: dataset=$dataset d=$beam" >&2
        exit 1
      }
      if ! complete_jsonl "$base/factual/results.jsonl" "$N"; then
        repair_no_frontier "$dataset" "$beam"
      fi
      complete_jsonl "$base/factual/results.jsonl" "$N"
      complete_jsonl "$base/final/frontier.jsonl" "$N"
    done
  done
}

check_protected_spacy
[[ -s "$MODEL_PATH/config.json" ]]
write_run_manifest
materialize_frozen_inputs
trap stop_readers EXIT

echo "[receiver-beam] stage=graph-frontier beams=${BEAMS[*]}"
run_graph_and_frontier 0 hotpotqa finqa quartz 2wiki &
left_pid=$!
run_graph_and_frontier 1 timeqa musique triviaqa popqa &
right_pid=$!
wait "$left_pid"
wait "$right_pid"

echo "[receiver-beam] stage=reader-verification"
start_readers
run_factual_evaluations
stop_readers
check_protected_spacy
date -Is >"$OUT_ROOT/COMPLETE"
echo "[receiver-beam] complete root=$OUT_ROOT"
