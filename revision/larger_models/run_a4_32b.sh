#!/usr/bin/env bash
set -euo pipefail

# A4: Qwen2.5-32B-Instruct, adapted from run_cross_model_reflow_server3.sh (model sweep).
# One tensor-parallel reader on both GPUs; graphs built with device_map=balanced.
# Run one ReFlow backbone on server 3. Retrieval and both replacement pools
# are immutable inputs; only clean answers, contribution graphs, frontiers,
# and factual/control reader executions are model-specific.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 MODEL_KEY MODEL_PATH SERVED_MODEL [N]" >&2
  exit 2
fi

MODEL_KEY=$1
MODEL_PATH=$2
SERVED_MODEL=$3
N=${4:-1000}

REPO=${REPO:-/data1/yujia/CausalityRAG_release_model_sweep_20260807}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
FROZEN_ROOT=${FROZEN_ROOT:-/data1/yujia/CausalityRAG/out}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a4_32b/out}
RUN_ROOT="$OUTPUT_ROOT/$MODEL_KEY"
if [[ "$N" != "1000" ]]; then
  RUN_ROOT="$OUTPUT_ROOT/smoke_${N}/$MODEL_KEY"
fi
LOG_ROOT="$RUN_ROOT/logs"
export CAUSALITYRAG_READER_MAX_TOKENS=${CAUSALITYRAG_READER_MAX_TOKENS:-128}
DATASETS=(hotpotqa finqa popqa)

mkdir -p "$LOG_ROOT"
cd "$REPO"

line_count() {
  if [[ ! -f "$1" ]]; then
    echo 0
  else
    awk 'NF {count++} END {print count+0}' "$1"
  fi
}

complete_jsonl() {
  [[ "$(line_count "$1")" == "$N" ]]
}

input_path() {
  echo "$FROZEN_ROOT/$1/retrieval/top10_1000.jsonl"
}

units_path() {
  case "$1" in
    hotpotqa) echo "$FROZEN_ROOT/$1/inputs/token_units_top5_1000.jsonl" ;;
    triviaqa) echo "$FROZEN_ROOT/$1/inputs/context_units_top5_1000.jsonl" ;;
    *) echo "$FROZEN_ROOT/$1/analysis_v1/inputs/token_units_top5_1000.jsonl" ;;
  esac
}

factual_pool_path() {
  case "$1" in
    hotpotqa|triviaqa)
      echo "$FROZEN_ROOT/$1/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl"
      ;;
    *)
      echo "$FROZEN_ROOT/$1/replacements/shared_pool_top10_v1/shared_pool.jsonl"
      ;;
  esac
}

paraphrase_pool_path() {
  case "$1" in
    triviaqa) echo "$FROZEN_ROOT/$1/controls/paraphrase_pool_top5_v1.jsonl" ;;
    *) echo "$FROZEN_ROOT/$1/controls/paraphrase_pool_top5_v2.jsonl" ;;
  esac
}

# PopQA lives in the Table 3 directory; its top-5 unit file is derived from the
# top-10 cache (a4_32b/inputs). Other datasets keep the sweep's paths.
POPQA=/data1/yujia/CausalityRAG/out_table3_final8_20260811/popqa
eval "orig_$(declare -f input_path)"
eval "orig_$(declare -f units_path)"
eval "orig_$(declare -f factual_pool_path)"
eval "orig_$(declare -f paraphrase_pool_path)"
input_path() { if [[ $1 == popqa ]]; then echo "$POPQA/retrieval/top10_1000.jsonl"; else orig_input_path "$1"; fi; }
units_path() { if [[ $1 == popqa ]]; then echo /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a4_32b/inputs/popqa_token_units_top5_1000.jsonl; else orig_units_path "$1"; fi; }
factual_pool_path() { if [[ $1 == popqa ]]; then echo "$POPQA/replacements/shared_pool_top10_v1/shared_pool.jsonl"; else orig_factual_pool_path "$1"; fi; }
paraphrase_pool_path() { if [[ $1 == popqa ]]; then echo "$POPQA/controls/paraphrase_pool_top5_v2.jsonl"; else orig_paraphrase_pool_path "$1"; fi; }

reader_mode() {
  case "$1" in
    pubmedqa|quartz) echo "$1" ;;
    *) echo short_answer ;;
  esac
}

validate_frozen_inputs() {
  local dataset=$1
  local input units factual paraphrase
  input=$(input_path "$dataset")
  units=$(units_path "$dataset")
  factual=$(factual_pool_path "$dataset")
  paraphrase=$(paraphrase_pool_path "$dataset")
  [[ $(line_count "$input") -ge "$N" ]] || {
    echo "$dataset: retrieval has fewer than $N rows" >&2; exit 1;
  }
  [[ $(line_count "$units") -ge "$N" ]] || {
    echo "$dataset: token units have fewer than $N rows" >&2; exit 1;
  }
  [[ -s "$factual" ]] || { echo "$dataset: missing factual pool" >&2; exit 1; }
  [[ -s "$paraphrase" ]] || { echo "$dataset: missing paraphrase pool" >&2; exit 1; }
}

for dataset in "${DATASETS[@]}"; do
  validate_frozen_inputs "$dataset"
done

if [[ ! -s "$MODEL_PATH/config.json" ]]; then
  echo "model config is missing: $MODEL_PATH" >&2
  exit 1
fi
if find "$MODEL_PATH" \
  -path '*/quarantine_*' -prune -o \
  -name '*.incomplete' -print -quit | grep -q .; then
  echo "model download is incomplete: $MODEL_PATH" >&2
  exit 1
fi

write_manifest() {
  local frozen_table="$RUN_ROOT/frozen_artifacts.tsv"
  : >"$frozen_table"
  for dataset in "${DATASETS[@]}"; do
    for kind in retrieval units factual_pool paraphrase_pool; do
      local path
      case "$kind" in
        retrieval) path=$(input_path "$dataset") ;;
        units) path=$(units_path "$dataset") ;;
        factual_pool) path=$(factual_pool_path "$dataset") ;;
        paraphrase_pool) path=$(paraphrase_pool_path "$dataset") ;;
      esac
      printf '%s\t%s\t%s\t%s\n' "$dataset" "$kind" "$path" \
        "$(sha256sum "$path" | awk '{print $1}')" >>"$frozen_table"
    done
  done
  "$PYTHON" - "$RUN_ROOT/manifest.json" "$MODEL_KEY" "$MODEL_PATH" \
    "$SERVED_MODEL" "$N" "$REPO" "$FROZEN_ROOT" "$frozen_table" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

out, key, model_path, served, n, repo, frozen, frozen_table = sys.argv[1:]
repo_path = Path(repo)
try:
    commit = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    commit = "unknown"
value = {
    "schema": "causalityrag.cross_model_reflow.v1",
    "server": "3haoji-A100",
    "model_key": key,
    "model_path": str(Path(model_path).resolve()),
    "served_model": served,
    "queries_per_dataset": int(n),
    "datasets": [
        "hotpotqa", "timeqa", "finqa", "musique",
        "quartz", "triviaqa", "2wiki", "pubmedqa",
    ],
    "retrieved_chunks": 5,
    "receiver_beam": 48,
    "replacement_seed": 0,
    "frozen_root": str(Path(frozen).resolve()),
    "code_commit": commit,
    "baselines_run": False,
    "pool_policy": "read_only_frozen_factual_and_paraphrase_pools",
}
artifacts = {}
for line in Path(frozen_table).read_text(encoding="utf-8").splitlines():
    dataset, kind, path, sha256 = line.split("\t")
    artifacts.setdefault(dataset, {})[kind] = {
        "path": str(Path(path).resolve()),
        "sha256": sha256,
    }
value["frozen_artifacts"] = artifacts
path = Path(out)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
PY
}

stop_owned_group() {
  local pid_file=$1
  local expected=$2
  [[ -f "$pid_file" ]] || return 0

  local pid cmd
  pid=$(cat "$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]]; then
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ "$cmd" == *"$expected"* ]]; then
      kill -- "-$pid" 2>/dev/null || true
      local deadline=$((SECONDS + 15))
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
  stop_owned_group "$RUN_ROOT/reader_8002.pid" "vllm.entrypoints.openai.api_server"
  stop_owned_group "$RUN_ROOT/reader_8003.pid" "vllm.entrypoints.openai.api_server"
  stop_owned_group "$RUN_ROOT/proxy_8000.pid" "lb_proxy_two_gpu.py"
  for port in 8000 8002 8003; do
    local deadline=$((SECONDS + 120))
    while fuser "${port}/tcp" >/dev/null 2>&1; do
      if (( SECONDS >= deadline )); then
        echo "owned reader did not release port ${port}" >&2
        exit 1
      fi
      sleep 1
    done
  done
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
  CUDA_VISIBLE_DEVICES=0,1 setsid "$PYTHON" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
    --dtype bfloat16 --trust-remote-code --max-model-len 16384 --tensor-parallel-size 2 \
    --enable-chunked-prefill --max-num-seqs 32 --enforce-eager \
    --disable-log-requests --gpu-memory-utilization 0.90 --port 8000 \
    >"$LOG_ROOT/vllm_8000.log" 2>&1 < /dev/null &
  echo $! >"$RUN_ROOT/reader_8002.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models
}

generate_clean_targets() {
  for dataset in "${DATASETS[@]}"; do
    local dir="$RUN_ROOT/$dataset/clean"
    local out="$dir/clean_targets_top5.jsonl"
    mkdir -p "$dir"
    if complete_jsonl "$out"; then
      echo "[cross-model] reuse clean targets: $dataset"
      continue
    fi
    CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
      "$PYTHON" "$REPO/scripts/generate_reader_targets.py" \
      --input "$(input_path "$dataset")" --out "$out" \
      --summary-out "$dir/summary.json" --n "$N" --k 5 --workers 32 \
      --base-url http://127.0.0.1:8000/v1 --served-model "$SERVED_MODEL" \
      >"$LOG_ROOT/clean_${dataset}.log" 2>&1
    complete_jsonl "$out" || { echo "$dataset: incomplete clean targets" >&2; exit 1; }
  done
}

run_graph_and_frontier() {
  local gpu=$1
  shift
  for dataset in "$@"; do
    local graph_dir="$RUN_ROOT/$dataset/graph"
    local frontier_dir="$RUN_ROOT/$dataset/frontier"
    local graph="$graph_dir/contribution_graph_top5.jsonl"
    local frontier="$frontier_dir/frontier_top5.jsonl"
    mkdir -p "$graph_dir" "$frontier_dir"
    if ! complete_jsonl "$graph"; then
      local graph_args=()
      if [[ "$MODEL_KEY" == "mistral-nemo-12b" || "$MODEL_KEY" == "qwen2.5-32b" ]]; then
        graph_args+=(--device-map balanced)
      fi
      CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
        "$REPO/scripts/build_contribution_graph.py" \
        --input "$(input_path "$dataset")" --out "$graph" \
        --summary-out "$graph_dir/summary.json" --model-path "$MODEL_PATH" \
        --n "$N" --k 5 --target results \
        --target-results "$RUN_ROOT/$dataset/clean/clean_targets_top5.jsonl" \
        --device cuda --dtype bfloat16 --max-receivers-per-layer 48 \
        "${graph_args[@]}" \
        >"$LOG_ROOT/graph_${dataset}.log" 2>&1
    fi
    complete_jsonl "$graph" || { echo "$dataset: incomplete graph" >&2; exit 1; }

    if ! complete_jsonl "$frontier"; then
      local factual sha
      factual=$(factual_pool_path "$dataset")
      sha=$(sha256sum "$factual" | awk '{print $1}')
      "$PYTHON" "$REPO/exp/run_reflow.py" \
        --input "$(input_path "$dataset")" --graphs "$graph" \
        --units-cache "$(units_path "$dataset")" --shared-pool "$factual" \
        --expected-pool-sha256 "$sha" --out "$frontier" \
        --summary-out "$frontier_dir/summary.json" --n "$N" --k 5 \
        >"$LOG_ROOT/frontier_${dataset}.log" 2>&1
    fi
    complete_jsonl "$frontier" || { echo "$dataset: incomplete frontier" >&2; exit 1; }
  done
}

write_empty_baselines() {
  local dataset=$1
  local out=$2
  "$PYTHON" - "$(input_path "$dataset")" "$out" "$N" <<'PY'
import json
import sys
from pathlib import Path

source, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
with open(source, encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            row = json.loads(line)
            identifier = str(row.get("id", row.get("query_id", "")))
            rows.append({"id": identifier, "methods": {}})
            if len(rows) == n:
                break
path = Path(out)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
PY
}

run_reader_evaluations() {
  for dataset in "${DATASETS[@]}"; do
    local factual_dir="$RUN_ROOT/$dataset/factual"
    local control_dir="$RUN_ROOT/$dataset/synonym"
    local factual_out="$factual_dir/results.jsonl"
    local control_out="$control_dir/results.jsonl"
    local factual factual_sha paraphrase paraphrase_sha empty
    mkdir -p "$factual_dir" "$control_dir"
    factual=$(factual_pool_path "$dataset")
    factual_sha=$(sha256sum "$factual" | awk '{print $1}')
    if ! complete_jsonl "$factual_out"; then
      CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
        "$PYTHON" "$REPO/scripts/evaluate_reflow.py" \
        --input "$(input_path "$dataset")" \
        --units-cache "$(units_path "$dataset")" \
        --frontier "$RUN_ROOT/$dataset/frontier/frontier_top5.jsonl" \
        --shared-pool "$factual" --expected-pool-sha256 "$factual_sha" \
        --out "$factual_out" --summary-out "$factual_dir/summary.json" \
        --n "$N" --k 5 --workers 32 --replacement-seed 0 \
        --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
        >"$LOG_ROOT/factual_${dataset}.log" 2>&1
    fi
    complete_jsonl "$factual_out" || { echo "$dataset: incomplete factual evaluation" >&2; exit 1; }

    paraphrase=$(paraphrase_pool_path "$dataset")
    paraphrase_sha=$(sha256sum "$paraphrase" | awk '{print $1}')
    empty="$control_dir/empty_baseline_alignment.jsonl"
    write_empty_baselines "$dataset" "$empty"
    if ! complete_jsonl "$control_out"; then
      CAUSALITYRAG_READER_MODE="$(reader_mode "$dataset")" \
        "$PYTHON" "$REPO/scripts/evaluate_paraphrase_controls.py" \
        --input "$(input_path "$dataset")" \
        --units-cache "$(units_path "$dataset")" \
        --reflow-results "$factual_out" --baseline-results "$empty" \
        --paraphrase-pool "$paraphrase" \
        --expected-pool-sha256 "$paraphrase_sha" \
        --out "$control_out" --summary-out "$control_dir/summary.json" \
        --n "$N" --k 5 --workers 32 --replacement-seed 0 \
        --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$SERVED_MODEL" \
        >"$LOG_ROOT/synonym_${dataset}.log" 2>&1
    fi
    complete_jsonl "$control_out" || { echo "$dataset: incomplete synonym control" >&2; exit 1; }
  done
}

write_manifest
trap stop_readers EXIT

echo "[cross-model] model=$MODEL_KEY n=$N stage=clean"
start_readers
generate_clean_targets
stop_readers

echo "[cross-model] model=$MODEL_KEY n=$N stage=graph-frontier"
if [[ "$MODEL_KEY" == "mistral-nemo-12b" || "$MODEL_KEY" == "qwen2.5-32b" ]]; then
  run_graph_and_frontier 0,1 "${DATASETS[@]}"
else
  run_graph_and_frontier 0 hotpotqa finqa quartz 2wiki &
  left_pid=$!
  run_graph_and_frontier 1 timeqa musique triviaqa pubmedqa &
  right_pid=$!
  wait "$left_pid"
  wait "$right_pid"
fi

echo "[cross-model] model=$MODEL_KEY n=$N stage=reader-verification"
start_readers
run_reader_evaluations
stop_readers

echo "[cross-model] complete model=$MODEL_KEY n=$N root=$RUN_ROOT"
