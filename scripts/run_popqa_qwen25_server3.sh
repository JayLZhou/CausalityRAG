#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

REPO=${REPO:-/data1/yujia/CausalityRAG_release_popqa_20260810}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
SPACY_PYTHON=${SPACY_PYTHON:-/data1/yujia/envs/spacyner/bin/python}
SOURCE=${SOURCE:-/data1/yujia/RAGData/paper_datasets/popqa/user_provided_20260810}
NORMALIZED=${NORMALIZED:-/data1/yujia/RAGData/paper_datasets/popqa/normalized_20260810}
OUT_ROOT=${OUT_ROOT:-/data1/yujia/CausalityRAG/out_popqa_20260810}
RUN_ROOT=$OUT_ROOT/popqa
LOG_ROOT=$RUN_ROOT/logs
PID_ROOT=$RUN_ROOT/pids
EMBED_MODEL=${EMBED_MODEL:-/data1/yujia/models/Qwen3-Embedding-0.6B}
QWEN25=${QWEN25:-/data1/yujia/models/Qwen2.5-7B-Instruct}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-$REPO/scripts/compat/transformers_452}
N=${N:-1000}
BEAM=${BEAM:-48}
RESUME_AFTER_GRAPH=${RESUME_AFTER_GRAPH:-0}
mkdir -p "$LOG_ROOT" "$PID_ROOT" "$NORMALIZED"
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

start_readers() {
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
  setsid "$PYTHON" scripts/lb_proxy_two_gpu.py \
    >"$LOG_ROOT/proxy_8000.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models 120
}

run_graph_shard() {
  local gpu=$1 start=$2 count=$3
  local shard_dir=$RUN_ROOT/graphs/shards
  local shard=$shard_dir/contribution_graph_top5_${start}_$((start + count - 1)).jsonl
  local summary=${shard%.jsonl}.summary.json
  mkdir -p "$shard_dir"
  if complete_jsonl "$shard" "$count" && [[ -s "$summary" ]]; then return; fi
  CAUSALITYRAG_READER_MODE=short_answer CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    scripts/build_contribution_graph.py \
    --input "$RUN_ROOT/retrieval/top10_1000.jsonl" \
    --out "$shard" --summary-out "$summary" --model-path "$QWEN25" \
    --start "$start" --n "$count" --k 5 --target results \
    --target-results "$RUN_ROOT/inputs/clean_targets_top5_1000.jsonl" \
    --device cuda --dtype bfloat16 --max-receivers-per-layer "$BEAM" \
    >"$LOG_ROOT/graph_${start}_$((start + count - 1)).log" 2>&1
}

if [[ "$RESUME_AFTER_GRAPH" != 1 ]]; then
echo '[popqa] stage=normalize-bundle'
"$PYTHON" scripts/prepare_popqa_bundle.py \
  --questions "$SOURCE/Question.json" --corpus "$SOURCE/Corpus.json" \
  --out-dir "$NORMALIZED" --manifest-out "$NORMALIZED/manifest.json" \
  >"$LOG_ROOT/normalize.log" 2>&1

if ! complete_jsonl "$RUN_ROOT/retrieval/top10_1000.jsonl" "$N"; then
  echo '[popqa] stage=retrieval'
  start_embedding
  "$PYTHON" scripts/prepare_dataset_retrieval.py \
    --dataset popqa --questions "$NORMALIZED/questions.jsonl" \
    --corpus "$NORMALIZED/corpus.jsonl" --out-root "$RUN_ROOT" \
    --tokenizer-path "$EMBED_MODEL" \
    --embedding-base-url http://127.0.0.1:8017/v1 \
    --embedding-model Qwen3-Embedding-0.6B --embedding-batch-size 16 \
    --n "$N" --top-k 10 >"$LOG_ROOT/retrieval.log" 2>&1
  stop_embedding
else
  echo '[popqa] stage=retrieval reuse-complete'
fi

echo '[popqa] stage=units-pool-clean'
GLOBAL_SEED=$RUN_ROOT/replacements/global_seed.jsonl
if [[ ! -s "$GLOBAL_SEED" ]]; then
  seed_args=()
  for path in /data1/yujia/CausalityRAG/out/*/replacements/shared_pool_top10_v1/typed_candidates.jsonl; do
    [[ -s "$path" ]] && seed_args+=(--seed "$path")
  done
  "$PYTHON" scripts/merge_shared_replacement_seeds.py \
    "${seed_args[@]}" --out "$GLOBAL_SEED" \
    --manifest-out "$RUN_ROOT/replacements/global_seed.manifest.json" \
    >"$LOG_ROOT/global_seed.log" 2>&1
fi
mkdir -p "$RUN_ROOT/replacements/smoke10" \
  "$RUN_ROOT/replacements/shared_pool_top10_v1"
ln -sfn "$GLOBAL_SEED" "$RUN_ROOT/replacements/smoke10/global_seed.jsonl"
ln -sfn "$GLOBAL_SEED" \
  "$RUN_ROOT/replacements/shared_pool_top10_v1/global_seed.jsonl"
start_readers
"$PYTHON" scripts/run_seven_dataset_pools.py \
  --out-root "$OUT_ROOT" --datasets popqa \
  --max-generation-passes 100 --exclude-unresolved-after-max-passes \
  >"$LOG_ROOT/pool.log" 2>&1
CAUSALITYRAG_READER_MODE=short_answer "$PYTHON" \
  scripts/generate_reader_targets.py \
  --input "$RUN_ROOT/retrieval/top10_1000.jsonl" \
  --out "$RUN_ROOT/inputs/clean_targets_top5_1000.jsonl" \
  --summary-out "$RUN_ROOT/inputs/clean_targets_top5_1000.summary.json" \
  --n "$N" --k 5 --workers 48 \
  --base-url http://127.0.0.1:8000/v1 --served-model qwen2.5-7b \
  >"$LOG_ROOT/clean.log" 2>&1
stop_readers

echo '[popqa] stage=graph'
run_graph_shard 0 0 500 & left=$!
run_graph_shard 1 500 500 & right=$!
left_status=0
right_status=0
wait "$left" || left_status=$?
wait "$right" || right_status=$?
if (( left_status != 0 || right_status != 0 )); then
  left_shard="$RUN_ROOT/graphs/shards/contribution_graph_top5_0_499.jsonl"
  right_shard="$RUN_ROOT/graphs/shards/contribution_graph_top5_500_999.jsonl"
  if ! complete_jsonl "$left_shard" 500 || \
     ! complete_jsonl "$right_shard" 500 || \
     [[ ! -s "${left_shard%.jsonl}.summary.json" ]] || \
     [[ ! -s "${right_shard%.jsonl}.summary.json" ]]; then
    echo "graph shard failed before producing complete artifacts: left=$left_status right=$right_status" >&2
    exit 1
  fi
  echo "[popqa] graph commands reported nonzero but both frozen shards are complete: left=$left_status right=$right_status"
fi
else
  left_shard="$RUN_ROOT/graphs/shards/contribution_graph_top5_0_499.jsonl"
  right_shard="$RUN_ROOT/graphs/shards/contribution_graph_top5_500_999.jsonl"
  if ! complete_jsonl "$left_shard" 500 || \
     ! complete_jsonl "$right_shard" 500 || \
     [[ ! -s "${left_shard%.jsonl}.summary.json" ]] || \
     [[ ! -s "${right_shard%.jsonl}.summary.json" ]]; then
    echo "RESUME_AFTER_GRAPH=1 requires two complete 500-row graph shards and summaries" >&2
    exit 1
  fi
  echo '[popqa] stage=graph reuse-complete'
fi
"$PYTHON" scripts/merge_contribution_graph_shards.py \
  --retrieval "$RUN_ROOT/retrieval/top10_1000.jsonl" \
  --shard "$RUN_ROOT/graphs/shards/contribution_graph_top5_0_499.jsonl" \
  --shard "$RUN_ROOT/graphs/shards/contribution_graph_top5_500_999.jsonl" \
  --out "$RUN_ROOT/graphs/contribution_graph_top5_1000.jsonl" \
  --summary-out "$RUN_ROOT/graphs/contribution_graph_top5_1000.summary.json" \
  --n "$N" >"$LOG_ROOT/merge_graph.log" 2>&1

echo '[popqa] stage=frontier'
"$PYTHON" scripts/run_table3_frontiers.py \
  --out-root "$OUT_ROOT" --datasets popqa --n "$N" --k 5 \
  >"$LOG_ROOT/frontier.log" 2>&1

echo '[popqa] stage=factual-reader'
start_readers
pool_sha=$("$PYTHON" -c \
  "import json; print(json.load(open('$RUN_ROOT/replacements/shared_pool_top10_v1/shared_pool.manifest.json'))['pool_sha256'])")
CAUSALITYRAG_READER_MODE=short_answer "$PYTHON" scripts/evaluate_reflow.py \
  --input "$RUN_ROOT/retrieval/top10_1000.jsonl" \
  --units-cache "$RUN_ROOT/inputs/token_units_top10_1000.jsonl" \
  --frontier "$RUN_ROOT/methods/reflow/frontier_top5_1000.jsonl" \
  --shared-pool "$RUN_ROOT/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  --expected-pool-sha256 "$pool_sha" \
  --out "$RUN_ROOT/methods/reflow/results_top5_1000.jsonl" \
  --summary-out "$RUN_ROOT/methods/reflow/results_top5_1000.summary.json" \
  --n "$N" --k 5 --workers 48 --replacement-seed 0 \
  --llm-base-url http://127.0.0.1:8000/v1 --llm-model qwen2.5-7b \
  >"$LOG_ROOT/factual.log" 2>&1
stop_readers

echo '[popqa] stage=audit'
"$PYTHON" - "$RUN_ROOT" "$NORMALIZED/manifest.json" <<'PY'
import collections
import hashlib
import json
import pathlib
import statistics
import sys

from causalityrag.reader import answer_token_f1, answers_exact_match, normalize_answer

root = pathlib.Path(sys.argv[1])

def rows(relative):
    with (root / relative).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]

def sha(relative):
    digest = hashlib.sha256()
    with (root / relative).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

retrieval = rows("retrieval/top10_1000.jsonl")
clean = rows("inputs/clean_targets_top5_1000.jsonl")
graphs = rows("graphs/contribution_graph_top5_1000.jsonl")
frontier = rows("methods/reflow/frontier_top5_1000.jsonl")
factual = rows("methods/reflow/results_top5_1000.jsonl")
expected_ids = [row["id"] for row in retrieval]
for name, artifact in (
    ("clean", clean), ("graph", graphs), ("frontier", frontier),
    ("factual", factual),
):
    if [row["id"] for row in artifact] != expected_ids:
        raise SystemExit(f"{name} IDs do not match frozen retrieval order")

def aliases(row):
    values = row.get("gold_answers") or [row.get("gold_answer", "")]
    return [str(value) for value in values if str(value).strip()]

def best_f1(answer, accepted):
    return max((answer_token_f1(answer, gold) for gold in accepted), default=0.0)

retrieval_coverage = {}
for k in (5, 10):
    covered = 0
    for row in retrieval:
        text = normalize_answer(" ".join(
            str(item.get("text", "")) for item in row.get("retrieved", [])[:k]
        ))
        accepted = [normalize_answer(value) for value in row.get("answers", [])]
        covered += int(any(value and value in text for value in accepted))
    retrieval_coverage[f"top{k}_alias_span"] = covered

f1_population = 0
f1_flips = 0
for row in factual:
    accepted = aliases(row)
    clean_f1 = best_f1(str(row.get("clean_answer", "")), accepted)
    edited_f1 = best_f1(str(row.get("edited_answer", "")), accepted)
    clean_correct = clean_f1 >= 1.0 - 1e-12
    f1_population += int(clean_correct)
    f1_flips += int(clean_correct and edited_f1 < clean_f1 - 1e-12)

summary = json.loads(
    (root / "methods/reflow/results_top5_1000.summary.json").read_text()
)
pool_sha_audit = json.loads(
    (root / "replacements/shared_pool_top10_v1/shared_pool.manifest.json").read_text()
)["pool_sha256"]
audit = {
    "schema": "causalityrag.popqa_qwen25.v1",
    "queries": len(retrieval),
    "reader_mode": "short_answer",
    "guided_decoding": False,
    "source_manifest": json.loads(pathlib.Path(sys.argv[2]).read_text()),
    "retrieval_answer_coverage": retrieval_coverage,
    "clean_correct_any_alias": sum(bool(row.get("clean_correct")) for row in clean),
    "clean_status": dict(collections.Counter(row.get("status") for row in clean)),
    "graph_status": dict(collections.Counter(row.get("status") for row in graphs)),
    "frontier_status": dict(collections.Counter(row.get("status") for row in frontier)),
    "factual_status": dict(collections.Counter(row.get("evaluation_status") for row in factual)),
    "answer_flips": int(summary["verified_flips"]),
    "answer_flip_rate": float(summary["raw_flip_rate"]),
    "f1_clean_correct_queries": f1_population,
    "f1_flips": f1_flips,
    "f1_flip_rate": f1_flips / max(1, f1_population),
    "mean_modified_tokens_all": float(summary["mean_modified_tokens_all_queries"]),
    "mean_modified_tokens_flips": float(summary["mean_modified_tokens_flips"]),
    "reader_calls": int(summary["reader_calls"]),
    "protocol_violations": int(summary["protocol_violations"]),
    "pool_sha256": pool_sha_audit,
    "artifact_sha256": {
        "retrieval": sha("retrieval/top10_1000.jsonl"),
        "units": sha("inputs/token_units_top10_1000.jsonl"),
        "clean": sha("inputs/clean_targets_top5_1000.jsonl"),
        "graph": sha("graphs/contribution_graph_top5_1000.jsonl"),
        "frontier": sha("methods/reflow/frontier_top5_1000.jsonl"),
        "factual": sha("methods/reflow/results_top5_1000.jsonl"),
    },
}
(root / "audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(audit, indent=2, ensure_ascii=False))
PY

date -Is >"$RUN_ROOT/COMPLETE"
echo "[popqa] complete root=$RUN_ROOT"
