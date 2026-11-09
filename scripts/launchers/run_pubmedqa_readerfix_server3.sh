#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

REPO=${REPO:-/data1/yujia/CausalityRAG_release_pubmedqa_readerfix_20260809}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL=${MODEL:-/data1/yujia/models/Qwen2.5-7B-Instruct}
SOURCE_ROOT=${SOURCE_ROOT:-/data1/yujia/CausalityRAG/out_label_protocol_20260805/pubmedqa}
OUT_ROOT=${OUT_ROOT:-/data1/yujia/CausalityRAG/out_pubmedqa_readerfix_20260809}
RUN_ROOT=${RUN_ROOT:-$OUT_ROOT/pubmedqa}
LOG_ROOT=$RUN_ROOT/logs
PID_ROOT=$RUN_ROOT/pids
RESULT_ROOT=$RUN_ROOT/methods/reflow
CONTROL_ROOT=$RUN_ROOT/controls
POOL_REL=replacements/shared_pool_top10_v1
PARAPHRASE_POOL=${PARAPHRASE_POOL:-/data1/yujia/CausalityRAG/out/pubmedqa/controls/paraphrase_pool_top5_v2.jsonl}

mkdir -p "$LOG_ROOT" "$PID_ROOT" "$RESULT_ROOT" "$CONTROL_ROOT"
cd "$REPO"

line_count() {
  if [[ -f "$1" ]]; then
    awk 'NF {n++} END {print n+0}' "$1"
  else
    echo 0
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
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
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
  stop_owned_group "$PID_ROOT/reader_8002.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/reader_8003.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/proxy_8000.pid" lb_proxy_two_gpu.py
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
      --model "$MODEL" --served-model-name qwen2.5-7b \
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

link_prerequisite() {
  local relative=$1
  if [[ ! -f "$SOURCE_ROOT/$relative" ]]; then
    echo "missing prerequisite: $SOURCE_ROOT/$relative" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$RUN_ROOT/$relative")"
  ln -sfn "$SOURCE_ROOT/$relative" "$RUN_ROOT/$relative"
}

link_prerequisite retrieval/top10_1000.jsonl
link_prerequisite inputs/token_units_top10_1000.jsonl
link_prerequisite inputs/clean_targets_top5_1000.jsonl
link_prerequisite methods/reflow/frontier_top5_1000.jsonl
link_prerequisite "$POOL_REL/shared_pool.jsonl"
link_prerequisite "$POOL_REL/shared_pool.manifest.json"

for prerequisite in \
  "$RUN_ROOT/retrieval/top10_1000.jsonl" \
  "$RUN_ROOT/inputs/token_units_top10_1000.jsonl" \
  "$RUN_ROOT/inputs/clean_targets_top5_1000.jsonl" \
  "$RUN_ROOT/methods/reflow/frontier_top5_1000.jsonl"; do
  if [[ "$(line_count "$prerequisite")" != 1000 ]]; then
    echo "prerequisite is not exactly 1000 rows: $prerequisite" >&2
    exit 1
  fi
done

POOL_MANIFEST=$RUN_ROOT/$POOL_REL/shared_pool.manifest.json
POOL=$RUN_ROOT/$POOL_REL/shared_pool.jsonl
POOL_SHA=$(
  "$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["pool_sha256"])' \
    "$POOL_MANIFEST"
)
ACTUAL_POOL_SHA=$(sha256sum "$POOL" | awk '{print $1}')
if [[ "$POOL_SHA" != "$ACTUAL_POOL_SHA" ]]; then
  echo "frozen pool SHA mismatch" >&2
  exit 1
fi

archive_partial() {
  local path=$1
  if [[ -e "$path" ]]; then
    mv "$path" "$path.partial.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
}

run_evaluation() {
  local n=$1 out=$2 summary=$3 workers=$4
  CAUSALITYRAG_READER_MODE=pubmedqa "$PYTHON" scripts/evaluate_reflow.py \
    --input "$RUN_ROOT/retrieval/top10_1000.jsonl" \
    --units-cache "$RUN_ROOT/inputs/token_units_top10_1000.jsonl" \
    --frontier "$RUN_ROOT/methods/reflow/frontier_top5_1000.jsonl" \
    --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
    --out "$out" --summary-out "$summary" --n "$n" --k 5 \
    --workers "$workers" --replacement-seed 0 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model qwen2.5-7b
}

validate_labels() {
  local path=$1 audit=$2
  "$PYTHON" - "$path" "$audit" <<'PY'
import collections
import json
import sys

source, audit_path = sys.argv[1:]
labels = {"yes", "no", "maybe"}
rows = [json.loads(line) for line in open(source, encoding="utf-8") if line.strip()]
bad = []
edited = collections.Counter()
for row in rows:
    answer = str(row.get("edited_answer", "")).strip().casefold()
    edited[answer] += 1
    if answer not in labels:
        bad.append({"id": row.get("id"), "field": "edited_answer", "value": answer})
    for attempt in row.get("attempts", []):
        if not attempt.get("reader_called"):
            continue
        attempt_answer = str(attempt.get("edited_answer", "")).strip().casefold()
        if attempt_answer not in labels:
            bad.append({
                "id": row.get("id"),
                "field": "attempt.edited_answer",
                "value": attempt_answer,
            })
audit = {
    "rows": len(rows),
    "edited_answer_histogram": dict(sorted(edited.items())),
    "invalid_label_count": len(bad),
    "invalid_labels": bad,
}
with open(audit_path, "w", encoding="utf-8") as output:
    json.dump(audit, output, indent=2, ensure_ascii=False)
    output.write("\n")
print(json.dumps(audit, ensure_ascii=False))
if bad:
    raise SystemExit("PubMedQA label audit failed")
PY
}

write_empty_baselines() {
  local path=$1
  "$PYTHON" - "$RUN_ROOT/retrieval/top10_1000.jsonl" "$path" <<'PY'
import json
import sys
from pathlib import Path

from causalityrag.io import load_records, record_id

rows = [
    {"id": record_id(record), "methods": {}}
    for record in load_records(sys.argv[1])
]
Path(sys.argv[2]).write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
PY
}

validate_control_labels() {
  local path=$1 audit=$2
  "$PYTHON" - "$path" "$audit" <<'PY'
import collections
import json
import sys

source, audit_path = sys.argv[1:]
labels = {"yes", "no", "maybe"}
rows = [json.loads(line) for line in open(source, encoding="utf-8") if line.strip()]
statuses = collections.Counter()
answers = collections.Counter()
invalid = []
protocol_violations = 0
answer_flips = 0
f1_flips = 0
for row in rows:
    method = row.get("methods", {}).get("reflow", {})
    status = str(method.get("status", ""))
    statuses[status] += 1
    protocol_violations += int(status.startswith("protocol_violation"))
    answer_flips += int(bool(method.get("answer_flip")))
    f1_flips += int(bool(method.get("f1_flip")))
    answer = str(method.get("edited_answer", "")).strip().casefold()
    if answer:
        answers[answer] += 1
        if answer not in labels:
            invalid.append({"id": row.get("id"), "value": answer})
audit = {
    "rows": len(rows),
    "status_histogram": dict(sorted(statuses.items())),
    "edited_answer_histogram": dict(sorted(answers.items())),
    "answer_flips": answer_flips,
    "f1_flips": f1_flips,
    "invalid_label_count": len(invalid),
    "protocol_violations": protocol_violations,
    "invalid_labels": invalid,
}
with open(audit_path, "w", encoding="utf-8") as output:
    json.dump(audit, output, indent=2, ensure_ascii=False)
    output.write("\n")
print(json.dumps(audit, ensure_ascii=False))
if invalid or protocol_violations:
    raise SystemExit("PubMedQA control protocol audit failed")
PY
}

archive_partial "$RUN_ROOT/COMPLETE"
start_readers

SMOKE=$RESULT_ROOT/smoke_1.jsonl
SMOKE_SUMMARY=$RESULT_ROOT/smoke_1.summary.json
archive_partial "$SMOKE"
archive_partial "$SMOKE_SUMMARY"
echo '[pubmedqa-readerfix] stage=protocol-smoke n=1'
run_evaluation 1 "$SMOKE" "$SMOKE_SUMMARY" 1 \
  >"$LOG_ROOT/protocol_smoke.log" 2>&1
validate_labels "$SMOKE" "$RESULT_ROOT/smoke_1.label_audit.json"

RESULTS=$RESULT_ROOT/results_top5_1000.jsonl
SUMMARY=$RESULT_ROOT/results_top5_1000.summary.json
if [[ "$(line_count "$RESULTS")" == 1000 && -s "$SUMMARY" ]]; then
  echo '[pubmedqa-readerfix] stage=full-reader reuse-complete'
else
  archive_partial "$RESULTS"
  archive_partial "$SUMMARY"
  echo '[pubmedqa-readerfix] stage=full-reader n=1000'
  run_evaluation 1000 "$RESULTS" "$SUMMARY" 48 \
    >"$LOG_ROOT/full_reader.log" 2>&1
fi

validate_labels "$RESULTS" "$RESULT_ROOT/results_top5_1000.label_audit.json"
cp "$SOURCE_ROOT/methods/reflow/frontier_top5_1000.summary.json" \
  "$RESULT_ROOT/frontier_top5_1000.source.summary.json"

if [[ ! -s "$PARAPHRASE_POOL" ]]; then
  echo "missing frozen paraphrase pool: $PARAPHRASE_POOL" >&2
  exit 1
fi
PARAPHRASE_SHA=$(sha256sum "$PARAPHRASE_POOL" | awk '{print $1}')
EMPTY_BASELINES=$CONTROL_ROOT/empty_baseline_alignment.jsonl
write_empty_baselines "$EMPTY_BASELINES"
CONTROL_RESULTS=$CONTROL_ROOT/paraphrase_results_top5_1000_v2.jsonl
CONTROL_SUMMARY=$CONTROL_ROOT/paraphrase_results_top5_1000_v2.summary.json
if [[ "$(line_count "$CONTROL_RESULTS")" == 1000 && -s "$CONTROL_SUMMARY" ]]; then
  echo '[pubmedqa-readerfix] stage=synonym-control reuse-complete'
else
  archive_partial "$CONTROL_RESULTS"
  archive_partial "$CONTROL_SUMMARY"
  echo '[pubmedqa-readerfix] stage=synonym-control n=1000'
  CAUSALITYRAG_READER_MODE=pubmedqa "$PYTHON" \
    scripts/evaluate_paraphrase_controls.py \
    --input "$RUN_ROOT/retrieval/top10_1000.jsonl" \
    --units-cache "$RUN_ROOT/inputs/token_units_top10_1000.jsonl" \
    --reflow-results "$RESULTS" --baseline-results "$EMPTY_BASELINES" \
    --paraphrase-pool "$PARAPHRASE_POOL" \
    --expected-pool-sha256 "$PARAPHRASE_SHA" \
    --out "$CONTROL_RESULTS" --summary-out "$CONTROL_SUMMARY" \
    --n 1000 --k 5 --workers 48 --replacement-seed 0 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model qwen2.5-7b \
    >"$LOG_ROOT/synonym_control.log" 2>&1
fi
validate_control_labels \
  "$CONTROL_RESULTS" \
  "$CONTROL_ROOT/paraphrase_results_top5_1000_v2.label_audit.json"
date -Is >"$RUN_ROOT/COMPLETE"
echo "[pubmedqa-readerfix] complete: $RUN_ROOT"
