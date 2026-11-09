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
RUN_ROOT=${RUN_ROOT:-/data1/yujia/CausalityRAG/out_pubmedqa_readerfix_20260809/pubmedqa}
SCORE_ROOT=${SCORE_ROOT:-/data1/yujia/CausalityRAG/out/pubmedqa/methods/baselines}
OUT_DIR=$RUN_ROOT/audits/final_top10pool_k5
LOG_ROOT=$RUN_ROOT/logs
PID_ROOT=$RUN_ROOT/pids/baseline_readerfix
POOL_REL=replacements/shared_pool_top10_v1
OUT=$OUT_DIR/baselines_1000.jsonl
SUMMARY=$OUT_DIR/baselines_1000.summary.json
AUDIT=$OUT_DIR/baselines_1000.label_budget_audit.json

mkdir -p "$OUT_DIR" "$LOG_ROOT" "$PID_ROOT"
cd "$REPO"

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

cleanup() {
  stop_owned_group "$PID_ROOT/reader_8002.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/reader_8003.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/proxy_8000.pid" lb_proxy_two_gpu.py
}
trap cleanup EXIT

require_free_ports 8000 8002 8003
for spec in '0 8002' '1 8003'; do
  read -r gpu port <<<"$spec"
  CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen2.5-7b \
    --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
    --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
    --disable-log-requests --gpu-memory-utilization 0.90 --port "$port" \
    >"$LOG_ROOT/baseline_vllm_${port}.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/reader_${port}.pid"
done
wait_endpoint http://127.0.0.1:8002/v1/models 900
wait_endpoint http://127.0.0.1:8003/v1/models 900
setsid "$PYTHON" "$REPO/scripts/lb_proxy_two_gpu.py" \
  >"$LOG_ROOT/baseline_proxy_8000.log" 2>&1 < /dev/null &
echo $! >"$PID_ROOT/proxy_8000.pid"
wait_endpoint http://127.0.0.1:8000/v1/models 120

POOL=$RUN_ROOT/$POOL_REL/shared_pool.jsonl
POOL_SHA=$(sha256sum "$POOL" | awk '{print $1}')

CAUSALITYRAG_READER_MODE=pubmedqa "$PYTHON" \
  scripts/evaluate_matched_budget_baselines.py \
  --input "$RUN_ROOT/retrieval/top10_1000.jsonl" \
  --units-cache "$RUN_ROOT/inputs/token_units_top10_1000.jsonl" \
  --reflow-results "$RUN_ROOT/methods/reflow/results_top5_1000.jsonl" \
  --scores "attention=$SCORE_ROOT/attention_top5_1000.jsonl" \
  --scores "gradient_x_input=$SCORE_ROOT/gradient_x_input_top5_1000.jsonl" \
  --scores "integrated_gradients=$SCORE_ROOT/integrated_gradients_top5_1000.jsonl" \
  --scores "mirage=$SCORE_ROOT/mirage_top5_1000.jsonl" \
  --scores "arc_jsd=$SCORE_ROOT/arc_jsd_top5_1000.jsonl" \
  --shared-pool "$POOL" --expected-pool-sha256 "$POOL_SHA" \
  --out "$OUT" --summary-out "$SUMMARY" \
  --n 1000 --k 5 --workers 48 --replacement-seed 0 \
  --llm-base-url http://127.0.0.1:8000/v1 --llm-model qwen2.5-7b

"$PYTHON" - "$OUT" "$AUDIT" <<'PY'
import collections
import json
import sys

source, audit_path = sys.argv[1:]
labels = {"yes", "no", "maybe"}
rows = [json.loads(line) for line in open(source, encoding="utf-8") if line.strip()]
methods = sorted({name for row in rows for name in row.get("methods", {})})
bad = []
status_histograms = {name: collections.Counter() for name in methods}
for row in rows:
    budget = int(row.get("matched_token_budget", 0))
    if budget <= 0:
        if row.get("methods"):
            bad.append({"id": row.get("id"), "error": "methods_on_zero_budget"})
        continue
    for name in methods:
        result = row.get("methods", {}).get(name)
        if not isinstance(result, dict):
            bad.append({"id": row.get("id"), "method": name, "error": "missing_method"})
            continue
        status_histograms[name][str(result.get("status", ""))] += 1
        if not result.get("reader_called"):
            bad.append({"id": row.get("id"), "method": name, "error": "reader_not_called"})
            continue
        actual = int(result.get("n_modified_tokens", -1))
        if actual != budget:
            bad.append({
                "id": row.get("id"),
                "method": name,
                "error": "budget_mismatch",
                "expected": budget,
                "actual": actual,
            })
        answer = str(result.get("edited_answer", "")).strip().casefold()
        if answer not in labels:
            bad.append({
                "id": row.get("id"),
                "method": name,
                "error": "invalid_label",
                "value": answer,
            })

audit = {
    "rows": len(rows),
    "unique_ids": len({str(row.get("id", "")) for row in rows}),
    "methods": methods,
    "positive_budget_queries": sum(int(row.get("matched_token_budget", 0)) > 0 for row in rows),
    "zero_budget_queries": sum(int(row.get("matched_token_budget", 0)) == 0 for row in rows),
    "status_histograms": {
        name: dict(sorted(histogram.items()))
        for name, histogram in status_histograms.items()
    },
    "violations": len(bad),
    "examples": bad[:20],
}
with open(audit_path, "w", encoding="utf-8") as output:
    json.dump(audit, output, indent=2, ensure_ascii=False)
    output.write("\n")
print(json.dumps(audit, indent=2, ensure_ascii=False))
if len(rows) != 1000 or audit["unique_ids"] != 1000 or bad:
    raise SystemExit("PubMedQA baseline label/budget audit failed")
PY

sha256sum "$OUT" "$SUMMARY" "$AUDIT"
date -Is >"$RUN_ROOT/BASELINES_COMPLETE"
echo "[pubmedqa-readerfix-baselines] complete"
