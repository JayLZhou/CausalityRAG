#!/usr/bin/env bash
set -euo pipefail

# Repair QuaRTz choice-level metrics without modifying the historical JSONL files.
# Only factual rows whose old run stopped at a same-choice text change are rerun.

REPO=${REPO:-/data1/yujia/CausalityRAG_release_quartz_repair_20260809}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
FROZEN_ROOT=${FROZEN_ROOT:-/data1/yujia/CausalityRAG/out}
SWEEP_ROOT=${SWEEP_ROOT:-$FROZEN_ROOT/model_sensitivity_v1}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
QUARTZ_INPUT="$FROZEN_ROOT/quartz/retrieval/top10_1000.jsonl"
POOL="$FROZEN_ROOT/quartz/replacements/shared_pool_top10_v1/shared_pool.jsonl"
POOL_SHA=$(sha256sum "$POOL" | awk '{print $1}')

line_count() {
  if [[ -f "$1" ]]; then
    awk 'NF {count++} END {print count+0}' "$1"
  else
    echo 0
  fi
}

runtime_dir() {
  echo "$SWEEP_ROOT/quartz_choice_repair_runtime"
}

stop_owned_readers() {
  local runtime
  runtime=$(runtime_dir)
  local pid_file
  for pid_file in "$runtime"/*.pid; do
    [[ -f "$pid_file" ]] || continue
    local pid cmd=''
    pid=$(cat "$pid_file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/cmdline" ]]; then
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    fi
    if [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* || \
          "$cmd" == *"lb_proxy_two_gpu.py"* ]]; then
      kill -- "-$pid" 2>/dev/null || true
      local deadline=$((SECONDS + 30))
      while kill -0 -- "-$pid" 2>/dev/null && ((SECONDS < deadline)); do
        sleep 1
      done
      if kill -0 -- "-$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || true
      fi
    fi
    mv "$pid_file" "$pid_file.stopped.$RUN_ID"
  done
}

require_free_ports() {
  local port
  for port in "$@"; do
    if fuser "${port}/tcp" >/dev/null 2>&1 || \
       ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
      echo "refusing to interrupt an unowned service on port $port" >&2
      exit 1
    fi
  done
}

wait_for_port_release() {
  local deadline=$((SECONDS + 120)) port occupied
  while true; do
    occupied=0
    for port in "$@"; do
      if fuser "${port}/tcp" >/dev/null 2>&1 || \
         ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
        occupied=1
      fi
    done
    ((occupied == 0)) && break
    if ((SECONDS >= deadline)); then
      echo "owned reader ports did not release within 120 seconds" >&2
      exit 1
    fi
    sleep 2
  done
  sleep 3
}

wait_endpoint() {
  local url=$1
  local deadline=$((SECONDS + 900))
  until curl -fsS "$url" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "reader endpoint did not become ready: $url" >&2
      exit 1
    fi
    sleep 5
  done
}

start_readers() {
  local model_key=$1 model_path=$2 served_model=$3 log_dir=$4
  local runtime reader_python proxy_port backend_a backend_b
  runtime=$(runtime_dir)
  reader_python=$PYTHON
  case "$model_key" in
    qwen2.5-7b) proxy_port=8000; backend_a=8002; backend_b=8003 ;;
    qwen3-4b)
      proxy_port=8100; backend_a=8102; backend_b=8103
      reader_python=/data1/yujia/envs/qwen3_vllm_0102/bin/python
      ;;
    llama3.1-8b) proxy_port=8110; backend_a=8112; backend_b=8113 ;;
    mistral-nemo-12b) proxy_port=8120; backend_a=8122; backend_b=8123 ;;
    *) echo "unknown model: $model_key" >&2; exit 2 ;;
  esac
  mkdir -p "$runtime" "$log_dir"
  stop_owned_readers
  wait_for_port_release "$proxy_port" "$backend_a" "$backend_b"
  require_free_ports "$proxy_port" "$backend_a" "$backend_b"
  for spec in "0 $backend_a" "1 $backend_b"; do
    read -r gpu port <<<"$spec"
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$reader_python" \
      -m vllm.entrypoints.openai.api_server \
      --model "$model_path" --served-model-name "$served_model" \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 --port "$port" \
      >"$log_dir/vllm_${port}.log" 2>&1 < /dev/null &
    echo $! >"$runtime/${port}.pid"
  done
  wait_endpoint "http://127.0.0.1:$backend_a/v1/models"
  wait_endpoint "http://127.0.0.1:$backend_b/v1/models"
  CAUSALITYRAG_PROXY_BACKENDS="http://127.0.0.1:$backend_a,http://127.0.0.1:$backend_b" \
    CAUSALITYRAG_PROXY_PORT="$proxy_port" \
    setsid "$PYTHON" "$REPO/scripts/lb_proxy_two_gpu.py" \
    >"$log_dir/proxy_${proxy_port}.log" 2>&1 < /dev/null &
  echo $! >"$runtime/${proxy_port}.pid"
  wait_endpoint "http://127.0.0.1:$proxy_port/v1/models"
  echo "[quartz-repair] readers ready model=$model_key"
}

proxy_port() {
  case "$1" in
    qwen2.5-7b) echo 8000 ;;
    qwen3-4b) echo 8100 ;;
    llama3.1-8b) echo 8110 ;;
    mistral-nemo-12b) echo 8120 ;;
    *) return 2 ;;
  esac
}

model_root() {
  case "$1" in
    qwen2.5-7b) echo "$FROZEN_ROOT" ;;
    *) echo "$SWEEP_ROOT/$1" ;;
  esac
}

model_path() {
  case "$1" in
    qwen2.5-7b) echo /data1/yujia/models/Qwen2.5-7B-Instruct ;;
    qwen3-4b) echo /data1/yujia/models/Qwen3-4B-Instruct-2507 ;;
    llama3.1-8b) echo /data1/yujia/models/Llama-3.1-8B-Instruct ;;
    mistral-nemo-12b) echo /data1/yujia/models/Mistral-Nemo-Instruct-2407 ;;
    *) return 2 ;;
  esac
}

served_model() {
  case "$1" in
    qwen2.5-7b) echo qwen2.5-7b ;;
    qwen3-4b) echo qwen3-4b-instruct-2507 ;;
    llama3.1-8b) echo llama-3.1-8b-instruct ;;
    mistral-nemo-12b) echo mistral-nemo-instruct-2407 ;;
    *) return 2 ;;
  esac
}

raw_factual() {
  local root
  root=$(model_root "$1")
  if [[ "$1" == qwen2.5-7b ]]; then
    echo "$root/quartz/methods/reflow/results_top5_1000.jsonl"
  else
    echo "$root/quartz/factual/results.jsonl"
  fi
}

raw_control() {
  local root
  root=$(model_root "$1")
  if [[ "$1" == qwen2.5-7b ]]; then
    echo "$root/quartz/controls/paraphrase_results_top5_1000_v2.jsonl"
  else
    echo "$root/quartz/synonym/results.jsonl"
  fi
}

units_path() {
  echo "$FROZEN_ROOT/quartz/analysis_v1/inputs/token_units_top5_1000.jsonl"
}

frontier_path() {
  local root
  root=$(model_root "$1")
  if [[ "$1" == qwen2.5-7b ]]; then
    echo "$root/quartz/methods/reflow/frontier_top5_1000.jsonl"
  else
    echo "$root/quartz/frontier/frontier_top5.jsonl"
  fi
}

postprocess() {
  local model_key=$1 factual_source=$2 output_root=$3 report_name=$4
  local command=(
    "$PYTHON" "$REPO/scripts/reprocess_quartz_choice_results.py"
    --retrieval "$QUARTZ_INPUT"
    --factual "$factual_source"
    --control "$(raw_control "$model_key")"
    --factual-out "$output_root/factual/results.jsonl"
    --control-out "$output_root/control/results.jsonl"
    --report-out "$output_root/$report_name"
  )
  if [[ "$model_key" == qwen2.5-7b ]]; then
    command+=(
      --baselines "$FROZEN_ROOT/quartz/audits/final_top10pool_k5/baselines_1000.jsonl"
      --baselines-out "$output_root/baselines/results.jsonl"
    )
  fi
  "${command[@]}"
}

validate_final() {
  local model_key=$1 output_root=$2
  "$PYTHON" - "$QUARTZ_INPUT" "$output_root" "$model_key" <<'PY'
import json
import sys
from pathlib import Path

retrieval, root, model = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]

expected = [str(row["id"]) for row in rows(retrieval)]
if len(expected) != 1000 or len(set(expected)) != 1000:
    raise SystemExit("invalid retrieval query manifest")
for name in ("factual", "control"):
    actual = [str(row["id"]) for row in rows(root / name / "results.jsonl")]
    if actual != expected:
        raise SystemExit(f"{name} artifact is incomplete or misaligned")
if model == "qwen2.5-7b":
    actual = [str(row["id"]) for row in rows(root / "baselines/results.jsonl")]
    if actual != expected:
        raise SystemExit("baseline artifact is incomplete or misaligned")
report = json.loads((root / "report.json").read_text())
if report.get("requires_frontier_resume") != 0:
    raise SystemExit("factual frontier repair is incomplete")
PY
}

write_provenance() {
  local model_key=$1 output_root=$2
  "$PYTHON" - "$output_root/provenance.json" "$model_key" \
    "$QUARTZ_INPUT" "$(units_path "$model_key")" \
    "$(frontier_path "$model_key")" "$POOL" \
    "$(raw_factual "$model_key")" "$(raw_control "$model_key")" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, model, *sources = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
value = {
    "schema": "causalityrag.quartz_choice_repair_provenance.v1",
    "model": model,
    "sources": {str(Path(path).resolve()): sha(path) for path in sources},
}
Path(out).write_text(json.dumps(value, indent=2) + "\n")
PY
}

repair_model() {
  local model_key=$1
  local root output_root staging subset count resumed merged logs
  root=$(model_root "$model_key")
  output_root="$root/quartz/choice_postprocess_v1"
  staging="$output_root/staging/$RUN_ID"
  subset="$staging/subset"
  resumed="$staging/resumed/results.jsonl"
  merged="$staging/merged_raw_factual.jsonl"
  logs="$output_root/logs/$RUN_ID"
  mkdir -p "$staging" "$logs"

  if [[ -f "$output_root/COMPLETE" ]]; then
    validate_final "$model_key" "$output_root"
    echo "[quartz-repair] skip complete model=$model_key"
    return
  fi
  for source in "$QUARTZ_INPUT" "$(raw_factual "$model_key")" \
                "$(raw_control "$model_key")" "$(units_path "$model_key")" \
                "$(frontier_path "$model_key")" "$POOL"; do
    [[ -s "$source" ]] || { echo "missing source: $source" >&2; exit 1; }
  done
  [[ $(line_count "$(raw_factual "$model_key")") == 1000 ]]
  [[ $(line_count "$(raw_control "$model_key")") == 1000 ]]

  postprocess "$model_key" "$(raw_factual "$model_key")" "$staging/initial" report.json
  "$PYTHON" "$REPO/scripts/prepare_quartz_frontier_resume.py" prepare \
    --input "$QUARTZ_INPUT" --units "$(units_path "$model_key")" \
    --frontier "$(frontier_path "$model_key")" \
    --postprocessed-factual "$staging/initial/factual/results.jsonl" \
    --out-dir "$subset" >"$logs/prepare.json"
  count=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["queries"])' \
    "$subset/manifest.json")
  echo "[quartz-repair] model=$model_key resume=$count/1000"

  if ((count > 0)); then
    start_readers "$model_key" "$(model_path "$model_key")" \
      "$(served_model "$model_key")" "$logs"
    CAUSALITYRAG_READER_MODE=quartz "$PYTHON" \
      "$REPO/scripts/evaluate_reflow.py" \
      --input "$subset/input.jsonl" --units-cache "$subset/units.jsonl" \
      --frontier "$subset/frontier.jsonl" --shared-pool "$POOL" \
      --expected-pool-sha256 "$POOL_SHA" --out "$resumed" \
      --summary-out "$staging/resumed/summary.json" --n "$count" --k 5 \
      --workers 48 --replacement-seed 0 \
      --llm-base-url "http://127.0.0.1:$(proxy_port "$model_key")/v1" \
      --llm-model "$(served_model "$model_key")" \
      >"$logs/factual_resume.log" 2>&1
    stop_owned_readers
    [[ $(line_count "$resumed") == "$count" ]] || {
      echo "incomplete factual resume for $model_key" >&2; exit 1;
    }
    "$PYTHON" "$REPO/scripts/prepare_quartz_frontier_resume.py" merge \
      --original-factual "$(raw_factual "$model_key")" \
      --resumed-factual "$resumed" \
      --postprocessed-factual "$staging/initial/factual/results.jsonl" \
      --out "$merged" >"$logs/merge.json"
  else
    merged=$(raw_factual "$model_key")
  fi

  postprocess "$model_key" "$merged" "$output_root" report.json
  CAUSALITYRAG_READER_MODE=quartz "$PYTHON" \
    "$REPO/scripts/summarize_paraphrase_controls.py" \
    --input "$output_root/control/results.jsonl" \
    --out "$output_root/control/summary.json" --reader-mode quartz \
    >"$logs/control_summary.log" 2>&1
  if [[ "$model_key" == qwen2.5-7b ]]; then
    CAUSALITYRAG_READER_MODE=quartz "$PYTHON" \
      "$REPO/scripts/summarize_clean_anchor_flips.py" \
      --reflow "$output_root/factual/results.jsonl" \
      --baselines "$output_root/baselines/results.jsonl" \
      --out "$output_root/factual_metrics_1000.json" --reader-mode quartz \
      >"$logs/factual_metrics.log" 2>&1
    "$PYTHON" "$REPO/scripts/summarize_adjusted_flips.py" \
      --factual "$output_root/factual_metrics_1000.json" \
      --meaning-preserving "$output_root/control/summary.json" \
      --out "$output_root/table3_metrics_1000_v2.json" \
      >"$logs/adjusted_metrics.log" 2>&1
  fi
  validate_final "$model_key" "$output_root"
  write_provenance "$model_key" "$output_root"
  "$PYTHON" - "$output_root/COMPLETE" "$model_key" "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema": "causalityrag.quartz_choice_repair_complete.v1",
    "model": sys.argv[2],
    "run_id": sys.argv[3],
}, indent=2) + "\n")
PY
  echo "[quartz-repair] complete model=$model_key"
}

trap stop_owned_readers EXIT
cd "$REPO"
for model_key in qwen2.5-7b qwen3-4b llama3.1-8b mistral-nemo-12b; do
  repair_model "$model_key"
done
stop_owned_readers
trap - EXIT

"$PYTHON" "$REPO/scripts/render_table3_rows.py" \
  --metrics-root "$FROZEN_ROOT" --datasets quartz \
  --out "$FROZEN_ROOT/quartz/choice_postprocess_v1/table3_quartz_rows.tex"

echo "[quartz-repair] all models complete"
