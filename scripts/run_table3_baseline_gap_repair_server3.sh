#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
BASE_ROOT=${BASE_ROOT:-/data1/yujia/CausalityRAG/out}
SEVEN_ROOT=${SEVEN_ROOT:-/data1/yujia/CausalityRAG/out_no_frontier_repair_20260810}
FINAL_ROOT=${FINAL_ROOT:-/data1/yujia/CausalityRAG/out_table3_final8_20260811}
SOURCE=$BASE_ROOT/hotpotqa
REPAIRED=$SEVEN_ROOT/hotpotqa
FINAL=$FINAL_ROOT/hotpotqa
WORK=$FINAL/baseline_gap_repair_v1
MANIFEST=$RELEASE/artifacts/table3/hotpotqa/baseline_execution_gap_manifest.json
QWEN25=${QWEN25:-/data1/yujia/models/Qwen2.5-7B-Instruct}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-/data1/yujia/CausalityRAG_release_popqa_20260810/scripts/compat/transformers_452}
BASE_URL=http://127.0.0.1:8000/v1
MODEL=qwen2.5-7b
PID_ROOT=$WORK/pids
LOG_ROOT=$WORK/logs
mkdir -p "$PID_ROOT" "$LOG_ROOT"
cd "$RELEASE"

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
    echo "required pre-existing spaCy service is not listening on 8021" >&2
    exit 1
  fi
  local spec gpu reader_port
  for spec in '0 8002' '1 8003'; do
    read -r gpu reader_port <<<"$spec"
    PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
      -m vllm.entrypoints.openai.api_server \
      --model "$QWEN25" --served-model-name "$MODEL" \
      --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
      --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
      --disable-log-requests --gpu-memory-utilization 0.90 \
      --port "$reader_port" \
      >"$LOG_ROOT/vllm_${reader_port}.log" 2>&1 < /dev/null &
    echo $! >"$PID_ROOT/reader_${reader_port}.pid"
  done
  wait_endpoint http://127.0.0.1:8002/v1/models 900
  wait_endpoint http://127.0.0.1:8003/v1/models 900
  setsid "$PYTHON" scripts/lb_proxy_two_gpu.py \
    >"$LOG_ROOT/proxy_8000.log" 2>&1 < /dev/null &
  echo $! >"$PID_ROOT/proxy_8000.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models 120
}

safe_link() {
  local source=$1 target=$2
  if [[ -L "$target" ]]; then
    return 0
  fi
  if [[ -e "$target" ]]; then
    echo "refusing to replace existing non-symlink: $target" >&2
    exit 1
  fi
  ln -s "$source" "$target"
}

replace_owned_symlink() {
  local source=$1 target=$2 temporary=$2.next.$$
  if [[ ! -L "$target" ]]; then
    echo "refusing to replace non-symlink final artifact: $target" >&2
    exit 1
  fi
  ln -s "$source" "$temporary"
  mv -Tf "$temporary" "$target"
}

build_control_pool() {
  local existing=$SEVEN_ROOT/hotpotqa/table3_resume/paraphrase_pool.jsonl
  local pool=$WORK/paraphrase_pool.jsonl
  local manifest=$WORK/paraphrase_pool.manifest.json
  local unresolved=$WORK/paraphrase_pool.unresolved.jsonl
  local pass
  for pass in $(seq 1 20); do
    if [[ -s "$manifest" ]] && "$PYTHON" -c \
      'import json,sys; m=json.load(open(sys.argv[1])); raise SystemExit(0 if m.get("coverage")==1.0 and m.get("unresolved_positions")==0 else 1)' \
      "$manifest"; then
      return 0
    fi
    echo "[baseline-gap] paraphrase pass=$pass"
    CAUSALITYRAG_READER_MODE=short_answer "$PYTHON" \
      scripts/build_paraphrase_control_pool.py \
      --input "$WORK/input.jsonl" --units-cache "$WORK/units.jsonl" \
      --reflow-results "$WORK/reflow_results.jsonl" \
      --baseline-results "$WORK/baselines_v2.jsonl" \
      --existing-pool "$existing" --out "$pool" \
      --manifest-out "$manifest" --unresolved-out "$unresolved" \
      --n 5 --batch-size 16 --workers 24 --attempts 8 --max-candidates 3 \
      --llm-base-url "$BASE_URL" --llm-model "$MODEL"
    existing=$pool
  done
  echo "targeted paraphrase pool remains unresolved" >&2
  exit 1
}

echo '[baseline-gap] stage=prepare'
"$PYTHON" scripts/prepare_targeted_table3_resume.py prepare \
  --manifest "$MANIFEST" \
  --input "$SOURCE/retrieval/top10_1000.jsonl" \
  --units-cache "$SOURCE/inputs/token_units_top10_1000.jsonl" \
  --reflow-results "$REPAIRED/results_1000.jsonl" \
  --old-baselines "$FINAL/audits/final_top10pool_k5/baselines_1000.jsonl" \
  --scores "attention_prompt_last=$SOURCE/methods/baselines/attention_prompt_last_1000.jsonl" \
  --scores "gradient_x_input=$SOURCE/methods/baselines/gradient_x_input_top5_1000.jsonl" \
  --scores "integrated_gradients=$SOURCE/methods/baselines/integrated_gradients_top5_1000.jsonl" \
  --scores "mirage=$SOURCE/methods/baselines/mirage_gamma1_setting_q1000_scores.jsonl" \
  --scores "arc_jsd=$SOURCE/methods/baselines/arc_jsd_projection_1000.jsonl" \
  --out-dir "$WORK" --n 1000

start_readers

echo '[baseline-gap] stage=factual-readers'
CAUSALITYRAG_READER_MODE=short_answer "$PYTHON" \
  scripts/evaluate_matched_budget_baselines.py \
  --input "$WORK/input.jsonl" --units-cache "$WORK/units.jsonl" \
  --reflow-results "$WORK/reflow_results.jsonl" \
  --scores "attention_prompt_last=$WORK/scores/attention_prompt_last.jsonl" \
  --scores "gradient_x_input=$WORK/scores/gradient_x_input.jsonl" \
  --scores "integrated_gradients=$WORK/scores/integrated_gradients.jsonl" \
  --scores "mirage=$WORK/scores/mirage.jsonl" \
  --scores "arc_jsd=$WORK/scores/arc_jsd.jsonl" \
  --shared-pool "$SOURCE/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl" \
  --expected-pool-sha256 5f161ed2405becbda7cd39517d1d1291c562574a404c975e42f2d91ad8b31bef \
  --out "$WORK/baselines_v2.jsonl" \
  --summary-out "$WORK/baselines_v2.summary.json" \
  --random-seeds 0,1,2,3,4 --replacement-seed 0 \
  --n 5 --k 5 --workers 24 \
  --llm-base-url "$BASE_URL" --llm-model "$MODEL"

"$PYTHON" scripts/prepare_targeted_table3_resume.py merge-baselines \
  --manifest "$MANIFEST" \
  --old-baselines "$FINAL/audits/final_top10pool_k5/baselines_1000.jsonl" \
  --resumed-baselines "$WORK/baselines_v2.jsonl" \
  --reflow-results "$REPAIRED/results_1000.jsonl" \
  --out "$WORK/baselines_1000.jsonl" \
  --summary-out "$WORK/baselines_1000.summary.json" --n 1000

"$PYTHON" scripts/summarize_clean_anchor_flips.py \
  --reflow "$REPAIRED/results_1000.jsonl" \
  --baselines "$WORK/baselines_1000.jsonl" \
  --out "$WORK/factual_metrics_1000.json" --reader-mode short_answer

echo '[baseline-gap] stage=control-pool-reader'
build_control_pool
CONTROL_SHA=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["pool_sha256"])' \
  "$WORK/paraphrase_pool.manifest.json")
CAUSALITYRAG_READER_MODE=short_answer "$PYTHON" \
  scripts/evaluate_paraphrase_controls.py \
  --input "$WORK/input.jsonl" --units-cache "$WORK/units.jsonl" \
  --reflow-results "$WORK/reflow_results.jsonl" \
  --baseline-results "$WORK/baselines_v2.jsonl" \
  --paraphrase-pool "$WORK/paraphrase_pool.jsonl" \
  --expected-pool-sha256 "$CONTROL_SHA" \
  --out "$WORK/controls_v2.jsonl" \
  --summary-out "$WORK/controls_v2.summary.json" \
  --n 5 --k 5 --workers 16 --replacement-seed 0 \
  --llm-base-url "$BASE_URL" --llm-model "$MODEL"

"$PYTHON" scripts/prepare_targeted_table3_resume.py merge-controls \
  --manifest "$MANIFEST" \
  --old-controls "$FINAL/controls/paraphrase_results_top5_1000_v2.jsonl" \
  --resumed-controls "$WORK/controls_v2.jsonl" \
  --reflow-results "$REPAIRED/results_1000.jsonl" \
  --baseline-results "$WORK/baselines_1000.jsonl" \
  --reader-mode short_answer --out "$WORK/controls_1000.jsonl" \
  --summary-out "$WORK/controls_1000.summary.json" --n 1000

"$PYTHON" scripts/summarize_adjusted_flips.py \
  --factual "$WORK/factual_metrics_1000.json" \
  --meaning-preserving "$WORK/controls_1000.summary.json" \
  --out "$WORK/table3_metrics_1000.json"

"$PYTHON" scripts/prepare_targeted_table3_resume.py audit \
  --manifest "$MANIFEST" \
  --old-baselines "$FINAL/audits/final_top10pool_k5/baselines_1000.jsonl" \
  --new-baselines "$WORK/baselines_1000.jsonl" \
  --old-controls "$FINAL/controls/paraphrase_results_top5_1000_v2.jsonl" \
  --new-controls "$WORK/controls_1000.jsonl" \
  --reflow-results "$REPAIRED/results_1000.jsonl" \
  --out "$WORK/audit.json" --n 1000

stop_readers

echo '[baseline-gap] stage=publish-final-bundle'
replace_owned_symlink "$WORK/baselines_1000.jsonl" \
  "$FINAL/audits/final_top10pool_k5/baselines_1000.jsonl"
replace_owned_symlink "$WORK/factual_metrics_1000.json" \
  "$FINAL/audits/final_top10pool_k5/factual_metrics_1000.json"
replace_owned_symlink "$WORK/controls_1000.jsonl" \
  "$FINAL/controls/paraphrase_results_top5_1000_v2.jsonl"
replace_owned_symlink "$WORK/controls_1000.summary.json" \
  "$FINAL/controls/paraphrase_results_top5_1000_v2.summary.json"
replace_owned_symlink "$WORK/table3_metrics_1000.json" \
  "$FINAL/audits/final_top10pool_k5/table3_metrics_1000_v2.json"

for dataset in hotpotqa timeqa finqa musique quartz triviaqa 2wiki; do
  safe_link "$BASE_ROOT/$dataset/retrieval" "$FINAL_ROOT/$dataset/retrieval"
done

"$PYTHON" scripts/render_table3_rows.py \
  --metrics-root "$FINAL_ROOT" --out "$FINAL_ROOT/table3_rows.tex"
"$PYTHON" scripts/audit_table3_baseline_execution.py \
  --root "$FINAL_ROOT" --out "$FINAL_ROOT/baseline_execution_audit.json"
"$PYTHON" scripts/audit_table3_final8.py \
  --root "$FINAL_ROOT" --out "$FINAL_ROOT/audit.json"
"$PYTHON" scripts/audit_table3_token_counts.py \
  --root "$FINAL_ROOT" --out "$FINAL_ROOT/table3_token_count_audit.json"
"$PYTHON" scripts/render_table3_mean_edited_tokens.py \
  --root "$FINAL_ROOT" \
  --out-json "$FINAL_ROOT/table3_mean_edited_tokens.json" \
  --out-tex "$FINAL_ROOT/table3_mean_edited_tokens.tex"
date -Is >"$FINAL_ROOT/FINAL8_COMPLETE"
date -Is >"$WORK/COMPLETE"
echo "[baseline-gap] complete root=$FINAL_ROOT"
