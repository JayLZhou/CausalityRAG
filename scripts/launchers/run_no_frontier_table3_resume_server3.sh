#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HOST=${EXPECTED_HOST:-3haoji-A100}
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/data1/yujia/CausalityRAG/out}
REPAIR_ROOT=${REPAIR_ROOT:-/data1/yujia/CausalityRAG/out_no_frontier_repair_20260810}
POPQA_ROOT=${POPQA_ROOT:-/data1/yujia/CausalityRAG/out_popqa_20260810/popqa}
QWEN25=${QWEN25:-/data1/yujia/models/Qwen2.5-7B-Instruct}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-/data1/yujia/CausalityRAG_release_popqa_20260810/scripts/compat/transformers_452}
BASE_URL=http://127.0.0.1:8000/v1
MODEL=qwen2.5-7b
PID_ROOT=$REPAIR_ROOT/table3_resume_pids
LOG_ROOT=$REPAIR_ROOT/table3_resume_logs
mkdir -p "$PID_ROOT" "$LOG_ROOT"
cd "$RELEASE"

line_count() {
  if [[ -f "$1" ]]; then awk 'NF {n++} END {print n+0}' "$1"; else echo 0; fi
}

complete_jsonl() {
  [[ "$(line_count "$1")" == "$2" ]]
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
  if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:)8021$'; then
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

wait_for_popqa() {
  while screen -ls 2>/dev/null | grep -q '[.]popqa_qwen25'; do
    echo "[targeted-table3] waiting for healthy PopQA screen"
    sleep 60
  done
  if [[ ! -f "$POPQA_ROOT/COMPLETE" ]]; then
    echo "PopQA screen ended without COMPLETE; refusing to claim its GPUs" >&2
    exit 3
  fi
}

build_control_pool() {
  local work=$1 old_pool=$2 mode=$3 count=$4
  local pool=$work/paraphrase_pool.jsonl
  local manifest=$work/paraphrase_pool.manifest.json
  local unresolved=$work/paraphrase_pool.unresolved.jsonl
  local existing=$old_pool
  local pass
  for pass in $(seq 1 20); do
    if [[ -s "$manifest" ]] && "$PYTHON" -c \
      'import json,sys; m=json.load(open(sys.argv[1])); raise SystemExit(0 if m.get("coverage")==1.0 and m.get("unresolved_positions")==0 else 1)' \
      "$manifest"; then
      return 0
    fi
    echo "[targeted-table3] paraphrase pass=$pass work=$work"
    CAUSALITYRAG_READER_MODE="$mode" "$PYTHON" \
      scripts/build_paraphrase_control_pool.py \
      --input "$work/input.jsonl" \
      --units-cache "$work/units.jsonl" \
      --reflow-results "$work/reflow_results.jsonl" \
      --baseline-results "$work/baselines_v2.jsonl" \
      --existing-pool "$existing" \
      --out "$pool" --manifest-out "$manifest" \
      --unresolved-out "$unresolved" --n "$count" \
      --batch-size 16 --workers 24 --attempts 8 --max-candidates 3 \
      --llm-base-url "$BASE_URL" --llm-model "$MODEL"
    existing=$pool
  done
  echo "paraphrase pool remains unresolved: $work" >&2
  exit 4
}

rerank_hotpotqa_empty_scores() {
  local work=$1
  local reranked=$work/reranked
  mkdir -p "$reranked"
  local method gpu output summary log pid
  local pids=()
  for method in gradient_x_input integrated_gradients; do
    if [[ "$method" == gradient_x_input ]]; then gpu=0; else gpu=1; fi
    output=$reranked/$method.jsonl
    summary=$reranked/$method.summary.json
    log=$reranked/$method.log
    if complete_jsonl "$output" 1 && [[ -s "$summary" ]]; then
      continue
    fi
    CAUSALITYRAG_READER_MODE=short_answer CUDA_VISIBLE_DEVICES="$gpu" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" exp/run_gradient_attribution_baselines.py \
      --input "$work/input.jsonl" --clean-reference "$work/reflow_results.jsonl" \
      --context-units "$work/units.jsonl" --method "$method" \
      --out "$output" --summary-out "$summary" --model-path "$QWEN25" \
      --start 0 --n 1 --k 5 --device cuda --dtype bfloat16 \
      --top-k 32 --ig-steps 16 >"$log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  for method in gradient_x_input integrated_gradients; do
    complete_jsonl "$reranked/$method.jsonl" 1 || {
      echo "targeted HotpotQA rerank incomplete: $method" >&2
      exit 5
    }
  done
}

prepare_hotpotqa_rerank_inputs() {
  local base=$SOURCE_ROOT/hotpotqa
  local repair=$REPAIR_ROOT/hotpotqa
  local work=$repair/table3_resume
  local audit=$base/audits/final_top10pool_k5
  mkdir -p "$work/scores"
  "$PYTHON" scripts/prepare_targeted_table3_resume.py prepare \
    --manifest "$repair/resume/manifest.json" \
    --input "$base/retrieval/top10_1000.jsonl" \
    --units-cache "$base/inputs/token_units_top10_1000.jsonl" \
    --reflow-results "$repair/results_1000.jsonl" \
    --old-baselines "$audit/baselines_1000.jsonl" \
    --scores "attention_prompt_last=$base/methods/baselines/attention_prompt_last_1000.jsonl" \
    --scores "gradient_x_input=$base/methods/baselines/gradient_x_input_top5_1000.jsonl" \
    --scores "integrated_gradients=$base/methods/baselines/integrated_gradients_top5_1000.jsonl" \
    --scores "mirage=$base/methods/baselines/mirage_gamma1_setting_q1000_scores.jsonl" \
    --scores "arc_jsd=$base/methods/baselines/arc_jsd_projection_1000.jsonl" \
    --out-dir "$work" --n 1000
}

run_dataset() {
  local dataset=$1 count=$2 mode=$3 pool=$4 pool_sha=$5
  local old_control_pool=$6 old_control_results=$7
  local base=$SOURCE_ROOT/$dataset
  local repair=$REPAIR_ROOT/$dataset
  local work=$repair/table3_resume
  local audit=$base/audits/final_top10pool_k5
  local attention_name=attention
  local attention_scores=$base/methods/baselines/attention_top5_1000.jsonl
  local mirage_scores=$base/methods/baselines/mirage_top5_1000.jsonl
  local arc_scores=$base/methods/baselines/arc_jsd_top5_1000.jsonl
  if [[ "$dataset" == hotpotqa ]]; then
    attention_name=attention_prompt_last
    attention_scores=$base/methods/baselines/attention_prompt_last_1000.jsonl
    mirage_scores=$base/methods/baselines/mirage_gamma1_setting_q1000_scores.jsonl
    arc_scores=$base/methods/baselines/arc_jsd_projection_1000.jsonl
  fi
  mkdir -p "$work/scores"
  echo "[targeted-table3] dataset=$dataset targets=$count mode=$mode"

  "$PYTHON" scripts/prepare_targeted_table3_resume.py prepare \
    --manifest "$repair/resume/manifest.json" \
    --input "$base/retrieval/top10_1000.jsonl" \
    --units-cache "$base/inputs/token_units_top10_1000.jsonl" \
    --reflow-results "$repair/results_1000.jsonl" \
    --old-baselines "$audit/baselines_1000.jsonl" \
    --scores "$attention_name=$attention_scores" \
    --scores "gradient_x_input=$base/methods/baselines/gradient_x_input_top5_1000.jsonl" \
    --scores "integrated_gradients=$base/methods/baselines/integrated_gradients_top5_1000.jsonl" \
    --scores "mirage=$mirage_scores" \
    --scores "arc_jsd=$arc_scores" \
    --out-dir "$work" --n 1000

  local gradient_scores=$work/scores/gradient_x_input.jsonl
  local ig_scores=$work/scores/integrated_gradients.jsonl
  if [[ "$dataset" == hotpotqa ]]; then
    gradient_scores=$work/reranked/gradient_x_input.jsonl
    ig_scores=$work/reranked/integrated_gradients.jsonl
  fi

  if ! complete_jsonl "$work/baselines_v2.jsonl" "$count"; then
    CAUSALITYRAG_READER_MODE="$mode" "$PYTHON" \
      scripts/evaluate_matched_budget_baselines.py \
      --input "$work/input.jsonl" --units-cache "$work/units.jsonl" \
      --reflow-results "$work/reflow_results.jsonl" \
      --scores "$attention_name=$work/scores/$attention_name.jsonl" \
      --scores "gradient_x_input=$gradient_scores" \
      --scores "integrated_gradients=$ig_scores" \
      --scores "mirage=$work/scores/mirage.jsonl" \
      --scores "arc_jsd=$work/scores/arc_jsd.jsonl" \
      --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
      --out "$work/baselines_v2.jsonl" \
      --summary-out "$work/baselines_v2.summary.json" \
      --random-seeds 0,1,2,3,4 --replacement-seed 0 \
      --n "$count" --k 5 --workers 24 \
      --llm-base-url "$BASE_URL" --llm-model "$MODEL"
  fi

  "$PYTHON" scripts/prepare_targeted_table3_resume.py merge-baselines \
    --manifest "$repair/resume/manifest.json" \
    --old-baselines "$audit/baselines_1000.jsonl" \
    --resumed-baselines "$work/baselines_v2.jsonl" \
    --reflow-results "$repair/results_1000.jsonl" \
    --out "$work/baselines_1000.jsonl" \
    --summary-out "$work/baselines_1000.summary.json" --n 1000

  "$PYTHON" scripts/summarize_clean_anchor_flips.py \
    --reflow "$repair/results_1000.jsonl" \
    --baselines "$work/baselines_1000.jsonl" \
    --out "$work/factual_metrics_1000.json" --reader-mode "$mode"

  build_control_pool "$work" "$old_control_pool" "$mode" "$count"
  local control_sha
  control_sha=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["pool_sha256"])' \
    "$work/paraphrase_pool.manifest.json")
  if ! complete_jsonl "$work/controls_v2.jsonl" "$count"; then
    CAUSALITYRAG_READER_MODE="$mode" "$PYTHON" \
      scripts/evaluate_paraphrase_controls.py \
      --input "$work/input.jsonl" --units-cache "$work/units.jsonl" \
      --reflow-results "$work/reflow_results.jsonl" \
      --baseline-results "$work/baselines_v2.jsonl" \
      --paraphrase-pool "$work/paraphrase_pool.jsonl" \
      --expected-pool-sha256 "$control_sha" \
      --out "$work/controls_v2.jsonl" \
      --summary-out "$work/controls_v2.summary.json" \
      --n "$count" --k 5 --workers 16 --replacement-seed 0 \
      --llm-base-url "$BASE_URL" --llm-model "$MODEL"
  fi

  "$PYTHON" scripts/prepare_targeted_table3_resume.py merge-controls \
    --manifest "$repair/resume/manifest.json" \
    --old-controls "$old_control_results" \
    --resumed-controls "$work/controls_v2.jsonl" \
    --reflow-results "$repair/results_1000.jsonl" \
    --baseline-results "$work/baselines_1000.jsonl" \
    --reader-mode "$mode" --out "$work/controls_1000.jsonl" \
    --summary-out "$work/controls_1000.summary.json" --n 1000

  "$PYTHON" scripts/summarize_adjusted_flips.py \
    --factual "$work/factual_metrics_1000.json" \
    --meaning-preserving "$work/controls_1000.summary.json" \
    --out "$work/table3_metrics_1000.json"

  "$PYTHON" scripts/prepare_targeted_table3_resume.py audit \
    --manifest "$repair/resume/manifest.json" \
    --old-baselines "$audit/baselines_1000.jsonl" \
    --new-baselines "$work/baselines_1000.jsonl" \
    --old-controls "$old_control_results" \
    --new-controls "$work/controls_1000.jsonl" \
    --reflow-results "$repair/results_1000.jsonl" \
    --out "$work/audit.json" --n 1000
}

wait_for_popqa
prepare_hotpotqa_rerank_inputs
rerank_hotpotqa_empty_scores "$REPAIR_ROOT/hotpotqa/table3_resume"
start_readers

run_dataset hotpotqa 1 short_answer \
  "$SOURCE_ROOT/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl" \
  5f161ed2405becbda7cd39517d1d1291c562574a404c975e42f2d91ad8b31bef \
  "$SOURCE_ROOT/hotpotqa/controls/paraphrase_pool_top5_v2.jsonl" \
  "$SOURCE_ROOT/hotpotqa/controls/paraphrase_results_top5_1000_v2.jsonl"
run_dataset timeqa 64 short_answer \
  "$SOURCE_ROOT/timeqa/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  01f19f09fe2842006b036823ca820d649ce14fd23d41a74793f7f21f62fe8b44 \
  "$SOURCE_ROOT/timeqa/controls/paraphrase_pool_top5_v2.jsonl" \
  "$SOURCE_ROOT/timeqa/controls/paraphrase_results_top5_1000_v2.jsonl"
run_dataset finqa 74 short_answer \
  "$SOURCE_ROOT/finqa/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  e2276863d19c5c4f9148016eef4550128a4516baaa039e1e36e62db03ccdadb6 \
  "$SOURCE_ROOT/finqa/controls/paraphrase_pool_top5_v2.jsonl" \
  "$SOURCE_ROOT/finqa/controls/paraphrase_results_top5_1000_v2.jsonl"
run_dataset musique 27 short_answer \
  "$SOURCE_ROOT/musique/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  335bc1027e62c72973c97f57f4b20fc769601b16698411b42be2dd3319eac31e \
  "$SOURCE_ROOT/musique/controls/paraphrase_pool_top5_v2.jsonl" \
  "$SOURCE_ROOT/musique/controls/paraphrase_results_top5_1000_v2.jsonl"
run_dataset quartz 42 quartz \
  "$SOURCE_ROOT/quartz/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  724115a67b2b8af90f74cfaeea47270e0412f7250b75a66d421f232d6cd148e6 \
  "$SOURCE_ROOT/quartz/controls/paraphrase_pool_top5_v2.jsonl" \
  "$SOURCE_ROOT/quartz/controls/paraphrase_results_top5_1000_v2.jsonl"
run_dataset triviaqa 10 short_answer \
  "$SOURCE_ROOT/triviaqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl" \
  3703c4fa8e05a41b6d67f2dc247705d14b410867f56fe6ad8cf918c521d14d0f \
  "$SOURCE_ROOT/triviaqa/controls/paraphrase_pool_top5_v1.jsonl" \
  "$SOURCE_ROOT/triviaqa/controls/paraphrase_results_top5_1000.jsonl"
run_dataset 2wiki 11 short_answer \
  "$SOURCE_ROOT/2wiki/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  968488f69f032fbea66405b385fe8c227ce92c018632812d6d6486ddfe2be19d \
  "$SOURCE_ROOT/2wiki/controls/paraphrase_pool_top5_v2.jsonl" \
  "$SOURCE_ROOT/2wiki/controls/paraphrase_results_top5_1000_v2.jsonl"

date -Is >"$REPAIR_ROOT/TABLE3_COMPLETE"
echo "[targeted-table3] complete root=$REPAIR_ROOT"
