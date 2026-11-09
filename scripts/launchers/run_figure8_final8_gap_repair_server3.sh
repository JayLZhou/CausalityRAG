#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_popqa_models_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
VLLM_COMPAT_ROOT=${VLLM_COMPAT_ROOT:-/data1/yujia/CausalityRAG_release_popqa_20260810/scripts/compat/transformers_452}
SOURCE_ROOT=${SOURCE_ROOT:-/data1/yujia/CausalityRAG/out}
FINAL8_ROOT=${FINAL8_ROOT:-/data1/yujia/CausalityRAG/out_table3_final8_20260811}
OUT=${OUT:-/data1/yujia/CausalityRAG/out_figure8_final8_20260811/gap_repair}
MODEL_PATH=${MODEL_PATH:-/data1/yujia/models/Qwen2.5-7B-Instruct}
MODEL=${MODEL:-qwen2.5-7b}
LOG_ROOT=$OUT/logs
PID_ROOT=$OUT/pids
mkdir -p "$LOG_ROOT" "$PID_ROOT"
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
  rm -f "$pid_file"
}

stop_readers() {
  stop_owned_group "$PID_ROOT/reader_8002.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/reader_8003.pid" vllm.entrypoints.openai.api_server
  stop_owned_group "$PID_ROOT/proxy_8000.pid" lb_proxy_two_gpu.py
}
trap stop_readers EXIT

while pgrep -u "$USER" -f run_popqa_cross_model_sweep_server3.sh >/dev/null; do
  echo "[figure8-gap-repair] waiting for healthy PopQA model sweep"
  sleep 60
done

for port in 8000 8002 8003; do
  if ! port_is_free "$port"; then
    echo "refusing to interrupt unowned service on port $port" >&2
    exit 1
  fi
done
if port_is_free 8021; then
  echo "required existing spaCy service is not listening on 8021" >&2
  exit 1
fi

for spec in '0 8002' '1 8003'; do
  read -r gpu port <<<"$spec"
  PYTHONPATH="$VLLM_COMPAT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$MODEL" \
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

repair_method() {
  local dataset=$1 method=$2 budgets=$3 units=$4 frontier=$5 scores=$6 pool=$7 pool_sha=$8
  local allow_absent=${9:-false}
  local expanded_budgets
  expanded_budgets=$("$PYTHON" -c \
    'import sys; from scripts.materialize_figure8_seed import parse_budget_values; print(",".join(map(str, parse_budget_values(sys.argv[1]))))' \
    "$budgets")
  local tag=${budgets//,/_}
  tag=${tag//-/_}
  local dataset_root=$SOURCE_ROOT/$dataset
  local repair_dir=$dataset_root/audits/baseline_budget_to_reflow/final8_repair
  local work_dir=$OUT/$dataset/$method
  local seed=$work_dir/seed_${tag}.jsonl
  local rows=$repair_dir/${method}_alias_gap_${tag}_1000.jsonl
  local summary=$repair_dir/${method}_alias_gap_${tag}_1000.summary.json
  mkdir -p "$repair_dir" "$work_dir"
  LAST_SUMMARY=$summary
  if [[ -s "$summary" ]] && [[ -s "$rows" ]] && \
    [[ "$(wc -l < "$rows")" -eq 1000 ]]; then
    echo "[figure8-gap-repair] reuse dataset=$dataset method=$method budgets=$expanded_budgets"
    return 0
  fi

  local seed_args=()
  if [[ "$allow_absent" == true ]]; then
    seed_args+=(--allow-absent)
  fi
  "$PYTHON" scripts/materialize_figure8_seed.py \
    --dataset-root "$dataset_root" \
    --input "$FINAL8_ROOT/$dataset/retrieval/top10_1000.jsonl" \
    --reflow-results "$FINAL8_ROOT/$dataset/methods/reflow/results_top5_1000.jsonl" \
    --points "$method=$budgets" --n 1000 \
    --out "$seed" --summary-out "$work_dir/seed_${tag}.summary.json" \
    "${seed_args[@]}" \
    >"$LOG_ROOT/${dataset}_${method}_seed.log" 2>&1

  CAUSALITYRAG_READER_MODE=short_answer \
    CAUSALITYRAG_READER_MAX_TOKENS=128 \
    "$PYTHON" scripts/evaluate_fixed_budget_acc.py \
    --input "$FINAL8_ROOT/$dataset/retrieval/top10_1000.jsonl" \
    --units-cache "$units" --frontier "$frontier" \
    --scores "$method=$scores" \
    --shared-pool "$pool" --expected-pool-sha256 "$pool_sha" \
    --out "$rows" --summary-out "$summary" --budgets "$expanded_budgets" \
    --seed-results "$seed" --skip-reflow --replacement-seed 0 \
    --n 1000 --k 5 --workers 48 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$MODEL" \
    --reader-max-tokens 128 >"$LOG_ROOT/${dataset}_${method}_repair.log" 2>&1
  [[ "$(wc -l < "$rows")" -eq 1000 ]]
}

first_crossing() {
  local summary=$1 method=$2 target=$3
  "$PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); m=sys.argv[2]; t=float(sys.argv[3]); pts=sorted((int(k),v) for k,v in d["methods"][m].items()); print(next((k for k,v in pts if float(v["acc_cfr"]) >= t), ""))' \
    "$summary" "$method" "$target"
}

repair_until_crossing() {
  local dataset=$1 method=$2 start=$3 maximum=$4 block=$5 target=$6
  local units=$7 frontier=$8 scores=$9 pool=${10} pool_sha=${11}
  local end crossing
  while (( start <= maximum )); do
    end=$((start + block - 1))
    (( end > maximum )) && end=$maximum
    repair_method "$dataset" "$method" "$start-$end" \
      "$units" "$frontier" "$scores" "$pool" "$pool_sha" true
    crossing=$(first_crossing "$LAST_SUMMARY" "$method" "$target")
    if [[ -n "$crossing" ]]; then
      echo "[figure8-gap-repair] crossing dataset=$dataset method=$method budget=$crossing"
      return 0
    fi
    start=$((end + 1))
  done
  echo "[figure8-gap-repair] not_reached dataset=$dataset method=$method max_budget=$maximum"
}

repair_method timeqa mirage 1-9 \
  "$SOURCE_ROOT/timeqa/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/timeqa/methods/reflow/frontier_top5_1000.jsonl" \
  "$SOURCE_ROOT/timeqa/methods/baselines/mirage_top5_1000.jsonl" \
  "$SOURCE_ROOT/timeqa/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  01f19f09fe2842006b036823ca820d649ce14fd23d41a74793f7f21f62fe8b44
repair_method timeqa arc_jsd 1-9 \
  "$SOURCE_ROOT/timeqa/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/timeqa/methods/reflow/frontier_top5_1000.jsonl" \
  "$SOURCE_ROOT/timeqa/methods/baselines/arc_jsd_top5_1000.jsonl" \
  "$SOURCE_ROOT/timeqa/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  01f19f09fe2842006b036823ca820d649ce14fd23d41a74793f7f21f62fe8b44

repair_method musique mirage 1-24 \
  "$SOURCE_ROOT/musique/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/musique/methods/reflow/frontier_top5_1000.jsonl" \
  "$SOURCE_ROOT/musique/methods/baselines/mirage_top5_1000.jsonl" \
  "$SOURCE_ROOT/musique/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  335bc1027e62c72973c97f57f4b20fc769601b16698411b42be2dd3319eac31e
repair_method musique arc_jsd 1-16 \
  "$SOURCE_ROOT/musique/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/musique/methods/reflow/frontier_top5_1000.jsonl" \
  "$SOURCE_ROOT/musique/methods/baselines/arc_jsd_top5_1000.jsonl" \
  "$SOURCE_ROOT/musique/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  335bc1027e62c72973c97f57f4b20fc769601b16698411b42be2dd3319eac31e

repair_method triviaqa mirage 1,2,4,8,16,32,64,128 \
  "$SOURCE_ROOT/triviaqa/inputs/context_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/triviaqa/methods/reflow/frontier_top5_1000.jsonl" \
  "$SOURCE_ROOT/triviaqa/methods/baselines/mirage_top5_1000.jsonl" \
  "$SOURCE_ROOT/triviaqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl" \
  3703c4fa8e05a41b6d67f2dc247705d14b410867f56fe6ad8cf918c521d14d0f
repair_method triviaqa arc_jsd 1,2,4,8,16,32,64,128 \
  "$SOURCE_ROOT/triviaqa/inputs/context_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/triviaqa/methods/reflow/frontier_top5_1000.jsonl" \
  "$SOURCE_ROOT/triviaqa/methods/baselines/arc_jsd_top5_1000.jsonl" \
  "$SOURCE_ROOT/triviaqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl" \
  3703c4fa8e05a41b6d67f2dc247705d14b410867f56fe6ad8cf918c521d14d0f

repair_until_crossing hotpotqa mirage 89 128 8 0.8521256931608133 \
  "$SOURCE_ROOT/hotpotqa/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/hotpotqa/audits/final_top10pool_k5/reflow_frontier_1000.jsonl" \
  "$SOURCE_ROOT/hotpotqa/methods/baselines/mirage_gamma1_setting_q1000_scores.jsonl" \
  "$SOURCE_ROOT/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl" \
  5f161ed2405becbda7cd39517d1d1291c562574a404c975e42f2d91ad8b31bef
repair_until_crossing hotpotqa arc_jsd 102 128 8 0.8521256931608133 \
  "$SOURCE_ROOT/hotpotqa/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/hotpotqa/audits/final_top10pool_k5/reflow_frontier_1000.jsonl" \
  "$SOURCE_ROOT/hotpotqa/methods/baselines/arc_jsd_projection_1000.jsonl" \
  "$SOURCE_ROOT/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl" \
  5f161ed2405becbda7cd39517d1d1291c562574a404c975e42f2d91ad8b31bef

repair_until_crossing 2wiki mirage 33 128 8 0.8816568047337278 \
  "$SOURCE_ROOT/2wiki/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/2wiki/methods/reflow/frontier_top5_1000.jsonl" \
  "$SOURCE_ROOT/2wiki/methods/baselines/mirage_top5_1000.jsonl" \
  "$SOURCE_ROOT/2wiki/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  968488f69f032fbea66405b385fe8c227ce92c018632812d6d6486ddfe2be19d
repair_until_crossing 2wiki arc_jsd 65 128 8 0.8816568047337278 \
  "$SOURCE_ROOT/2wiki/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE_ROOT/2wiki/methods/reflow/frontier_top5_1000.jsonl" \
  "$SOURCE_ROOT/2wiki/methods/baselines/arc_jsd_top5_1000.jsonl" \
  "$SOURCE_ROOT/2wiki/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  968488f69f032fbea66405b385fe8c227ce92c018632812d6d6486ddfe2be19d

POPQA_SOURCE=/data1/yujia/CausalityRAG/out_popqa_20260810/popqa
POPQA_REPAIRED=/data1/yujia/CausalityRAG/out_popqa_no_frontier_repair_20260811/popqa
POPQA_FIGURE8=/data1/yujia/CausalityRAG/out_popqa_figures_20260811/popqa/figure8
POPQA_ROWS=$POPQA_FIGURE8/exact_k1_32_1000.jsonl
POPQA_SUMMARY=$POPQA_FIGURE8/exact_k1_32_1000.summary.json
if [[ -s "$POPQA_SUMMARY" ]] && [[ -s "$POPQA_ROWS" ]] && \
  [[ "$(wc -l < "$POPQA_ROWS")" -eq 1000 ]]; then
  echo "[figure8-gap-repair] reuse dataset=popqa budgets=1..32"
else
  CAUSALITYRAG_READER_MODE=short_answer \
    CAUSALITYRAG_READER_MAX_TOKENS=128 \
    "$PYTHON" scripts/evaluate_fixed_budget_acc.py \
    --input "$POPQA_SOURCE/retrieval/top10_1000.jsonl" \
    --units-cache "$POPQA_SOURCE/inputs/token_units_top10_1000.jsonl" \
    --frontier "$POPQA_REPAIRED/frontier_1000.jsonl" \
    --scores "mirage=$POPQA_SOURCE/methods/baselines/mirage_top5_1000.jsonl" \
    --scores "arc_jsd=$POPQA_SOURCE/methods/baselines/arc_jsd_top5_1000.jsonl" \
    --shared-pool "$POPQA_SOURCE/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
    --expected-pool-sha256 42dd41818cdefac31ce788a1bf13402b07353ed032806c9a874f5fb8b19313e7 \
    --out "$POPQA_ROWS" --summary-out "$POPQA_SUMMARY" \
    --budgets "$(seq -s, 1 32)" \
    --seed-results "$POPQA_FIGURE8/coarse_1000.jsonl" --skip-reflow \
    --replacement-seed 0 --n 1000 --k 5 --workers 48 \
    --llm-base-url http://127.0.0.1:8000/v1 --llm-model "$MODEL" \
    --reader-max-tokens 128 >"$LOG_ROOT/popqa_exact_k1_32.log" 2>&1
  [[ "$(wc -l < "$POPQA_ROWS")" -eq 1000 ]]
fi

stop_readers

RESCORE_OUT=$OUT/final_rescore_phase1
mkdir -p "$RESCORE_OUT"
for dataset in hotpotqa timeqa musique triviaqa 2wiki; do
  "$PYTHON" scripts/rescore_figure8_dataset.py \
    --dataset "$dataset" --dataset-root "$SOURCE_ROOT/$dataset" \
    --input "$FINAL8_ROOT/$dataset/retrieval/top10_1000.jsonl" \
    --reflow-results "$FINAL8_ROOT/$dataset/methods/reflow/results_top5_1000.jsonl" \
    --allow-incomplete --out "$RESCORE_OUT/$dataset.json" \
    >"$LOG_ROOT/${dataset}_final_rescore.log" 2>&1
done

date -Is >"$OUT/PHASE1_COMPLETE"
date -Is >"$OUT/COMPLETE"
echo "[figure8-gap-repair] complete"
