#!/usr/bin/env bash
set -euo pipefail

RELEASE=/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810
SOURCE_ROOT=/data1/yujia/CausalityRAG/out
REPAIR_ROOT=/data1/yujia/CausalityRAG/out_no_frontier_repair_20260810
BASE_URL=http://127.0.0.1:8000/v1
MODEL=qwen2.5-7b

run_dataset() {
  local dataset=$1
  local count=$2
  local reader_mode=$3
  local pool=$4
  local pool_sha=$5
  local old_results=$6
  local source_dir="${SOURCE_ROOT}/${dataset}"
  local repair_dir="${REPAIR_ROOT}/${dataset}"
  local resumed_results="${repair_dir}/resume/results.jsonl"

  if [[ -e "${resumed_results}" ]]; then
    echo "Refusing to append to existing output: ${resumed_results}" >&2
    return 2
  fi
  echo "[no-frontier-reader] ${dataset} targets=${count} mode=${reader_mode}"
  CAUSALITYRAG_READER_MODE="${reader_mode}" python3 \
    "${RELEASE}/scripts/evaluate_reflow.py" \
    --input "${repair_dir}/resume/input.jsonl" \
    --units-cache "${repair_dir}/resume/units.jsonl" \
    --frontier "${repair_dir}/resume/frontier.jsonl" \
    --shared-pool "${pool}" \
    --expected-pool-sha256 "${pool_sha}" \
    --out "${resumed_results}" \
    --summary-out "${repair_dir}/resume/results.summary.json" \
    --n "${count}" \
    --k 5 \
    --workers 2 \
    --replacement-seed 0 \
    --llm-base-url "${BASE_URL}" \
    --llm-model "${MODEL}"

  python3 "${RELEASE}/scripts/prepare_no_frontier_resume.py" merge-results \
    --manifest "${repair_dir}/resume/manifest.json" \
    --old-results "${old_results}" \
    --resumed-results "${resumed_results}" \
    --out "${repair_dir}/results_1000.jsonl" \
    --summary-out "${repair_dir}/results_1000.summary.json"
}

run_dataset \
  timeqa 64 short_answer \
  "${SOURCE_ROOT}/timeqa/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  01f19f09fe2842006b036823ca820d649ce14fd23d41a74793f7f21f62fe8b44 \
  "${SOURCE_ROOT}/timeqa/methods/reflow/results_top5_1000.jsonl"

run_dataset \
  finqa 74 short_answer \
  "${SOURCE_ROOT}/finqa/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  e2276863d19c5c4f9148016eef4550128a4516baaa039e1e36e62db03ccdadb6 \
  "${SOURCE_ROOT}/finqa/methods/reflow/results_top5_1000.jsonl"

run_dataset \
  musique 27 short_answer \
  "${SOURCE_ROOT}/musique/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  335bc1027e62c72973c97f57f4b20fc769601b16698411b42be2dd3319eac31e \
  "${SOURCE_ROOT}/musique/methods/reflow/results_top5_1000.jsonl"

run_dataset \
  quartz 42 quartz \
  "${SOURCE_ROOT}/quartz/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  724115a67b2b8af90f74cfaeea47270e0412f7250b75a66d421f232d6cd148e6 \
  "${SOURCE_ROOT}/quartz/methods/reflow/results_top5_1000.jsonl"

run_dataset \
  triviaqa 10 short_answer \
  "${SOURCE_ROOT}/triviaqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl" \
  3703c4fa8e05a41b6d67f2dc247705d14b410867f56fe6ad8cf918c521d14d0f \
  "${SOURCE_ROOT}/triviaqa/methods/reflow/results_top5_1000.jsonl"

run_dataset \
  2wiki 11 short_answer \
  "${SOURCE_ROOT}/2wiki/replacements/shared_pool_top10_v1/shared_pool.jsonl" \
  968488f69f032fbea66405b385fe8c227ce92c018632812d6d6486ddfe2be19d \
  "${SOURCE_ROOT}/2wiki/methods/reflow/results_top5_1000.jsonl"

echo "[no-frontier-reader] complete"
