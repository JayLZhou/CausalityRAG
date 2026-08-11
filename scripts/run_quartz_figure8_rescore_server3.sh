#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_popqa_figures_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
SOURCE=${SOURCE:-/data1/yujia/CausalityRAG/out/quartz/audits/baseline_budget_to_reflow}
OUT=${OUT:-/data1/yujia/CausalityRAG/out_figure8_final8_20260811/quartz_final_anchor}
INPUT=/data1/yujia/CausalityRAG/out/quartz/retrieval/top10_1000.jsonl
CLEAN=${CLEAN:-/data1/yujia/CausalityRAG/out_no_frontier_repair_20260810/quartz/results_1000.jsonl}

mkdir -p "$OUT"
args=()
for path in "$SOURCE"/exact_points_v2/arc_jsd_*.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k1_8_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k9_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k22_24_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k25_32_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k33_40_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k41_48_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k49_56_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k57_64_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k65_72_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k73_80_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k81_88_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k89_96_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k97_104_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k105_112_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k113_120_1000.jsonl \
  "$SOURCE"/exact_points_v2/mirage_k121_128_1000.jsonl; do
  args+=(--results "$path")
done
for budget in {10..21}; do
  args+=(--results "$SOURCE/exact_points/mirage_k${budget}_1000.jsonl")
done

cd "$RELEASE"
"$PYTHON" scripts/rescore_fixed_budget_results.py \
  --input "$INPUT" --clean-results "$CLEAN" "${args[@]}" \
  --reader-mode quartz --n 1000 --out "$OUT/results_1000.jsonl" \
  --summary-out "$OUT/summary_1000.json"
