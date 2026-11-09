#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_popqa_models_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/data1/yujia/CausalityRAG/out}
FINAL8_ROOT=${FINAL8_ROOT:-/data1/yujia/CausalityRAG/out_table3_final8_20260811}
OUT=${OUT:-/data1/yujia/CausalityRAG/out_figure8_final8_20260811/final_rescore}
mkdir -p "$OUT"
cd "$RELEASE"

datasets=(hotpotqa timeqa finqa musique triviaqa 2wiki)
for dataset in "${datasets[@]}"; do
  output="$OUT/${dataset}.json"
  if [[ -s "$output" ]] && "$PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("dataset")==sys.argv[2] and d.get("complete") is True else 1)' \
    "$output" "$dataset"; then
    echo "[figure8-rescore] dataset=$dataset already complete"
    continue
  fi
  echo "[figure8-rescore] dataset=$dataset"
  "$PYTHON" scripts/rescore_figure8_dataset.py \
    --dataset "$dataset" \
    --dataset-root "$SOURCE_ROOT/$dataset" \
    --input "$FINAL8_ROOT/$dataset/retrieval/top10_1000.jsonl" \
    --reflow-results "$FINAL8_ROOT/$dataset/methods/reflow/results_top5_1000.jsonl" \
    --allow-incomplete --out "$output" >"$OUT/${dataset}.log" 2>&1
done

if "$PYTHON" - "$OUT" "${datasets[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
datasets = sys.argv[2:]
complete = all(json.loads((root / f"{dataset}.json").read_text())["complete"] for dataset in datasets)
raise SystemExit(0 if complete else 1)
PY
then
  date -Is >"$OUT/RESCORE_COMPLETE"
  echo "[figure8-rescore] complete root=$OUT"
else
  rm -f "$OUT/RESCORE_COMPLETE"
  echo "[figure8-rescore] audit complete with targeted gaps root=$OUT"
fi
