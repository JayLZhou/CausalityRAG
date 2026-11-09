#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "3haoji-A100" ]] || {
  echo "This runner is restricted to server3." >&2
  exit 1
}

SOURCE=/data1/yujia/CausalityRAG
POPQA_SOURCE=/data1/yujia/CausalityRAG/out_popqa_20260810/popqa
ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
OUT="$ROOT/out/sentence_sufficiency_ts_appendix_v1"
IMPORT="$OUT/import_server2"
EVAL="$OUT/eval"
LOG="$OUT/logs"
PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python

DATASETS=(2wiki timeqa finqa quartz triviaqa popqa)
declare -A EXPECTED=(
  [2wiki]=983
  [timeqa]=931
  [finqa]=918
  [quartz]=958
  [triviaqa]=969
  [popqa]=992
)

mkdir -p "$EVAL" "$LOG"
for dataset in "${DATASETS[@]}"; do
  mkdir -p "$EVAL/$dataset"
done

export PYTHONPATH="$ROOT"
export CAUSALITYRAG_READER_MODE=short_answer

if ! curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models \
  | grep -q 'qwen2.5-7b'; then
  echo "The existing qwen2.5-7b reader on port 8001 is unavailable." >&2
  exit 1
fi
if ! ps -p 1483747 -o cmd= | grep -q 'spacy_annotation_server.py --port 8021'; then
  echo "Protected spaCy PID 1483747 is unavailable; refusing to proceed." >&2
  exit 1
fi

evaluate() {
  local dataset=$1 source_root=$2 expected=${EXPECTED[$1]}
  local output="$EVAL/$dataset/per_query.jsonl"
  local summary="$EVAL/$dataset/summary.json"

  "$PYTHON" "$ROOT/scripts/run_sentence_sufficiency_ts.py" \
    --retrieval "$source_root/retrieval/top10_1000.jsonl" \
    --units "$source_root/inputs/token_units_top10_1000.jsonl" \
    --reflow "$source_root/methods/reflow/results_top5_1000.jsonl" \
    --mirage "$source_root/methods/baselines/mirage_top5_1000.jsonl" \
    --arc-jsd "$source_root/methods/baselines/arc_jsd_top5_1000.jsonl" \
    --token-shapley "$IMPORT/$dataset"/shard_*.jsonl \
    --out "$output" --summary-out "$summary" \
    --workers 8 --max-tokens 64 \
    --llm-base-url http://127.0.0.1:8001/v1 --llm-model qwen2.5-7b \
    >"$LOG/eval_${dataset}.log" 2>&1

  "$PYTHON" - "$output" "$summary" "$expected" <<'PY'
import json, sys
from pathlib import Path

output, summary, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in output.open() if line.strip()]
meta = json.loads(summary.read_text())
ids = [str(row.get("id", "")) for row in rows]
assert len(rows) == len(set(ids)) == expected
assert meta["queries"] == expected
assert meta["population"]["included_queries"] == expected
assert meta["alignment"]["aligned_queries"] == expected
print(f"[eval-audit] dataset={output.parent.name} rows={expected} unique={expected}")
PY
}

declare -a pids=()
for dataset in "${DATASETS[@]}"; do
  source_root="$SOURCE/out/$dataset"
  [[ "$dataset" == popqa ]] && source_root="$POPQA_SOURCE"
  evaluate "$dataset" "$source_root" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 ))

{
  for dataset in "${DATASETS[@]}"; do
    sha256sum "$IMPORT/$dataset"/shard_*.jsonl
    sha256sum "$EVAL/$dataset/per_query.jsonl" "$EVAL/$dataset/summary.json"
  done
} >"$OUT/SHA256SUMS"

touch "$OUT/COMPLETE"
echo "[done] $(date -Is) Figure 20 Token Shapley provenance evaluation complete"
