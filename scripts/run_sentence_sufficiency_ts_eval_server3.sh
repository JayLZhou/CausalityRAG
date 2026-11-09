#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "3haoji-A100" ]] || {
  echo "This runner is restricted to server3." >&2
  exit 1
}

SOURCE=/data1/yujia/CausalityRAG
ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
OUT="$ROOT/out/sentence_sufficiency_ts_v1"
IMPORT="$OUT/import_server2"
EVAL="$OUT/eval"
LOG="$OUT/logs"
PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python

mkdir -p "$EVAL/hotpotqa" "$EVAL/musique" "$LOG"
export PYTHONPATH="$ROOT"
export CAUSALITYRAG_READER_MODE=short_answer

if ! curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models \
  | grep -q 'qwen2.5-7b'; then
  echo "The existing qwen2.5-7b reader on port 8001 is unavailable." >&2
  exit 1
fi

evaluate() {
  local dataset=$1 retrieval=$2 units=$3 reflow=$4 mirage=$5 arc=$6 expected=$7
  local output="$EVAL/$dataset/per_query.jsonl"
  local summary="$EVAL/$dataset/summary.json"
  "$PYTHON" "$ROOT/scripts/run_sentence_sufficiency_ts.py" \
    --retrieval "$retrieval" --units "$units" --reflow "$reflow" \
    --mirage "$mirage" --arc-jsd "$arc" \
    --token-shapley "$IMPORT/$dataset"/shard_{0,1,2,3}.jsonl \
    --out "$output" --summary-out "$summary" \
    --workers 16 --max-tokens 64 \
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

evaluate hotpotqa \
  "$SOURCE/out/hotpotqa/retrieval/top10_1000.jsonl" \
  "$SOURCE/out/hotpotqa/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE/out/hotpotqa/audits/final_top10pool_k5/reflow_1000_v2.jsonl" \
  "$SOURCE/out/hotpotqa/methods/baselines/mirage_gamma1_setting_q1000_scores.jsonl" \
  "$SOURCE/out/hotpotqa/methods/baselines/arc_jsd_projection_1000.jsonl" 985 &
hqa_pid=$!

evaluate musique \
  "$SOURCE/out/musique/retrieval/top10_1000.jsonl" \
  "$SOURCE/out/musique/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE/out/musique/methods/reflow/results_top5_1000.jsonl" \
  "$SOURCE/out/musique/methods/baselines/mirage_top5_1000.jsonl" \
  "$SOURCE/out/musique/methods/baselines/arc_jsd_top5_1000.jsonl" 938 &
msq_pid=$!

status=0
wait "$hqa_pid" || status=1
wait "$msq_pid" || status=1
(( status == 0 ))

sha256sum \
  "$IMPORT/hotpotqa"/shard_{0,1,2,3}.jsonl \
  "$IMPORT/musique"/shard_{0,1,2,3}.jsonl \
  "$EVAL/hotpotqa/per_query.jsonl" "$EVAL/hotpotqa/summary.json" \
  "$EVAL/musique/per_query.jsonl" "$EVAL/musique/summary.json" \
  >"$OUT/SHA256SUMS"
touch "$OUT/COMPLETE"
echo "[done] $(date -Is) HQA/MSQ Token Shapley provenance evaluation complete"
