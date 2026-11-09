#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "3haoji-A100" ]] || {
  echo "This runner is restricted to server3." >&2
  exit 1
}

SOURCE=/data1/yujia/CausalityRAG
ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
DATA=/data1/yujia/shapley_rank_bundle
PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python
MODEL=/data1/yujia/models/Qwen2.5-7B-Instruct
OUT="$ROOT/out/sentence_sufficiency_ts_v1"
RANK="$OUT/rank"
EVAL="$OUT/eval"
LOG="$OUT/logs"
GPU=1

mkdir -p "$RANK/hotpotqa" "$RANK/musique" "$EVAL" "$LOG"
export PYTHONPATH="$ROOT"
export CAUSALITYRAG_READER_MODE=short_answer

if ! curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models \
  | grep -q 'qwen2.5-7b'; then
  echo "The existing qwen2.5-7b reader on port 8001 is unavailable." >&2
  exit 1
fi

free_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.free \
  --format=csv,noheader,nounits | tr -d ' ')
if (( free_mib < 32000 )); then
  echo "GPU${GPU} has only ${free_mib} MiB free; refusing to disturb other jobs." >&2
  exit 1
fi

rank_dataset() {
  local dataset=$1 input=$2 clean=$3 units=$4 batch=$5
  local output="$RANK/$dataset/scores_1000.jsonl"
  local summary="$RANK/$dataset/summary.json"
  echo "[rank] $(date -Is) dataset=$dataset existing=$(test -f "$output" && wc -l < "$output" || echo 0)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/exp/run_token_leverage_shap.py" \
    --input "$input" \
    --clean-reference "$clean" \
    --context-units "$units" \
    --model-path "$MODEL" \
    --out "$output" \
    --summary-out "$summary" \
    --start 0 --n 1000 --k 5 --top-k 128 \
    --attention-candidates 128 --budget-multiplier 3 \
    --coalition-batch-size "$batch" --seed 42 --resume \
    >"$LOG/rank_${dataset}.log" 2>&1
  "$PYTHON" - "$input" "$output" "$summary" <<'PY'
import json, sys
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]

source, output, summary = map(Path, sys.argv[1:])
src = rows(source)
out = rows(output)
src_ids = [str(row.get("id", "")) for row in src]
out_ids = [str(row.get("id", "")) for row in out]
assert len(src) == len(out) == 1000, (len(src), len(out))
assert len(set(out_ids)) == 1000
assert set(src_ids) == set(out_ids)
statuses = {}
for row in out:
    status = str(row.get("status"))
    statuses[status] = statuses.get(status, 0) + 1
meta = json.loads(summary.read_text())
assert meta["queries"] == 1000
print(
    f"[rank-audit] rows=1000 unique=1000 statuses={statuses} "
    f"summary_queries={meta['queries']}"
)
PY
}

evaluate_dataset() {
  local dataset=$1 retrieval=$2 units=$3 reflow=$4 mirage=$5 arc=$6 expected=$7
  local output="$EVAL/$dataset/per_query.jsonl"
  local summary="$EVAL/$dataset/summary.json"
  mkdir -p "$EVAL/$dataset"
  "$PYTHON" "$ROOT/scripts/run_sentence_sufficiency_ts.py" \
    --retrieval "$retrieval" --units "$units" --reflow "$reflow" \
    --mirage "$mirage" --arc-jsd "$arc" \
    --token-shapley "$RANK/$dataset/scores_1000.jsonl" \
    --out "$output" --summary-out "$summary" \
    --workers 24 --max-tokens 64 \
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
print(f"[eval-audit] rows={expected} unique={expected} reader_failures=0")
PY
}

rank_dataset hotpotqa \
  "$SOURCE/out/hotpotqa/retrieval/top10_1000.jsonl" \
  "$SOURCE/out/hotpotqa/inputs/clean_reference_1000.jsonl" \
  "$SOURCE/out/hotpotqa/inputs/token_units_top5_1000.jsonl" 8

evaluate_dataset hotpotqa \
  "$SOURCE/out/hotpotqa/retrieval/top10_1000.jsonl" \
  "$SOURCE/out/hotpotqa/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE/out/hotpotqa/audits/final_top10pool_k5/reflow_1000_v2.jsonl" \
  "$SOURCE/out/hotpotqa/methods/baselines/mirage_gamma1_setting_q1000_scores.jsonl" \
  "$SOURCE/out/hotpotqa/methods/baselines/arc_jsd_projection_1000.jsonl" 985 &
hqa_eval_pid=$!

rank_dataset musique \
  "$DATA/musique/input.jsonl" "$DATA/musique/clean.jsonl" \
  "$DATA/musique/units.jsonl" 32

wait "$hqa_eval_pid"
evaluate_dataset musique \
  "$SOURCE/out/musique/retrieval/top10_1000.jsonl" \
  "$SOURCE/out/musique/inputs/token_units_top10_1000.jsonl" \
  "$SOURCE/out/musique/methods/reflow/results_top5_1000.jsonl" \
  "$SOURCE/out/musique/methods/baselines/mirage_top5_1000.jsonl" \
  "$SOURCE/out/musique/methods/baselines/arc_jsd_top5_1000.jsonl" 938

sha256sum \
  "$RANK/hotpotqa/scores_1000.jsonl" "$RANK/musique/scores_1000.jsonl" \
  "$EVAL/hotpotqa/per_query.jsonl" "$EVAL/hotpotqa/summary.json" \
  "$EVAL/musique/per_query.jsonl" "$EVAL/musique/summary.json" \
  >"$OUT/SHA256SUMS"
touch "$OUT/COMPLETE"
echo "[done] $(date -Is) HQA/MSQ Token Shapley provenance complete"
