#!/usr/bin/env bash
# Rerun the Table 3 reader stages (ReFlow verification, matched-budget baselines,
# meaning-preserving controls, adjusted flips) with one counterfactual pool swapped in.
# Mirrors CausalityRAG_metric_contract_20260811/scripts/run_table3_reader_stage.py.
# Usage: run_pool_sensitivity.sh DATASET TAG POOL_JSONL READER_PORT
set -euo pipefail
[[ "$(hostname)" == "3haoji-A100" ]] || exit 1
DS=$1 TAG=$2 POOL=$3 PORT=$4
REPO=/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810
case "$DS" in
  popqa)  # PopQA's Table 3 run (standard layout, no resume pass)
    DATA=/data1/yujia/CausalityRAG/out_table3_final8_20260811/popqa
    MAIN=$DATA/methods/reflow; FRONTIER=$MAIN/frontier_top5_1000.jsonl ;;
  *)
    DATA=/data1/yujia/CausalityRAG/out/$DS
    MAIN=/data1/yujia/CausalityRAG/out_no_frontier_repair_20260810/$DS; FRONTIER=$MAIN/frontier_1000.jsonl ;;
esac
OUT=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/$TAG/$DS
# Same interpreter as the Table 3 runs (run_no_frontier_table3_resume_server3.sh); it has nltk/WordNet.
PY=/data1/yujia/envs/graphrag/bin/python
# Reader HTTP wait: 30 min by default, since a few edited contexts make the reader
# generate up to the context limit; decoding itself is unchanged.
READER_TIMEOUT=${READER_TIMEOUT:-1800}
RUNPY=("$PY" /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity/scripts/run_with_reader_timeout.py "$READER_TIMEOUT")
# A stage whose summary is missing is rerun from scratch; keep any partial output aside.
stash_partial() { [[ -s $1.jsonl && ! -s $1.summary.json ]] && mv "$1.jsonl" "$1.partial_$(date +%s).jsonl" || true; }
URL=http://127.0.0.1:$PORT/v1 MODEL=qwen2.5-7b
# Optional: REPL_SEED picks which valid counterfactual alternative each token gets
# (A7c); the meaning-preserving control keeps seed 0.
REPL_SEED=${REPL_SEED:-0}
case "$DS" in quartz) MODE=quartz ;; *) MODE=short_answer ;; esac
export CAUSALITYRAG_READER_MODE=$MODE PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/scores" "$OUT/controls"
RET=$DATA/retrieval/top10_1000.jsonl
UNITS=$DATA/inputs/token_units_top10_1000.jsonl
[[ $DS == triviaqa ]] && UNITS=$DATA/inputs/context_units_top10_1000.jsonl
SHA=$(sha256sum "$POOL" | cut -d' ' -f1)
echo "pool $POOL sha $SHA" | tee "$OUT/pool.txt"
cd "$REPO"

# Baseline score files that reproduce the main-table selections, with the
# per-query rows that the Table 3 resume recomputed substituted in.
declare -A SRC
if [[ $DS == hotpotqa ]]; then
  SRC=([attention_prompt_last]=attention_prompt_last_1000 [gradient_x_input]=gradient_x_input_top5_1000
       [integrated_gradients]=integrated_gradients_top5_1000 [mirage]=mirage_gamma1_setting_q1000_scores
       [arc_jsd]=arc_jsd_projection_1000)
else
  SRC=([attention]=attention_top5_1000 [gradient_x_input]=gradient_x_input_top5_1000
       [integrated_gradients]=integrated_gradients_top5_1000 [mirage]=mirage_top5_1000 [arc_jsd]=arc_jsd_top5_1000)
fi
SCORE_ARGS=()
for m in "${!SRC[@]}"; do
  f=$OUT/scores/$m.jsonl
  [[ -s $f ]] || python3 - "$DATA/methods/baselines/${SRC[$m]}.jsonl" "$MAIN/table3_resume/scores/$m.jsonl" "$f" <<'PY'
import json, os, sys
base, resumed, out = sys.argv[1:]
rows = [json.loads(l) for l in open(base)]
sub = {r["id"]: r for r in map(json.loads, open(resumed))} if os.path.exists(resumed) else {}
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(sub.get(r["id"], r)) + "\n")
PY
  SCORE_ARGS+=(--scores "$m=$f")
done
# Optional: EXTRA_SCORES="name=path name=path" adds rankings of the revision baselines.
for spec in ${EXTRA_SCORES:-}; do SCORE_ARGS+=(--scores "$spec"); done

stash_partial "$OUT/reflow_1000"
[[ -s $OUT/reflow_1000.summary.json ]] || "${RUNPY[@]}" scripts/evaluate_reflow.py \
  --input "$RET" --units-cache "$UNITS" --frontier "$FRONTIER" \
  --shared-pool "$POOL" --expected-pool-sha256 "$SHA" \
  --out "$OUT/reflow_1000.jsonl" --summary-out "$OUT/reflow_1000.summary.json" \
  --n 1000 --k 5 --workers 48 --replacement-seed "$REPL_SEED" --llm-base-url "$URL" --llm-model "$MODEL"
echo "DONE reflow $(date -Is)"

stash_partial "$OUT/baselines_1000"
[[ -s $OUT/baselines_1000.summary.json ]] || "${RUNPY[@]}" scripts/evaluate_matched_budget_baselines.py \
  --input "$RET" --units-cache "$UNITS" --reflow-results "$OUT/reflow_1000.jsonl" \
  --shared-pool "$POOL" --expected-pool-sha256 "$SHA" \
  --out "$OUT/baselines_1000.jsonl" --summary-out "$OUT/baselines_1000.summary.json" \
  --random-seeds 0,1,2,3,4 --replacement-seed "$REPL_SEED" --n 1000 --k 5 --workers 48 \
  --llm-base-url "$URL" --llm-model "$MODEL" "${SCORE_ARGS[@]}"
echo "DONE baselines $(date -Is)"

"${RUNPY[@]}" scripts/summarize_clean_anchor_flips.py --reflow "$OUT/reflow_1000.jsonl" \
  --baselines "$OUT/baselines_1000.jsonl" --out "$OUT/factual_metrics_1000.json" --reader-mode "$MODE"
echo "DONE factual $(date -Is)"

# Meaning-preserving control: start from the main run's paraphrase pool (Qwen editor,
# unchanged) and add only positions newly selected under this pool.
PP=$OUT/controls/paraphrase_pool.jsonl
[[ -s $PP ]] || python3 - "$DATA/controls/paraphrase_pool_top5_v2.jsonl" "$MAIN/table3_resume/paraphrase_pool.jsonl" "$PP" <<'PY'
import json, os, sys
rows = {}
for path in sys.argv[1:3]:  # main pool, then the Table 3 resume additions (if any)
    if not os.path.exists(path):
        continue
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["unit_id"]] = r
with open(sys.argv[3], "w") as f:
    for r in rows.values():
        f.write(json.dumps(r) + "\n")
PY
for pass in $(seq 1 20); do
  "${RUNPY[@]}" scripts/build_paraphrase_control_pool.py --input "$RET" --units-cache "$UNITS" \
    --reflow-results "$OUT/reflow_1000.jsonl" --baseline-results "$OUT/baselines_1000.jsonl" \
    --out "$PP.next" --manifest-out "$PP.manifest.json" --unresolved-out "$PP.unresolved.jsonl" \
    --n 1000 --workers 32 --batch-size 16 --attempts 8 --max-candidates 3 \
    --llm-base-url "$URL" --llm-model "$MODEL" --existing-pool "$PP"
  mv "$PP.next" "$PP"
  unresolved=$(python3 -c "import json;print(json.load(open('$PP.manifest.json')).get('unresolved_positions',-1))")
  echo "paraphrase pass $pass unresolved=$unresolved"
  [[ $unresolved == 0 ]] && break
done
PSHA=$(sha256sum "$PP" | cut -d' ' -f1)
stash_partial "$OUT/controls_1000"
[[ -s $OUT/controls_1000.summary.json ]] || "${RUNPY[@]}" scripts/evaluate_paraphrase_controls.py \
  --input "$RET" --units-cache "$UNITS" --reflow-results "$OUT/reflow_1000.jsonl" \
  --baseline-results "$OUT/baselines_1000.jsonl" --paraphrase-pool "$PP" --expected-pool-sha256 "$PSHA" \
  --out "$OUT/controls_1000.jsonl" --summary-out "$OUT/controls_1000.summary.json" \
  --n 1000 --k 5 --workers 48 --replacement-seed 0 --llm-base-url "$URL" --llm-model "$MODEL"
"${RUNPY[@]}" scripts/summarize_paraphrase_controls.py --input "$OUT/controls_1000.jsonl" \
  --out "$OUT/controls_1000.summary.json" --reader-mode "$MODE"
"${RUNPY[@]}" scripts/summarize_adjusted_flips.py --factual "$OUT/factual_metrics_1000.json" \
  --meaning-preserving "$OUT/controls_1000.summary.json" --out "$OUT/table3_metrics_1000.json"
echo "DONE $DS $TAG $(date -Is)"
