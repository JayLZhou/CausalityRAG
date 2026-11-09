#!/usr/bin/env bash
# QTZ follow-ups on the option-stop run: flow ablation variants and sentence sufficiency.
set -euo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; S=/data1/yujia/CausalityRAG; V=$R/out/qtz_option_stop_v2_20261109
TS=/data1/yujia/CausalityRAG_token_shapley_smoke
PY=/data1/yujia/envs/graphrag/bin/python; PY2=/data1/yujia/envs/qwen3_vllm_0102/bin/python
POOL=$S/out/quartz/replacements/shared_pool_top10_v1/shared_pool.jsonl
SHA=724115a67b2b8af90f74cfaeea47270e0412f7250b75a66d421f232d6cd148e6
RET=$S/out/quartz/retrieval/top10_1000.jsonl; UNITS=$S/out/quartz/inputs/token_units_top10_1000.jsonl
export CAUSALITYRAG_READER_MODE=short_answer
CUDA_VISIBLE_DEVICES="$GPU" setsid "$PY" -m vllm.entrypoints.openai.api_server \
  --model /data1/yujia/models/Qwen2.5-7B-Instruct --served-model-name qwen2.5-7b \
  --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
  --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
  --disable-log-requests --gpu-memory-utilization 0.90 --port "$PORT" \
  > "$V/logs/vllm_fu.log" 2>&1 < /dev/null &
VP=$!
trap 'kill -- -"$VP" 2>/dev/null || true' EXIT
ready=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then ready=1; break; fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "reader did not start"; exit 1; }
cd "$R"
"$PY" scripts/evaluate_matched_budget_baselines.py --input "$RET" --units-cache "$UNITS" \
  --reflow-results "$V/frontier/results.jsonl" --shared-pool "$POOL" --expected-pool-sha256 "$SHA" \
  --out "$V/flow_ablation.jsonl" --summary-out "$V/flow_ablation.summary.json" --random-seeds "" \
  --n 1000 --k 5 --workers 24 --llm-base-url "http://127.0.0.1:$PORT/v1" --llm-model qwen2.5-7b \
  --scores "direct_score=$V/flow_scores/direct_score.jsonl" --scores "individual_flow=$V/flow_scores/individual_flow.jsonl" \
  > "$V/logs/flow_ablation.log" 2>&1
cd "$TS"
mkdir -p "$V/suff"
"$PY2" scripts/run_sentence_sufficiency_case_study.py --retrieval "$RET" --units "$UNITS" \
  --reflow "$V/frontier/results.jsonl" --mirage "$S/out/quartz/methods/baselines/mirage_top5_1000.jsonl" \
  --arc-jsd "$S/out/quartz/methods/baselines/arc_jsd_top5_1000.jsonl" \
  --out "$V/suff/per_query.jsonl" --summary-out "$V/suff/summary.json" --workers 8 --max-tokens 64 \
  --llm-base-url "http://127.0.0.1:$PORT/v1" --llm-model qwen2.5-7b > "$V/logs/suff.log" 2>&1
"$PY2" scripts/run_sentence_sufficiency_ts.py --retrieval "$RET" --units "$UNITS" \
  --reflow "$V/frontier/results.jsonl" --mirage "$S/out/quartz/methods/baselines/mirage_top5_1000.jsonl" \
  --arc-jsd "$S/out/quartz/methods/baselines/arc_jsd_top5_1000.jsonl" \
  --token-shapley "$TS/out/sentence_sufficiency_ts_appendix_v1/import_server2/quartz"/shard_*.jsonl \
  --out "$V/suff/per_query_ts.jsonl" --summary-out "$V/suff/summary_ts.json" --workers 8 --max-tokens 64 \
  --llm-base-url "http://127.0.0.1:$PORT/v1" --llm-model qwen2.5-7b > "$V/logs/suff_ts.log" 2>&1
touch "$V/FOLLOWUPS_COMPLETE"
