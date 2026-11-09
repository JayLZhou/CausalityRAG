#!/usr/bin/env bash
# QTZ with the original reader prompt: only the stopping rule changes (stop when the selected option changes).
set -euo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; P=$R/pool_sensitivity; S=/data1/yujia/CausalityRAG
V=$R/out/qtz_option_stop_v2_20261109; OPT=$R/out/qtz_option_stop_20261109
INPUT=$S/out/quartz/retrieval/top10_1000.jsonl; UNITS=$S/out/quartz/inputs/token_units_top10_1000.jsonl
FRONT=$S/out_no_frontier_repair_20260810/quartz/frontier_1000.jsonl
POOL=$S/out/quartz/replacements/shared_pool_top10_v1/shared_pool.jsonl
SHA=724115a67b2b8af90f74cfaeea47270e0412f7250b75a66d421f232d6cd148e6
PY=/data1/yujia/envs/graphrag/bin/python
export EXTRA_SCORES="ts=$OPT/ts_scores.jsonl contextcite=$R/baselines_new/quartz/contextcite_1000.jsonl mice_rag=$R/baselines_new/quartz/mice_rag_1000.jsonl rage=$R/baselines_new/quartz/rage_1000.jsonl"
mkdir -p "$V/logs"
CUDA_VISIBLE_DEVICES="$GPU" setsid "$PY" -m vllm.entrypoints.openai.api_server \
  --model /data1/yujia/models/Qwen2.5-7B-Instruct --served-model-name qwen2.5-7b \
  --dtype bfloat16 --trust-remote-code --max-model-len 16384 \
  --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
  --disable-log-requests --gpu-memory-utilization 0.90 --port "$PORT" \
  > "$V/logs/vllm.log" 2>&1 < /dev/null &
VP=$!
trap 'kill -- -"$VP" 2>/dev/null || true' EXIT
ready=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then ready=1; break; fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "reader did not start"; exit 1; }
for M in frontier grid one_shot; do
  case "$M" in frontier) F="$FRONT" ;; grid) F="$OPT/grid_frontier.jsonl" ;; one_shot) F="$OPT/one_shot_frontier.jsonl" ;; esac
  mkdir -p "$V/$M"
  "$PY" "$V/opt_eval.py" --input "$INPUT" --units-cache "$UNITS" --frontier "$F" --shared-pool "$POOL" \
    --expected-pool-sha256 "$SHA" --out "$V/$M/results.jsonl" --summary-out "$V/$M/summary.json" \
    --n 1000 --k 5 --workers 24 --replacement-seed 0 \
    --llm-base-url "http://127.0.0.1:$PORT/v1" --llm-model qwen2.5-7b > "$V/logs/$M.log" 2>&1
done
touch "$V/SEARCH_COMPLETE"
for TAG in qtz_optstop2 qtz_optstop2_miceflip; do
  O=$P/$TAG/quartz; mkdir -p "$O"
  [ -s "$O/reflow_1000.jsonl" ] || cp "$V/frontier/results.jsonl" "$O/reflow_1000.jsonl"
  [ -s "$O/reflow_1000.summary.json" ] || echo "{\"source\": \"$V/frontier/results.jsonl\"}" > "$O/reflow_1000.summary.json"
  if [ "$TAG" = qtz_optstop2_miceflip ]; then
    [ -s "$O/baselines_1000.summary.json" ] || "$PY" "$V/mice_flip_v2.py" quartz "$PORT"
  fi
  bash "$V/run_pool_short.sh" quartz "$TAG" "$POOL" "$PORT"
done
touch "$V/COMPLETE"
