#!/usr/bin/env bash
# A2/A3 on one GPU, HotpotQA k=5: tracing check, then same-hardware timing of ARC-JSD and TS (100 queries).
set -uo pipefail
A=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a2_runtime; S=/data1/yujia/CausalityRAG
PY=/data1/yujia/envs/graphrag/bin/python; M=/data1/yujia/models/Qwen2.5-7B-Instruct
IN=$S/out/hotpotqa/retrieval/top10_1000.jsonl; CLEAN=$S/out/hotpotqa/inputs/clean_reference_1000.jsonl
MEAS="$PY /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a3_memory/measure_gpu.py --gpu $GPU --n 100 --out $A/timing.jsonl"
export CUDA_VISIBLE_DEVICES=$GPU CAUSALITYRAG_READER_MODE=short_answer TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
case $1 in
  a3) $PY $A/a3_trace_check.py --input $IN --clean-reference $CLEAN --model-path $M --n 1000 --out $A/a3_trace_check.jsonl ;;
  arcjsd) # ARC-JSD segments sentences through the spaCy annotation service (CPU).
    (cd $S && setsid /data1/yujia/envs/spacyner/bin/python scripts/spacy_annotation_server.py --port 8061 > $A/out/spacy_arcjsd.log 2>&1 < /dev/null) &
    for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8061/health | grep -q ok && break; sleep 5; done
    cd /data1/yujia/CausalityRAG_release_no_frontier_repair_20260810
    $MEAS --name arc_jsd_k5 -- $PY exp/run_arc_jsd.py --input $IN --clean-reference $CLEAN --out $A/out/arc_jsd_k5.jsonl \
      --summary-out $A/out/arc_jsd_k5.summary.json --model-path $M --n 100 --k 5 --spacy-base-url http://127.0.0.1:8061
    pkill -f "spacy_annotation_server.py --port 8061" ;;
  ts) T=/data1/yujia/CausalityRAG_token_shapley_smoke; cd $T; export PYTHONPATH=$T
    $MEAS --name ts_k5 -- /data1/yujia/envs/qwen3_vllm_0102/bin/python exp/run_token_leverage_shap.py --input $IN \
      --clean-reference $CLEAN --context-units $S/out/hotpotqa/inputs/token_units_top5_1000.jsonl --model-path $M \
      --out $A/out/ts_k5.jsonl --summary-out $A/out/ts_k5.summary.json --n 100 --k 5 --top-k 128 \
      --attention-candidates 128 --budget-multiplier 3 --coalition-batch-size 32 --seed 42 ;;
esac
