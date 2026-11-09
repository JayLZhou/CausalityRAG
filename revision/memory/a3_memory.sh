#!/usr/bin/env bash
# A2/A3: peak GPU memory and time per query for plain inference, contribution-graph
# construction and MIRAGE, Qwen2.5-7B, HotpotQA, k=5 and k=10, 100 queries each.
set -uo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; A=$R/a3_memory; S=/data1/yujia/CausalityRAG
REPO=/data1/yujia/CausalityRAG_release_model_sweep_20260807; PY=/data1/yujia/envs/graphrag/bin/python
M=/data1/yujia/models/Qwen2.5-7B-Instruct; IN=$S/out/hotpotqa/retrieval/top10_1000.jsonl; N=100
mkdir -p $A/out; cd $REPO
export CUDA_VISIBLE_DEVICES=$GPU CAUSALITYRAG_READER_MODE=short_answer TOKENIZERS_PARALLELISM=false
run() { $PY $A/measure_gpu.py --gpu $GPU --name "$1" --n $N --out $A/measurements.jsonl -- "${@:2}"; }
for k in 5 10; do
  if [[ $k == 5 ]]; then CLEAN=$S/out/hotpotqa/inputs/clean_reference_1000.jsonl; UNITS=$S/out/hotpotqa/inputs/token_units_top5_1000.jsonl
  else CLEAN=$S/out/hotpotqa/inputs/clean_reference_top10_1000.jsonl; UNITS=$S/out/hotpotqa/inputs/token_units_top10_1000.jsonl; fi
  run plain_k$k $PY $A/plain_generate.py --input $IN --n $N --k $k --model-path $M --out $A/out/plain_k$k.jsonl
  run graph_k$k $PY scripts/build_contribution_graph.py --input $IN --out $A/out/graph_k$k.jsonl \
    --summary-out $A/out/graph_k$k.summary.json --model-path $M --n $N --k $k --target results \
    --target-results $CLEAN --device cuda --dtype bfloat16 --max-receivers-per-layer 48
  run mirage_k$k $PY exp/run_mirage.py --input $IN --clean-reference $CLEAN --units-cache $UNITS \
    --out $A/out/mirage_k$k.jsonl --summary-out $A/out/mirage_k$k.summary.json --model-path $M --n $N --k $k
done
