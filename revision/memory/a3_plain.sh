#!/usr/bin/env bash
# A3 reference rerun: plain greedy generation only (the first run failed on an import).
set -uo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; A=$R/a3_memory; S=/data1/yujia/CausalityRAG
PY=/data1/yujia/envs/graphrag/bin/python; M=/data1/yujia/models/Qwen2.5-7B-Instruct; IN=$S/out/hotpotqa/retrieval/top10_1000.jsonl
export CUDA_VISIBLE_DEVICES=$GPU
for k in 5 10; do
  $PY $A/measure_gpu.py --gpu $GPU --name plain_k$k --n 100 --out $A/measurements.jsonl -- $PY $A/plain_generate.py --input $IN --n 100 --k $k --model-path $M --out $A/out/plain_k$k.jsonl
done
