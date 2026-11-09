#!/usr/bin/env bash
# A8 for one dataset on $GPU: 100 queries, Table 3 inputs (k=5, frozen clean responses).
# Usage: a8_run.sh DATASET r|remove
set -euo pipefail
DS=$1 STAGE=$2; C=/data1/yujia/CausalityRAG; D=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a8_first_order
if [[ $DS == popqa ]]; then IN=$C/out_table3_final8_20260811/popqa/retrieval/top10_1000.jsonl; RES=$C/out_table3_final8_20260811/popqa/methods/reflow/results_top5_1000.jsonl
else IN=$C/out/$DS/retrieval/top10_1000.jsonl; RES=$C/out_no_frontier_repair_20260810/$DS/results_1000.jsonl; fi
[[ $DS == quartz ]] && MODE=quartz || MODE=short_answer
mkdir -p $D/messages
# the remove stage starts once the r stage (possibly on the other GPU) has written the messages
[[ $STAGE == remove ]] && while [[ ! -f $D/messages/$DS.pt ]]; do sleep 60; done
CUDA_VISIBLE_DEVICES=$GPU CAUSALITYRAG_READER_MODE=$MODE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  /data1/yujia/envs/graphrag/bin/python $D/a8_first_order.py --stage $STAGE --input $IN --results $RES \
  --model-path /data1/yujia/models/Qwen2.5-7B-Instruct --messages $D/messages/$DS.pt --out $D/a8_$DS.jsonl --n 100
