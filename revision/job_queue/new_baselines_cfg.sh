#!/usr/bin/env bash
# new_baselines_cfg.sh DATASET : ContextCite, MiCE-RAG, RAGE (adapted + native) with the
# inputs, units, ReFlow rows and pool recorded in the A1 config for this dataset.
set -uo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; S=/data1/yujia/CausalityRAG
DS=$1; CFG=$R/out/flow_ablation_full8_20261024/configs/$DS.json
get() { python3 -c "import json,sys;print(json.load(open('$CFG'))[sys.argv[1]])" "$1"; }
MODE=$(get reader_mode)
export CAUSALITYRAG_READER_MODE=$MODE PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p $R/baselines_new/$DS
status=0
for m in contextcite mice_rag rage rage_native; do
  CUDA_VISIBLE_DEVICES=$GPU /data1/yujia/envs/graphrag/bin/python $R/baselines_new/scripts/run_new_baselines.py --method $m     --input $(get input) --units-cache $(get units) --reflow-results $(get reflow)     --prompt-attention $S/out/$DS/methods/baselines/attention_top5_1000.jsonl     --gxi-scores $S/out/$DS/methods/baselines/gradient_x_input_top5_1000.jsonl     --shared-pool $(get pool) --reader-mode $MODE     --out $R/baselines_new/$DS/${m}_1000.jsonl --n 1000 > $R/baselines_new/logs/${m}_$DS.log 2>&1 || status=1
done
exit $status
