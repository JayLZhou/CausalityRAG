#!/usr/bin/env bash
# miceflip_job.sh DATASET : first-flip MiCE-RAG, then factual metrics, controls and CFR
# through the Table 3 stages (ReFlow rows copied from the main-table run, so budgets match).
set -euo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; P=$R/pool_sensitivity; S=/data1/yujia/CausalityRAG
DS=$1; O=$P/qwen_a5_miceflip/$DS
case $DS in
  hotpotqa) POOL=$S/out/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl; SRC=$P/qwen_a5/$DS/reflow_1000.jsonl ;;
  popqa)    POOL=$S/out_table3_final8_20260811/popqa/replacements/shared_pool_top10_v1/shared_pool.jsonl; SRC=$P/qwen_a5/$DS/reflow_1000.jsonl ;;
  2wiki|timeqa|quartz) POOL=$S/out/$DS/replacements/shared_pool_top10_v1/shared_pool.jsonl; SRC=$R/a5_cfg/$DS/reflow_1000.jsonl ;;
  triviaqa) POOL=$S/out/triviaqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl; SRC=$R/a5_cfg/$DS/reflow_1000.jsonl ;;
  *)        POOL=$S/out/$DS/replacements/shared_pool_top10_v1/shared_pool.jsonl; SRC=$P/qwen_a5/$DS/reflow_1000.jsonl ;;
esac
mkdir -p $O
[[ -s $O/baselines_1000.summary.json ]] || /data1/yujia/envs/graphrag/bin/python $R/queue/mice_flip.py $DS $PORT
cp -n $SRC $O/reflow_1000.jsonl
[[ -s $O/reflow_1000.summary.json ]] || echo "{\"source\": \"$SRC\"}" > $O/reflow_1000.summary.json
[[ $DS == quartz ]] && export LENIENT_QUARTZ=1
bash $P/scripts/run_pool_sensitivity.sh $DS qwen_a5_miceflip $POOL $PORT
