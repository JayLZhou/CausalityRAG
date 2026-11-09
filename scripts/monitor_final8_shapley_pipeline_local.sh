#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
RANK="$ROOT/out/token_leverage_shap_final8_m128_3m"
EVAL="$ROOT/out/token_leverage_shap_final8_m128_3m_eval"

until ssh yujia-server3 "test -f '$RANK/RANKING_COMPLETE'"; do
  sleep 60
done

ssh yujia-server3 \
  "tar -czf /data1/yujia/shapley_trivia_rank.tgz -C '$RANK' triviaqa"
ssh -A yujia-server2 \
  "scp -o BatchMode=yes yujia@10.249.47.119:/data1/yujia/shapley_trivia_rank.tgz /data1/yujia/shapley_trivia_rank.tgz"
ssh yujia-server2 \
  "tar -xzf /data1/yujia/shapley_trivia_rank.tgz -C '$RANK'"

until ssh yujia-server2 "test -f '$RANK/2WIKI_COMPLETE'"; do
  sleep 60
done

until ssh yujia-server2 \
  "gzip -t /data1/yujia/shapley_eval_bundle.tgz 2>/dev/null"; do
  sleep 60
done
ssh yujia-server2 \
  "tar -xzf /data1/yujia/shapley_eval_bundle.tgz -C '$ROOT'"

ssh yujia-server2 \
  "mkdir -p '$EVAL'; nohup '$ROOT/scripts/run_final8_token_shapley_eval_server2.sh' >'$EVAL/master.log' 2>&1 < /dev/null & echo \$! >'$EVAL/master.pid'"

until ssh yujia-server2 "test -f '$EVAL/FINAL8_COMPLETE'"; do
  sleep 60
done

ssh yujia-server2 \
  "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
