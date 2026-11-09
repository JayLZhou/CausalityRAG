#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "2haoji-A100" ]] || exit 1

ROOT=/data1/yujia/CausalityRAG_token_shapley_smoke
RANK="$ROOT/out/token_leverage_shap_final8_m128_3m"
EVAL="$ROOT/out/token_leverage_shap_final8_m128_3m_eval"

# The primary server2 evaluator owns the GPUs until these five datasets are
# complete.  Evaluate the two server3-ranked datasets only after it exits.
for dataset in timeqa finqa musique quartz popqa; do
  while [[ ! -f "$EVAL/$dataset/COMPLETE" ]]; do
    sleep 60
  done
done

for dataset in triviaqa 2wiki; do
  for shard in 0 1; do
    while [[ ! -s "$RANK/$dataset/shard_${shard}.jsonl" ]]; do
      sleep 60
    done
  done
done

"$ROOT/scripts/launchers/run_final8_token_shapley_eval_server2_subset.sh"

touch "$EVAL/SERVER3_IMPORTED"
