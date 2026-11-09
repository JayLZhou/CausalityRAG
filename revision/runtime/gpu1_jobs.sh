#!/usr/bin/env bash
# A2 with the vLLM reader on this GPU (PORT): serial ReFlow verification on 100 HotpotQA
# queries, and Qwen-editor pool construction on a sample of typed keys.
set -uo pipefail
A=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a2_runtime; S=/data1/yujia/CausalityRAG
PY=/data1/yujia/envs/graphrag/bin/python; URL=http://127.0.0.1:$PORT/v1
export CAUSALITYRAG_READER_MODE=short_answer TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p $A/out $A/pool_qwen
case $1 in
  verify) POOL=$S/out/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl
    cd /data1/yujia/CausalityRAG_release_no_frontier_repair_20260810
    t0=$(date +%s.%N)
    $PY scripts/evaluate_reflow.py --input $S/out/hotpotqa/retrieval/top10_1000.jsonl \
      --units-cache $S/out/hotpotqa/inputs/token_units_top10_1000.jsonl \
      --frontier $S/out_no_frontier_repair_20260810/hotpotqa/frontier_1000.jsonl \
      --shared-pool $POOL --expected-pool-sha256 $(sha256sum $POOL | cut -d" " -f1) \
      --out $A/out/verify_serial_k5.jsonl --summary-out $A/out/verify_serial_k5.summary.json \
      --n 100 --k 5 --workers 1 --replacement-seed 0 --llm-base-url $URL --llm-model qwen2.5-7b
    echo "wall_seconds $(echo "$(date +%s.%N) - $t0" | bc)" ;;
  pool) cd $S
    setsid /data1/yujia/envs/spacyner/bin/python scripts/spacy_annotation_server.py --port 8061 > $A/pool_qwen/spacy.log 2>&1 < /dev/null &
    SP=$!; for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8061/ >/dev/null 2>&1 && break; sleep 5; done
    : > $A/pool_qwen/empty_seed.jsonl; rm -f $A/pool_qwen/typed_candidates.jsonl
    t0=$(date +%s.%N)
    /data1/yujia/envs/qwen3_vllm_0102/bin/python scripts/generate_shared_typed_replacement_pool.py \
      --typed-keys $S/out/hotpotqa/replacements/shared_pool_top10_v1/typed_keys.jsonl --seed $A/pool_qwen/empty_seed.jsonl \
      --out $A/pool_qwen/typed_candidates.jsonl --unresolved-out $A/pool_qwen/unresolved.jsonl \
      --manifest-out $A/pool_qwen/manifest.json --workers 32 --batch-size 24 --generation-rounds 8 --limit 4800 \
      --llm-base-url $URL --llm-model qwen2.5-7b --spacy-base-url http://127.0.0.1:8061
    echo "wall_seconds $(echo "$(date +%s.%N) - $t0" | bc) keys_total $(wc -l < $S/out/hotpotqa/replacements/shared_pool_top10_v1/typed_keys.jsonl)"
    kill $SP ;;
esac
