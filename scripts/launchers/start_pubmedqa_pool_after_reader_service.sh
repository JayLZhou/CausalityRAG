#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data1/yujia/CausalityRAG_release_table3_20260801}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}

until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; do
  echo "[pubmedqa-pool-launch] waiting for reader service $(date -Is)"
  sleep 30
done

cd "$REPO"
exec "$PYTHON" scripts/finish_pubmedqa_table3_pool.py \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model qwen2.5-7b \
  --spacy-base-url http://127.0.0.1:8021
