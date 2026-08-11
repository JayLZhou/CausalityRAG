#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data1/yujia/CausalityRAG_release_table3_20260801}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
MODEL=${MODEL:-/data1/yujia/models/Qwen2.5-7B-Instruct}
OUT=${OUT:-/data1/yujia/CausalityRAG/out}
LOGDIR=${LOGDIR:-/data1/yujia/vllm_logs/table3}

mkdir -p "$LOGDIR"

while pgrep -f "run_table3_rankers.py" >/dev/null || \
      pgrep -f "run_.*attribution_baselines.py|run_attention.py|run_mirage.py|run_arc_jsd.py" >/dev/null; do
  echo "[table3-reader-launch] waiting for rankers $(date -Is)"
  sleep 60
done

for port in 8000 8001 8002 8017; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done

COMMON=(
  --model "$MODEL"
  --served-model-name qwen2.5-7b
  --dtype bfloat16
  --trust-remote-code
  --max-model-len 32768
  --enable-chunked-prefill
  --max-num-seqs 256
  --disable-log-requests
)

CUDA_VISIBLE_DEVICES=0 setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
  "${COMMON[@]}" --gpu-memory-utilization 0.90 --port 8001 \
  >"$LOGDIR/vllm_8001.log" 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
  "${COMMON[@]}" --gpu-memory-utilization 0.82 --port 8002 \
  >"$LOGDIR/vllm_8002.log" 2>&1 < /dev/null &

for port in 8001 8002; do
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    echo "[table3-reader-launch] waiting for vLLM :${port} $(date -Is)"
    sleep 15
  done
done

setsid "$PYTHON" "$REPO/scripts/lb_proxy_two_gpu.py" \
  >"$LOGDIR/proxy_8000.log" 2>&1 < /dev/null &
until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; do
  sleep 2
done

cd "$REPO"
exec "$PYTHON" scripts/run_table3_reader_stage.py \
  --out-root "$OUT" \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model qwen2.5-7b
