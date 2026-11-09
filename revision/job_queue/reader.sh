#!/usr/bin/env bash
# reader.sh up|down : Qwen2.5-7B vLLM reader on $GPU at $PORT, same flags as the Table 3 runs.
set -euo pipefail
Q=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/queue
pidf=$Q/reader_$PORT.pid
if [[ $1 == up ]]; then
  cd /tmp
  CUDA_VISIBLE_DEVICES=$GPU setsid /data1/yujia/envs/qwen3_vllm_0102/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data1/yujia/models/Qwen2.5-7B-Instruct --served-model-name qwen2.5-7b --dtype bfloat16 \
    --trust-remote-code --max-model-len 16384 --enable-chunked-prefill --max-num-seqs 64 --enforce-eager \
    --disable-log-requests --gpu-memory-utilization 0.90 --port "$PORT" > "$Q/logs/reader_$PORT.log" 2>&1 < /dev/null &
  echo $! > "$pidf"
  for i in $(seq 1 60); do curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" | grep -q qwen && exit 0; sleep 5; done
  echo "reader did not come up"; exit 1
else
  [[ -f $pidf ]] && kill "$(cat "$pidf")" 2>/dev/null || true
  rm -f "$pidf"; sleep 20
fi
