#!/usr/bin/env bash
# One resident worker per GPU: pops "NAME<TAB>COMMAND" lines from queue_gpu$GPU.txt in
# order and runs them with GPU/PORT exported; polls when the queue is empty.
set -uo pipefail
GPU=$1
Q=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/queue
export GPU PORT=$((8070 + GPU)) Q
exec 8>"$Q/worker_gpu$GPU.lock"; flock -n 8 || { echo "worker for GPU$GPU already running"; exit 1; }
touch "$Q/queue_gpu$GPU.txt"
while true; do
  line=$(flock "$Q/queue_gpu$GPU.txt" bash -c 'f=$0; l=$(head -n 1 "$f"); [ -n "$l" ] && sed -i 1d "$f"; printf "%s" "$l"' "$Q/queue_gpu$GPU.txt")
  if [[ -z $line ]]; then sleep 60; continue; fi
  name=${line%%$'\t'*}; cmd=${line#*$'\t'}
  echo "$(date -Is) START $name" >> "$Q/status_gpu$GPU.log"
  if bash -c "$cmd" > "$Q/logs/${name}.log" 2>&1; then s=DONE; else s=FAILED; fi
  echo "$(date -Is) $s $name" >> "$Q/status_gpu$GPU.log"
done
