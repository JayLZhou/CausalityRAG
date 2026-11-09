#!/usr/bin/env bash
# Runs from the GPU0 queue once GPU1 has yielded and the 32B download is complete.
set -euo pipefail
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1; Q=$R/queue; A=$R/a4_32b
M=/data1/yujia/models/Qwen2.5-32B-Instruct
until grep -q 'START hold_for_a4' $Q/status_gpu1.log; do sleep 60; done
until grep -q DOWNLOAD_DONE /data1/yujia/models/download_qwen25_32b.log; do
  pgrep -f 'snapshot_download' >/dev/null || { echo 'download stopped before completion'; exit 1; }
  sleep 60
done
rm -f $A/A4_FINISHED
trap 'touch $A/A4_FINISHED' EXIT   # releases GPU1 whatever happens
bash $A/run_a4_32b.sh qwen2.5-32b $M qwen2.5-32b-instruct 10
bash $A/run_a4_32b.sh qwen2.5-32b $M qwen2.5-32b-instruct 1000
