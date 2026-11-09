#!/usr/bin/env bash
set -euo pipefail

SERVER3="yujia@10.249.47.119"
RUN_ROOT="/data1/yujia/CausalityRAG_rmf_b20_20260827"
SOURCE_ROOT="${RUN_ROOT}/source/k5"
LOG="${RUN_ROOT}/graph_sync.log"

mkdir -p "${SOURCE_ROOT}" "${RUN_ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1

sync_graph() {
  local dataset="$1"
  local source="$2"
  local destination="${SOURCE_ROOT}/${dataset}/graph/contribution_graph.jsonl"
  mkdir -p "$(dirname "${destination}")"
  rsync -a --info=progress2 "${SERVER3}:${source}" "${destination}"
}

sync_graph hotpotqa \
  /data1/yujia/CausalityRAG/out/retrieval_depth_qwen25_v1/k5/hotpotqa/graph/contribution_graph.jsonl
sync_graph timeqa \
  /data1/yujia/CausalityRAG/out/retrieval_depth_qwen25_v1/k5/timeqa/graph/contribution_graph.jsonl
sync_graph finqa \
  /data1/yujia/CausalityRAG/out/finqa/graphs/contribution_graph_top5_1000.jsonl
sync_graph musique \
  /data1/yujia/CausalityRAG/out/musique/graphs/contribution_graph_top5_1000.jsonl
sync_graph quartz \
  /data1/yujia/CausalityRAG/out_label_protocol_20260805/quartz/graphs/contribution_graph_top5_1000.jsonl
sync_graph triviaqa \
  /data1/yujia/CausalityRAG/out/triviaqa/graphs/contribution_graph_top5_1000.jsonl
sync_graph 2wiki \
  /data1/yujia/CausalityRAG/out/2wiki/graphs/contribution_graph_top5_1000.jsonl
sync_graph popqa \
  /data1/yujia/CausalityRAG/out_popqa_no_frontier_repair_20260811/popqa/graphs.jsonl

touch "${RUN_ROOT}/GRAPHS_SYNC_COMPLETE"

