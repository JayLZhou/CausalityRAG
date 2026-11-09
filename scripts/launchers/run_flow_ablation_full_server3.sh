#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == "3haoji-A100" ]] || exit 1
cd /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1
PYTHON=/data1/yujia/envs/qwen3_vllm_0102/bin/python
ROOT="$PWD/out/flow_ablation_full8_20261024"
LANE="$1"
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/lane_${LANE}.lock"
flock -n 9 || exit 1
case "$LANE" in
  1)
    export CUDA_VISIBLE_DEVICES=GPU-4a836dad-0a67-0358-2529-5185dce33356
    DATASETS=(hotpotqa musique triviaqa popqa)
    ;;
  2)
    export CUDA_VISIBLE_DEVICES=GPU-380960e6-b5a9-3c70-b124-ab3c12950d68
    DATASETS=(quartz timeqa finqa 2wiki)
    ;;
  *) exit 1 ;;
esac
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export CAUSALITYRAG_READER_MODE=short_answer
for dataset in "${DATASETS[@]}"; do
  if [[ -s "$ROOT/$dataset/summary.json" ]]; then
    echo "ALREADY COMPLETE $dataset"
    continue
  fi
  # A free device must have both low memory use and no compute process.
  "$PYTHON" - "$CUDA_VISIBLE_DEVICES" <<'PY'
import csv, subprocess, sys
uuid = sys.argv[1]
gpus = csv.reader(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines())
memory = {row[0].strip(): int(row[1]) for row in gpus}
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines()
if memory[uuid] >= 2048 or uuid in [s.strip() for s in apps]:
    raise SystemExit('Device busy; refusing to start a competing task')
PY
  reuse=()
  if [[ "$dataset" == hotpotqa ]]; then
    reuse=(--reuse-pilot "$PWD/out/hotpotqa_pilot100")
  fi
  echo "START $dataset $(date -Is)"
  if "$PYTHON" scripts/run_flow_ablation_full.py --root "$ROOT" --dataset "$dataset" "${reuse[@]}" >"$ROOT/logs/$dataset.log" 2>&1; then
    echo "DONE $dataset $(date -Is)"
  else
    echo "FAILED $dataset $(date -Is); inspect dataset log, other datasets continue"
  fi
done
echo "LANE $LANE FINISHED $(date -Is)"
