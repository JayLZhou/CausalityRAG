#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != "${EXPECTED_HOST:-3haoji-A100}" ]]; then
  echo "refusing to run outside server 3: $(hostname)" >&2
  exit 1
fi

RELEASE=${RELEASE:-/data1/yujia/CausalityRAG_release_popqa_models_20260811}
PYTHON=${PYTHON:-/data1/yujia/envs/graphrag/bin/python}
QWEN25_ROOT=${QWEN25_ROOT:-/data1/yujia/CausalityRAG/out_table3_final8_20260811}
OLD_SWEEP_ROOT=${OLD_SWEEP_ROOT:-/data1/yujia/CausalityRAG/out/model_sensitivity_v1}
POPQA_SWEEP_ROOT=${POPQA_SWEEP_ROOT:-/data1/yujia/CausalityRAG/out/model_sensitivity_popqa_v1}
FINAL_ROOT=${FINAL_ROOT:-/data1/yujia/CausalityRAG/out/model_sensitivity_final8_v2}
QWEN25_QUARTZ_REPAIR=${QWEN25_QUARTZ_REPAIR:-$QWEN25_ROOT/quartz/choice_postprocess_v1}
RENDERED=$FINAL_ROOT/rendered

DATASETS=(hotpotqa timeqa finqa musique quartz triviaqa 2wiki)
MODELS=(qwen3-4b llama3.1-8b mistral-nemo-12b)

link_exact() {
  local source=$1 target=$2
  [[ -e "$source" ]] || { echo "missing source: $source" >&2; exit 1; }
  mkdir -p "$(dirname "$target")"
  if [[ -L "$target" ]]; then
    [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]] || {
      echo "unexpected symlink target: $target" >&2
      exit 1
    }
  elif [[ -e "$target" ]]; then
    echo "refusing to replace non-symlink path: $target" >&2
    exit 1
  else
    ln -s "$source" "$target"
  fi
}

[[ -f "$POPQA_SWEEP_ROOT/COMPLETE" ]] || {
  echo "PopQA added-model sweep is incomplete" >&2
  exit 1
}
[[ -f "$QWEN25_QUARTZ_REPAIR/COMPLETE" ]] || {
  echo "Qwen2.5 QuaRTz choice repair is incomplete" >&2
  exit 1
}
for dataset in "${DATASETS[@]}"; do
  qwen_result="$QWEN25_ROOT/$dataset/methods/reflow/results_top5_1000.jsonl"
  [[ -f "$qwen_result" ]] || { echo "missing Qwen2.5 result: $dataset" >&2; exit 1; }
  qwen_frontier="$(dirname "$(readlink -f "$qwen_result")")/frontier_1000.jsonl"
  link_exact "$qwen_frontier" \
    "$QWEN25_ROOT/$dataset/methods/reflow/frontier_top5_1000.jsonl"
done

for model in "${MODELS[@]}"; do
  [[ -f "$POPQA_SWEEP_ROOT/$model/COMPLETE" ]] || {
    echo "missing PopQA completion marker: $model" >&2
    exit 1
  }
  for dataset in "${DATASETS[@]}"; do
    link_exact "$OLD_SWEEP_ROOT/$model/$dataset" \
      "$FINAL_ROOT/$model/$dataset"
  done
  link_exact "$POPQA_SWEEP_ROOT/$model/popqa" \
    "$FINAL_ROOT/$model/popqa"
  link_exact "$OLD_SWEEP_ROOT/$model/manifest.json" \
    "$FINAL_ROOT/$model/manifest.json"
  link_exact "$POPQA_SWEEP_ROOT/$model/audit.json" \
    "$FINAL_ROOT/$model/popqa_audit.json"
done

mkdir -p "$RENDERED"
"$PYTHON" - "$OLD_SWEEP_ROOT" "$POPQA_SWEEP_ROOT" \
  "$QWEN25_ROOT" "$RENDERED/strict_input_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

old_root, popqa_root, qwen25_root, output = map(Path, sys.argv[1:])
models = ("qwen3-4b", "llama3.1-8b", "mistral-nemo-12b")
datasets = ("hotpotqa", "timeqa", "finqa", "musique", "quartz", "triviaqa", "2wiki")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

manifests = {
    model: json.loads((old_root / model / "manifest.json").read_text())
    for model in models
}
reference = {
    dataset: manifests[models[0]]["frozen_artifacts"][dataset]
    for dataset in datasets
}
for model, manifest in manifests.items():
    if manifest.get("queries_per_dataset") != 1000:
        raise ValueError(f"unexpected query count in {model} manifest")
    for dataset in datasets:
        if manifest["frozen_artifacts"][dataset] != reference[dataset]:
            raise ValueError(f"frozen artifact mismatch: {model}/{dataset}")

verified_sources = {}
for dataset, artifacts in reference.items():
    for kind, artifact in artifacts.items():
        path = Path(artifact["path"])
        actual = sha(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"SHA mismatch: {dataset}/{kind}")
        verified_sources[str(path.resolve())] = actual

popqa_audits = {
    model: json.loads((popqa_root / model / "audit.json").read_text())
    for model in models
}
popqa_reference = popqa_audits[models[0]]["frozen_sources"]
for model, audit in popqa_audits.items():
    if audit.get("queries") != 1000 or audit.get("remaining_no_frontier") != 0:
        raise ValueError(f"incomplete PopQA audit: {model}")
    if audit["frozen_sources"] != popqa_reference:
        raise ValueError(f"PopQA frozen artifact mismatch: {model}")
for artifact in popqa_reference.values():
    path = Path(artifact["path"])
    actual = sha(path)
    if actual != artifact["sha256"]:
        raise ValueError(f"PopQA SHA mismatch: {path}")
    verified_sources[str(path.resolve())] = actual

qwen25_popqa = qwen25_root / "popqa"
for relative in (
    "methods/reflow/results_top5_1000.jsonl",
    "controls/paraphrase_results_top5_1000_v2.jsonl",
    "methods/reflow/frontier_top5_1000.jsonl",
):
    path = qwen25_popqa / relative
    if not path.is_file():
        raise FileNotFoundError(path)

report = {
    "schema": "causalityrag.cross_model_final8_input_audit.v1",
    "server": "3haoji-A100",
    "models": list(models),
    "datasets": [*datasets, "popqa"],
    "queries_per_dataset": 1000,
    "verified_frozen_sources": dict(sorted(verified_sources.items())),
    "popqa_audits": popqa_audits,
}
output.write_text(json.dumps(report, indent=2) + "\n")
PY

"$PYTHON" "$RELEASE/scripts/render_cross_model_sensitivity.py" \
  --model "qwen2.5-7b|Qwen2.5-7B|Qwen2.5|default|$QWEN25_ROOT" \
  --model "llama3.1-8b|Llama-3.1-8B|Llama3.1|sweep|$FINAL_ROOT/llama3.1-8b" \
  --model "mistral-nemo-12b|Mistral-Nemo-12B|Mistral|sweep|$FINAL_ROOT/mistral-nemo-12b" \
  --model "qwen3-4b|Qwen3-4B|Qwen3|sweep|$FINAL_ROOT/qwen3-4b" \
  --quartz-retrieval "$QWEN25_ROOT/quartz/retrieval/top10_1000.jsonl" \
  --out-json "$RENDERED/cross_model_sensitivity.json" \
  --out-values-tex "$RENDERED/cross_model_sensitivity_values.tex" \
  --out-appendix-tex "$RENDERED/cross_model_sensitivity_rows.tex"

"$PYTHON" - "$RENDERED/cross_model_sensitivity.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
if len(data.get("models", [])) != 4:
    raise ValueError("expected four models")
for model in data["models"]:
    rows = model.get("datasets", [])
    if len(rows) != 8:
        raise ValueError(f"expected eight datasets for {model['key']}")
    for row in rows:
        if row["queries"] != 1000 or row["raw_paired_queries"] != 1000:
            raise ValueError(f"row/paired failure: {model['key']}/{row['dataset']}")
        if not 0 <= row["paired_queries"] <= 1000:
            raise ValueError(f"invalid paired denominator: {model['key']}/{row['dataset']}")
        if not 0.0 <= row["graph_coverage"] <= 1.0:
            raise ValueError(f"invalid coverage: {model['key']}/{row['dataset']}")
PY

date -Is >"$FINAL_ROOT/COMPLETE"
echo "cross-model final8 rendered: $RENDERED"
