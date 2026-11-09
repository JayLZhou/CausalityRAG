"""C1 aligned with Table IV, first-flip MiCE-RAG only (replaces the greedy-probability MiCE prompts).

Same construction as build_tableiv_prompts.py; MiCE-RAG selections and its control
positions come from the qwen_a5_miceflip run, and the clean/ReFlow prompts are not rewritten.

Original description:

Datasets HQA / FinQA / PopQA, all valid-clean queries of the 1000. For ReFlow and
the strongest baseline (MiCE-RAG), the counterfactual edit (shared pool, seed 0)
and the meaning-preserving control edit (paraphrase pool, seed 0, numeric/date
positions excluded) are rebuilt exactly as in the Table 3 code from the main
run's records, and the reader prompt is written for clean, edited and control
contexts. Controls with no eligible position are recorded without a prompt.
"""
import hashlib
import json
import sys

sys.path.insert(0, "/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810")
from causalityrag.io import load_records, record_id, retrieved_contexts  # noqa: E402
from causalityrag.reader import reader_prompt  # noqa: E402
from causalityrag.revision import apply_token_replacements  # noqa: E402
from causalityrag.shared_replacement_pool import (  # noqa: E402
    FrozenSharedReplacementPool,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row  # noqa: E402

S = "/data1/yujia/CausalityRAG"
P = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity"
DATA = {
    "hotpotqa": (f"{S}/out/hotpotqa", "shared_pool_top10_final.jsonl"),
    "finqa": (f"{S}/out/finqa", "shared_pool.jsonl"),
    "popqa": (f"{S}/out_table3_final8_20260811/popqa", "shared_pool.jsonl"),
}
METHODS = ("mice_rag",)
TAG = "qwen_a5_miceflip"


def stable_candidate(candidates, unit_id, seed=0):  # evaluate_paraphrase_controls.stable_candidate
    index = int.from_bytes(hashlib.sha256(f"{seed}\0{unit_id}".encode()).digest()[:8], "big")
    return dict(candidates[index % len(candidates)])


out = open(sys.argv[1], "w")
meta = open(sys.argv[1].replace(".jsonl", ".meta.jsonl"), "w")
for ds, (base, pool_name) in DATA.items():
    records = {record_id(r): r for r in load_records(f"{base}/retrieval/top10_1000.jsonl")}
    units = {str(r["id"]): r for r in load_records(f"{base}/inputs/token_units_top10_1000.jsonl")}
    pool = FrozenSharedReplacementPool(f"{base}/replacements/shared_pool_top10_v1/{pool_name}")
    para = {r["unit_id"]: r for r in load_records(f"{P}/{TAG}/{ds}/controls/paraphrase_pool.jsonl")}
    reflow = {r["id"]: r for r in load_records(f"{P}/qwen_a5/{ds}/reflow_1000.jsonl")}
    baselines = {r["id"]: r for r in load_records(f"{P}/{TAG}/{ds}/baselines_1000.jsonl")}
    controls = {r["id"]: r for r in load_records(f"{P}/{TAG}/{ds}/controls_1000.jsonl")}
    n_q = n_p = 0
    for q, rf in reflow.items():
        if rf.get("evaluation_status") == "invalid_clean_answer" or not str(rf.get("clean_answer", "")).strip():
            continue
        n_q += 1
        record = records[q]
        question = str(record["question"])
        by_id = {str(u["unit_id"]): u for u in units_from_cache_row(record, units[q], k=5)}

        def emit(method, contexts, **extra):
            global n_p
            system, user = reader_prompt(question, contexts, reader_mode="short_answer")
            out.write(json.dumps({"dataset": ds, "id": q, "method": method, "system": system, "user": user},
                                 ensure_ascii=False) + "\n")
            n_p += 1

        info = {"dataset": ds, "id": q, "gold_answers": record.get("answers") or record.get("answer"),
                "question": question, "qwen_clean_answer": rf["clean_answer"], "methods": {}}
        for m in METHODS:
            selected = (rf if m == "reflow" else baselines[q].get("methods", {}).get(m, {})).get("selected_ids") or []
            entry = {"n_tokens": len(selected)}
            if selected:
                rows = pool.require(selected)
                repl = {u: stable_shared_candidate(rows[u]["candidates"], unit_id=u, seed=0) for u in selected}
                rev = apply_token_replacements(record, [by_id[u] for u in selected], repl, k=5)
                if int(rev["n_failed_edits"]) == 0 and int(rev["n_edits"]) == len(selected):
                    emit("mice_flip", rev["edited_contexts"])
                    entry["edited"] = True
            ctrl_ids = controls[q].get("methods", {}).get(m, {}).get("control_selected_ids") or []
            entry["control_ids"] = len(ctrl_ids)
            if ctrl_ids and all(u in para for u in ctrl_ids):
                repl = {u: stable_candidate(para[u]["candidates"], u) for u in ctrl_ids}
                rev = apply_token_replacements(record, [by_id[u] for u in ctrl_ids], repl, k=5,
                                               allow_whitespace=True, allow_case_only=True)
                if int(rev["n_failed_edits"]) == 0 and int(rev["n_edits"]) == len(ctrl_ids):
                    emit("mice_flip_ctrl", rev["edited_contexts"])
                    entry["control"] = True
            info["methods"]["mice_flip"] = entry
        meta.write(json.dumps(info, ensure_ascii=False) + "\n")
    print(ds, "valid queries", n_q, "prompts", n_p, flush=True)
out.close()
meta.close()
