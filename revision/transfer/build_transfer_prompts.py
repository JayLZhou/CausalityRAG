"""C1: reader prompts for transferring Qwen2.5-7B explanations to API models.

For a fixed random subset of valid-clean queries, each method's matched-budget
edit from the main run (original Qwen pool, replacement seed 0) is applied with
the Table 3 code, and the exact reader prompt is written for the clean context
and for every method's edited context.
"""
import json
import random
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
    "musique": (f"{S}/out/musique", "shared_pool.jsonl"),
    "popqa": (f"{S}/out_table3_final8_20260811/popqa", "shared_pool.jsonl"),
}
METHODS = ["mice_rag", "arc_jsd", "mirage", "contextcite"]  # from qwen_a5; TS from qwen_a5_tsorig
N, SEED = int(sys.argv[2]), 20261028
out = open(sys.argv[1], "w")
for ds, (base, pool_name) in DATA.items():
    records = {record_id(r): r for r in load_records(f"{base}/retrieval/top10_1000.jsonl")}
    units = {str(r["id"]): r for r in load_records(f"{base}/inputs/token_units_top10_1000.jsonl")}
    pool = FrozenSharedReplacementPool(f"{base}/replacements/shared_pool_top10_v1/{pool_name}")
    reflow = {r["id"]: r for r in load_records(f"{P}/qwen_a5/{ds}/reflow_1000.jsonl")}
    base_rows = {r["id"]: r for r in load_records(f"{P}/qwen_a5/{ds}/baselines_1000.jsonl")}
    ts_rows = {r["id"]: r for r in load_records(f"{P}/qwen_a5_tsorig/{ds}/baselines_1000.jsonl")}
    valid = sorted(q for q, r in reflow.items() if r.get("evaluation_status") != "invalid_clean_answer"
                   and str(r.get("clean_answer", "")).strip())
    chosen = sorted(random.Random(f"{SEED}:{ds}").sample(valid, N))
    for q in chosen:
        record = records[q]
        question = str(record["question"])
        by_id = {str(u["unit_id"]): u for u in units_from_cache_row(record, units[q], k=5)}
        selections = {"reflow": reflow[q].get("selected_ids") or []}
        for m in METHODS:
            selections[m] = base_rows[q].get("methods", {}).get(m, {}).get("selected_ids") or []
        selections["ts"] = ts_rows[q].get("methods", {}).get("ts", {}).get("selected_ids") or []
        system, user = reader_prompt(question, retrieved_contexts(record)[:5], reader_mode="short_answer")
        out.write(json.dumps({"dataset": ds, "id": q, "method": "clean", "n_edits": 0,
                              "qwen_clean_answer": reflow[q]["clean_answer"],
                              "system": system, "user": user}, ensure_ascii=False) + "\n")
        for method, selected in selections.items():
            if not selected:
                continue
            rows = pool.require(selected)
            repl = {u: stable_shared_candidate(rows[u]["candidates"], unit_id=u, seed=0) for u in selected}
            revision = apply_token_replacements(record, [by_id[u] for u in selected], repl, k=5)
            if int(revision["n_failed_edits"]) > 0 or int(revision["n_edits"]) != len(selected):
                continue
            system, user = reader_prompt(question, revision["edited_contexts"], reader_mode="short_answer")
            flip = (reflow[q] if method == "reflow" else (ts_rows if method == "ts" else base_rows)[q]
                    .get("methods", {}).get(method, {})).get("verified_flip")
            out.write(json.dumps({"dataset": ds, "id": q, "method": method, "n_edits": len(selected),
                                  "qwen_verified_flip": bool(flip), "system": system, "user": user},
                                 ensure_ascii=False) + "\n")
    print(ds, "queries", len(chosen), flush=True)
out.close()
