"""MiCE-RAG with ReFlow's verification strategy (replaces the greedy-probability variant).

MiCE's gradient-guided masking orders candidate tokens (Gradient x Input, no
access to the replacement pool). Like ReFlow, the search proposes candidate
sets in order -- the top-1, top-2, ... gradient tokens, up to ReFlow's per-query
budget -- verifies each with the shared reader and pool replacement, and stops
at the first answer change. The output keeps the evaluator's row schema with
the `mice_rag` entry replaced, so the Table 3 summaries and controls apply as is.
Usage: mice_flip.py DATASET PORT [EDITOR_TAG]
With EDITOR_TAG (llama31_8b, gpt56_sol), the search and verification use that editor's
pool and start from its A7(b) rows (<tag>_a5); output goes to <tag>_miceflip.
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810")
from causalityrag.io import load_records, record_id, retrieved_contexts  # noqa: E402
from causalityrag.reader import ReaderClient, reader_answers_exact_match  # noqa: E402
from causalityrag.revision import apply_token_replacements  # noqa: E402
from causalityrag.shared_replacement_pool import (  # noqa: E402
    FrozenSharedReplacementPool,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row  # noqa: E402

S = "/data1/yujia/CausalityRAG"
P = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity"
BASES = {"hotpotqa": (f"{S}/out/hotpotqa", "shared_pool_top10_final.jsonl"),
         "musique": (f"{S}/out/musique", "shared_pool.jsonl"),
         "finqa": (f"{S}/out/finqa", "shared_pool.jsonl"),
         "popqa": (f"{S}/out_table3_final8_20260811/popqa", "shared_pool.jsonl"),
         "2wiki": (f"{S}/out/2wiki", "shared_pool.jsonl"),
         "timeqa": (f"{S}/out/timeqa", "shared_pool.jsonl"),
         "triviaqa": (f"{S}/out/triviaqa", "shared_pool_top10_final.jsonl"),
         "quartz": (f"{S}/out/quartz", "shared_pool.jsonl")}
UNITS = {"triviaqa": "context_units_top10_1000.jsonl"}
# Evaluator rows to start from: the qwen_a5 run, or the A1-config run for 2Wiki/TimeQA.
ROWS = {ds: f"{P}/qwen_a5/{ds}/baselines_1000.jsonl" for ds in ("hotpotqa", "musique", "finqa", "popqa")}
ROWS.update({ds: f"/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a5_cfg/{ds}/baselines_1000.jsonl"
             for ds in ("2wiki", "timeqa", "triviaqa", "quartz")})
TAG = "qwen_a5_miceflip"


def main():
    ds, port = sys.argv[1], sys.argv[2]
    editor = sys.argv[3] if len(sys.argv) > 3 else ""
    base, pool_name = BASES[ds]
    mode = "quartz" if ds == "quartz" else "short_answer"
    records = {record_id(r): r for r in load_records(f"{base}/retrieval/top10_1000.jsonl")}
    units = {str(r["id"]): r for r in load_records(f"{base}/inputs/{UNITS.get(ds, 'token_units_top10_1000.jsonl')}")}
    pool_path = (f"/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/editor_pools/{editor}/{ds}/shared_pool.jsonl"
                 if editor else f"{base}/replacements/shared_pool_top10_v1/{pool_name}")
    pool = FrozenSharedReplacementPool(pool_path)
    gxi = {}
    for r in load_records(f"{base}/methods/baselines/gradient_x_input_top5_1000.jsonl"):
        ranked = r.get("ranked_ids") or [k for k, _ in sorted((r.get("token_scores") or {}).items(), key=lambda kv: -kv[1])]
        gxi[str(r["id"])] = [str(u) for u in ranked]
    rows = load_records(f"{P}/{editor}_a5/{ds}/baselines_1000.jsonl" if editor else ROWS[ds])
    out_dir = f"{P}/{editor}_miceflip/{ds}" if editor else f"{P}/{TAG}/{ds}"
    os.makedirs(out_dir, exist_ok=True)
    if ds == "quartz":
        # Same lenient QuaRTz matching as run_with_reader_timeout.py (LENIENT_QUARTZ):
        # three clean answers map to no choice; Table 3 kept them as valid queries.
        import causalityrag.reader as reader_module
        _label, _qc, _canon = (reader_module.quartz_choice_label, reader_module.canonicalize_quartz_answer,
                               reader_module.canonicalize_reader_answer)

        def lenient_label(question, answer):
            try:
                return _label(question, answer)
            except reader_module.ReaderProtocolError:
                return str(answer).strip().casefold()

        def lenient_qc(question, answer):
            try:
                return _qc(question, answer)
            except (reader_module.ReaderProtocolError, KeyError):
                return str(answer).strip()

        def lenient_canon(question, answer, *, reader_mode):
            try:
                return _canon(question, answer, reader_mode=reader_mode)
            except (reader_module.ReaderProtocolError, KeyError):
                return str(answer).strip()

        reader_module.quartz_choice_label = lenient_label
        reader_module.canonicalize_quartz_answer = lenient_qc
        reader_module.canonicalize_reader_answer = lenient_canon
    reader = ReaderClient(base_url=f"http://127.0.0.1:{port}/v1", model="qwen2.5-7b", timeout=1800, reader_mode=mode)

    def run(row):
        q = row["id"]
        if row.get("evaluation_status") != "evaluated" or "mice_rag" not in row.get("methods", {}):
            return row, 0
        record = records[q]
        question = str(record["question"])
        by_id = {str(u["unit_id"]): u for u in units_from_cache_row(record, units[q], k=5)}
        eligible = [u for u in gxi.get(q, []) if u in by_id and pool.is_eligible(u)]
        budget = int(row.get("matched_token_budget", 0))
        entry = {"status": "no_candidate", "matched_token_budget": budget, "selected_ids": [],
                 "n_modified_tokens": 0, "verified_flip": False, "reader_called": False,
                 "search": "gradient_order_first_flip", "attempts": 0}
        calls = 0
        for k in range(1, min(budget, len(eligible)) + 1):
            sel = eligible[:k]
            prow = pool.require(sel)
            repl = {u: stable_shared_candidate(prow[u]["candidates"], unit_id=u, seed=0) for u in sel}
            rev = apply_token_replacements(record, [by_id[u] for u in sel], repl, k=5)
            if int(rev["n_failed_edits"]) > 0 or int(rev["n_edits"]) != len(sel):
                entry.update(status="protocol_violation_failed_edit", selected_ids=sel)
                break
            answer = reader.answer(question, rev["edited_contexts"])
            calls += 1
            try:
                flip = not reader_answers_exact_match(question, answer, str(row["clean_answer"]), reader_mode=mode)
            except Exception:  # QuaRTz answers that map to no choice: compare raw strings
                flip = answer.strip().casefold() != str(row["clean_answer"]).strip().casefold()
            entry.update(status="verified_flip" if flip else "verified_no_flip", selected_ids=sel,
                         selected_tokens=[by_id[u]["text"] for u in sel], n_modified_tokens=len(sel),
                         edits=rev["edits"], edited_answer=answer, verified_flip=flip,
                         reader_called=True, attempts=k)
            if flip:
                break
        new = dict(row)
        new["methods"] = dict(row["methods"], mice_rag=entry)
        new["reader_calls"] = int(row.get("reader_calls", 0)) - int(bool(row["methods"]["mice_rag"].get("reader_called"))) + calls
        return new, calls

    results, lock, total = [None] * len(rows), threading.Lock(), [0, 0]
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, (new, calls) in enumerate(ex.map(run, rows)):
            results[i] = new
            total[0] += calls
            total[1] += bool(new.get("methods", {}).get("mice_rag", {}).get("verified_flip"))
    with open(f"{out_dir}/baselines_1000.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"dataset": ds, "editor_pool": editor or "qwen_original", "variant": "mice_rag gradient order, first-flip verification",
               "mice_reader_calls": total[0], "mice_verified_flips": total[1], "queries": len(rows)}
    json.dump(summary, open(f"{out_dir}/baselines_1000.summary.json", "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
