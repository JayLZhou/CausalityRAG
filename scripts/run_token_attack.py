"""Run token ILP, apply typed revisions from the rule library, and optionally verify with a reader."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, retrieved_contexts
from causalityrag.reader import ReaderClient, answers_match
from causalityrag.revision import apply_typed_token_revisions
from causalityrag.rules import TypedRuleLibrary
from causalityrag.token_units import run_token_ilp_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cf-pools", required=True, help="previous cf_pools.json")
    parser.add_argument("--type-rules", default="", help="optional previous type_rules_llm.yaml")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--objective", choices=["min-cost", "budgeted", "search-flip"], default="budgeted")
    parser.add_argument("--tau-graph", type=float, default=0.2)
    parser.add_argument("--budget", type=float, default=8.0)
    parser.add_argument("--start-budget", type=int, default=1, help="for search-flip")
    parser.add_argument("--max-budget", type=int, default=20, help="for search-flip")
    parser.add_argument("--solver", choices=["auto", "scipy", "dp", "branch"], default="auto")
    parser.add_argument("--max-edits", type=int, default=0, help="cap applied edits after ILP selection; 0 = all")
    parser.add_argument("--verify-reader", action="store_true")
    parser.add_argument("--include-contexts", action="store_true")
    args = parser.parse_args()

    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None)
    reader = ReaderClient() if (args.verify_reader or args.objective == "search-flip") else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as out:
        for record in iter_records(args.input, args.n):
            contexts = retrieved_contexts(record)[: args.k]
            clean_answer = reader.answer(str(record.get("question", "")), contexts) if reader else ""

            attempts = []
            if args.objective == "search-flip":
                final_ilp = None
                final_revision = None
                final_edited_answer = ""
                for budget in range(args.start_budget, args.max_budget + 1):
                    ilp = run_token_ilp_record(
                        record,
                        k=args.k,
                        objective="budgeted",
                        budget=float(budget),
                        solver=args.solver,
                        top_candidates=10,
                    )
                    revision = apply_typed_token_revisions(record, ilp["selected_units"], library, k=args.k)
                    edited_answer = reader.answer(ilp["question"], revision["edited_contexts"])
                    changed = not answers_match(clean_answer, edited_answer)
                    attempts.append({
                        "budget": budget,
                        "cut_cost": ilp["cut_cost"],
                        "n_edits": revision["n_edits"],
                        "n_failed_edits": revision["n_failed_edits"],
                        "support_removed": ilp["support_removed"],
                        "edited_answer": edited_answer,
                        "answer_changed": changed,
                    })
                    final_ilp, final_revision, final_edited_answer = ilp, revision, edited_answer
                    if changed:
                        break
                ilp = final_ilp
                revision = final_revision
                edited_answer = final_edited_answer
            else:
                ilp = run_token_ilp_record(
                    record,
                    k=args.k,
                    objective=args.objective,
                    tau_graph=args.tau_graph,
                    budget=args.budget,
                    solver=args.solver,
                    top_candidates=10,
                )
                revision = apply_typed_token_revisions(
                    record,
                    ilp["selected_units"],
                    library,
                    k=args.k,
                    max_edits=args.max_edits,
                )
                edited_answer = reader.answer(ilp["question"], revision["edited_contexts"]) if reader else ""

            row = {
                "id": ilp["id"],
                "question": ilp["question"],
                "gold_answer": str(record.get("answer", "")),
                "context_ids": ilp["context_ids"],
                "ilp": {
                    "objective": ilp["objective"],
                    "solver": ilp["solver"],
                    "status": ilp["status"],
                    "cut_cost": ilp["cut_cost"],
                    "support_removed": ilp["support_removed"],
                    "target_support": ilp["target_support"],
                    "n_units": ilp["n_units"],
                    "selected_cut": ilp["selected_cut"],
                },
                "selected_units": ilp["selected_units"],
                "edits": revision["edits"],
                "n_edits": revision["n_edits"],
                "n_failed_edits": revision["n_failed_edits"],
            }
            if attempts:
                row["search_attempts"] = attempts
                row["min_flip_budget"] = attempts[-1]["budget"] if attempts[-1]["answer_changed"] else None
            if args.include_contexts:
                row["clean_contexts"] = contexts
                row["edited_contexts"] = revision["edited_contexts"]
            if reader:
                row["reader"] = {
                    "clean_answer": clean_answer,
                    "edited_answer": edited_answer,
                    "answer_changed": not answers_match(clean_answer, edited_answer),
                    "gold_to_clean_match": answers_match(str(record.get("answer", "")), clean_answer),
                }
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "records": len(rows),
        "avg_edits": round(sum(row["n_edits"] for row in rows) / max(1, len(rows)), 2),
        "verified": bool(reader),
        "answer_changed": sum(1 for row in rows if row.get("reader", {}).get("answer_changed")),
        "out": args.out,
    }
    print("[token-attack summary]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
