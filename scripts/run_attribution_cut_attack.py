"""Find attribution-support cuts, edit their tokens, and verify answer changes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.graph_cut import project_graph_support_to_editable_units, top_budgeted_cut_sets
from causalityrag.io import iter_records, load_records, retrieved_contexts
from causalityrag.reader import ReaderClient, answers_match
from causalityrag.revision import apply_typed_token_revisions
from causalityrag.rules import TypedRuleLibrary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cf-pools", required=True)
    parser.add_argument("--type-rules", default="")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-budget", type=int, default=8)
    parser.add_argument("--cuts-per-budget", type=int, default=3)
    parser.add_argument("--include-contexts", action="store_true")
    args = parser.parse_args()

    records = list(iter_records(args.input, args.n))
    graphs = load_records(args.graphs)[: len(records)]
    if len(graphs) != len(records):
        raise ValueError(f"graph rows ({len(graphs)}) do not match records ({len(records)})")
    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None)
    reader = ReaderClient()
    import spacy

    nlp = spacy.load(os.environ.get("YVETTE_SPACY_MODEL", "en_core_web_lg"))

    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for index, (record, graph_row) in enumerate(zip(records, graphs), 1):
            started = time.monotonic()
            contexts = retrieved_contexts(record)[: args.k]
            clean_answer = reader.answer(str(record.get("question", "")), contexts)
            units = project_graph_support_to_editable_units(record, graph_row, library, k=args.k, nlp=nlp)
            total_support = sum(float(unit["support"]) for unit in units)
            attempts = []
            winner = None

            for budget in range(1, min(args.max_budget, len(units)) + 1):
                cuts = top_budgeted_cut_sets(units, budget, limit=args.cuts_per_budget)
                for candidate_rank, cut in enumerate(cuts, 1):
                    revision = apply_typed_token_revisions(record, cut, library, k=args.k)
                    edited_answer = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
                    changed = not answers_match(clean_answer, edited_answer)
                    attempt = {
                        "budget": budget,
                        "candidate_rank": candidate_rank,
                        "cut": [unit["unit_id"] for unit in cut],
                        "selected_units": cut,
                        "support_removed": sum(float(unit["support"]) for unit in cut),
                        "support_fraction": (
                            sum(float(unit["support"]) for unit in cut) / total_support
                            if total_support else 0.0
                        ),
                        "edits": revision["edits"],
                        "n_edits": revision["n_edits"],
                        "n_failed_edits": revision["n_failed_edits"],
                        "edited_answer": edited_answer,
                        "answer_changed": changed,
                    }
                    attempts.append(attempt)
                    if changed:
                        winner = {**attempt, "edited_contexts": revision["edited_contexts"]}
                        break
                if winner:
                    break

            row = {
                "id": graph_row.get("id", ""),
                "question": str(record.get("question", "")),
                "gold_answer": str(record.get("answer", "")),
                "clean_answer": clean_answer,
                "graph_target_answer": graph_row.get("target_answer", ""),
                "graph_target_matches_clean": answers_match(str(graph_row.get("target_answer", "")), clean_answer),
                "method": "gradient_attention_budgeted_support_cut_with_reader_verification",
                "n_editable_units": len(units),
                "total_editable_support": total_support,
                "top_editable_units": units[:20],
                "attempts": attempts,
                "verified_cut_found": winner is not None,
                "min_verified_cut_size": winner["n_edits"] if winner else None,
                "verified_cut": winner,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            if args.include_contexts:
                row["clean_contexts"] = contexts
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[attribution-cut] {index}/{len(records)} units={len(units)} "
                f"found={row['verified_cut_found']} size={row['min_verified_cut_size']} "
                f"attempts={len(attempts)} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    found = [row for row in rows if row["verified_cut_found"]]
    print("[attribution-cut summary]", json.dumps({
        "records": len(rows),
        "verified_cuts": len(found),
        "avg_verified_cut_size": (
            round(sum(row["min_verified_cut_size"] for row in found) / len(found), 3) if found else None
        ),
        "reader_calls": sum(1 + len(row["attempts"]) for row in rows),
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
