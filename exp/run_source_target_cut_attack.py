"""Experimental baseline: run source-to-target editable-token vertex cuts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.graph_cut import project_source_target_graph, solve_source_target_vertex_cut
from causalityrag.io import iter_records, load_records, retrieved_contexts
from causalityrag.reader import ReaderClient, answers_match
from causalityrag.revision import apply_token_deletions, apply_typed_token_revisions
from causalityrag.rules import TypedRuleLibrary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cf-pools", default="")
    parser.add_argument("--type-rules", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--use-graph-target", action="store_true")
    parser.add_argument("--all-tokens", action="store_true")
    parser.add_argument("--edit-mode", choices=["typed-replace", "delete"], default="typed-replace")
    args = parser.parse_args()

    if args.start < 0:
        raise ValueError("--start must be non-negative")
    records = list(iter_records(args.input, args.start + args.n))[args.start:]
    graphs = load_records(args.graphs)[args.start : args.start + len(records)]
    if len(graphs) != len(records):
        raise ValueError(f"graph rows ({len(graphs)}) do not match records ({len(records)})")
    if args.all_tokens and args.edit_mode != "delete":
        parser.error("--all-tokens requires --edit-mode delete")
    if not args.all_tokens and not args.cf_pools:
        parser.error("--cf-pools is required unless --all-tokens is used")
    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None) if args.cf_pools else None
    reader = ReaderClient()
    nlp = None
    if not args.all_tokens:
        import spacy

        nlp = spacy.load(os.environ.get("YVETTE_SPACY_MODEL", "en_core_web_lg"))
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for index, (record, graph_row) in enumerate(zip(records, graphs), 1):
            started = time.monotonic()
            contexts = retrieved_contexts(record)[: args.k]
            clean_answer = (
                str(graph_row.get("target_answer", ""))
                if args.use_graph_target
                else reader.answer(str(record.get("question", "")), contexts)
            )
            units, source_edges, interactions, target_edges, projection = project_source_target_graph(
                record,
                graph_row,
                library,
                k=args.k,
                nlp=nlp,
                all_context_tokens=args.all_tokens,
            )
            cut = solve_source_target_vertex_cut(units, source_edges, interactions, target_edges)
            selected_ids = set(cut["selected_ids"])
            selected = [unit for unit in units if unit["unit_id"] in selected_ids]
            revision = (
                apply_token_deletions(record, selected, k=args.k)
                if args.edit_mode == "delete"
                else apply_typed_token_revisions(record, selected, library, k=args.k)
            )
            if selected:
                edited_answer = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
                changed = not answers_match(clean_answer, edited_answer)
            else:
                edited_answer = clean_answer
                changed = False
            row = {
                "id": graph_row.get("id", ""),
                "question": str(record.get("question", "")),
                "gold_answer": str(record.get("answer", "")),
                "clean_answer": clean_answer,
                "edited_answer": edited_answer,
                "answer_changed": changed,
                "method": "exact_query_source_to_target_score_minimum_chunk_token_vertex_cut",
                "selection_domain": "editable_chunk_tokens_only",
                "unit_domain": "all_context_words" if args.all_tokens else "typed_editable_words",
                "edit_mode": args.edit_mode,
                "projection": projection,
                "cut": cut,
                "n_editable_units": len(units),
                "selected_units": selected,
                "n_selected": len(selected),
                "edits": revision["edits"],
                "n_edits": revision["n_edits"],
                "n_failed_edits": revision["n_failed_edits"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[source-target-cut] {index}/{len(records)} status={cut['status']} "
                f"selected={row['n_selected']} flip={changed} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    optimal = [row for row in rows if row["cut"]["status"] == "optimal"]
    flips = [row for row in rows if row["answer_changed"]]
    print("[source-target-cut summary]", json.dumps({
        "records": len(rows),
        "optimal_cuts": len(optimal),
        "flips": len(flips),
        "flip_rate": round(len(flips) / max(1, len(rows)), 4),
        "conditional_flip_rate": round(len(flips) / max(1, len(optimal)), 4),
        "avg_selected": round(sum(row["n_selected"] for row in optimal) / max(1, len(optimal)), 3),
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
