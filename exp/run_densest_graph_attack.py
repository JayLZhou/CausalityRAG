"""Experimental baseline: evaluate a densest attribution word subgraph."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.graph_cut import (
    project_graph_interactions,
    solve_weighted_densest_subgraph,
)
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
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--use-graph-target", action="store_true")
    parser.add_argument(
        "--objective",
        choices=["interaction-only"],
        default="interaction-only",
    )
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
            clean_answer = (
                str(graph_row.get("target_answer", ""))
                if args.use_graph_target
                else reader.answer(str(record.get("question", "")), retrieved_contexts(record)[: args.k])
            )
            units, interactions = project_graph_interactions(record, graph_row, library, k=args.k, nlp=nlp)
            dense = solve_weighted_densest_subgraph(units, interactions)
            selected = [unit for unit in units if unit["unit_id"] in set(dense["selected_ids"])]
            revision = apply_typed_token_revisions(record, selected, library, k=args.k)
            edited_answer = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
            changed = not answers_match(clean_answer, edited_answer)
            row = {
                "id": graph_row.get("id", ""),
                "question": str(record.get("question", "")),
                "gold_answer": str(record.get("answer", "")),
                "clean_answer": clean_answer,
                "edited_answer": edited_answer,
                "answer_changed": changed,
                "method": f"exact_{args.objective}_densest_subgraph_on_attribution_graph",
                "graph_method": str(graph_row.get("method", "")),
                "objective": args.objective,
                "selection_domain": "editable_chunk_tokens_only",
                "selection_scope": "context-induced interaction projection only",
                "n_editable_units": len(units),
                "n_projected_interactions": len(interactions),
                "densest": dense,
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
                f"[densest-graph] {index}/{len(records)} selected={row['n_selected']} "
                f"pairs={len(interactions)} flip={changed} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    flips = [row for row in rows if row["answer_changed"]]
    nonempty = [row for row in rows if row["n_selected"]]
    print("[densest-graph summary]", json.dumps({
        "records": len(rows),
        "flips": len(flips),
        "flip_rate": round(len(flips) / max(1, len(rows)), 4),
        "nonempty": len(nonempty),
        "avg_selected": round(sum(row["n_selected"] for row in nonempty) / max(1, len(nonempty)), 3),
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
