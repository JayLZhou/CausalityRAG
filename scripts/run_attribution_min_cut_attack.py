"""Verify answer flips from unbudgeted minimum attribution-support cuts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.graph_cut import minimum_support_cut, project_graph_support_to_editable_units
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
    parser.add_argument("--remaining-tau", type=float, default=0.2)
    parser.add_argument("--use-graph-target", action="store_true")
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
            clean_answer = (
                str(graph_row.get("target_answer", ""))
                if args.use_graph_target
                else reader.answer(str(record.get("question", "")), contexts)
            )
            units = project_graph_support_to_editable_units(record, graph_row, library, k=args.k, nlp=nlp)
            total_support = sum(float(unit["support"]) for unit in units)
            cut = minimum_support_cut(units, remaining_tau=args.remaining_tau)
            removed_support = sum(float(unit["support"]) for unit in cut)
            revision = apply_typed_token_revisions(record, cut, library, k=args.k)
            edited_answer = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
            changed = not answers_match(clean_answer, edited_answer)
            row = {
                "id": graph_row.get("id", ""),
                "question": str(record.get("question", "")),
                "gold_answer": str(record.get("answer", "")),
                "clean_answer": clean_answer,
                "edited_answer": edited_answer,
                "answer_changed": changed,
                "method": "unbudgeted_minimum_gradient_support_cut_with_reader_verification",
                "remaining_tau": args.remaining_tau,
                "n_editable_units": len(units),
                "total_editable_support": total_support,
                "cut_support_removed": removed_support,
                "cut_support_fraction": removed_support / total_support if total_support else 0.0,
                "selected_cut": [unit["unit_id"] for unit in cut],
                "selected_units": cut,
                "n_selected": len(cut),
                "edits": revision["edits"],
                "n_edits": revision["n_edits"],
                "n_failed_edits": revision["n_failed_edits"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[attribution-min-cut] {index}/{len(records)} selected={row['n_selected']} "
                f"edits={row['n_edits']} flip={changed} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    flips = [row for row in rows if row["answer_changed"]]
    print("[attribution-min-cut summary]", json.dumps({
        "records": len(rows),
        "flips": len(flips),
        "flip_rate": round(len(flips) / max(1, len(rows)), 4),
        "avg_selected": round(sum(row["n_selected"] for row in rows) / max(1, len(rows)), 3),
        "avg_selected_when_flipped": (
            round(sum(row["n_selected"] for row in flips) / len(flips), 3) if flips else None
        ),
        "reader_calls": len(rows) * (1 if args.use_graph_target else 2),
        "remaining_tau": args.remaining_tau,
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
