"""Experimental baseline: peel source-target densest chunk subgraphs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.graph_cut import (
    project_source_target_graph,
    solve_source_target_densest_subgraph,
)
from causalityrag.io import iter_records, load_records, retrieved_contexts
from causalityrag.reader import ReaderClient, answers_match
from causalityrag.replacement import GenericReplacementClient
from causalityrag.revision import apply_token_replacements
from causalityrag.rules import TypedRuleLibrary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cf-pools", required=True)
    parser.add_argument("--type-rules", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--use-graph-target", action="store_true")
    args = parser.parse_args()

    if args.start < 0 or args.max_rounds < 0:
        raise ValueError("--start and --max-rounds must be non-negative")
    records = list(iter_records(args.input, args.start + args.n))[args.start:]
    graphs = load_records(args.graphs)[args.start : args.start + len(records)]
    if len(graphs) != len(records):
        raise ValueError(f"graph rows ({len(graphs)}) do not match records ({len(records)})")
    reader = ReaderClient()
    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None)
    generic_editor = GenericReplacementClient()
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
                record, graph_row, None, k=args.k, all_context_tokens=True
            )
            by_id = {str(unit["unit_id"]): unit for unit in units}
            remaining_ids = set(by_id)
            selected_ids: set[str] = set()
            replacements: dict[str, dict] = {}
            attempts = []
            edited_answer = clean_answer
            changed = False
            while remaining_ids and (args.max_rounds <= 0 or len(attempts) < args.max_rounds):
                candidate = solve_source_target_densest_subgraph(
                    [by_id[unit_id] for unit_id in sorted(remaining_ids)],
                    source_edges,
                    interactions,
                    target_edges,
                )
                if not candidate["selected_ids"]:
                    break
                selected_ids.update(candidate["selected_ids"])
                remaining_ids -= set(candidate["selected_ids"])
                for unit_id in candidate["selected_ids"]:
                    unit = by_id[unit_id]
                    typed = library.replacement_for_token(
                        unit["text"], str(unit.get("type", "")), str(record.get("question", ""))
                    )
                    if typed.get("ok"):
                        replacements[unit_id] = typed
                    else:
                        context = next(
                            (ctx["text"] for ctx in contexts if ctx["chunk_id"] == unit["chunk_id"]),
                            "",
                        )
                        replacements[unit_id] = generic_editor.replace(
                            unit["text"], context, str(unit.get("type", ""))
                        )
                selected = [by_id[unit_id] for unit_id in sorted(selected_ids)]
                revision = apply_token_replacements(record, selected, replacements, k=args.k)
                edited_answer = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
                changed = not answers_match(clean_answer, edited_answer)
                attempts.append({
                    **candidate,
                    "cumulative_selected_ids": sorted(selected_ids),
                    "cumulative_n_selected": len(selected_ids),
                    "edited_answer": edited_answer,
                    "answer_changed": changed,
                })
                if changed:
                    break

            selected = [by_id[unit_id] for unit_id in sorted(selected_ids)]
            revision = apply_token_replacements(record, selected, replacements, k=args.k)
            row = {
                "id": graph_row.get("id", ""),
                "question": str(record.get("question", "")),
                "gold_answer": str(record.get("answer", "")),
                "clean_answer": clean_answer,
                "edited_answer": edited_answer,
                "answer_changed": changed,
                "method": "iterative_source_target_densest_subgraph_decomposition_with_cumulative_deletion",
                "selection_domain": "all_chunk_word_tokens",
                "edit_mode": "typed_or_generic_replacement",
                "projection": projection,
                "max_rounds": args.max_rounds,
                "attempts": attempts,
                "rounds_tried": len(attempts),
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
                f"[densest-decomposition] {index}/{len(records)} rounds={row['rounds_tried']} "
                f"selected={row['n_selected']} flip={changed} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    flips = [row for row in rows if row["answer_changed"]]
    with_attempts = [row for row in rows if row["rounds_tried"]]
    print("[densest-decomposition summary]", json.dumps({
        "records": len(rows),
        "flips": len(flips),
        "flip_rate": round(len(flips) / max(1, len(rows)), 4),
        "with_attempts": len(with_attempts),
        "conditional_flip_rate": round(len(flips) / max(1, len(with_attempts)), 4),
        "avg_rounds": round(sum(row["rounds_tried"] for row in rows) / max(1, len(rows)), 3),
        "avg_selected": round(sum(row["n_selected"] for row in flips) / max(1, len(flips)), 3),
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
