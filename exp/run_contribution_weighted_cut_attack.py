"""Evaluate the contribution-weighted token-gated min-cut."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.graph_cut import (  # noqa: E402
    project_cached_units_source_target_graph,
    solve_contribution_weighted_vertex_cut,
)
from causalityrag.io import iter_records, load_records, record_id  # noqa: E402
from causalityrag.reader import ReaderClient, answers_match  # noqa: E402
from causalityrag.revision import apply_token_replacements  # noqa: E402
from causalityrag.token_units import units_from_cache_row  # noqa: E402


def summarize(rows: list[dict], started: float) -> dict:
    valid = [row for row in rows if row["status"] == "optimal"]
    flips = [row for row in valid if row["answer_changed"]]
    return {
        "queries": len(rows),
        "valid": len(valid),
        "flips": len(flips),
        "overall_flip_rate": len(flips) / max(1, len(rows)),
        "conditional_flip_rate": len(flips) / max(1, len(valid)),
        "mean_tokens_valid": statistics.fmean(row["n_selected"] for row in valid) if valid else 0.0,
        "median_tokens_valid": statistics.median(row["n_selected"] for row in valid) if valid else 0.0,
        "reader_calls": sum(row["reader_calls"] for row in rows),
        "status_histogram": dict(Counter(row["status"] for row in rows)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--report-every", type=int, default=100)
    args = parser.parse_args()

    unit_rows = {str(row["id"]): row for row in load_records(args.units_cache)}
    registries = {
        str(row["id"]): row.get("replacements", {})
        for row in load_records(args.replacement_registry)
    }
    records = list(islice(iter_records(args.input), args.n))
    graphs = list(islice(iter_records(args.graphs), args.n))
    if len(records) != len(graphs):
        raise ValueError(f"input/graph length mismatch: {len(records)} != {len(graphs)}")
    started = time.monotonic()

    def process(index: int) -> dict:
        record = records[index]
        graph = graphs[index]
        identifier = record_id(record)
        if str(graph.get("id", "")) != identifier:
            raise ValueError(f"unaligned graph at query {index}")
        clean_answer = str(graph.get("clean_answer", "")).strip()
        if graph.get("status") != "ok" or not clean_answer:
            return {
                "index": index,
                "id": identifier,
                "status": str(graph.get("status", "missing_clean_answer")),
                "answer_changed": False,
                "n_selected": 0,
                "reader_calls": 0,
                "selected_ids": [],
                "selected_tokens": [],
                "cut": {},
            }
        replacements = registries.get(identifier, {})
        units = [
            unit
            for unit in units_from_cache_row(record, unit_rows[identifier], k=args.k)
            if replacements.get(str(unit["unit_id"]), {}).get("ok")
        ]
        by_id = {str(unit["unit_id"]): unit for unit in units}
        source, interactions, target, projection = project_cached_units_source_target_graph(graph, units)
        cut = solve_contribution_weighted_vertex_cut(units, source, interactions, target)
        selected_ids = list(cut.get("selected_ids", [])) if cut.get("status") == "optimal" else []
        selected_units = [by_id[unit_id] for unit_id in selected_ids]
        if selected_units:
            revision = apply_token_replacements(record, selected_units, replacements, k=args.k)
            edited_answer = ReaderClient().answer(
                str(record.get("question", "")),
                revision["edited_contexts"],
            )
            changed = not answers_match(clean_answer, edited_answer)
            edits = revision["edits"]
            reader_calls = 1
        else:
            edited_answer = clean_answer
            changed = False
            edits = []
            reader_calls = 0
        return {
            "index": index,
            "id": identifier,
            "question": str(record.get("question", "")),
            "clean_answer": clean_answer,
            "edited_answer": edited_answer,
            "answer_changed": changed,
            "status": str(cut.get("status", "empty")),
            "selected_ids": selected_ids,
            "selected_tokens": [str(by_id[unit_id].get("text", "")) for unit_id in selected_ids],
            "n_selected": len(selected_ids),
            "reader_calls": reader_calls,
            "edits": edits,
            "cut": cut,
            "projection": projection,
            "n_candidate_tokens": len(units),
            "method": "contribution_weighted_token_gated_min_cut",
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows: list[dict] = []
    next_report = args.report_every
    with open(args.out, "w", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, index) for index in range(len(records))]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(rows) >= next_report or len(rows) == len(records):
                report = summarize(rows, started)
                report["processed"] = len(rows)
                print("[contribution-weighted-cut progress] " + json.dumps(report, ensure_ascii=False), flush=True)
                with open(args.summary_out + ".progress.json", "w", encoding="utf-8") as progress:
                    json.dump(report, progress, ensure_ascii=False, indent=2)
                next_report += args.report_every
        output.flush()

    summary = summarize(rows, started)
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print("[contribution-weighted-cut summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
