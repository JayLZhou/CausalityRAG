"""Contract saved closed message-flow graphs into final token-label graphs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from itertools import zip_longest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import contract_message_flow_row
from causalityrag.io import record_id
from causalityrag.token_units import units_from_cache_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--raw-graphs", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.k <= 0 or args.progress_every <= 0:
        parser.error("--k and --progress-every must be positive")
    if os.path.abspath(args.raw_graphs) == os.path.abspath(args.out):
        parser.error("--out must not overwrite --raw-graphs")

    started = time.monotonic()
    temporary = args.out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    status_counts: Counter[str] = Counter()
    rows = 0
    token_nodes = 0
    source_edges = 0
    interaction_edges = 0
    target_edges = 0

    try:
        with (
            open(args.input, encoding="utf-8") as input_file,
            open(args.raw_graphs, encoding="utf-8") as graph_file,
            open(args.units_cache, encoding="utf-8") as units_file,
            open(temporary, "w", encoding="utf-8") as output,
        ):
            for index, lines in enumerate(
                zip_longest(input_file, graph_file, units_file),
                1,
            ):
                if any(line is None for line in lines):
                    raise ValueError(
                        "input, raw graph, and token-unit files have different lengths"
                    )
                record, message_flow, units_row = (
                    json.loads(line) for line in lines
                )
                identifier = record_id(record)
                if str(message_flow.get("id", "")) != identifier:
                    raise ValueError(f"raw graph ID mismatch at row {index}")
                if str(units_row.get("id", "")) != identifier:
                    raise ValueError(f"token-unit ID mismatch at row {index}")

                units = units_from_cache_row(record, units_row, k=args.k)
                target_answer = str(
                    message_flow.get(
                        "clean_answer",
                        message_flow.get("target_answer", ""),
                    )
                )
                row = contract_message_flow_row(
                    record,
                    target_answer,
                    message_flow,
                    token_units=units,
                    k=args.k,
                )
                row["clean_answer"] = target_answer
                row["clean_correct"] = bool(
                    message_flow.get("clean_correct", False)
                )
                row["target_source"] = str(
                    message_flow.get("target_source", "saved_raw_message_flow")
                )
                row["raw_graph_source"] = os.path.abspath(args.raw_graphs)
                graph = row["contribution_graph"]
                expected = {str(unit["unit_id"]) for unit in units}
                actual = {str(unit_id) for unit_id in graph["token_nodes"]}
                if actual != expected:
                    raise ValueError(
                        f"incomplete token domain at row {index}: "
                        f"missing={len(expected - actual)}, "
                        f"unexpected={len(actual - expected)}"
                    )

                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                status_counts[str(row["status"])] += 1
                token_nodes += len(graph["token_nodes"])
                source_edges += len(graph["source_edges"])
                interaction_edges += len(graph["interaction_edges"])
                target_edges += len(graph["target_edges"])
                if rows % args.progress_every == 0:
                    print(
                        f"[contract-contribution-graph] rows={rows} "
                        f"status={row['status']} "
                        f"tokens={len(graph['token_nodes'])} "
                        f"interactions={len(graph['interaction_edges'])}",
                        flush=True,
                    )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    summary = {
        "rows": rows,
        "status_histogram": dict(sorted(status_counts.items())),
        "mean_token_nodes": token_nodes / max(1, rows),
        "mean_source_edges": source_edges / max(1, rows),
        "mean_interaction_edges": interaction_edges / max(1, rows),
        "mean_target_edges": target_edges / max(1, rows),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "raw_graphs": os.path.abspath(args.raw_graphs),
        "out": os.path.abspath(args.out),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[contract-contribution-graph summary] " + rendered, flush=True)
    if args.summary_out:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.summary_out)),
            exist_ok=True,
        )
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


if __name__ == "__main__":
    main()
