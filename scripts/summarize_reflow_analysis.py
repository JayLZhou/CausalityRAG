#!/usr/bin/env python3
"""Aggregate the four Section 6.4 analyses from frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.evaluation_metrics import answer_changed, valid_answer
from causalityrag.reader import answer_token_f1


DATASETS = (
    "hotpotqa",
    "timeqa",
    "finqa",
    "musique",
    "quartz",
    "triviaqa",
    "2wiki",
    "popqa",
)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_metrics(path: Path) -> dict:
    rows = read_jsonl(path)
    valid_rows = [
        row
        for row in rows
        if valid_answer(row.get("clean_answer", ""))
        and (
            int(row.get("reader_calls", 0)) == 0
            or valid_answer(row.get("edited_answer", ""))
        )
    ]
    clean_f1 = [
        row for row in valid_rows
        if answer_token_f1(str(row.get("clean_answer", "")), str(row.get("gold_answer", ""))) == 1.0
    ]
    return {
        "queries": len(rows),
        "valid_answer_queries": len(valid_rows),
        "answer_flip_rate": sum(
            int(row.get("reader_calls", 0)) > 0
            and answer_changed(row.get("clean_answer", ""), row.get("edited_answer", ""))
            for row in valid_rows
        ) / len(valid_rows),
        "f1_cfr": sum(
            answer_token_f1(str(row.get("edited_answer", "")), str(row.get("gold_answer", ""))) < 1.0
            for row in clean_f1
        ) / max(1, len(clean_f1)),
        "mean_final_tokens": statistics.fmean(int(row.get("n_modified_tokens", 0)) for row in valid_rows),
        "mean_reader_calls": statistics.fmean(int(row.get("reader_calls", 0)) for row in valid_rows),
        "mean_wall_seconds": statistics.fmean(float(row.get("elapsed_seconds", 0.0)) for row in valid_rows),
        "sha256": sha256(path),
    }


def graph_paths(root: Path, dataset: str) -> tuple[Path, Path, Path]:
    base = root / dataset
    if dataset == "hotpotqa":
        return (
            base / "graphs/contribution_graph_token_label_1000.jsonl",
            base / "audits/final_top10pool_k5/reflow_frontier_1000.jsonl",
            base / "audits/final_top10pool_k5/reflow_1000_v2.jsonl",
        )
    return (
        base / "graphs/contribution_graph_top5_1000.jsonl",
        base / "methods/reflow/frontier_top5_1000.jsonl",
        base / "methods/reflow/results_top5_1000.jsonl",
    )


def scaling_record(root: Path, dataset: str) -> dict:
    graph_path, frontier_path, result_path = graph_paths(root, dataset)
    graphs = read_jsonl(graph_path)
    frontiers = read_jsonl(frontier_path)
    results = read_jsonl(result_path)
    if not (len(graphs) == len(frontiers) == len(results) == 1000):
        raise ValueError(f"{dataset}: scaling artifacts must each contain 1000 rows")
    nodes = [len(row.get("contribution_graph", {}).get("token_nodes", [])) for row in graphs]
    edges = [
        sum(
            len(row.get("contribution_graph", {}).get(key, []))
            for key in ("source_edges", "interaction_edges", "target_edges")
        )
        for row in graphs
    ]
    graph_times = [
        float(row.get("elapsed_seconds", 0.0))
        for row in graphs
        if float(row.get("elapsed_seconds", 0.0)) > 0.0
    ]
    if dataset == "hotpotqa":
        raw_path = root / dataset / "graphs/contribution_graph_closed_beam48_1000.jsonl"
        graph_times = [
            float(row.get("elapsed_seconds", 0.0)) for row in iter_jsonl(raw_path)
        ]

    def editable_count(row: dict) -> int:
        value = row.get("editable_token_units", [])
        return int(value) if isinstance(value, (int, float)) else len(value)

    return {
        "dataset": dataset,
        "mean_nodes": statistics.fmean(nodes),
        "mean_editable_tokens": statistics.fmean(
            editable_count(row) for row in frontiers
        ),
        "mean_positive_edges": statistics.fmean(edges),
        "mean_frontier_candidates": statistics.fmean(
            int(row.get("n_frontier", 0)) for row in frontiers
        ),
        "mean_maxflow_calls": statistics.fmean(
            int(row.get("frontier_diagnostics", {}).get("total_maxflow_calls", 0))
            for row in frontiers
        ),
        "mean_graph_seconds": statistics.fmean(graph_times),
        "mean_frontier_seconds": statistics.fmean(
            float(row.get("elapsed_seconds", 0.0)) for row in frontiers
        ),
        "mean_reader_seconds": statistics.fmean(
            float(row.get("elapsed_seconds", 0.0)) for row in results
        ),
        "mean_reader_calls": statistics.fmean(
            int(row.get("reader_calls", 0)) for row in results
        ),
        "source_sha256": {
            "graph": sha256(graph_path),
            "frontier": sha256(frontier_path),
            "results": sha256(result_path),
        },
    }


def scaling(root: Path) -> list[dict]:
    return [scaling_record(root, dataset) for dataset in DATASETS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--one-shot", type=Path)
    parser.add_argument("--sweep", type=Path)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--fidelity-summary", type=Path)
    parser.add_argument("--oracle-summary", type=Path)
    parser.add_argument(
        "--base-analysis",
        type=Path,
        help="inherit seven frozen final-suite datasets and replace PubMedQA with PopQA",
    )
    parser.add_argument("--popqa-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.base_analysis:
        if args.popqa_root is None:
            parser.error("--popqa-root is required with --base-analysis")
        base = json.loads(args.base_analysis.read_text(encoding="utf-8"))
        inherited = {
            row["dataset"]: row
            for row in base["scaling"]
            if row["dataset"] in DATASETS and row["dataset"] != "popqa"
        }
        if set(inherited) != set(DATASETS) - {"popqa"}:
            parser.error("base analysis does not contain the seven frozen datasets")
        popqa = scaling_record(args.popqa_root.parent, args.popqa_root.name)
        result = dict(base)
        result.update(
            {
                "schema": "causalityrag.reflow_analysis.v3",
                "scaling": [
                    popqa if dataset == "popqa" else inherited[dataset]
                    for dataset in DATASETS
                ],
                "provenance": {
                    "base_analysis": str(args.base_analysis.resolve()),
                    "base_analysis_sha256": sha256(args.base_analysis),
                    "inherited_scaling_datasets": list(DATASETS[:-1]),
                    "recomputed_scaling_datasets": ["popqa"],
                },
            }
        )
    else:
        required = (
            args.out_root,
            args.one_shot,
            args.sweep,
            args.frontier,
            args.fidelity_summary,
            args.oracle_summary,
        )
        if any(value is None for value in required):
            parser.error("legacy full aggregation requires all artifact arguments")
        result = {
            "schema": "causalityrag.reflow_analysis.v3",
            "selection_protocols": {
                "one_shot": protocol_metrics(args.one_shot),
                "fixed_grid_sweep": protocol_metrics(args.sweep),
                "exact_frontier": protocol_metrics(args.frontier),
            },
            "scaling": scaling(args.out_root),
            "fidelity": json.loads(args.fidelity_summary.read_text(encoding="utf-8")),
            "restricted_oracle": json.loads(args.oracle_summary.read_text(encoding="utf-8")),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
