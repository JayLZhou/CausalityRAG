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

from causalityrag.reader import answer_token_f1, answers_exact_match


DATASETS = ("hotpotqa", "timeqa", "finqa", "musique", "quartz", "qasper", "2wiki", "pubmedqa")


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
    clean_f1 = [
        row for row in rows
        if answer_token_f1(str(row.get("clean_answer", "")), str(row.get("gold_answer", ""))) == 1.0
    ]
    return {
        "queries": len(rows),
        "answer_flip_rate": sum(bool(row.get("verified_flip")) for row in rows) / len(rows),
        "f1_cfr": sum(
            answer_token_f1(str(row.get("edited_answer", "")), str(row.get("gold_answer", ""))) < 1.0
            for row in clean_f1
        ) / max(1, len(clean_f1)),
        "mean_final_tokens": statistics.fmean(int(row.get("n_modified_tokens", 0)) for row in rows),
        "mean_reader_calls": statistics.fmean(int(row.get("reader_calls", 0)) for row in rows),
        "mean_wall_seconds": statistics.fmean(float(row.get("elapsed_seconds", 0.0)) for row in rows),
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


def scaling(root: Path) -> list[dict]:
    output = []
    for dataset in DATASETS:
        graph_path, frontier_path, result_path = graph_paths(root, dataset)
        graphs = read_jsonl(graph_path)
        frontiers = read_jsonl(frontier_path)
        results = read_jsonl(result_path)
        nodes = [len(row.get("contribution_graph", {}).get("token_nodes", [])) for row in graphs]
        edges = [
            sum(len(row.get("contribution_graph", {}).get(key, [])) for key in ("source_edges", "interaction_edges", "target_edges"))
            for row in graphs
        ]
        graph_times = [float(row.get("elapsed_seconds", 0.0)) for row in graphs if float(row.get("elapsed_seconds", 0.0)) > 0.0]
        if dataset == "hotpotqa":
            raw_path = root / dataset / "graphs/contribution_graph_closed_beam48_1000.jsonl"
            graph_times = [
                float(row.get("elapsed_seconds", 0.0)) for row in iter_jsonl(raw_path)
            ]
        output.append({
            "dataset": dataset,
            "mean_nodes": statistics.fmean(nodes),
            "mean_positive_edges": statistics.fmean(edges),
            "mean_frontier_candidates": statistics.fmean(int(row.get("n_frontier", 0)) for row in frontiers),
            "mean_maxflow_calls": statistics.fmean(
                int(row.get("frontier_diagnostics", {}).get("total_maxflow_calls", 0)) for row in frontiers
            ),
            "mean_graph_seconds": statistics.fmean(graph_times),
            "mean_frontier_seconds": statistics.fmean(float(row.get("elapsed_seconds", 0.0)) for row in frontiers),
            "mean_reader_seconds": statistics.fmean(float(row.get("elapsed_seconds", 0.0)) for row in results),
            "mean_reader_calls": statistics.fmean(int(row.get("reader_calls", 0)) for row in results),
            "source_sha256": {
                "graph": sha256(graph_path),
                "frontier": sha256(frontier_path),
                "results": sha256(result_path),
            },
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--one-shot", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--fidelity-summary", type=Path, required=True)
    parser.add_argument("--oracle-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "causalityrag.reflow_analysis.v1",
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
