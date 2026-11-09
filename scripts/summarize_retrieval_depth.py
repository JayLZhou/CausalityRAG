#!/usr/bin/env python3
"""Summarize one retrieval-depth run from final, audited artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


DATASETS = (
    ("hotpotqa", "HotpotQA"),
    ("timeqa", "TimeQA"),
    ("finqa", "FinQA"),
    ("musique", "MuSiQue"),
    ("quartz", "QuaRTz"),
    ("triviaqa", "TriviaQA"),
    ("2wiki", "2Wiki"),
    ("popqa", "PopQA"),
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from causalityrag.evaluation_metrics import (  # pylint: disable=import-outside-toplevel
        answer_changed,
        gold_answer_spec,
        gold_f1,
        valid_answer,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_base = args.run_root / f"k{args.k}"
    run_manifest = load_json(args.run_root / "manifest.json")
    dataset_results = []

    for dataset, label in DATASETS:
        base = run_base / dataset
        paths = {
            "input": base / "inputs/retrieval.jsonl",
            "clean": base / "clean/targets.jsonl",
            "graph_initial": base / "graph/contribution_graph.jsonl",
            "frontier_initial": base / "frontier/initial.jsonl",
            "factual_initial": base / "factual/initial_results.jsonl",
            "graph_final": base / "final/graphs.jsonl",
            "frontier_final": base / "final/frontier.jsonl",
            "factual_final": base / "factual/results.jsonl",
        }
        rows = {name: load_jsonl(path) for name, path in paths.items()}
        input_ids = [str(row["id"]) for row in rows["input"]]
        if len(input_ids) != 1000 or len(set(input_ids)) != 1000:
            raise ValueError(f"{dataset}: input must contain 1000 unique IDs")
        for name, values in rows.items():
            if [str(row["id"]) for row in values] != input_ids:
                raise ValueError(f"{dataset}: {name} IDs do not align with input")

        factual_summary = load_json(base / "factual/summary.json")
        clean_summary = load_json(base / "clean/summary.json")
        graph_summary = load_json(base / "graph/summary.json")
        graph_repair = load_json(base / "final/graphs.summary.json")
        frontier_repair = load_json(base / "final/frontier.summary.json")
        final_factual = rows["factual_final"]
        final_graph = rows["graph_final"]

        answer_denominator = []
        f1_denominator = []
        answer_flips = 0
        f1_flips = 0
        for row in final_factual:
            status = str(row.get("evaluation_status", ""))
            clean_answer = str(row.get("clean_answer", ""))
            edited_answer = str(row.get("edited_answer", ""))
            if not row.get("eligible", False) or not valid_answer(clean_answer):
                continue
            if status.startswith("protocol_violation") or not valid_answer(edited_answer):
                continue
            answer_denominator.append(row)
            executed = int(row.get("reader_calls", 0)) > 0
            answer_flips += int(executed and answer_changed(clean_answer, edited_answer))
            spec = gold_answer_spec(row, dataset)
            clean_f1 = gold_f1(clean_answer, spec)
            if spec.is_valid and clean_f1 >= 1.0 - 1e-12:
                f1_denominator.append(row)
                f1_flips += int(executed and gold_f1(edited_answer, spec) < clean_f1)

        if answer_flips != int(factual_summary["verified_flips"]):
            raise ValueError(f"{dataset}: recomputed Ans-FR flips disagree with summary")
        if len(answer_denominator) != int(factual_summary["answer_denominator_queries"]):
            raise ValueError(f"{dataset}: recomputed Ans-FR denominator disagrees with summary")

        graph_ok = [row for row in final_graph if row.get("status") == "ok"]
        node_counts = []
        edge_counts = []
        source_edge_counts = []
        interaction_edge_counts = []
        target_edge_counts = []
        for row in graph_ok:
            graph = row.get("contribution_graph", {})
            source_count = len(graph.get("source_edges", []))
            interaction_count = len(graph.get("interaction_edges", []))
            target_count = len(graph.get("target_edges", []))
            node_counts.append(len(graph.get("token_nodes", [])))
            source_edge_counts.append(source_count)
            interaction_edge_counts.append(interaction_count)
            target_edge_counts.append(target_count)
            edge_counts.append(source_count + interaction_count + target_count)

        clean_seconds = float(clean_summary.get("elapsed_seconds", 0.0))
        graph_seconds = sum(float(row.get("elapsed_seconds", 0.0)) for row in final_graph)
        frontier_seconds = sum(
            float(row.get("elapsed_seconds", 0.0)) for row in rows["frontier_final"]
        )
        factual_seconds = sum(
            float(row.get("elapsed_seconds", 0.0)) for row in final_factual
        )
        pipeline_seconds = clean_seconds + graph_seconds + frontier_seconds + factual_seconds
        statuses = Counter(str(row.get("evaluation_status", "")) for row in final_factual)
        invalid_clean = statuses.get("invalid_clean_answer", 0)
        invalid_no_executable = statuses.get("invalid_no_executable_domain", 0)
        protocol_violations = sum(
            count for status, count in statuses.items() if status.startswith("protocol_violation")
        )
        reader_calls = sum(int(row.get("reader_calls", 0)) for row in final_factual)
        mean_modified = mean(int(row.get("n_modified_tokens", 0)) for row in answer_denominator)
        if abs(mean_modified - float(factual_summary["mean_modified_tokens_all_queries"])) > 1e-12:
            raise ValueError(f"{dataset}: mean edited-token count disagrees with summary")

        manifest = load_json(base / "inputs/manifest.json")
        artifacts = {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        }
        artifacts.update({
            "input_manifest": {
                "path": str(base / "inputs/manifest.json"),
                "sha256": sha256(base / "inputs/manifest.json"),
            },
            "factual_summary": {
                "path": str(base / "factual/summary.json"),
                "sha256": sha256(base / "factual/summary.json"),
            },
        })
        result = {
            "dataset": dataset,
            "label": label,
            "retrieval_depth": args.k,
            "queries": 1000,
            "answer_flips": answer_flips,
            "answer_denominator": len(answer_denominator),
            "ans_fr": answer_flips / len(answer_denominator),
            "f1_flips": f1_flips,
            "f1_denominator": len(f1_denominator),
            "f1_fr": f1_flips / len(f1_denominator) if f1_denominator else None,
            "mean_edited_tokens": mean_modified,
            "graph_ok": len(graph_ok),
            "graph_coverage": len(graph_ok) / 1000,
            "mean_graph_nodes": mean(node_counts),
            "mean_graph_edges": mean(edge_counts),
            "mean_source_edges": mean(source_edge_counts),
            "mean_interaction_edges": mean(interaction_edge_counts),
            "mean_target_edges": mean(target_edge_counts),
            "mean_clean_seconds": clean_seconds / 1000,
            "mean_graph_seconds": graph_seconds / 1000,
            "mean_frontier_seconds": frontier_seconds / 1000,
            "mean_factual_seconds": factual_seconds / 1000,
            "mean_terminal_pipeline_seconds": pipeline_seconds / 1000,
            "total_terminal_pipeline_seconds": pipeline_seconds,
            "reader_calls": reader_calls,
            "invalid_clean": invalid_clean,
            "invalid_no_executable": invalid_no_executable,
            "protocol_violations": protocol_violations,
            "no_frontier_repaired": int(graph_repair.get("repaired_queries", 0)),
            "remaining_no_frontier": int(factual_summary.get("remaining_no_frontier", 0)),
            "remaining_empty_frontiers": int(frontier_repair.get("remaining_empty_frontiers", 0)),
            "pool_sha256": manifest["outputs"]["shared_pool_sha256"],
            "artifacts": artifacts,
            "status_histogram": dict(sorted(statuses.items())),
            "initial_graph_summary": graph_summary,
        }
        dataset_results.append(result)

    answer_denominator = sum(row["answer_denominator"] for row in dataset_results)
    f1_denominator = sum(row["f1_denominator"] for row in dataset_results)
    graph_queries = sum(row["graph_ok"] for row in dataset_results)
    aggregate = {
        "queries": sum(row["queries"] for row in dataset_results),
        "answer_flips": sum(row["answer_flips"] for row in dataset_results),
        "answer_denominator": answer_denominator,
        "ans_fr_micro": sum(row["answer_flips"] for row in dataset_results) / answer_denominator,
        "ans_fr_macro": mean(row["ans_fr"] for row in dataset_results),
        "f1_flips": sum(row["f1_flips"] for row in dataset_results),
        "f1_denominator": f1_denominator,
        "f1_fr_micro": sum(row["f1_flips"] for row in dataset_results) / f1_denominator,
        "f1_fr_macro": mean(row["f1_fr"] for row in dataset_results if row["f1_fr"] is not None),
        "mean_edited_tokens": (
            sum(row["mean_edited_tokens"] * row["answer_denominator"] for row in dataset_results)
            / answer_denominator
        ),
        "graph_ok": graph_queries,
        "graph_coverage": graph_queries / 8000,
        "mean_graph_nodes": (
            sum(row["mean_graph_nodes"] * row["graph_ok"] for row in dataset_results)
            / graph_queries
        ),
        "mean_graph_edges": (
            sum(row["mean_graph_edges"] * row["graph_ok"] for row in dataset_results)
            / graph_queries
        ),
        "mean_terminal_pipeline_seconds": (
            sum(row["total_terminal_pipeline_seconds"] for row in dataset_results) / 8000
        ),
        "reader_calls": sum(row["reader_calls"] for row in dataset_results),
        "invalid_clean": sum(row["invalid_clean"] for row in dataset_results),
        "invalid_no_executable": sum(
            row["invalid_no_executable"] for row in dataset_results
        ),
        "protocol_violations": sum(row["protocol_violations"] for row in dataset_results),
        "no_frontier_repaired": sum(row["no_frontier_repaired"] for row in dataset_results),
        "remaining_no_frontier": sum(row["remaining_no_frontier"] for row in dataset_results),
    }
    output = {
        "schema": "causalityrag.retrieval_depth_result.v1",
        "model": run_manifest["model"],
        "retrieval_depth": args.k,
        "queries_per_dataset": 1000,
        "metric_contract": {
            "ans_fr": "normalized clean-versus-edited answer change among valid clean and valid factual reader answers",
            "f1_fr": "loss of normalized token F1 among valid factual rows whose clean token F1 equals one",
            "invalid_exclusion": "ineligible clean answers, zero-executable-domain prefixes, and invalid factual reader answers are excluded from both metric numerators and denominators",
            "mean_edited_tokens": "mean terminal intervention size over the Ans-FR denominator",
            "graph_coverage": "final graph rows with status=ok divided by all 1000 queries",
            "graph_size": "mean final token nodes and all source/interactions/target edges over status=ok graphs",
            "latency": "sum of saved terminal clean, final graph, final frontier, and final factual elapsed time; excludes service startup and queue waiting",
            "control": "not defined by the retrieval-depth scheduler; this artifact reports factual ReFlow only",
        },
        "run_manifest": run_manifest,
        "datasets": dataset_results,
        "aggregate": aggregate,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    columns = [
        "dataset", "queries", "answer_flips", "answer_denominator", "ans_fr",
        "f1_flips", "f1_denominator", "f1_fr", "mean_edited_tokens",
        "graph_ok", "graph_coverage", "mean_graph_nodes", "mean_graph_edges",
        "mean_terminal_pipeline_seconds", "reader_calls", "invalid_clean",
        "invalid_no_executable", "protocol_violations", "no_frontier_repaired",
        "remaining_no_frontier",
        "pool_sha256",
    ]
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dataset_results)

    tex_lines = [
        f"% Qwen2.5-7B retrieval-depth k={args.k}; factual ReFlow only.",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Dataset & Ans-FR & F1-FR & Edits & Coverage & Nodes & Edges & E2E (s) \\\\",
        "\\midrule",
    ]
    for row in dataset_results:
        tex_lines.append(
            f"{row['label']} & {100 * row['ans_fr']:.1f} & {100 * row['f1_fr']:.1f} & "
            f"{row['mean_edited_tokens']:.2f} & {100 * row['graph_coverage']:.1f} & "
            f"{row['mean_graph_nodes']:.1f} & {row['mean_graph_edges']:.1f} & "
            f"{row['mean_terminal_pipeline_seconds']:.2f} \\\\" 
        )
    tex_lines.extend([
        "\\midrule",
        f"Micro/weighted & {100 * aggregate['ans_fr_micro']:.1f} & "
        f"{100 * aggregate['f1_fr_micro']:.1f} & {aggregate['mean_edited_tokens']:.2f} & "
        f"{100 * aggregate['graph_coverage']:.1f} & {aggregate['mean_graph_nodes']:.1f} & "
        f"{aggregate['mean_graph_edges']:.1f} & {aggregate['mean_terminal_pipeline_seconds']:.2f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ])
    (args.out_dir / "table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    readme_lines = [
        f"# Qwen2.5-7B retrieval-depth result: k={args.k}",
        "",
        "All values below are recomputed from the final audited 1,000-query artifacts.",
        "Invalid clean answers, zero-executable-domain prefixes, and invalid factual reader outputs are excluded from both metric numerators and denominators.",
        "The retrieval-depth scheduler has no separate control stage, so control is reported as N/A.",
        "",
        "| Dataset | Ans-FR | F1-FR | Mean edits | Graph coverage | Nodes | Edges | E2E s/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_results:
        readme_lines.append(
            f"| {row['label']} | {100 * row['ans_fr']:.2f}% ({row['answer_flips']}/{row['answer_denominator']}) "
            f"| {100 * row['f1_fr']:.2f}% ({row['f1_flips']}/{row['f1_denominator']}) "
            f"| {row['mean_edited_tokens']:.3f} | {100 * row['graph_coverage']:.2f}% "
            f"| {row['mean_graph_nodes']:.2f} | {row['mean_graph_edges']:.2f} "
            f"| {row['mean_terminal_pipeline_seconds']:.3f} |"
        )
    readme_lines.extend([
        f"| **Micro/weighted** | **{100 * aggregate['ans_fr_micro']:.2f}%** "
        f"| **{100 * aggregate['f1_fr_micro']:.2f}%** "
        f"| **{aggregate['mean_edited_tokens']:.3f}** "
        f"| **{100 * aggregate['graph_coverage']:.2f}%** "
        f"| **{aggregate['mean_graph_nodes']:.2f}** "
        f"| **{aggregate['mean_graph_edges']:.2f}** "
        f"| **{aggregate['mean_terminal_pipeline_seconds']:.3f}** |",
        "",
        "`summary.json` contains per-dataset status histograms, reader calls, latency components, pool hashes, and artifact hashes.",
        "`audit.json` is the independent final integrity audit.",
    ])
    (args.out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
