#!/usr/bin/env python3
"""Render Table 2, Figure 7, and runtime TeX inputs from final Shapley outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DATASETS = (
    ("hotpotqa", "Hotpot"),
    ("timeqa", "TQA"),
    ("finqa", "FQA"),
    ("musique", "MSQ"),
    ("quartz", "QTZ"),
    ("triviaqa", "TrQA"),
    ("2wiki", "2Wiki"),
    ("popqa", "PopQA"),
)
RUNTIME_LABELS = {
    "hotpotqa": "HQA",
    "timeqa": "TQA",
    "finqa": "FQA",
    "musique": "MSQ",
    "quartz": "QTZ",
    "triviaqa": "TrQA",
    "2wiki": "2Wiki",
    "popqa": "PQA",
}
DISPLAY_NAMES = {
    "hotpotqa": "HotpotQA",
    "timeqa": "TimeQA",
    "finqa": "FinQA",
    "musique": "MuSiQue",
    "quartz": "QuaRTz",
    "triviaqa": "TriviaQA",
    "2wiki": "2Wiki",
    "popqa": "PopQA",
}
ALL_METRICS = (
    "Ans-FR", "F1-FR", "EM-FR", "Acc-FR",
    "Ans-CFR", "F1-CFR", "EM-CFR", "Acc-CFR",
)
# Token F1 and normalized EM coincide on these short-answer outputs.  The
# main table keeps EM and removes the duplicate F1 rows.
METRICS = ("Ans-FR", "EM-FR", "Acc-FR", "Ans-CFR", "EM-CFR", "Acc-CFR")
MACROS = {
    "hotpotqa": "HotpotQAMetricRows",
    "timeqa": "TimeQAMetricRows",
    "finqa": "FinQAMetricRows",
    "musique": "MuSiQueMetricRows",
    "quartz": "QuartzMetricRows",
    "triviaqa": "TriviaQAMetricRows",
    "2wiki": "TwoWikiMetricRows",
    "popqa": "PopQAMetricRows",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def method_values(method: dict) -> dict[str, float]:
    values = {}
    for label, family, key in (
        ("Ans-FR", "fr", "answer"),
        ("F1-FR", "fr", "f1"),
        ("EM-FR", "fr", "em"),
        ("Acc-FR", "fr", "acc"),
        ("Ans-CFR", "cfr", "answer"),
        ("F1-CFR", "cfr", "f1"),
        ("EM-CFR", "cfr", "em"),
        ("Acc-CFR", "cfr", "acc"),
    ):
        values[label] = 100 * float(method[family][key]["rate"])
    return values


def random_values(methods: dict) -> dict[str, float]:
    """Average the five frozen Random seeds, matching the existing table."""
    seeds = [method_values(methods[f"random_seed{seed}"]) for seed in range(5)]
    return {
        metric: sum(seed[metric] for seed in seeds) / len(seeds)
        for metric in METRICS
    }


def existing_cells(row_file: Path) -> dict[str, dict[str, list[str]]]:
    text = row_file.read_text()
    output: dict[str, dict[str, list[str]]] = {}
    for dataset, macro in MACROS.items():
        start = text.index(f"\\newcommand{{\\{macro}}}")
        next_start = text.find("\\newcommand{", start + 12)
        block = text[start : None if next_start < 0 else next_start]
        output[dataset] = {}
        for metric in ALL_METRICS:
            marker = f"& {metric} &"
            if marker not in block:
                continue
            line = next(line for line in block.splitlines() if marker in line)
            cells = [part.strip() for part in line.split("&")[2:]]
            output[dataset][metric] = cells
        # The historical QuaRTz block reported the identical F1 rows under
        # their F1 labels and omitted a duplicate EM label.
        if dataset == "quartz":
            output[dataset].setdefault("EM-FR", output[dataset].get("F1-FR", []))
            output[dataset].setdefault("EM-CFR", output[dataset].get("F1-CFR", []))
    return output


def shade_cells(values: list[float], *, cfr: bool) -> list[str]:
    colors = (
        ("D7EAE4", "C9E0D9", "B6D9CD", "A6D0C2", "9ECCC0", "96C8B7", "8CC3B1")
        if cfr
        else ("D7E2EF", "C7D7E8", "B4CAE2", "A8C1DD", "9FBBD9", "97B5D7", "7BA1CC")
    )
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    rank_by_index = {index: rank for rank, index in enumerate(order)}
    best = max(values)
    rendered = []
    for index, value in enumerate(values):
        color = colors[min(rank_by_index[index], len(colors) - 1)]
        text = f"{value:.1f}"
        rendered.append(
            f"\\ragbest{{{color}}}{{{text}}}"
            if value >= best - 1e-9
            else f"\\ragcell{{{color}}}{{{text}}}"
        )
    return rendered


def render_rows(args: argparse.Namespace, shapley: dict[str, dict[str, float]]) -> None:
    unified = read(args.unified_table)
    cells = existing_cells(args.existing_rows)
    methods = ("random", "attention", "gradient_x_input", "token_shapley", "mirage", "arc_jsd", "reflow")
    method_tables = {
        dataset: {
            name: (
                shapley[dataset]
                if name == "token_shapley"
                else random_values(unified["datasets"][dataset]["methods"])
                if name == "random"
                else method_values(unified["datasets"][dataset]["methods"][name])
            )
            for name in methods
        }
        for dataset, _ in DATASETS
    }
    lines = []
    for dataset, _ in DATASETS:
        macro = MACROS[dataset]
        available = [metric for metric in METRICS if metric in cells[dataset]]
        lines.append(f"\\newcommand{{\\{macro}}}{{%")
        lines.append(f"  \\multirow{{{len(available)}}}{{*}}{{{DISPLAY_NAMES[dataset]}}}")
        for offset, metric in enumerate(available):
            values = [method_tables[dataset][name][metric] for name in methods]
            rendered = shade_cells(values, cfr="CFR" in metric)
            suffix = " \\\\" if offset < len(available) - 1 else ""
            if metric == "Acc-FR" or (dataset == "quartz" and metric == "EM-FR"):
                suffix = " \\\\[-0.5pt]"
            lines.append(f"  & {metric} & " + " & ".join(rendered) + suffix)
            if metric == "Acc-FR" or (dataset == "quartz" and metric == "EM-FR"):
                lines.append("  \\cmidrule(lr){2-9}")
        lines.append("}")
        lines.append("")
    args.rows_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--hotpot-metrics", type=Path, required=True)
    parser.add_argument("--rank-root", type=Path, required=True)
    parser.add_argument("--hotpot-rank", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, required=True)
    parser.add_argument("--unified-table", type=Path, required=True)
    parser.add_argument("--existing-rows", type=Path, required=True)
    parser.add_argument("--reflow-runtime", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--figure7-out", type=Path, required=True)
    parser.add_argument("--runtime-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()

    shapley = {}
    rank_seconds = {}
    budget = {}
    for dataset, _ in DATASETS:
        metric_path = args.hotpot_metrics if dataset == "hotpotqa" else args.metrics_root / dataset / "eight_metrics.json"
        metric_data = read(metric_path)
        shapley[dataset] = {
            metric: float(row["rate_percent"])
            for metric, row in metric_data["metrics"].items()
        }
        rank_dir = args.hotpot_rank if dataset == "hotpotqa" else args.rank_root / dataset
        summaries = sorted(rank_dir.glob("shard_*.summary.json"))
        total_queries = sum(int(read(path)["queries"]) for path in summaries)
        total_seconds = sum(float(read(path)["total_seconds"]) for path in summaries)
        if total_queries != 1000:
            raise ValueError(f"{dataset}: expected 1,000 ranked queries")
        rank_seconds[dataset] = total_seconds / total_queries
        budget[dataset] = read(args.budget_root / dataset / "summary.json")

    render_rows(args, shapley)
    figure7 = []
    nr = []
    for dataset, x in DATASETS:
        first = budget[dataset].get(
            "first_matching_budget",
            budget[dataset].get("first_matching_tested_budget"),
        )
        if first is None:
            y = 7.45
            nr.append(dataset)
        else:
            y = math.log2(float(first))
        figure7.append(f"({x},{y:.5f})")
    args.figure7_out.write_text(
        "% Generated from the final Token Shapley budget sweep.\n"
        f"\\def\\FigureEightShapleyCoordinates{{{' '.join(figure7)}}}\n"
        f"\\def\\FigureEightShapleyNRNote{{{', '.join(nr)}}}\n",
        encoding="utf-8",
    )

    reflow = {
        row["dataset"]: row
        for row in read(args.reflow_runtime)["scaling"]
    }
    reflow_coords = []
    shapley_coords = []
    for dataset, _ in DATASETS:
        y = RUNTIME_LABELS[dataset]
        row = reflow[dataset]
        reflow_seconds = float(row["mean_graph_seconds"]) + float(row["mean_frontier_seconds"])
        reflow_coords.append(f"({reflow_seconds:.3f},{y})")
        shapley_coords.append(f"({rank_seconds[dataset]:.3f},{y})")
    args.runtime_out.write_text(
        "% Generated from measured per-query selector time.\n"
        f"\\def\\ReFlowSelectorRuntimeCoordinates{{{' '.join(reflow_coords)}}}\n"
        f"\\def\\ShapleySelectorRuntimeCoordinates{{{' '.join(shapley_coords)}}}\n",
        encoding="utf-8",
    )
    args.manifest_out.write_text(
        json.dumps({"shapley_metrics": shapley, "shapley_rank_seconds": rank_seconds, "budget": budget}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
