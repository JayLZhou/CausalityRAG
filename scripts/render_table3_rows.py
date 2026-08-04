#!/usr/bin/env python3
"""Render the verified Table 3 metric JSON files as LaTeX row macros."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DATASETS = (
    ("hotpotqa", "HotpotQA", "HotpotQAMetricRows"),
    ("timeqa", "TimeQA", "TimeQAMetricRows"),
    ("finqa", "FinQA", "FinQAMetricRows"),
    ("musique", "MuSiQue", "MuSiQueMetricRows"),
    ("quartz", "QuaRTz", "QuartzMetricRows"),
    ("qasper", "Qasper", "QasperMetricRows"),
    ("2wiki", "2Wiki", "TwoWikiMetricRows"),
    ("pubmedqa", "PubMedQA", "PubMedQAMetricRows"),
)
METHODS = (
    ("random", "random"),
    ("attention", "attention"),
    ("gradient_x_input", "gradient_x_input"),
    ("integrated_gradients", "integrated_gradients"),
    ("mirage", "mirage"),
    ("arc_jsd", "arc_jsd"),
    ("reflow", "reflow"),
)
METRICS = (
    ("Ans-FR", "answer", False),
    ("F1-FR", "f1", False),
    ("EM-FR", "em", False),
    ("Acc-FR", "acc", False),
    ("Ans-CFR", "answer", True),
    ("F1-CFR", "f1", True),
    ("EM-CFR", "em", True),
    ("Acc-CFR", "acc", True),
)
PUBMEDQA_METRICS = tuple(
    row for row in METRICS if row[1] != "acc"
)
FACTUAL_KEYS = {
    "answer": "answer_flip_ratio",
    "f1": "f1_flip_ratio",
    "em": "em_flip_ratio",
    "acc": "acc_flip_ratio",
}
CONTROL_KEYS = {
    "answer": "answer_flip_rate_itt",
    "f1": "f1_flip_rate_itt",
    "em": "em_flip_rate_itt",
    "acc": "acc_flip_rate_itt",
}
BLUE = ("D7E2EF", "C7D7E8", "B4CAE2", "A8C1DD", "9FBBD9", "97B5D7", "7BA1CC")
GREEN = ("D7EAE4", "C9E0D9", "B6D9CD", "A6D0C2", "9ECCC0", "96C8B7", "8CC3B1")
ALIASES = {
    "attention": ("attention", "attention_prompt_last"),
    "gradient_x_input": ("gradient_x_input",),
    "integrated_gradients": ("integrated_gradients",),
    "mirage": ("mirage",),
    "arc_jsd": ("arc_jsd",),
    "reflow": ("reflow",),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def find_method(methods: dict, canonical: str) -> dict:
    for alias in ALIASES[canonical]:
        if alias in methods:
            return methods[alias]
    raise KeyError(f"missing method {canonical}; available={sorted(methods)}")


def metric_values(factual: dict, control: dict, metric: str, adjusted: bool) -> list[float]:
    values = []
    random_values = []
    for seed in range(5):
        name = f"random_seed{seed}"
        factual_value = float(factual["methods"][name][FACTUAL_KEYS[metric]])
        value = factual_value
        if adjusted:
            value -= float(control["methods"][name][CONTROL_KEYS[metric]])
        random_values.append(value * 100.0)
    values.append(max(0.0, statistics.fmean(random_values)))
    for canonical, _ in METHODS[1:]:
        factual_row = find_method(factual["methods"], canonical)
        value = float(factual_row[FACTUAL_KEYS[metric]])
        if adjusted:
            control_row = find_method(control["methods"], canonical)
            value -= float(control_row[CONTROL_KEYS[metric]])
        values.append(max(0.0, value) * 100.0)
    return values


def cells(values: list[float], *, adjusted: bool) -> list[str]:
    palette = GREEN if adjusted else BLUE
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    rank = {index: position for position, index in enumerate(ordered)}
    best = max(values)
    output = []
    for index, value in enumerate(values):
        color = palette[rank[index]]
        text = f"{value:.1f}"
        if abs(value - best) < 1e-9:
            output.append(f"\\ragbest{{{color}}}{{{text}}}")
        else:
            output.append(f"\\ragcell{{{color}}}{{{text}}}")
    return output


def dataset_paths(root: Path, dataset: str) -> tuple[Path, Path]:
    base = root / dataset
    factual_candidates = (
        base / "audits/final_top10pool_k5/factual_metrics_1000.json",
        base / "factual_metrics_1000.json",
    )
    control_candidates = (
        base / "controls/paraphrase_results_top5_1000_v2.summary.json",
        base / "controls/paraphrase_results_top5_1000.summary.json",
        base / "audits/final_top10pool_k5/paraphrase_eval_v3_1000.summary.json",
        base / "paraphrase_eval_v3_1000.summary.json",
    )
    factual = next((path for path in factual_candidates if path.is_file()), factual_candidates[0])
    control = next((path for path in control_candidates if path.is_file()), control_candidates[0])
    return factual, control


def render_dataset(root: Path, dataset: str, label: str, macro: str) -> str:
    metrics = PUBMEDQA_METRICS if dataset == "pubmedqa" else METRICS
    factual_path, control_path = dataset_paths(root, dataset)
    if not factual_path.is_file() or not control_path.is_file():
        lines = [f"\\newcommand{{\\{macro}}}{{%", f"  \\multirow{{{len(metrics)}}}{{*}}{{{label}}}"]
        for row_index, (row_label, _, _) in enumerate(metrics):
            suffix = " \\\\" if row_index < len(metrics) - 1 else ""
            lines.append("  & " + row_label + " & " + " & ".join(["--"] * 7) + suffix)
            if row_index == len(metrics) // 2 - 1:
                lines[-1] += "[-0.5pt]"
                lines.append("  \\cmidrule(lr){2-9}")
        lines.append("}")
        return "\n".join(lines)
    factual = load_json(factual_path)
    control = load_json(control_path)
    lines = [f"\\newcommand{{\\{macro}}}{{%", f"  \\multirow{{{len(metrics)}}}{{*}}{{{label}}}"]
    for row_index, (row_label, metric, adjusted) in enumerate(metrics):
        values = metric_values(factual, control, metric, adjusted)
        rendered = cells(values, adjusted=adjusted)
        prefix = "  & " if row_index else "  & "
        suffix = " \\\\" if row_index < len(metrics) - 1 else ""
        lines.append(prefix + row_label + " & " + " & ".join(rendered) + suffix)
        if row_index == len(metrics) // 2 - 1:
            lines[-1] += "[-0.5pt]"
            lines.append("  \\cmidrule(lr){2-9}")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[dataset for dataset, _, _ in DATASETS],
    )
    args = parser.parse_args()
    root = Path(args.metrics_root)
    requested = set(args.datasets)
    unknown = requested.difference(dataset for dataset, _, _ in DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    rendered = "\n\n".join(
        render_dataset(root, dataset, label, macro)
        for dataset, label, macro in DATASETS
        if dataset in requested
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
