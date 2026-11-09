#!/usr/bin/env python3
"""Plot Full/ReFlow/MIRAGE/ARC-JSD sentence-sufficiency bars."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHODS = ("Full", "ReFlow", "MIRAGE", "ARC-JSD")
COLORS = ("#4C78A8", "#E45756", "#72B7B2", "#F2CF5B")


def parse_summary(value: str) -> tuple[str, dict]:
    if "=" not in value:
        raise ValueError("--summary must use LABEL=PATH")
    label, path = value.split("=", 1)
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def plot_panel(axis, label: str, summary: dict, metric: str) -> None:
    values = [100.0 * summary["methods"][method][metric]["mean"] for method in METHODS]
    intervals = [summary["methods"][method][metric]["ci95"] for method in METHODS]
    lower = [value - 100.0 * interval[0] for value, interval in zip(values, intervals)]
    upper = [100.0 * interval[1] - value for value, interval in zip(values, intervals)]
    axis.bar(
        range(len(METHODS)),
        values,
        color=COLORS,
        edgecolor="black",
        linewidth=0.7,
        yerr=[lower, upper],
        capsize=3,
    )
    axis.set_xticks([])
    axis.set_ylim(0, 100)
    axis.set_ylabel(metric.upper() + " (%)")
    axis.set_title(f"{label} — {metric.upper()}")
    axis.grid(axis="y", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--datasets-per-row", type=int, default=1)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    summaries = [parse_summary(value) for value in args.summary]
    per_row = max(1, args.datasets_per_row)
    rows = math.ceil(len(summaries) / per_row)
    columns = 2 * per_row
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.4 * rows),
        squeeze=False,
    )
    for index, (label, summary) in enumerate(summaries):
        row = index // per_row
        dataset_column = index % per_row
        plot_panel(axes[row][2 * dataset_column], label, summary, "em")
        plot_panel(axes[row][2 * dataset_column + 1], label, summary, "f1")
    for index in range(len(summaries), rows * per_row):
        row = index // per_row
        dataset_column = index % per_row
        axes[row][2 * dataset_column].axis("off")
        axes[row][2 * dataset_column + 1].axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.legend(
        handles=[
            Patch(facecolor=color, edgecolor="black", label=method)
            for method, color in zip(METHODS, COLORS)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(METHODS),
        frameon=False,
    )
    output = Path(args.out_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
