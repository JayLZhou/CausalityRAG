#!/usr/bin/env python3
"""Render the prefix-search Ans-CFR bubble frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {"exhaustive": "#7E9DB8", "reflow": "#2F8A7C"}
LABELS = {"exhaustive": "Exhaustive", "reflow": "ReFlow"}


def bubble_area(tokens: float) -> float:
    return 42.0 + 44.0 * tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data.get("schema") != "causalityrag.prefix_answer_frontier.aggregate.v1":
        raise ValueError("unexpected aggregate schema")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.2,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(3.35, 2.48))
    for method in ("exhaustive", "reflow"):
        curve = data["macro_average"][method]
        x_values = [float(row["mean_independent_verifications"]) for row in curve]
        y_values = [100.0 * float(row["answer_cfr"]) for row in curve]
        sizes = [bubble_area(float(row["mean_modified_tokens"])) for row in curve]
        axis.plot(
            x_values,
            y_values,
            color=COLORS[method],
            linewidth=1.15,
            alpha=0.90,
            zorder=2,
        )
        axis.scatter(
            x_values,
            y_values,
            s=sizes,
            color=COLORS[method],
            edgecolor="#33404A",
            linewidth=0.65,
            alpha=0.72,
            zorder=3,
        )
        for index in (0, 2, 4, 9):
            row = curve[index]
            axis.annotate(
                f"{int(row['n'])}",
                (x_values[index], y_values[index]),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                color="white",
                fontsize=5.8,
                fontweight="bold",
                zorder=4,
            )

    axis.set_xscale("log")
    axis.set_xlabel("Independent reader verifications (log scale)")
    axis.set_ylabel("Ans-CFR (%)")
    axis.grid(True, which="major", color="#D8DEE3", linewidth=0.55, alpha=0.9)
    axis.grid(True, which="minor", axis="x", color="#E8ECEF", linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_color("#4E5963")
    axis.spines["right"].set_color("#4E5963")
    axis.spines["bottom"].set_color("#4E5963")
    axis.spines["left"].set_color("#4E5963")

    method_handles = [
        Line2D(
            [0], [0],
            color=COLORS[method],
            marker="o",
            markeredgecolor="#33404A",
            markeredgewidth=0.55,
            markersize=5.2,
            linewidth=1.1,
            label=LABELS[method],
        )
        for method in ("exhaustive", "reflow")
    ]
    method_legend = axis.legend(
        handles=method_handles,
        loc="upper left",
        frameon=False,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.15,
    )
    axis.add_artist(method_legend)

    size_handles = [
        axis.scatter(
            [], [],
            s=bubble_area(tokens),
            color="#B8D5CF",
            edgecolor="#5D6973",
            linewidth=0.55,
            alpha=0.75,
            label=str(tokens),
        )
        for tokens in (1, 2, 4)
    ]
    axis.legend(
        handles=size_handles,
        title="Mean edited tokens",
        loc="lower right",
        frameon=False,
        ncol=3,
        columnspacing=0.55,
        handletextpad=0.15,
        borderaxespad=0.15,
        title_fontsize=6.8,
    )

    figure.tight_layout(pad=0.45)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
