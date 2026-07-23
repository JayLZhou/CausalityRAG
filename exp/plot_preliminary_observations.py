"""Experimental plotting: render preliminary contribution-flow observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("attention", r"Attention $\alpha$", "#477DAA", "o", "-"),
    ("raw_r", r"Local $r$", "#9B9B9B", "^", "--"),
    ("conserved_f", r"Normalized $f$", "#D08A2E", "s", "--"),
    ("aggregated_c", r"Aggregated $c$", "#16877A", "D", "-"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-png", required=True)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.1,
        "axes.titlesize": 7.7,
        "axes.labelsize": 7.1,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    args = parse_args()
    configure_style()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.21, top=0.90, wspace=0.34)

    # Observation 1: matched-budget reader flips.
    axis = axes[0]
    selection = payload["selection"]
    budgets = np.asarray(selection["budgets"], dtype=np.int64)
    for method, label, color, marker, linestyle in METHODS:
        rates = np.asarray([
            100.0 * selection["rates"][method][str(int(budget))]["flip_rate"]
            for budget in budgets
        ])
        axis.plot(
            budgets,
            rates,
            label=label,
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.35,
            linestyle=linestyle,
        )
    attention_b3 = 100.0 * selection["rates"]["attention"]["3"]["flip_rate"]
    graph_b3 = 100.0 * selection["rates"]["aggregated_c"]["3"]["flip_rate"]
    axis.annotate(
        f"+{graph_b3 - attention_b3:.1f} pp",
        xy=(3, graph_b3),
        xytext=(3.38, 84.0),
        fontsize=6.1,
        color="#116A63",
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": "#116A63"},
    )
    axis.set_xticks(budgets)
    axis.set_xlim(0.7, 5.3)
    axis.set_ylim(45, 95)
    axis.set_yticks([50, 60, 70, 80, 90])
    axis.set_xlabel("Replaced tokens")
    axis.set_ylabel("Reader flip rate (%)")
    axis.grid(axis="y", color="#E4E7EC", linewidth=0.55)
    axis.legend(loc="lower right", frameon=False, handlelength=2.2)
    axis.set_title(
        "(a) Local scores miss joint support",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    paired = selection["attention_vs_aggregated_c"]["3"]
    axis.text(
        0.02,
        0.98,
        f"$c$-only / attention-only flips: {paired['graph_only']} / {paired['attention_only']}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color="#475467",
    )

    # Observation 2: raw receiver scale varies across layers.
    axis = axes[1]
    scale = payload["graphs"]["receiver_scale"]
    layers = np.asarray([row["layer"] for row in scale["by_layer"]])
    p25 = np.asarray([row["p25"] for row in scale["by_layer"]])
    median = np.asarray([row["median"] for row in scale["by_layer"]])
    p75 = np.asarray([row["p75"] for row in scale["by_layer"]])
    axis.fill_between(layers, p25, p75, color="#D4E6E2", alpha=0.9, linewidth=0)
    axis.plot(layers, median, color="#16877A", linewidth=1.45)
    axis.axhline(0.0, color="#98A2B3", linewidth=0.55, linestyle=":")
    axis.set_xlim(1, max(layers))
    axis.set_xticks([1, 7, 14, 21, 28])
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel(r"$\log_{10}\, Z_v$, $Z_v=\sum_u[r(u,v)]_+$")
    axis.grid(axis="y", color="#E4E7EC", linewidth=0.55)
    axis.set_title(
        "(b) Receiver scales are incomparable",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    axis.text(
        0.03,
        0.06,
        f"P5--P95 span: {scale['p05_to_p95_orders']:.2f} orders",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color="#116A63",
        fontweight="bold",
    )

    # Observation 3: contribution is spread over layer copies.
    axis = axes[2]
    fragmentation = payload["graphs"]["layer_fragmentation"]
    outside = 100.0 * np.asarray(fragmentation["outside_dominant_layer"])
    outside_sorted = np.sort(outside)
    cumulative = 100.0 * np.arange(1, len(outside_sorted) + 1) / len(outside_sorted)
    axis.plot(outside_sorted, cumulative, color="#6D5A9E", linewidth=1.55)
    axis.fill_between(outside_sorted, 0, cumulative, color="#E7E1F2", alpha=0.65)
    median_outside = 100.0 * fragmentation[
        "query_median_outside_percentiles"
    ]["median"]
    median_layers = fragmentation[
        "query_median_active_layers_percentiles"
    ]["median"]
    axis.axvline(median_outside, color="#6D5A9E", linestyle="--", linewidth=0.8)
    axis.text(
        median_outside + 2.0,
        20.0,
        f"query median: {median_outside:.1f}%\nacross {median_layers:.0f} layers",
        fontsize=6.1,
        color="#5D478E",
        fontweight="bold",
    )
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_xlabel("Mass outside strongest layer (%)")
    axis.set_ylabel("Context positions (CDF, %)")
    axis.grid(color="#E4E7EC", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.set_title(
        "(c) Answer support spans many layers",
        loc="left",
        fontweight="bold",
        pad=7,
    )

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_pdf, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(args.out_png, dpi=260, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)


if __name__ == "__main__":
    main()
