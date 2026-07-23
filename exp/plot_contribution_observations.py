"""Experimental plotting: render attention/contribution/flow observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("raw_r", r"Raw $r$", "#4C78A8"),
    ("conserved_f", r"Conserved $f$", "#E39C37"),
    ("aggregated_c", r"Aggregated $c$", "#27966F"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--source-data-out", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--exact-clean-only", action="store_true")
    return parser.parse_args()


def bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples,) + values.shape[1:], dtype=np.float64)
    for index in range(samples):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean(axis=0)
    return (
        np.quantile(estimates, 0.025, axis=0),
        np.quantile(estimates, 0.975, axis=0),
    )


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def chunk_ticks(chunk_ranks: list[int]) -> tuple[list[float], list[str], list[int]]:
    centers = []
    labels = []
    boundaries = []
    start = 0
    while start < len(chunk_ranks):
        rank = chunk_ranks[start]
        end = start + 1
        while end < len(chunk_ranks) and chunk_ranks[end] == rank:
            end += 1
        centers.append((start + end - 1) / 2)
        labels.append(rf"$C_{{{rank}}}$")
        if end < len(chunk_ranks):
            boundaries.append(end - 0.5)
        start = end
    return centers, labels, boundaries


def main() -> None:
    args = parse_args()
    configure_style()
    observations = json.load(open(args.observations, encoding="utf-8"))
    verification = [
        json.loads(line)
        for line in open(args.verification, encoding="utf-8")
        if line.strip()
    ]

    flow_rows = [
        row
        for row in observations["per_query"]
        if row.get("status") == "ok"
        and float(row.get("dense_attention_flow", 0.0)) > 0
    ]
    flow_values = np.asarray(
        [row["top_d_flow_fraction"] for row in flow_rows],
        dtype=np.float64,
    )
    flow_mean = flow_values.mean(axis=0)
    flow_lower, flow_upper = bootstrap_interval(
        flow_values,
        samples=args.bootstrap_samples,
        seed=20260722,
    )
    d_values = np.asarray(observations["d_values"], dtype=np.int64)

    verified_rows = [
        row for row in verification
        if not args.exact_clean_only or row.get("clean_exact")
    ]
    budgets = [1, 3, 5]
    flip_rates = np.zeros((len(METHODS), len(budgets)), dtype=np.float64)
    flip_lower = np.zeros_like(flip_rates)
    flip_upper = np.zeros_like(flip_rates)
    flip_counts: dict[str, dict[str, dict]] = {}
    for method_index, (method, _, _) in enumerate(METHODS):
        flip_counts[method] = {}
        for budget_index, budget in enumerate(budgets):
            key = f"budget_b{budget}:{method}"
            values = np.asarray(
                [float(row["methods"][key]["flip"]) for row in verified_rows],
                dtype=np.float64,
            )
            lower, upper = bootstrap_interval(
                values[:, None],
                samples=args.bootstrap_samples,
                seed=20260722 + 100 * method_index + budget,
            )
            flip_rates[method_index, budget_index] = values.mean()
            flip_lower[method_index, budget_index] = lower[0]
            flip_upper[method_index, budget_index] = upper[0]
            flip_counts[method][str(budget)] = {
                "queries": len(values),
                "flips": int(values.sum()),
                "rate": float(values.mean()),
                "bootstrap_95_ci": [float(lower[0]), float(upper[0])],
            }

    figure = plt.figure(figsize=(13.1, 3.25))
    outer = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.62, 1.0, 1.16],
        left=0.045,
        right=0.992,
        bottom=0.19,
        top=0.90,
        wspace=0.27,
    )

    heatmap = observations["heatmap"]
    heatmap_grid = outer[0].subgridspec(3, 1, hspace=0.10)
    heatmap_names = [
        ("raw_attention", r"Raw attention $A$"),
        ("local_r", r"Local contribution $r$"),
        ("conserved_f", r"Conserved flow $f$"),
    ]
    centers, chunk_labels, boundaries = chunk_ticks(heatmap["chunk_ranks"])
    heat_axes = []
    heat_image = None
    for index, (key, label) in enumerate(heatmap_names):
        axis = figure.add_subplot(heatmap_grid[index])
        heat_axes.append(axis)
        values = np.asarray(heatmap["maps"][key], dtype=np.float64)
        rendered = np.log10(values + 1e-7)
        heat_image = axis.imshow(
            rendered,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="magma",
            vmin=-6.0,
            vmax=-1.4,
        )
        for boundary in boundaries:
            axis.axvline(boundary, color="white", linewidth=0.38, alpha=0.75)
        axis.text(
            0.01,
            0.84,
            label,
            transform=axis.transAxes,
            color="white",
            fontsize=6.8,
            fontweight="bold",
            va="top",
        )
        axis.set_yticks([0, 13, 27], ["1", "14", "28"])
        axis.tick_params(length=2, pad=1.2)
        if index < 2:
            axis.set_xticks([])
        else:
            axis.set_xticks(centers, chunk_labels)
            axis.set_xlabel("Retrieved chunks (token positions)", labelpad=2)
    heat_axes[1].set_ylabel("Transformer layer", labelpad=2)
    heat_axes[0].set_title(
        "(a) Same-query information maps",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    heat_axes[0].text(
        0.99,
        1.04,
        (
            rf"corr$(A,r)$={heatmap['attention_r_pearson']:.2f}; "
            rf"Top-1% overlap={100 * heatmap['attention_r_top_1pct_overlap']:.1f}%"
        ),
        transform=heat_axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#343434",
    )
    colorbar = figure.colorbar(
        heat_image,
        ax=heat_axes,
        location="right",
        fraction=0.018,
        pad=0.012,
    )
    colorbar.set_label(r"$log_{10}$ normalized mass", labelpad=2)
    colorbar.set_ticks([-6, -4, -2])

    flow_axis = figure.add_subplot(outer[1])
    flow_axis.fill_between(
        d_values,
        100 * flow_lower,
        100 * flow_upper,
        color="#4C78A8",
        alpha=0.16,
        linewidth=0,
    )
    flow_axis.plot(
        d_values,
        100 * flow_mean,
        color="#2F6DA5",
        marker="o",
        markersize=3.2,
        linewidth=1.5,
    )
    flow_axis.axhline(80, color="#777777", linestyle=":", linewidth=0.7)
    flow_axis.axhline(90, color="#777777", linestyle=":", linewidth=0.7)
    d6_index = int(np.where(d_values == 6)[0][0])
    flow_axis.axvline(6, color="#C44E52", linestyle="--", linewidth=0.8)
    flow_axis.scatter(
        [6],
        [100 * flow_mean[d6_index]],
        color="#C44E52",
        s=18,
        zorder=4,
    )
    flow_axis.annotate(
        f"{100 * flow_mean[d6_index]:.1f}%",
        (6, 100 * flow_mean[d6_index]),
        xytext=(7, 75.5),
        textcoords="data",
        fontsize=7.0,
        color="#9B3035",
        arrowprops={"arrowstyle": "-", "color": "#9B3035", "lw": 0.6},
    )
    flow_axis.set_xscale("log", base=2)
    flow_axis.set_xlim(0.9, 70)
    flow_axis.set_ylim(45, 101)
    flow_axis.set_xticks([1, 2, 4, 6, 8, 16, 32, 64])
    flow_axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    flow_axis.set_yticks([50, 60, 70, 80, 90, 100])
    flow_axis.set_xlabel(r"Incoming edges retained per receiver ($d$)")
    flow_axis.set_ylabel("Cumulative attention-flow mass (%)")
    flow_axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    flow_axis.set_axisbelow(True)
    flow_axis.set_title(
        f"(b) Graph sparsity (n={len(flow_rows)})",
        loc="left",
        fontweight="bold",
        pad=5,
    )

    flip_axis = figure.add_subplot(outer[2])
    x_positions = np.arange(len(budgets), dtype=np.float64)
    width = 0.23
    for method_index, (_, label, color) in enumerate(METHODS):
        positions = x_positions + (method_index - 1) * width
        rates = 100 * flip_rates[method_index]
        errors = np.vstack((
            rates - 100 * flip_lower[method_index],
            100 * flip_upper[method_index] - rates,
        ))
        bars = flip_axis.bar(
            positions,
            rates,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            yerr=errors,
            capsize=2.0,
            error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
        )
        for bar, value in zip(bars, rates):
            flip_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.0,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.1,
                color="#333333",
            )
    flip_axis.set_xticks(x_positions, [str(value) for value in budgets])
    flip_axis.set_xlabel("Replacement budget (tokens)")
    flip_axis.set_ylabel("Answer flip rate (%)")
    flip_axis.set_ylim(30, 103)
    flip_axis.set_yticks([40, 50, 60, 70, 80, 90, 100])
    flip_axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    flip_axis.set_axisbelow(True)
    subset = "exact-clean" if args.exact_clean_only else "shared-valid"
    subset_suffix = ", exact-clean" if args.exact_clean_only else ""
    flip_axis.set_title(
        f"(c) Matched-budget flips (n={len(verified_rows)}{subset_suffix})",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    flip_axis.legend(
        loc="upper left",
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.25,
    )
    for axis in [flow_axis, flip_axis]:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    source_data = {
        "panel_a": {
            "id": heatmap["id"],
            "question": heatmap["question"],
            "target_answer": heatmap["target_answer"],
            "attention_r_pearson": heatmap["attention_r_pearson"],
            "attention_r_top_1pct_overlap": heatmap[
                "attention_r_top_1pct_overlap"
            ],
            "context_tokens": len(heatmap["context_tokens"]),
            "layers": len(heatmap["layers"]),
        },
        "panel_b": {
            "queries": len(flow_rows),
            "d_values": d_values.tolist(),
            "mean": flow_mean.tolist(),
            "bootstrap_95_ci_lower": flow_lower.tolist(),
            "bootstrap_95_ci_upper": flow_upper.tolist(),
        },
        "panel_c": {
            "subset": subset,
            "queries": len(verified_rows),
            "methods": flip_counts,
        },
    }
    if args.source_data_out:
        Path(args.source_data_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.source_data_out, "w", encoding="utf-8") as output:
            json.dump(source_data, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps(source_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
