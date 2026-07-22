"""Plot one token-ranking alignment heatmap for the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--display-top", type=int, default=7)
    return parser.parse_args()


def max_normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def top_indices(values: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(values)[-min(k, len(values)):])


def ndcg(scores: np.ndarray, effects: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    ranked = np.argsort(scores)[::-1][:k]
    ideal = np.argsort(effects)[::-1][:k]

    def dcg(indices: np.ndarray) -> float:
        return sum(
            (2.0 ** float(effects[index]) - 1.0) / math.log2(rank + 2.0)
            for rank, index in enumerate(indices)
        )

    denominator = dcg(ideal)
    return dcg(ranked) / denominator if denominator > 0.0 else 0.0


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload["tokens"]
    attention = np.asarray([row["attention_score"] for row in rows])
    reflow = np.asarray([row["reflow_score"] for row in rows])
    effect = np.asarray([row["intervention_effect"] for row in rows])
    metrics = payload["metrics"]
    top_key = next(
        key for key in metrics
        if key.startswith("attention_top_") and key.endswith("_overlap")
    )
    reflow_top_key = top_key.replace("attention", "reflow")
    k = int(top_key.split("_")[2])
    true_top = top_indices(effect, k)

    shown = sorted(
        top_indices(attention, args.display_top)
        | top_indices(reflow, args.display_top)
        | top_indices(effect, args.display_top),
        key=lambda index: (
            index not in true_top,
            -float(effect[index]),
            -float(reflow[index]),
        ),
    )
    values = np.vstack((
        max_normalize(attention)[shown],
        max_normalize(reflow)[shown],
        max_normalize(effect)[shown],
    ))
    labels = [
        f"{rows[index]['token']}→{rows[index]['replacement']}"
        for index in shown
    ]

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 8.6,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axis = plt.subplots(figsize=(7.15, 2.70))
    image = axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_yticks(
        [0, 1, 2],
        [
            (
                f"Raw attention\n"
                f"{100 * metrics[top_key]:.0f}% hits; "
                f"$\\Sigma D$={metrics[f'attention_top_{k}_effect']:.2f}"
            ),
            (
                f"ReFlow contribution\n"
                f"{100 * metrics[reflow_top_key]:.0f}% hits; "
                f"$\\Sigma D$={metrics[f'reflow_top_{k}_effect']:.2f}"
            ),
            "Executed effect $D_i$",
        ],
    )
    axis.set_xticks(range(len(labels)), labels, rotation=43, ha="right")
    axis.tick_params(length=0, pad=2)
    axis.axvspan(-0.5, k - 0.5, color="#148F77", alpha=0.055)
    axis.axvline(k - 0.5, color="#148F77", linewidth=1.1, linestyle="--")
    axis.text(
        (k - 1) / 2,
        1.035,
        "Empirical Top-5 by executed effect",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color="#087A67",
        fontsize=7.2,
        fontweight="bold",
    )
    axis.text(
        len(labels) - 0.5,
        1.035,
        r"$\circ$ = method's Top-5",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        color="#475467",
        fontsize=7.0,
    )

    selected_sets = [
        top_indices(attention, args.display_top),
        top_indices(reflow, args.display_top),
    ]
    for row_index, selected in enumerate(selected_sets):
        for column_index, token_index in enumerate(shown):
            if token_index in selected:
                axis.scatter(
                    column_index,
                    row_index,
                    marker="o",
                    s=13,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=0.7,
                )

    normalized_effect = max_normalize(effect)
    for column_index, token_index in enumerate(shown):
        axis.text(
            column_index,
            2,
            f"{effect[token_index]:.2f}",
            ha="center",
            va="center",
            color="white" if normalized_effect[token_index] >= 0.52 else "#263238",
            fontsize=6.7,
            fontweight="bold",
        )

    attention_ndcg = metrics.get(
        f"attention_ndcg_at_{k}", ndcg(attention, effect, k)
    )
    reflow_ndcg = metrics.get(
        f"reflow_ndcg_at_{k}", ndcg(reflow, effect, k)
    )
    effect_ratio = (
        metrics[f"reflow_top_{k}_effect"]
        / metrics[f"attention_top_{k}_effect"]
    )
    axis.set_title(
        (
            rf"Clean answer: {payload['target_answer']}  |  {payload['candidate_tokens']} legal replacements"
            "\n"
            rf"ReFlow doubles Top-{k} recovery "
            rf"({100*metrics[reflow_top_key]:.0f}\% vs. {100*metrics[top_key]:.0f}\%) "
            rf"and yields {effect_ratio:.1f}$\times$ cumulative executed effect "
            rf"(NDCG@{k}: {reflow_ndcg:.2f} vs. {attention_ndcg:.2f})"
        ),
        loc="left",
        fontweight="bold",
        pad=24,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.018, pad=0.015)
    colorbar.set_label("Relative score", labelpad=3)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0", "0.5", "1"])
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)

    figure.subplots_adjust(left=0.215, right=0.965, top=0.62, bottom=0.32)
    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


if __name__ == "__main__":
    main()
