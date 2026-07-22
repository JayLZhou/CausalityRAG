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

    shown = sorted(
        top_indices(attention, args.display_top)
        | top_indices(reflow, args.display_top)
        | top_indices(effect, args.display_top),
        key=lambda index: (
            int(rows[index]["chunk_rank"]),
            int(rows[index]["chunk_char_start"]),
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

    figure, axis = plt.subplots(figsize=(7.15, 2.55))
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
        ["Raw attention", "ReFlow contribution", "Executed replacement effect"],
    )
    axis.set_xticks(range(len(labels)), labels, rotation=48, ha="right")
    axis.tick_params(length=0, pad=2)

    selected_sets = [
        top_indices(attention, args.display_top),
        top_indices(reflow, args.display_top),
        top_indices(effect, args.display_top),
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

    metrics = payload["metrics"]
    top_key = next(
        key for key in metrics
        if key.startswith("attention_top_") and key.endswith("_overlap")
    )
    reflow_top_key = top_key.replace("attention", "reflow")
    k = top_key.split("_")[2]
    attention_ndcg = metrics.get(
        f"attention_ndcg_at_{k}", ndcg(attention, effect, int(k))
    )
    reflow_ndcg = metrics.get(
        f"reflow_ndcg_at_{k}", ndcg(reflow, effect, int(k))
    )
    axis.set_title(
        (
            rf"Clean answer: {payload['target_answer']}  |  {payload['candidate_tokens']} legal replacements"
            "\n"
            rf"Top-{k} alignment with executed effect: "
            rf"Attention Top-{k}={100*metrics[top_key]:.0f}\%, NDCG@{k}={attention_ndcg:.2f}; "
            rf"ReFlow Top-{k}={100*metrics[reflow_top_key]:.0f}\%, NDCG@{k}={reflow_ndcg:.2f}"
        ),
        loc="left",
        fontweight="bold",
        pad=10,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.018, pad=0.015)
    colorbar.set_label("Relative score", labelpad=3)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0", "0.5", "1"])
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)

    figure.subplots_adjust(left=0.185, right=0.965, top=0.70, bottom=0.32)
    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


if __name__ == "__main__":
    main()
