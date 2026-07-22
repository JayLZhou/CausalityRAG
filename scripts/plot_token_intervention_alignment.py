"""Plot a direct comparison of Top-k token intervention effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--display-top", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    k = args.display_top
    attention_effect = float(metrics[f"attention_top_{k}_effect"])
    reflow_effect = float(metrics[f"reflow_top_{k}_effect"])
    attention_recall = float(metrics[f"attention_top_{k}_overlap"])
    reflow_recall = float(metrics[f"reflow_top_{k}_overlap"])
    ratio = reflow_effect / attention_effect

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axis = plt.subplots(figsize=(4.45, 2.55))
    values = [attention_effect, reflow_effect]
    colors = ["#8795A8", "#148F77"]
    bars = axis.bar([0, 1], values, width=0.56, color=colors, zorder=3)

    axis.set_title(
        f"Effect of the selected Top-{k} tokens",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    axis.set_ylabel("Cumulative clean-answer\nlogit decrease")
    axis.set_xticks([0, 1], ["Raw attention", "ReFlow"])
    axis.set_ylim(0.0, reflow_effect * 1.34)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#D8DEE6", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#667085")
    axis.spines["bottom"].set_color("#667085")
    axis.tick_params(axis="x", length=0, pad=5)
    axis.tick_params(axis="y", colors="#475467")

    recalls = [attention_recall, reflow_recall]
    for bar, value, recall in zip(bars, values, recalls):
        center = bar.get_x() + bar.get_width() / 2
        axis.text(
            center,
            value + 0.07,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color="#1D2939",
        )
        axis.text(
            center,
            value * 0.50,
            f"{int(round(recall * k))}/{k} highest-effect\ntokens recovered",
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="white",
        )

    bracket_y = reflow_effect * 1.18
    bracket_drop = reflow_effect * 0.035
    axis.plot(
        [0, 0, 1, 1],
        [bracket_y - bracket_drop, bracket_y, bracket_y, bracket_y - bracket_drop],
        color="#344054",
        linewidth=0.9,
        clip_on=False,
    )
    axis.text(
        0.5,
        bracket_y + 0.05,
        f"{ratio:.1f}$\\times$ larger",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#087A67",
    )
    figure.subplots_adjust(left=0.20, right=0.98, top=0.84, bottom=0.19)
    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_pdf, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)


if __name__ == "__main__":
    main()
