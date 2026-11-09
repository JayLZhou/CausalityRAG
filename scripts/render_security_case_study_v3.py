"""Render the final attack/defense application-study figure for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    data = json.load(open(args.data, encoding="utf-8"))
    palette = {
        "ReFlow": "#0F766E",
        "PoisonedRAG": "#D55E00",
        "TPARAG": "#B2183B",
        "Keyword-Guard": "#8B5CF6",
        "Decoding-Guard": "#3B82F6",
        "ReFlow-Guard": "#0F766E",
    }
    markers = {
        "ReFlow": "*",
        "PoisonedRAG": "o",
        "TPARAG": "D",
        "ReFlow-Guard": "*",
        "Keyword-Guard": "o",
        "Decoding-Guard": "D",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.85,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (attack_ax, defense_ax) = plt.subplots(1, 2, figsize=(7.35, 2.63))
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.205, top=0.865, wspace=0.28)

    for point in data["attack_points"]:
        method = point["method"]
        x = float(point["mean_attack_token_footprint"])
        y = 100.0 * float(point["untargeted_asr"])
        low, high = (100.0 * float(value) for value in point["untargeted_asr_ci95"])
        attack_ax.errorbar(
            x,
            y,
            yerr=[[y - low], [high - y]],
            fmt=markers[method],
            color=palette[method],
            markeredgecolor="#334155" if method == "ReFlow" else palette[method],
            markeredgewidth=0.8,
            markersize=10.0 if method == "ReFlow" else 6.5,
            capsize=2.7,
            elinewidth=1.1,
            zorder=4,
        )
        if method == "ReFlow":
            offset, align = (7, -1), "left"
        elif method == "PoisonedRAG":
            offset, align = (-1, 8), "center"
        else:
            offset, align = (-7, -20), "right"
        attack_ax.annotate(
            f"{method}\n{x:.1f} tokens, {y:.1f}%",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=7.1,
            color=palette[method],
            fontweight="bold" if method == "ReFlow" else "normal",
        )

    attack_ax.set_xscale("log")
    attack_ax.set_xlim(1.1, 1050)
    attack_ax.set_ylim(42, 101)
    attack_ax.set_xlabel("Attack footprint (# tokens, log scale)")
    attack_ax.set_ylabel("Untargeted ASR (%)")
    attack_ax.set_title("(a) Attack efficiency", fontweight="bold")
    attack_ax.grid(axis="both", color="#DCE4EE", linewidth=0.65)
    attack_ax.spines[["top", "right"]].set_visible(False)

    for point in data["defense_points"]:
        method = point["method"]
        x = float(point["mean_context_tokens"])
        y = 100.0 * float(point["accuracy_under_attack"])
        low, high = (
            100.0 * float(value)
            for value in point["accuracy_under_attack_ci95"]
        )
        defense_ax.errorbar(
            x,
            y,
            yerr=[[y - low], [high - y]],
            fmt=markers[method],
            color=palette[method],
            markeredgecolor="#334155" if method == "ReFlow-Guard" else palette[method],
            markeredgewidth=0.8,
            markersize=10.0 if method == "ReFlow-Guard" else 6.5,
            capsize=2.7,
            elinewidth=1.1,
            zorder=4,
        )
        if method == "ReFlow-Guard":
            offset, align = (7, -1), "left"
        elif method == "Keyword-Guard":
            offset, align = (-7, 8), "right"
        else:
            offset, align = (-7, -10), "right"
        short_name = method.replace("-Guard", "")
        defense_ax.annotate(
            f"{short_name}\n{x:.1f} tokens, {y:.1f}%",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=7.1,
            color=palette[method],
            fontweight="bold" if method == "ReFlow-Guard" else "normal",
        )
    defense_ax.set_xscale("log")
    defense_ax.set_xlim(40, 2200)
    defense_ax.set_ylim(5, 91)
    defense_ax.set_xlabel("Context tokens per query (log scale)")
    defense_ax.set_ylabel("Accuracy under attack (%)")
    defense_ax.set_title("(b) Defense effectiveness", fontweight="bold")
    defense_ax.grid(axis="both", color="#DCE4EE", linewidth=0.65)
    defense_ax.set_axisbelow(True)
    defense_ax.spines[["top", "right"]].set_visible(False)

    output = Path(args.out_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output) + ".png", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
