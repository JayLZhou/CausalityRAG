"""Create the paper-ready two-panel security Pareto figure."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from causalityrag.reader import answers_match


def bootstrap_interval(values: list[float], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10_000):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return [means[249], means[9749]]


def reflow_curve(
    attack_rows: list[dict[str, Any]],
    reflow_rows: list[dict[str, Any]],
    budgets: list[int],
) -> list[dict[str, Any]]:
    reflow_by_id = {str(row.get("id", "")): row for row in reflow_rows}
    points = []
    for budget in budgets:
        successes = []
        footprints = []
        for attack in attack_rows:
            row = reflow_by_id[str(attack.get("id", ""))]
            gold = str(attack.get("gold_answer", ""))
            attempts = [
                attempt
                for attempt in row.get("attempts", [])
                if attempt.get("reader_called")
                and int(attempt.get("n_edits", 0)) <= budget
            ]
            success_attempt = next(
                (
                    attempt
                    for attempt in attempts
                    if not answers_match(str(attempt.get("edited_answer", "")), gold)
                ),
                None,
            )
            successes.append(int(success_attempt is not None))
            if success_attempt is not None:
                footprints.append(int(success_attempt.get("n_edits", 0)))
            elif attempts:
                footprints.append(max(int(attempt.get("n_edits", 0)) for attempt in attempts))
            else:
                footprints.append(0)
        points.append(
            {
                "method": "ReFlow",
                "budget": budget,
                "untargeted_asr": statistics.fmean(successes),
                "untargeted_asr_ci95": bootstrap_interval(successes, seed=budget + 101),
                "mean_attack_token_footprint": statistics.fmean(footprints),
            }
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-summary", required=True)
    parser.add_argument("--attack-rows", required=True)
    parser.add_argument("--reflow", required=True)
    parser.add_argument("--defense-summary", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--data-out", required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with open(args.attack_summary, encoding="utf-8") as handle:
        attack_summary = json.load(handle)
    with open(args.defense_summary, encoding="utf-8") as handle:
        defense_summary = json.load(handle)
    attack_rows = load_records(args.attack_rows)
    curve = reflow_curve(
        attack_rows,
        load_records(args.reflow),
        budgets=[1, 2, 4, 8, 16, 32],
    )
    attack_points = list(curve)
    for key, label in (("poisonedrag", "PoisonedRAG"), ("tparag", "TPARAG")):
        stats = attack_summary["methods"][key]
        attack_points.append(
            {
                "method": label,
                "untargeted_asr": stats["untargeted_asr"],
                "untargeted_asr_ci95": stats["untargeted_asr_ci95"],
                "mean_attack_token_footprint": stats["mean_attack_token_footprint"],
            }
        )

    defense_labels = {
        "no_defense": "No defense",
        "reflow_guard": "ReFlow-Guard",
        "random_guard": "Random-Guard",
        "robustrag_voting": "RobustRAG-Vote",
        "robustrag_keyword": "RobustRAG-Keyword",
    }
    defense_points = []
    for key, label in defense_labels.items():
        stats = defense_summary["macro"][key]
        defense_points.append(
            {
                "method": label,
                "attacked_accuracy": stats["attacked_accuracy"],
                "attacked_accuracy_ci95": stats["attacked_accuracy_ci95"],
                "mean_defense_token_footprint": stats["mean_defense_token_footprint"],
            }
        )

    palette = {
        "ReFlow": "#0F766E",
        "PoisonedRAG": "#B45309",
        "TPARAG": "#9F1239",
        "No defense": "#64748B",
        "ReFlow-Guard": "#0F766E",
        "Random-Guard": "#64748B",
        "RobustRAG-Vote": "#2563EB",
        "RobustRAG-Keyword": "#7C3AED",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.0))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.86, wspace=0.34)

    ax = axes[0]
    reflow = [point for point in attack_points if point["method"] == "ReFlow"]
    ax.plot(
        [point["mean_attack_token_footprint"] for point in reflow],
        [100 * point["untargeted_asr"] for point in reflow],
        color=palette["ReFlow"],
        marker="o",
        linewidth=1.8,
        markersize=4.8,
        label="ReFlow",
        zorder=3,
    )
    for point in attack_points:
        if point["method"] == "ReFlow":
            continue
        x = point["mean_attack_token_footprint"]
        y = 100 * point["untargeted_asr"]
        low, high = point["untargeted_asr_ci95"]
        ax.errorbar(
            x,
            y,
            yerr=[[y - 100 * low], [100 * high - y]],
            fmt="o",
            color=palette[point["method"]],
            capsize=2.5,
            markersize=6,
            label=point["method"],
            zorder=4,
        )
        offset = (-7, 7) if point["method"] == "TPARAG" else (5, 6)
        align = "right" if point["method"] == "TPARAG" else "left"
        ax.annotate(
            f"{point['method']}\n{x:.1f} tok, {y:.1f}%",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.9,
            ha=align,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Attack Footprint (# Token Edits, log scale)")
    ax.set_ylabel("Untargeted ASR (%)")
    ax.set_title("(a) Attack efficiency")
    ax.grid(axis="both", color="#E2E8F0", linewidth=0.7)
    ax.set_ylim(0, 103)
    ax.legend(frameon=False, fontsize=6.9, loc="lower right", handlelength=1.7)

    ax = axes[1]
    for point in defense_points:
        raw_x = point["mean_defense_token_footprint"]
        x = max(0.5, raw_x)
        y = 100 * point["attacked_accuracy"]
        low, high = point["attacked_accuracy_ci95"]
        ax.errorbar(
            x,
            y,
            yerr=[[y - 100 * low], [100 * high - y]],
            fmt="o",
            color=palette[point["method"]],
            capsize=2.5,
            markersize=6,
            label=f"{point['method']} ({'0' if raw_x == 0 else f'{raw_x:.1f}'} tok, {y:.1f}%)",
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Defense token footprint (log scale, lower is better)")
    ax.set_ylabel("Attacked accuracy (%)")
    ax.set_title("(b) Defense efficiency")
    ax.grid(axis="both", color="#E2E8F0", linewidth=0.7)
    ax.set_ylim(0, 103)
    ax.legend(
        frameon=False,
        fontsize=6.1,
        loc="upper left",
        ncol=1,
        handletextpad=0.35,
        labelspacing=0.3,
    )

    output_prefix = Path(args.out_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_prefix) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    data = {
        "attack_points": attack_points,
        "defense_points": defense_points,
        "notes": {
            "no_defense_log_position": 0.5,
            "no_defense_true_footprint": 0,
            "attack_population": "100 frozen clean-correct HotpotQA queries",
            "defense_macro_population": "PoisonedRAG and TPARAG attacks",
        },
    }
    with open(args.data_out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
