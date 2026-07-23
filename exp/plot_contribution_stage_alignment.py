"""Experimental plotting: compare contribution stages with token edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


STAGES = [
    ("attention", "Raw attention $\\alpha$\nreads only", "attention_score", "where the model reads"),
    ("raw_r", "Local contribution $r$\n+ value and answer gradient", "raw_r_score", "+ value and answer gradient"),
    ("conserved_f", "Normalized flow $f$\n+ receiver normalization", "conserved_f_score", "+ receiver-wise normalization"),
    ("aggregated_c", "Token capacity $c$\n+ cross-layer aggregation", "aggregated_c_score", "+ cross-layer token aggregation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--source-data-out", default="")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 4.8,
        "ytick.labelsize": 6.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def normalized(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > 0.0 else values


def compact_labels(rows: list[dict]) -> list[str]:
    counts: dict[tuple[int, str], int] = {}
    labels = []
    for row in rows:
        chunk = int(row["chunk_rank"])
        token = str(row["token"])
        key = (chunk, token.lower())
        counts[key] = counts.get(key, 0) + 1
        suffix = f"{counts[key]}" if counts[key] > 1 else ""
        shown = token if len(token) <= 10 else token[:9] + "..."
        labels.append(rf"{shown}{suffix}")
    return labels


def main() -> None:
    args = parse_args()
    configure_style()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = sorted(
        payload["tokens"],
        key=lambda row: (
            int(row["chunk_rank"]),
            int(row["chunk_char_start"]),
            str(row["unit_id"]),
        ),
    )
    effects = np.asarray(
        [float(row["intervention_effect"]) for row in rows],
        dtype=np.float64,
    )
    top_k = min(args.top_k, len(rows))
    ideal_indices = set(np.argsort(effects)[::-1][:top_k].tolist())
    ideal_effect = float(np.sort(effects)[::-1][:top_k].sum())

    score_rows = []
    selected_indices = []
    captured_fractions = []
    captured_effects = []
    for _, _, score_key, _ in STAGES:
        values = np.asarray([float(row[score_key]) for row in rows])
        score_rows.append(normalized(values))
        selected = np.argsort(values)[::-1][:top_k]
        selected_indices.append(set(selected.tolist()))
        captured = float(effects[selected].sum())
        captured_effects.append(captured)
        captured_fractions.append(captured / ideal_effect if ideal_effect else 0.0)
    score_rows.append(normalized(effects))
    matrix = np.vstack(score_rows)

    figure = plt.figure(figsize=(7.15, 3.05))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[4.5, 1.35],
        left=0.18,
        right=0.985,
        bottom=0.26,
        top=0.78,
        wspace=0.25,
    )
    heat_axis = figure.add_subplot(grid[0])
    image = heat_axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "reflow",
            ["#F7FAFC", "#D9EAF0", "#79B8B1", "#116A63"],
        ),
        vmin=0.0,
        vmax=1.0,
    )

    row_labels = [stage[1] for stage in STAGES] + [r"Executed effect $D_i$"]
    heat_axis.set_yticks(np.arange(len(row_labels)), row_labels)
    heat_axis.set_xticks(np.arange(len(rows)), compact_labels(rows), rotation=58, ha="right")
    heat_axis.tick_params(axis="x", length=0, pad=1.5)
    heat_axis.tick_params(axis="y", length=0, pad=5)

    previous_chunk = int(rows[0]["chunk_rank"])
    chunk_start = 0
    chunk_spans = []
    for index, row in enumerate(rows + [{"chunk_rank": -1}]):
        chunk = int(row["chunk_rank"])
        if chunk != previous_chunk:
            chunk_spans.append((previous_chunk, chunk_start, index - 1))
            if index < len(rows):
                heat_axis.axvline(index - 0.5, color="white", linewidth=2.0)
                previous_chunk = chunk
                chunk_start = index
    for chunk, start, end in chunk_spans:
        heat_axis.text(
            (start + end) / 2,
            -0.62,
            rf"chunk $C_{{{chunk}}}$",
            ha="center",
            va="bottom",
            fontsize=5.4,
            color="#475467",
        )

    for stage_index, selected in enumerate(selected_indices):
        for token_index in selected:
            heat_axis.add_patch(Rectangle(
                (token_index - 0.47, stage_index - 0.47),
                0.94,
                0.94,
                fill=False,
                edgecolor="#C73E1D",
                linewidth=0.9,
            ))
    for token_index in ideal_indices:
        heat_axis.add_patch(Rectangle(
            (token_index - 0.47, len(STAGES) - 0.47),
            0.94,
            0.94,
            fill=False,
            edgecolor="#F6C445",
            linewidth=1.1,
        ))

    for boundary in np.arange(0.5, len(STAGES), 1.0):
        heat_axis.axhline(boundary, color="white", linewidth=1.0)
    heat_axis.set_xlim(-0.5, len(rows) - 0.5)
    heat_axis.set_ylim(len(row_labels) - 0.5, -0.5)
    heat_axis.set_title(
        "(a) Same tokens, progressively refined contribution scores",
        loc="left",
        fontweight="bold",
        pad=13,
    )
    for spine in heat_axis.spines.values():
        spine.set_color("#98A2B3")
        spine.set_linewidth(0.65)

    bar_axis = figure.add_subplot(grid[1])
    y = np.arange(len(STAGES))
    colors = ["#7B9EBD", "#4C8F8A", "#2E7D72", "#116A63"]
    bars = bar_axis.barh(
        y,
        100.0 * np.asarray(captured_fractions),
        color=colors,
        height=0.58,
    )
    bar_axis.set_yticks(y, [r"$\alpha$", r"$r$", r"$f$", r"$c$"])
    bar_axis.invert_yaxis()
    bar_axis.set_xlim(0.0, 105.0)
    bar_axis.set_xticks([0, 25, 50, 75, 100])
    bar_axis.set_xlabel("Top-5 executed effect captured (%)")
    bar_axis.set_title(
        "(b) Alignment with actual edits",
        loc="left",
        fontweight="bold",
        pad=13,
    )
    bar_axis.grid(axis="x", color="#E4E7EC", linewidth=0.55)
    bar_axis.set_axisbelow(True)
    for bar, fraction in zip(bars, captured_fractions):
        bar_axis.text(
            min(101.0, 100.0 * fraction + 2.0),
            bar.get_y() + bar.get_height() / 2,
            f"{100.0 * fraction:.1f}%",
            ha="left",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="#344054",
        )
    bar_axis.spines[["top", "right", "left"]].set_visible(False)
    bar_axis.tick_params(axis="y", length=0)

    figure.text(
        0.02,
        0.965,
        f"HotpotQA example   |   Generated answer: {payload['target_answer']}",
        ha="left",
        va="top",
        fontsize=6.7,
        fontweight="bold",
        color="#344054",
    )
    figure.text(
        0.18,
        0.075,
        "Red outline: method Top-5; gold outline: oracle Top-5 under executed token replacements. "
        "Color intensity is normalized within each row.",
        ha="left",
        va="center",
        fontsize=5.4,
        color="#667085",
    )

    source_data = {
        "id": payload["id"],
        "question": payload["question"],
        "target_answer": payload["target_answer"],
        "top_k": top_k,
        "ideal_top_k_effect": ideal_effect,
        "stages": {
            name: {
                "score_key": score_key,
                "captured_effect": captured_effects[index],
                "captured_fraction": captured_fractions[index],
                "selected_unit_ids": [
                    str(rows[position]["unit_id"])
                    for position in sorted(selected_indices[index])
                ],
            }
            for index, (name, _, score_key, _) in enumerate(STAGES)
        },
        "tokens": rows,
    }
    if args.source_data_out:
        Path(args.source_data_out).write_text(
            json.dumps(source_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_pdf, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(args.out_png, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


if __name__ == "__main__":
    main()
