"""Visualize attention and ReFlow scores on the same retrieved text."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath


TOKEN_PATTERN = re.compile(r"\w+(?:[-'][\w]+)*|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--record-input", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--display-top", type=int, default=5)
    parser.add_argument("--chunk-ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--excerpt-chars", type=int, default=140)
    return parser.parse_args()


def load_record(path: str, index: int) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"record index {index} not found in {path}")


def rounded_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 0.8,
    radius: float = 0.012,
) -> None:
    axis.add_patch(FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    ))


def blended_color(base: str, strength: float) -> str:
    white = colors.to_rgb("#FFFFFF")
    target = colors.to_rgb(base)
    amount = 0.16 + 0.84 * max(0.0, min(1.0, strength)) ** 0.55
    return colors.to_hex(tuple(
        white[channel] * (1.0 - amount) + target[channel] * amount
        for channel in range(3)
    ))


def token_score(
    score_rows: list[dict],
    chunk_rank: int,
    start: int,
    end: int,
    score_key: str,
) -> float:
    score = 0.0
    for row in score_rows:
        if int(row["chunk_rank"]) != chunk_rank:
            continue
        row_start = int(row["chunk_char_start"])
        row_end = row_start + len(str(row["token"]))
        if row_start < end and start < row_end:
            score += float(row[score_key])
    return score


def text_width(
    text: str,
    font: FontProperties,
    font_size: float,
    axes_width_points: float,
) -> float:
    path = TextPath((0, 0), text, prop=font, size=font_size)
    return max(0.005, path.get_extents().width / axes_width_points)


def draw_excerpt(
    axis: plt.Axes,
    figure: plt.Figure,
    excerpt: str,
    score_rows: list[dict],
    score_key: str,
    chunk_rank: int,
    x0: float,
    x1: float,
    y_top: float,
    base_color: str,
) -> None:
    font_size = 6.65
    line_height = 0.044
    font = FontProperties(family="DejaVu Sans", size=font_size)
    axes_width_points = (
        figure.get_figwidth() * 72.0 * axis.get_position().width
    )
    matches = list(TOKEN_PATTERN.finditer(excerpt))
    raw_scores = [
        token_score(
            score_rows,
            chunk_rank,
            match.start(),
            match.end(),
            score_key,
        )
        for match in matches
    ]
    maximum = max(raw_scores, default=0.0)
    x = x0
    y = y_top
    for match, raw_score in zip(matches, raw_scores):
        token = match.group(0)
        width = text_width(token, font, font_size, axes_width_points)
        gap = 0.006 if token not in ",.;:!?)]}" else 0.002
        if x + width > x1:
            x = x0
            y -= line_height
        normalized = raw_score / maximum if maximum > 0.0 else 0.0
        kwargs = {}
        if raw_score > 0.0:
            facecolor = blended_color(base_color, normalized)
            kwargs["bbox"] = {
                "boxstyle": "round,pad=0.12,rounding_size=0.12",
                "facecolor": facecolor,
                "edgecolor": "none",
            }
            kwargs["color"] = "white" if normalized > 0.72 else "#1D2939"
        else:
            kwargs["color"] = "#475467"
        axis.text(
            x,
            y,
            token,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontproperties=font,
            zorder=4,
            **kwargs,
        )
        x += width + gap


def draw_legend(
    axis: plt.Axes,
    x: float,
    y: float,
    base_color: str,
) -> None:
    axis.text(x, y, "low", ha="right", va="center", fontsize=6.4, color="#E6F0F4")
    for index in range(5):
        axis.add_patch(Rectangle(
            (x + 0.008 + index * 0.012, y - 0.007),
            0.012,
            0.014,
            facecolor=blended_color(base_color, index / 4),
            edgecolor="none",
        ))
    axis.text(
        x + 0.075,
        y,
        "high",
        ha="left",
        va="center",
        fontsize=6.4,
        color="#E6F0F4",
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    record = load_record(args.record_input, args.index)
    retrieved = {
        int(chunk.get("rank", index + 1)): chunk
        for index, chunk in enumerate(record["retrieved"])
    }
    chunks = [retrieved[rank] for rank in args.chunk_ranks]

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(7.15, 3.75))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    rounded_box(
        axis, 0.035, 0.855, 0.93, 0.11,
        facecolor="#F8FAFC", edgecolor="#98A2B3", linewidth=0.8,
    )
    axis.text(
        0.055, 0.934, "Query", ha="left", va="center",
        fontsize=8.0, fontweight="bold", color="#344054",
    )
    question = "\n".join(textwrap.wrap(str(record["question"]), width=94))
    axis.text(
        0.055, 0.912, question, ha="left", va="top",
        fontsize=7.5, color="#101828", linespacing=1.1,
    )

    methods = [
        (0.035, "Raw attention", "attention_score", "#3B6FB6"),
        (0.515, "ReFlow contribution", "reflow_score", "#148F77"),
    ]
    card_width = 0.45
    for x, title, score_key, base_color in methods:
        rounded_box(
            axis, x, 0.18, card_width, 0.63,
            facecolor="#FFFFFF", edgecolor="#D0D5DD", linewidth=0.9,
        )
        axis.add_patch(FancyBboxPatch(
            (x, 0.75), card_width, 0.06,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=base_color, edgecolor=base_color, linewidth=0.0,
        ))
        axis.text(
            x + 0.018, 0.78, title, ha="left", va="center",
            fontsize=8.6, fontweight="bold", color="white",
        )
        draw_legend(axis, x + card_width - 0.13, 0.78, base_color)

        for chunk_index, chunk in enumerate(chunks):
            header_y = 0.705 if chunk_index == 0 else 0.445
            text_y = 0.676 if chunk_index == 0 else 0.416
            axis.text(
                x + 0.018,
                header_y,
                f"Retrieved chunk {chunk['rank']}",
                ha="left",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color=base_color,
            )
            excerpt = str(chunk["text"])[: args.excerpt_chars].rstrip()
            if len(str(chunk["text"])) > len(excerpt):
                excerpt = excerpt.rsplit(" ", 1)[0]
                excerpt += " ..."
            draw_excerpt(
                axis,
                figure,
                excerpt,
                payload["tokens"],
                score_key,
                int(chunk["rank"]),
                x + 0.02,
                x + card_width - 0.018,
                text_y,
                base_color,
            )

    for start_x in [0.26, 0.74]:
        axis.add_patch(FancyArrowPatch(
            (start_x, 0.175),
            (0.5, 0.122),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.8,
            color="#98A2B3",
            connectionstyle="arc3,rad=0.08" if start_x < 0.5 else "arc3,rad=-0.08",
        ))
    rounded_box(
        axis, 0.21, 0.035, 0.58, 0.085,
        facecolor="#F0FDF4", edgecolor="#148F77", linewidth=1.0,
    )
    axis.text(
        0.5, 0.077,
        f"Generated answer: {payload['target_answer']}",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#086B5A",
    )

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_pdf, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


if __name__ == "__main__":
    main()
