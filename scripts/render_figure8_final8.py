#!/usr/bin/env python3
"""Render the final eight-dataset Figure 8 provenance and LaTeX coordinates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from scripts.rescore_figure8_dataset import prepare_population, sha256


DATASETS = (
    ("hotpotqa", "Hotpot", "HQA"),
    ("timeqa", "TQA", "TQA"),
    ("finqa", "FQA", "FQA"),
    ("musique", "MSQ", "MSQ"),
    ("quartz", "QTZ", "QTZ"),
    ("triviaqa", "TrQA", "TrQA"),
    ("2wiki", "2Wiki", "2Wiki"),
    ("popqa", "PopQA", "PopQA"),
)
METHODS = ("mirage", "arc_jsd")
NR_PLOT_Y = 7.45


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_named_paths(specs: list[str]) -> dict[str, Path]:
    result = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid NAME=PATH value: {spec}")
        name, value = spec.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"duplicate or empty dataset name: {name!r}")
        result[name] = Path(value)
    return result


def anchor_from_rows(
    input_path: Path,
    reflow_path: Path,
    *,
    dataset: str,
    reader_mode: str,
) -> tuple[dict, dict]:
    records = load_records(input_path)[:1000]
    reflow_rows = load_records(reflow_path)[:1000]
    if len(records) != 1000 or len(reflow_rows) != 1000:
        raise ValueError("Figure 8 anchors require exactly 1000 rows")
    _, anchor = prepare_population(
        records,
        reflow_rows,
        dataset=dataset,
        reader_mode=reader_mode,
    )
    sources = {
        "input": {"path": str(input_path.resolve()), "sha256": sha256(input_path)},
        "reflow": {
            "path": str(reflow_path.resolve()),
            "sha256": sha256(reflow_path),
        },
    }
    return anchor, sources


def merge_summary_curves(paths: list[Path], clean_queries: int) -> tuple[dict, list[dict]]:
    curves = {method: {} for method in METHODS}
    sources = []
    for path in paths:
        summary = load_json(path)
        if int(summary.get("clean_acc_queries", -1)) != clean_queries:
            raise ValueError(
                f"{path}: clean population {summary.get('clean_acc_queries')} "
                f"!= {clean_queries}"
            )
        for method in METHODS:
            for budget, point in summary.get("methods", {}).get(method, {}).items():
                existing = curves[method].get(str(budget))
                if existing is not None and existing != point:
                    raise ValueError(f"conflicting {method}/{budget} in {path}")
                curves[method][str(budget)] = point
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    return curves, sources


def select_method(
    curve: dict[str, dict],
    *,
    target: float,
    clean_queries: int,
) -> dict:
    points = []
    for budget_raw, point in curve.items():
        if int(point.get("clean_acc_queries", -1)) != clean_queries:
            raise ValueError(f"population mismatch at budget {budget_raw}")
        missing = int(point.get("missing_population_queries", 0))
        if missing:
            continue
        valid = int(point.get("valid_answer_queries", 0))
        if valid <= 0:
            raise ValueError(f"empty valid denominator at budget {budget_raw}")
        points.append((int(budget_raw), point))
    if not points:
        raise ValueError("method has no fully evaluated Figure 8 points")
    points.sort(key=lambda item: item[0])
    crossing = next(
        (
            (budget, point)
            for budget, point in points
            if float(point["acc_fr"] if "acc_fr" in point else point["acc_cfr"])
            >= target
        ),
        None,
    )
    if crossing is not None:
        budget, point = crossing
        available = {value for value, _ in points}
        missing_budgets = sorted(set(range(1, budget + 1)) - available)
        if missing_budgets:
            raise ValueError(
                f"cannot claim first crossing at {budget}; missing lower budgets "
                f"{missing_budgets[:8]}"
            )
        return {
            "status": "matched",
            "minimum_budget": budget,
            "acc_fr": float(
                point["acc_fr"] if "acc_fr" in point else point["acc_cfr"]
            ),
            "valid_answer_queries": int(point["valid_answer_queries"]),
            "protocol_violations": int(point.get("protocol_violations", 0)),
            "mean_modified_tokens": float(point["mean_modified_tokens"]),
            "plot_y_log2": math.log2(budget),
        }

    maximum = max(budget for budget, _ in points)
    if maximum < 128:
        raise ValueError(f"NR method was only tested through budget {maximum}")
    best_budget, best = max(
        points,
        key=lambda item: (
            float(
                item[1]["acc_fr"]
                if "acc_fr" in item[1]
                else item[1]["acc_cfr"]
            ),
            -item[0],
        ),
    )
    return {
        "status": "not_reached",
        "max_tested_budget": maximum,
        "tested_budget_count": len(points),
        "best_budget": best_budget,
        "best_acc_fr": float(
            best["acc_fr"] if "acc_fr" in best else best["acc_cfr"]
        ),
        "best_valid_answer_queries": int(best["valid_answer_queries"]),
        "best_protocol_violations": int(best.get("protocol_violations", 0)),
        "best_mean_modified_tokens": float(best["mean_modified_tokens"]),
        "plot_y_log2": NR_PLOT_Y,
    }


def dataset_from_rescore(path: Path) -> dict:
    data = load_json(path)
    anchor = data["reflow"]
    clean_queries = int(anchor["clean_acc_queries"])
    methods = {
        method: select_method(
            data["methods"][method],
            target=float(anchor["acc_fr"]),
            clean_queries=clean_queries,
        )
        for method in METHODS
    }
    return {
        "clean_acc_queries": clean_queries,
        "reflow": {
            **anchor,
            "plot_y_log2": math.log2(float(anchor["mean_modified_tokens"])),
        },
        "methods": methods,
        "sources": [{"path": str(path.resolve()), "sha256": sha256(path)}],
    }


def dataset_from_summaries(
    summaries: list[Path],
    input_path: Path,
    reflow_path: Path,
    *,
    dataset: str,
    reader_mode: str,
) -> dict:
    anchor, anchor_sources = anchor_from_rows(
        input_path,
        reflow_path,
        dataset=dataset,
        reader_mode=reader_mode,
    )
    clean_queries = int(anchor["clean_acc_queries"])
    curves, summary_sources = merge_summary_curves(summaries, clean_queries)
    methods = {
        method: select_method(
            curves[method],
            target=float(anchor["acc_fr"]),
            clean_queries=clean_queries,
        )
        for method in METHODS
    }
    return {
        "clean_acc_queries": clean_queries,
        "reflow": {
            **anchor,
            "plot_y_log2": math.log2(float(anchor["mean_modified_tokens"])),
        },
        "methods": methods,
        "sources": [*anchor_sources.values(), *summary_sources],
    }


def latex_coordinates(data: dict, method: str) -> str:
    return " ".join(
        f"({symbol},{data['datasets'][name]['methods'][method]['plot_y_log2']:.5f})"
        for name, symbol, _ in DATASETS
    )


def render_latex(data: dict) -> str:
    reflow = " ".join(
        f"({symbol},{data['datasets'][name]['reflow']['plot_y_log2']:.5f})"
        for name, symbol, _ in DATASETS
    )
    nr = []
    for name, _, label in DATASETS:
        methods = data["datasets"][name]["methods"]
        missing = [
            "MIRAGE" if method == "mirage" else "ARC-JSD"
            for method in METHODS
            if methods[method]["status"] == "not_reached"
        ]
        if missing:
            nr.append(f"{label}: {'/'.join(missing)}")
    lines = [
        "% Auto-generated by scripts/render_figure8_final8.py; do not edit.",
        "\\def\\FigureEightMirageCoordinates{%s}" % latex_coordinates(data, "mirage"),
        "\\def\\FigureEightArcCoordinates{%s}" % latex_coordinates(data, "arc_jsd"),
        "\\def\\FigureEightReflowCoordinates{%s}" % reflow,
        "\\def\\FigureEightNRNote{%s}" % "; ".join(nr),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", action="append", default=[])
    parser.add_argument("--quartz-summary", type=Path, required=True)
    parser.add_argument("--quartz-input", type=Path, required=True)
    parser.add_argument("--quartz-reflow", type=Path, required=True)
    parser.add_argument("--popqa-summary", type=Path, action="append", required=True)
    parser.add_argument("--popqa-input", type=Path, required=True)
    parser.add_argument("--popqa-reflow", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-tex", type=Path, required=True)
    args = parser.parse_args()

    rescore_paths = parse_named_paths(args.rescore)
    required_rescores = {
        "hotpotqa", "timeqa", "finqa", "musique", "triviaqa", "2wiki"
    }
    if set(rescore_paths) != required_rescores:
        raise ValueError(
            f"expected rescores {sorted(required_rescores)}, got "
            f"{sorted(rescore_paths)}"
        )
    datasets = {
        name: dataset_from_rescore(path)
        for name, path in rescore_paths.items()
    }
    datasets["quartz"] = dataset_from_summaries(
        [args.quartz_summary],
        args.quartz_input,
        args.quartz_reflow,
        dataset="quartz",
        reader_mode="quartz",
    )
    datasets["popqa"] = dataset_from_summaries(
        args.popqa_summary,
        args.popqa_input,
        args.popqa_reflow,
        dataset="popqa",
        reader_mode="short_answer",
    )
    result = {
        "schema": "causalityrag.figure8_final8.v1",
        "metric": "Acc-FR",
        "contract": (
            "clean-accuracy population; invalid edited answers excluded from "
            "both numerator and denominator; matched bars use the first fully "
            "evaluated integer budget; NR bars are capped above 128"
        ),
        "dataset_order": [name for name, _, _ in DATASETS],
        "datasets": {name: datasets[name] for name, _, _ in DATASETS},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(render_latex(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
