#!/usr/bin/env python3
"""Aggregate cross-model ReFlow results and render paper-ready LaTeX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.reader import MEDQA_CHOICES, answer_token_f1
from scripts.reprocess_quartz_choice_results import reprocess as reprocess_quartz


DATASETS = (
    ("hotpotqa", "HotpotQA"),
    ("timeqa", "TimeQA"),
    ("finqa", "FinQA"),
    ("musique", "MuSiQue"),
    ("quartz", "QuaRTz"),
    ("triviaqa", "TriviaQA"),
    ("2wiki", "2Wiki"),
    ("medqa", "MedQA"),
)

MAIN_DATASETS = (
    ("hotpotqa", "Hotpot"),
    ("finqa", "FinQA"),
    ("triviaqa", "Trivia"),
    ("medqa", "MedQA"),
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    short_label: str
    root: Path
    layout: str


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("none of the expected artifacts exist: " + ", ".join(map(str, paths)))


def artifact_paths(model: ModelSpec, dataset: str) -> tuple[Path, Path, Path]:
    repaired = model.root / dataset / "choice_postprocess_v1"
    repaired_complete = repaired / "COMPLETE"
    if model.layout == "sweep":
        base = model.root / dataset
        if dataset == "quartz" and repaired_complete.is_file():
            return (
                repaired / "factual/results.jsonl",
                repaired / "control/results.jsonl",
                base / "graph/summary.json",
            )
        return (
            base / "factual/results.jsonl",
            base / "synonym/results.jsonl",
            base / "graph/summary.json",
        )
    if model.layout != "default":
        raise ValueError(f"unknown layout: {model.layout}")

    base = model.root / dataset
    if dataset == "quartz" and repaired_complete.is_file():
        graph = first_existing((
            base / "graphs/contribution_graph_top5_1000.summary.json",
            base / "methods/reflow/frontier_top5_1000.jsonl",
        ))
        return (
            repaired / "factual/results.jsonl",
            repaired / "control/results.jsonl",
            graph,
        )
    if dataset == "hotpotqa":
        factual = first_existing((
            base / "audits/final_top10pool_k5/reflow_1000_v2.jsonl",
            base / "audits/final_top10pool_k5/reflow_1000.jsonl",
        ))
        graph = first_existing((
            base / "graphs/contribution_graph_token_label_1000.summary.json",
            base / "graphs/contribution_graph_top5_1000.summary.json",
            base / "methods/reflow/frontier_top5_1000.jsonl",
        ))
    else:
        factual = base / "methods/reflow/results_top5_1000.jsonl"
        graph = first_existing((
            base / "graphs/contribution_graph_top5_1000.summary.json",
            base / "methods/reflow/frontier_top5_1000.jsonl",
        ))
    control = first_existing((
        base / "controls/paraphrase_results_top5_1000_v2.jsonl",
        base / "controls/paraphrase_results_top5_1000.jsonl",
    ))
    return factual, control, graph


def control_record(row: dict) -> dict:
    methods = row.get("methods", {})
    if "reflow" not in methods:
        raise KeyError(f"missing reflow control for query {row.get('id')}")
    return methods["reflow"]


def graph_coverage(summary: dict) -> tuple[int, int, float]:
    total = int(summary.get("records", summary.get("rows", summary.get("queries", 0))))
    ok = int(summary.get(
        "ok",
        summary.get("queries_with_graph", summary.get("status_histogram", {}).get("ok", 0)),
    ))
    if total <= 0:
        raise ValueError("graph summary has no positive record count")
    return ok, total, ok / total


def load_graph_coverage(path: Path) -> tuple[int, int, float, float | None]:
    if path.suffix == ".jsonl":
        rows = list(iter_jsonl(path))
        ok = sum(
            row.get("frontier_status", row.get("status")) == "ok"
            for row in rows
        )
        if not rows:
            raise ValueError(f"empty coverage artifact: {path}")
        return ok, len(rows), ok / len(rows), None
    summary = json.loads(path.read_text(encoding="utf-8"))
    ok, total, coverage = graph_coverage(summary)
    return ok, total, coverage, summary.get("avg_seconds")


def summarize_dataset(
    model: ModelSpec,
    dataset: str,
    *,
    quartz_records: list[dict] | None = None,
) -> dict:
    factual_path, control_path, graph_path = artifact_paths(model, dataset)
    factual_list = list(iter_jsonl(factual_path))
    control_list = list(iter_jsonl(control_path))
    quartz_report = None
    if dataset == "quartz":
        if quartz_records is None:
            raise ValueError("QuaRTz aggregation requires the frozen retrieval rows")
        factual_list, control_list, quartz_report = reprocess_quartz(
            quartz_records,
            factual_list,
            control_list,
        )
        factual_unresolved = sum(
            count
            for status, count in quartz_report["factual_statuses"].items()
            if status.startswith("unresolved")
        )
        reflow_control_statuses = {}
        for row in control_list:
            status = str(
                control_record(row).get("quartz_postprocess_status", "")
            )
            reflow_control_statuses[status] = (
                reflow_control_statuses.get(status, 0) + 1
            )
        quartz_report["reflow_control_postprocess_statuses"] = dict(
            sorted(reflow_control_statuses.items())
        )
        control_unresolved = sum(
            count
            for status, count in reflow_control_statuses.items()
            if status.startswith("unresolved")
        )
        unresolved = factual_unresolved + control_unresolved
        same_choice_controls = reflow_control_statuses.get(
            "same_choice_false_positive", 0
        )
        repaired_complete = (
            factual_path.parent.parent / "COMPLETE"
        ).is_file()
        if (
            quartz_report["requires_frontier_resume"]
            or same_choice_controls
            or (unresolved and not repaired_complete)
        ):
            raise ValueError(
                f"QuaRTz artifacts for {model.key} require choice-level repair: "
                f"requires_frontier_resume="
                f"{quartz_report['requires_frontier_resume']}, "
                f"unresolved={unresolved}, "
                f"same_choice_controls={same_choice_controls}"
            )

    factual_rows = {str(row["id"]): row for row in factual_list}
    control_rows = {str(row["id"]): row for row in control_list}
    if factual_rows.keys() != control_rows.keys():
        missing_control = sorted(factual_rows.keys() - control_rows.keys())
        missing_factual = sorted(control_rows.keys() - factual_rows.keys())
        raise ValueError(
            f"factual/control query mismatch for {model.key}/{dataset}: "
            f"missing_control={missing_control[:3]} missing_factual={missing_factual[:3]}"
        )
    paired = [
        (factual_rows[query_id], control_record(control_rows[query_id]))
        for query_id in factual_rows
    ]

    if not paired:
        raise ValueError(f"no paired factual/control records for {model.key}/{dataset}")
    if dataset == "medqa":
        allowed = set(MEDQA_CHOICES)
        for factual, control in paired:
            query_id = str(factual.get("id", ""))
            for field in ("clean_answer", "gold_answer", "edited_answer"):
                value = str(factual.get(field, "")).strip().upper()
                if value not in allowed:
                    raise ValueError(
                        f"invalid MedQA {field} for {model.key}/{query_id}: {value!r}"
                    )
            control_answer = str(control.get("edited_answer", "")).strip().upper()
            if control_answer and control_answer not in allowed:
                raise ValueError(
                    f"invalid MedQA control answer for {model.key}/{query_id}: "
                    f"{control_answer!r}"
                )

    ans_deltas = [
        int(bool(factual.get("verified_flip"))) - int(bool(control.get("answer_flip")))
        for factual, control in paired
    ]
    f1_pairs = [
        (factual, control)
        for factual, control in paired
        if answer_token_f1(str(factual.get("clean_answer", "")), str(factual.get("gold_answer", ""))) == 1.0
    ]
    f1_deltas = [
        int(
            answer_token_f1(
                str(factual.get("edited_answer", "")), str(factual.get("gold_answer", ""))
            )
            < 1.0
        )
        - int(bool(control.get("f1_flip")))
        for factual, control in f1_pairs
    ]

    graph_ok, graph_total, coverage, mean_graph_seconds = load_graph_coverage(graph_path)
    all_factual = list(factual_rows.values())
    result = {
        "dataset": dataset,
        "queries": len(all_factual),
        "paired_queries": len(paired),
        "f1_clean_paired_queries": len(f1_pairs),
        "ans_cfr": statistics.fmean(ans_deltas),
        "f1_cfr": statistics.fmean(f1_deltas) if f1_deltas else None,
        "ans_delta_sum": sum(ans_deltas),
        "f1_delta_sum": sum(f1_deltas),
        "mean_modified_tokens": statistics.fmean(
            int(row.get("n_modified_tokens", 0)) for row in all_factual
        ),
        "graph_ok": graph_ok,
        "graph_total": graph_total,
        "graph_coverage": coverage,
        "mean_graph_seconds": mean_graph_seconds,
        "artifacts": {
            "factual": {"path": str(factual_path), "sha256": sha256(factual_path)},
            "control": {"path": str(control_path), "sha256": sha256(control_path)},
            "graph_summary": {"path": str(graph_path), "sha256": sha256(graph_path)},
        },
    }
    if quartz_report is not None:
        result["quartz_postprocess"] = quartz_report
    return result


def aggregate_results(rows: list[dict]) -> dict:
    f1_rows = [row for row in rows if row["f1_cfr"] is not None]
    paired_queries = sum(row["paired_queries"] for row in rows)
    f1_queries = sum(row["f1_clean_paired_queries"] for row in rows)
    total_queries = sum(row["queries"] for row in rows)
    graph_total = sum(row["graph_total"] for row in rows)
    return {
        "ans_cfr": sum(row["ans_delta_sum"] for row in rows) / paired_queries,
        "f1_cfr": sum(row["f1_delta_sum"] for row in rows) / f1_queries,
        "mean_modified_tokens": sum(
            row["mean_modified_tokens"] * row["queries"] for row in rows
        ) / total_queries,
        "graph_coverage": sum(row["graph_ok"] for row in rows) / graph_total,
        "paired_queries": paired_queries,
        "f1_clean_paired_queries": f1_queries,
        "dataset_macro_ans_cfr": statistics.fmean(row["ans_cfr"] for row in rows),
        "dataset_macro_f1_cfr": statistics.fmean(row["f1_cfr"] for row in f1_rows),
    }


def dataset_latex_coordinates(
    models: list[dict], dataset: str, field: str, scale: float = 1.0
) -> str:
    def value(model: dict) -> float:
        row = next(item for item in model["datasets"] if item["dataset"] == dataset)
        result = row[field]
        if result is None:
            raise ValueError(f"missing {field} for {model['key']}/{dataset}")
        return float(result)

    return " ".join(
        f"({index},{value(model) * scale:.2f})"
        for index, model in enumerate(models, start=1)
    )


def render_values(models: list[dict]) -> str:
    lines = ["% Generated by scripts/render_cross_model_sensitivity.py."]
    for dataset, macro in MAIN_DATASETS:
        lines.extend((
            f"\\def\\CrossModel{macro}AnsCoordinates{{"
            f"{dataset_latex_coordinates(models, dataset, 'ans_cfr', 100.0)}}}",
            f"\\def\\CrossModel{macro}FOneCoordinates{{"
            f"{dataset_latex_coordinates(models, dataset, 'f1_cfr', 100.0)}}}",
        ))
    return "\n".join(lines) + "\n"


def appendix_cell(row: dict) -> str:
    f1 = "--" if row["f1_cfr"] is None else f"{100.0 * row['f1_cfr']:.1f}"
    return " & ".join((
        f"{100.0 * row['ans_cfr']:.1f}",
        f1,
        f"{row['mean_modified_tokens']:.2f}",
        f"{100.0 * row['graph_coverage']:.1f}",
        str(row["paired_queries"]),
        str(row["f1_clean_paired_queries"]),
    ))


def render_appendix_rows(models: list[dict]) -> str:
    by_model = {model["key"]: model for model in models}
    lines = [
        "% Generated by scripts/render_cross_model_sensitivity.py.",
        "\\def\\CrossModelAppendixRows{%",
    ]
    for dataset_index, (dataset, label) in enumerate(DATASETS):
        for index, model in enumerate(models):
            row = next(item for item in by_model[model["key"]]["datasets"] if item["dataset"] == dataset)
            dataset_cell = f"\\multirow{{{len(models)}}}{{*}}{{{label}}}" if index == 0 else ""
            if index < len(models) - 1:
                suffix = " \\\\"
            elif dataset_index < len(DATASETS) - 1:
                suffix = " \\\\ \\midrule"
            else:
                suffix = " \\\\"
            lines.append(f"{dataset_cell} & {model['label']} & {appendix_cell(row)}{suffix}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def parse_model(value: str) -> ModelSpec:
    parts = value.split("|", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("model must be KEY|LABEL|SHORT|LAYOUT|ROOT")
    key, label, short_label, layout, root = parts
    if layout not in {"default", "sweep"}:
        raise argparse.ArgumentTypeError("layout must be default or sweep")
    return ModelSpec(key, label, short_label, Path(root), layout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-values-tex", type=Path, required=True)
    parser.add_argument("--out-appendix-tex", type=Path, required=True)
    parser.add_argument("--quartz-retrieval", type=Path)
    args = parser.parse_args()

    quartz_retrieval = args.quartz_retrieval
    if quartz_retrieval is None:
        candidates = [
            model.root / "quartz/retrieval/top10_1000.jsonl"
            for model in args.model
            if model.layout == "default"
        ]
        quartz_retrieval = next(
            (path for path in candidates if path.is_file()),
            None,
        )
    if quartz_retrieval is None or not quartz_retrieval.is_file():
        raise FileNotFoundError(
            "QuaRTz frozen retrieval is required; pass --quartz-retrieval"
        )
    quartz_records = list(iter_jsonl(quartz_retrieval))

    rendered_models = []
    for model in args.model:
        rows = [
            summarize_dataset(
                model,
                dataset,
                quartz_records=quartz_records if dataset == "quartz" else None,
            )
            for dataset, _ in DATASETS
        ]
        rendered_models.append({
            "key": model.key,
            "label": model.label,
            "short_label": model.short_label,
            "layout": model.layout,
            "root": str(model.root),
            "datasets": rows,
            "aggregate": aggregate_results(rows),
        })

    result = {
        "schema": "causalityrag.cross_model_sensitivity.v1",
        "metric_contract": {
            "ans_cfr": (
                "paired factual answer flip minus paired synonym-control answer flip "
                "over all query records; unsupported or unexecuted interventions contribute zero"
            ),
            "f1_cfr": (
                "same paired difference restricted to queries with clean token-F1 equal to one; "
                "unsupported or unexecuted interventions contribute zero"
            ),
            "mean_modified_tokens": "mean final attempted intervention size over all 1,000 queries",
            "main_aggregation": "per-dataset paired queries for four representative datasets",
            "appendix_aggregation": "per-dataset paired queries for all eight datasets",
            "diagnostic": "unweighted dataset macros are retained in JSON but not plotted",
        },
        "models": rendered_models,
    }
    for path in (args.out_json, args.out_values_tex, args.out_appendix_tex):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.out_values_tex.write_text(render_values(rendered_models), encoding="utf-8")
    args.out_appendix_tex.write_text(render_appendix_rows(rendered_models), encoding="utf-8")


if __name__ == "__main__":
    main()
