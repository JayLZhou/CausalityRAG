"""Aggregate real answer flips for fixed-budget residual-flow bands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from causalityrag.reader import answer_token_f1


METHODS = [
    "flow_0_20",
    "flow_20_40",
    "flow_40_60",
    "flow_60_80",
    "flow_80_100",
]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    score_tables = {
        name: {
            str(row["id"]): row
            for row in read_jsonl(args.scores_dir / f"{name}.jsonl")
        }
        for name in METHODS
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluation_rows = read_jsonl(args.evaluations)
    executed_all = 0
    complete = []
    for evaluation in evaluation_rows:
        query_id = str(evaluation.get("id", ""))
        methods = evaluation.get("methods", {})
        if all(
            (
                name in methods
                and methods[name].get("reader_called")
                and int(methods[name].get("n_modified_tokens", 0)) == 20
            )
            for name in METHODS
        ):
            executed_all += 1
        if not all(
            (
                name in methods
                and methods[name].get("reader_called")
                and str(methods[name].get("edited_answer", "")).strip()
                and int(methods[name].get("n_modified_tokens", 0)) == 20
                and score_tables[name].get(query_id, {}).get("ranked_ids")
            )
            for name in METHODS
        ):
            continue
        complete.append(evaluation)
    if not complete:
        raise ValueError("no queries have five complete calibration results")

    bubbles = []
    for rank, name in enumerate(METHODS, start=1):
        residuals = []
        flips = 0
        distances = []
        for evaluation in complete:
            query_id = str(evaluation["id"])
            method = evaluation["methods"][name]
            residuals.append(float(score_tables[name][query_id]["residual_flow"]))
            flips += int(bool(method.get("verified_flip", False)))
            distances.append(
                1.0
                - answer_token_f1(
                    str(evaluation.get("clean_answer", "")),
                    str(method.get("edited_answer", "")),
                )
            )
        count = len(complete)
        bubbles.append(
            {
                "rank": rank,
                "count": count,
                "mean_residual_flow": sum(residuals) / count,
                "answer_flip_rate": flips / count,
                "flip_count": flips,
                "wilson_95": wilson_interval(flips, count),
                "mean_edit_count": 20.0,
                "mean_answer_distance_1_minus_f1": (
                    sum(distances) / count
                ),
            }
        )

    result = {
        "dataset": "HotpotQA",
        "real": True,
        "input_queries": len(evaluation_rows),
        "band_covered_queries": int(manifest["selected_queries"]),
        "executed_queries": executed_all,
        "query_count": len(complete),
        "interventions": len(complete) * len(METHODS),
        "capacity_mode": "normalized",
        "selection_policy": (
            "Five prespecified residual-flow bands per query. Select one graph "
            "candidate per band and pad it to exactly 20 replacements with "
            "eligible zero-flow token positions. Retain only queries with all "
            "five valid edited answers."
        ),
        "source_artifacts": [
            {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            {"path": str(args.evaluations), "sha256": sha256(args.evaluations)},
        ],
        "bubbles": bubbles,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
