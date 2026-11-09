"""Merge security guard/keyword and secure-decoding results."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records


ATTACKS = ("poisonedrag", "tparag")
METHODS = (
    "no_defense",
    "reflow_guard",
    "random_guard",
    "robustrag_keyword",
    "robustrag_decoding",
)


def bootstrap_interval(values: list[float], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(10_000)
    ]
    means.sort()
    return [means[249], means[9749]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-keyword", required=True)
    parser.add_argument("--decoding", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    guard_rows = load_records(args.guard_keyword)
    decoding_by_id = {
        str(row.get("id", "")): row for row in load_records(args.decoding)
    }
    rows = []
    for row in guard_rows:
        identifier = str(row.get("id", ""))
        decoding = decoding_by_id[identifier]
        merged = json.loads(json.dumps(row))
        for attack_name in ATTACKS:
            merged["attacks"][attack_name]["robustrag_decoding"] = dict(
                decoding["attacks"][attack_name]
            )
        rows.append(merged)

    result: dict = {"queries": len(rows), "attacks": {}, "macro": {}}
    for attack_name in ATTACKS:
        result["attacks"][attack_name] = {}
        for method in METHODS:
            values = [
                float(row["attacks"][attack_name][method]["correct"])
                for row in rows
            ]
            result["attacks"][attack_name][method] = {
                "attacked_accuracy": statistics.fmean(values),
                "attacked_accuracy_ci95": bootstrap_interval(
                    values,
                    seed=101 + 17 * len(result["attacks"]) + len(method),
                ),
            }
            if method in {"reflow_guard", "random_guard"}:
                detection_values = [
                    float(row["attacks"][attack_name][method]["detected"])
                    for row in rows
                    if row["attacks"][attack_name]["malicious_passage_present"]
                ]
                result["attacks"][attack_name][method].update({
                    "detection_rate": (
                        statistics.fmean(detection_values)
                        if detection_values
                        else None
                    ),
                    "mean_verified_sentences": statistics.fmean(
                        float(row["attacks"][attack_name][method]["verified_sentences"])
                        for row in rows
                    ),
                })
    for method in METHODS:
        values = [
            statistics.fmean(
                float(row["attacks"][attack_name][method]["correct"])
                for attack_name in ATTACKS
            )
            for row in rows
        ]
        result["macro"][method] = {
            "attacked_accuracy": statistics.fmean(values),
            "attacked_accuracy_ci95": bootstrap_interval(
                values,
                seed=701 + len(method),
            ),
        }
    result["protocol"] = {
        "retrieved_chunks": 10,
        "maximum_injected_passages": 1,
        "attacked_context_only": True,
        "reflow_verifier": "perfect provenance verifier",
        "robustrag_keyword": {"alpha": 0.2, "beta": 3.0},
        "robustrag_decoding": {"eta": 0.0},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
