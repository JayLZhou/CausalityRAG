#!/usr/bin/env python3
"""Recompute paraphrase-control metrics from saved reader outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from scripts.evaluate_paraphrase_controls import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = summarize(load_records(args.input))
    result["metric_contract"] = {
        "answer_population": "all rows",
        "correctness_population": (
            "metric-specific clean-correct queries: clean F1=1 for F1, "
            "clean exact match for EM, and clean normalized containment "
            "accuracy for Acc"
        ),
        "correctness_flip": "the edited metric is lower than the clean metric",
    }
    result["source"] = os.path.abspath(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
