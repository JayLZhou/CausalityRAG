#!/usr/bin/env python3
"""Merge a power-of-two sweep with its exact integer refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    coarse = read(args.coarse)
    coarse_first = coarse.get("first_matching_tested_budget")
    combined = dict(coarse.get("token_shapley", {}))
    exact_first = None
    if coarse_first is not None:
        if int(coarse_first) == 1:
            exact_first = 1
        else:
            if args.fine is None:
                raise ValueError("a fine summary is required after a coarse hit")
            fine = read(args.fine)
            combined.update(fine.get("token_shapley", {}))
            exact_first = fine.get("first_matching_tested_budget")
            if exact_first is None:
                raise ValueError("refinement did not reproduce the coarse hit")

    output = dict(coarse)
    output["token_shapley"] = {
        key: combined[key]
        for key in sorted(combined, key=lambda value: int(value))
    }
    output["first_matching_budget"] = exact_first
    output["exact_integer_search"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

