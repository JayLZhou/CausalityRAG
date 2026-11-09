"""Materialize one attacked retrieval/target file per security attack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records


ATTACKS = ("poisonedrag", "tparag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = load_records(args.attacks)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for attack_name in ATTACKS:
        retrieval_path = root / f"{attack_name}_retrieval.jsonl"
        targets_path = root / f"{attack_name}_targets.jsonl"
        with retrieval_path.open("w", encoding="utf-8") as retrieval, targets_path.open(
            "w", encoding="utf-8"
        ) as targets:
            for row in rows:
                method = row["methods"][attack_name]
                identifier = str(row.get("id", ""))
                contexts = [dict(value) for value in method["attacked_contexts"]]
                retrieval.write(json.dumps({
                    "id": identifier,
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("gold_answer", "")),
                    "contexts": contexts,
                }, ensure_ascii=False) + "\n")
                targets.write(json.dumps({
                    "id": identifier,
                    "clean_answer": str(method.get("attacked_answer", "")),
                }, ensure_ascii=False) + "\n")
        print(
            f"[security-materialize] {attack_name}: {len(rows)} rows -> {retrieval_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
