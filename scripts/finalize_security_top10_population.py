"""Combine retained and newly generated top-10 clean-correct attacks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from causalityrag.reader import answers_match


ATTACKS = ("poisonedrag", "tparag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", required=True)
    parser.add_argument("--supplemental", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    by_id = {
        str(row.get("id", "")): row
        for row in load_records(args.retained) + load_records(args.supplemental)
    }
    rows = []
    for identifier in manifest["ids"]:
        row = by_id[str(identifier)]
        row["methods"] = {
            name: row["methods"][name] for name in ATTACKS
        }
        if not answers_match(str(row.get("clean_answer", "")), str(row.get("gold_answer", ""))):
            raise ValueError(f"{identifier}: not clean-correct under top-10")
        for name in ATTACKS:
            contexts = row["methods"][name]["attacked_contexts"]
            if len(contexts) != 10:
                raise ValueError(f"{identifier}:{name}: expected 10 contexts")
            injected = sum(
                str(context.get("chunk_id", "")).startswith(name + ":")
                for context in contexts
            )
            if injected != 1:
                raise ValueError(f"{identifier}:{name}: expected one injection")
        rows.append(row)
    if len(rows) != int(manifest["queries"]):
        raise ValueError("final population size mismatch")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[security-population] wrote {len(rows)} clean-correct rows", flush=True)


if __name__ == "__main__":
    main()
