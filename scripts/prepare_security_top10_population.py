"""Prepare a 100-query population that is clean-correct under top-10 RAG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.reader import answers_match
from scripts.run_security_pareto_attacks import attack_outcome


ATTACKS = ("poisonedrag", "tparag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--clean-targets", required=True)
    parser.add_argument("--candidate-attacks", required=True)
    parser.add_argument("--retained-out", required=True)
    parser.add_argument("--supplement-reflow-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    retrieval = load_records(args.retrieval)
    retrieval_by_id = {record_id(row): row for row in retrieval}
    targets = load_records(args.clean_targets)
    target_by_id = {str(row.get("id", "")): row for row in targets}
    candidate_attacks = load_records(args.candidate_attacks)

    retained = []
    retained_ids = set()
    for source in candidate_attacks:
        identifier = str(source.get("id", ""))
        target = target_by_id.get(identifier)
        if target is None:
            continue
        clean = str(target.get("clean_answer", ""))
        gold = str(source.get("gold_answer", retrieval_by_id[identifier].get("answer", "")))
        if not answers_match(clean, gold):
            continue
        row = json.loads(json.dumps(source))
        row["clean_answer"] = clean
        for attack_name in ATTACKS:
            method = row["methods"][attack_name]
            method.update(attack_outcome(
                str(method.get("attacked_answer", "")),
                clean=clean,
                gold=gold,
                target=str(row.get("target_answer", "")),
            ))
        retained.append(row)
        retained_ids.add(identifier)

    needed = args.n - len(retained)
    if needed < 0:
        retained = retained[: args.n]
        retained_ids = {str(row["id"]) for row in retained}
        needed = 0
    supplemental = []
    for index, target in enumerate(targets):
        identifier = str(target.get("id", ""))
        if identifier in retained_ids or identifier not in retrieval_by_id:
            continue
        record = retrieval_by_id[identifier]
        clean = str(target.get("clean_answer", ""))
        gold = str(record.get("answer", ""))
        if not answers_match(clean, gold):
            continue
        supplemental.append({
            "index": index,
            "id": identifier,
            "clean_answer": clean,
            "gold_answer": gold,
        })
        if len(supplemental) == needed:
            break
    if len(retained) + len(supplemental) != args.n:
        raise ValueError(
            f"could form only {len(retained) + len(supplemental)} clean-correct rows"
        )

    for path in (args.retained_out, args.supplement_reflow_out, args.manifest_out):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.retained_out, "w", encoding="utf-8") as output:
        for row in retained:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.supplement_reflow_out, "w", encoding="utf-8") as output:
        for row in supplemental:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "queries": args.n,
        "clean_correct_metric": "answers_match",
        "retained_queries": len(retained),
        "supplemental_queries": len(supplemental),
        "ids": [str(row["id"]) for row in retained + supplemental],
        "retained_ids": [str(row["id"]) for row in retained],
        "supplemental_ids": [str(row["id"]) for row in supplemental],
    }
    with open(args.manifest_out, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
