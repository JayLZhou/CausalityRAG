"""Build a frozen paraphrase pool for the positions selected in an experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.paraphrase_control import (
    PARAPHRASE_POLICY,
    SentenceParaphraseClient,
    excluded_from_paraphrase_control,
    wordnet_support,
)
from causalityrag.semantic_filter import ensure_wordnet_available
from causalityrag.semantic_filter import lexical_paraphrase_candidates
from causalityrag.shared_replacement_pool import file_sha256


def selected_ids(reflow_row: dict, baseline_row: dict) -> set[str]:
    result = {str(value) for value in reflow_row.get("selected_ids", [])}
    for method in baseline_row.get("methods", {}).values():
        result.update(str(value) for value in method.get("selected_ids", []))
    result.discard("")
    return result


def target_from_unit(unit: dict, sentences: dict[str, dict]) -> dict:
    sentence = sentences[str(unit["sentence_id"])]
    start = int(unit["chunk_char_start"]) - int(sentence["chunk_char_start"])
    end = int(unit["chunk_char_end"]) - int(sentence["chunk_char_start"])
    text = str(sentence["text"])
    marked = text[:start] + "[[" + text[start:end] + "]]" + text[end:]
    return {
        "unit_id": str(unit["unit_id"]),
        "token": str(unit["text"]),
        "pos": str(unit.get("pos", "")),
        "tag": str(unit.get("tag", "")),
        "unit_type": str(unit.get("type", "")),
        "sentence": marked,
    }


def generate_batch(
    targets: list[dict],
    *,
    client: SentenceParaphraseClient,
    attempts: int,
    max_candidates: int,
) -> list[dict]:
    accepted: dict[str, list[dict]] = {str(row["unit_id"]): [] for row in targets}
    by_id = {str(row["unit_id"]): row for row in targets}
    forbidden = {unit_id: set() for unit_id in by_id}
    for attempt in range(attempts):
        unresolved = [
            {
                **target,
                "forbidden": sorted(forbidden[unit_id]),
            }
            for unit_id, target in by_id.items()
            if len(accepted[unit_id]) < max_candidates
        ]
        if not unresolved:
            break
        proposed = client.generate_many(
            unresolved,
            max_candidates=max_candidates,
            attempt=attempt,
        )
        for target in unresolved:
            unit_id = str(target["unit_id"])
            lexical = list(lexical_paraphrase_candidates(
                str(target["token"]),
                str(target.get("pos", "")),
                limit=2,
            ))
            merged = []
            seen = set()
            for candidate in lexical + proposed.get(unit_id, []):
                folded = candidate.casefold()
                if folded in seen or folded in forbidden[unit_id]:
                    continue
                seen.add(folded)
                merged.append(candidate)
            proposed[unit_id] = merged
        judgments = client.judge_many(unresolved, proposed)
        for target in unresolved:
            unit_id = str(target["unit_id"])
            for candidate in proposed.get(unit_id, []):
                folded = candidate.casefold()
                forbidden[unit_id].add(folded)
                if not judgments.get((unit_id, candidate), False):
                    continue
                if any(
                    row["new"].casefold() == folded for row in accepted[unit_id]
                ):
                    continue
                accepted[unit_id].append({
                    "new": candidate,
                    "policy": PARAPHRASE_POLICY,
                    "wordnet_supported": wordnet_support(target, candidate),
                    "llm_equivalence_judge": True,
                })
                if len(accepted[unit_id]) >= max_candidates:
                    break
    for unit_id, target in by_id.items():
        if accepted[unit_id]:
            continue
        fallback = equivalent_rendering(str(target["token"]))
        if fallback is not None:
            accepted[unit_id].append({
                "new": fallback,
                "policy": "deterministic_equivalent_rendering_v2",
                "wordnet_supported": False,
                "llm_equivalence_judge": False,
                "fallback": True,
            })
    return [
        {
            "unit_id": unit_id,
            "original": str(by_id[unit_id]["token"]),
            "candidates": candidates,
            "status": "covered" if candidates else "unresolved",
            "policy": PARAPHRASE_POLICY,
        }
        for unit_id, candidates in accepted.items()
    ]


def equivalent_rendering(token: str) -> str | None:
    """Return a deterministic information-preserving surface rendering."""

    value = str(token).strip()
    if any(character.isalpha() for character in value):
        upper = value.upper()
        if upper != value:
            return upper
        lower = value.lower()
        if lower != value:
            return lower
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--reflow-results", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--unresolved-out", required=True)
    parser.add_argument(
        "--existing-pool",
        default="",
        help=(
            "Reuse covered rows from an existing frozen pool and generate "
            "only newly selected positions."
        ),
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    args = parser.parse_args()

    ensure_wordnet_available()
    records = load_records(args.input)[: args.n]
    unit_rows = load_records(args.units_cache)[: args.n]
    reflow_rows = load_records(args.reflow_results)[: args.n]
    baseline_rows = load_records(args.baseline_results)[: args.n]
    if len({len(records), len(unit_rows), len(reflow_rows), len(baseline_rows)}) != 1:
        raise ValueError("all inputs must have aligned row counts")

    targets = {}
    for record, units_row, reflow_row, baseline_row in zip(
        records,
        unit_rows,
        reflow_rows,
        baseline_rows,
    ):
        identifier = record_id(record)
        aligned = {
            str(units_row.get("id", "")),
            str(reflow_row.get("id", "")),
            str(baseline_row.get("id", "")),
        }
        if aligned != {identifier}:
            raise ValueError(f"misaligned query {identifier}: {aligned}")
        wanted = selected_ids(reflow_row, baseline_row)
        by_id = {
            str(unit["unit_id"]): unit for unit in units_row.get("units", [])
        }
        sentences = {
            str(row["sentence_id"]): row
            for row in units_row.get("sentences", [])
        }
        for unit_id in wanted:
            if unit_id not in by_id:
                raise KeyError(f"selected token missing from unit cache: {unit_id}")
            targets[unit_id] = target_from_unit(by_id[unit_id], sentences)

    excluded_targets = {
        unit_id: target
        for unit_id, target in targets.items()
        if excluded_from_paraphrase_control(target)
    }
    eligible_targets = {
        unit_id: target
        for unit_id, target in targets.items()
        if unit_id not in excluded_targets
    }

    existing_rows = {}
    if args.existing_pool:
        existing_rows = {
            str(row["unit_id"]): row
            for row in load_records(args.existing_pool)
            if row.get("unit_id") and row.get("candidates")
        }
    reused = [
        existing_rows[unit_id]
        for unit_id in sorted(eligible_targets)
        if unit_id in existing_rows
    ]
    ordered = [
        eligible_targets[unit_id]
        for unit_id in sorted(eligible_targets)
        if unit_id not in existing_rows
    ]
    batches = [
        ordered[start : start + args.batch_size]
        for start in range(0, len(ordered), args.batch_size)
    ]
    client = SentenceParaphraseClient(
        base_url=args.llm_base_url or None,
        model=args.llm_model or None,
    )
    rows = list(reused)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                generate_batch,
                batch,
                client=client,
                attempts=args.attempts,
                max_candidates=args.max_candidates,
            )
            for batch in batches
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if completed % 25 == 0:
                covered = sum(bool(row["candidates"]) for row in rows)
                print(
                    f"[{completed}/{len(futures)}] "
                    f"covered={covered}/{len(rows)} calls={client.calls}",
                    flush=True,
                )
    rows.sort(key=lambda row: row["unit_id"])
    covered = [row for row in rows if row["candidates"]]
    unresolved = [row for row in rows if not row["candidates"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for row in covered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.unresolved_out, "w", encoding="utf-8") as output:
        for row in unresolved:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "policy": PARAPHRASE_POLICY,
        "selected_positions": len(targets),
        "paraphrase_eligible_positions": len(eligible_targets),
        "excluded_numeric_date_positions": len(excluded_targets),
        "covered_positions": len(covered),
        "unresolved_positions": len(unresolved),
        "coverage": (
            len(covered) / len(eligible_targets)
            if eligible_targets
            else 1.0
        ),
        "wordnet_supported_positions": sum(
            any(candidate.get("wordnet_supported") for candidate in row["candidates"])
            for row in covered
        ),
        "llm_verified_positions": sum(
            any(candidate.get("llm_equivalence_judge") for candidate in row["candidates"])
            for row in covered
        ),
        "deterministic_fallback_positions": sum(
            any(candidate.get("fallback") for candidate in row["candidates"])
            for row in covered
        ),
        "llm_calls": client.calls,
        "reused_positions": len(reused),
        "generated_positions": len(ordered),
        "existing_pool": (
            os.path.abspath(args.existing_pool)
            if args.existing_pool
            else None
        ),
        "existing_pool_sha256": (
            file_sha256(args.existing_pool)
            if args.existing_pool
            else None
        ),
        "pool": os.path.abspath(args.out),
        "pool_sha256": file_sha256(args.out),
    }
    with open(args.manifest_out, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
