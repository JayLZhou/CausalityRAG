"""Build one global, validated multi-candidate replacement inventory.

The expensive part is deliberately global: repeated surface tokens are grouped
by token/type/grammar signature before the LLM is called.  The generated list
is then validated at every occurrence and projected back to query-keyed pools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.replacement import GenericReplacementClient
from causalityrag.token_units import units_from_cache_row


SEMANTIC_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
NON_SEMANTIC_TYPES = {"", "PUNCT", "STOPWORD"}


def take_jsonl(path: str, start: int, count: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index < start:
                continue
            if len(rows) >= count:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_semantic_unit(unit: dict) -> bool:
    token = str(unit.get("text", "")).strip()
    return bool(
        token
        and any(character.isalnum() for character in token)
        and str(unit.get("type", "")).upper() not in NON_SEMANTIC_TYPES
        and str(unit.get("pos", "")).upper() in SEMANTIC_POS
    )


def inventory_key(unit: dict) -> str:
    """Group repeated tokens without conflating different entity slots."""

    fields = (
        str(unit.get("text", "")).casefold(),
        str(unit.get("type", "")).upper(),
        str(unit.get("pos", "")).upper(),
        str(unit.get("tag", "")).upper(),
        str(unit.get("entity_token_index", "")),
        str(unit.get("entity_token_count", "")),
    )
    return "\t".join(fields)


def stable_batch_id(key: str) -> str:
    return "batch:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def target_from_occurrence(key: str, occurrence: dict) -> dict:
    context = str(occurrence.get("context", ""))
    start = max(0, int(occurrence.get("chunk_char_start", 0)) - 160)
    end = min(
        len(context),
        int(occurrence.get("chunk_char_end", 0)) + 160,
    )
    return {
        **occurrence,
        "unit_id": stable_batch_id(key),
        "token": str(occurrence["text"]),
        "unit_type": str(occurrence.get("type", "")),
        "pos_hint": str(occurrence.get("pos", "")),
        "tag_hint": str(occurrence.get("tag", "")),
        "context": context[start:end],
        "forbidden": (),
    }


def collect_inventory(
    records: list[dict],
    unit_rows: list[dict],
    *,
    k: int,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    occurrences: dict[str, list[dict]] = defaultdict(list)
    query_units: dict[str, list[dict]] = defaultdict(list)
    for record, unit_row in zip(records, unit_rows):
        query_id = record_id(record)
        contexts = {
            str(context["chunk_id"]): str(context["text"])
            for context in retrieved_contexts(record)[:k]
        }
        units = units_from_cache_row(record, unit_row, k=k)
        for unit in units:
            if not is_semantic_unit(unit):
                continue
            occurrence = {
                **unit,
                "query_id": query_id,
                "context": contexts.get(str(unit["chunk_id"]), ""),
                "occurrence_id": f"{query_id}:{unit['unit_id']}",
            }
            key = inventory_key(unit)
            occurrences[key].append(occurrence)
            query_units[query_id].append({"key": key, **occurrence})
    return dict(occurrences), dict(query_units)


def unique_candidates(rows: list[dict]) -> list[dict]:
    output = []
    seen = set()
    for row in rows:
        candidate = str(row.get("new", "")).strip()
        lowered = candidate.casefold()
        if not candidate or lowered in seen:
            continue
        seen.add(lowered)
        output.append({**row, "new": candidate})
    return output


def validate_new_candidates(
    keys: list[str],
    generated: dict[str, list[dict]],
    occurrences: dict[str, list[dict]],
    validated: dict[str, dict[str, list[dict]]],
    nlp: SpacyAnnotationClient,
) -> None:
    items = []
    locations = []
    for key in keys:
        for occurrence in occurrences[key]:
            occurrence_id = occurrence["occurrence_id"]
            seen = {
                str(row["new"]).casefold()
                for row in validated[key].get(occurrence_id, [])
            }
            original = str(occurrence["text"]).casefold()
            for candidate in generated.get(key, []):
                value = str(candidate.get("new", "")).strip()
                if not value or value.casefold() in seen or value.casefold() == original:
                    continue
                items.append({
                    "unit": occurrence,
                    "context": occurrence["context"],
                    "replacement": candidate,
                })
                locations.append((key, occurrence_id, candidate))
    if not items:
        return
    results = nlp.validate_many(items)
    for (key, occurrence_id, candidate), validation in zip(locations, results):
        if validation.get("valid"):
            validated[key].setdefault(occurrence_id, []).append(
                {**candidate, "validation": validation}
            )


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inventory-out", required=True)
    parser.add_argument("--cf-pools", default="")
    parser.add_argument("--spacy-base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--llm-base-urls",
        default="",
        help="Comma-separated OpenAI-compatible base URLs, one per worker.",
    )
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    records = take_jsonl(args.input, args.start, args.n)
    unit_rows = take_jsonl(args.units_cache, args.start, args.n)
    if len(records) != len(unit_rows):
        raise ValueError("input and unit cache rows are misaligned")
    nlp = SpacyAnnotationClient(args.spacy_base_url)
    if not nlp.health().get("ok"):
        raise RuntimeError("spaCy annotation service is unhealthy")

    occurrences, query_units = collect_inventory(records, unit_rows, k=args.k)
    keys = sorted(occurrences)
    targets = [target_from_occurrence(key, occurrences[key][0]) for key in keys]
    base_urls = [
        value.strip()
        for value in (
            args.llm_base_urls
            or os.environ.get("CAUSALITYRAG_LLM_BASE_URL", "")
        ).split(",")
        if value.strip()
    ]
    if not base_urls:
        base_urls = [None]
    editors = [GenericReplacementClient(base_url=url) for url in base_urls]
    worker_count = max(1, min(args.workers, len(editors)))
    validated: dict[str, dict[str, list[dict]]] = {
        key: defaultdict(list) for key in keys
    }
    all_generated: dict[str, list[dict]] = defaultdict(list)
    unresolved = set(keys)
    forbidden: dict[str, set[str]] = {key: set() for key in keys}
    target_by_key = dict(zip(keys, targets))
    stats = Counter()
    started = time.monotonic()

    for round_index in range(args.max_rounds):
        if not unresolved:
            break
        round_keys = sorted(unresolved)
        next_unresolved = set()
        pending = []
        for begin in range(0, len(round_keys), args.batch_size):
            batch_keys = round_keys[begin : begin + args.batch_size]
            batch = []
            for key in batch_keys:
                target = dict(target_by_key[key])
                target["forbidden"] = tuple(sorted(forbidden[key]))
                batch.append(target)
            pending.append((batch_keys, batch, editors[len(pending) % len(editors)]))

        def submit_batch(item):
            batch_keys, batch, editor = item
            return batch_keys, batch, editor.replace_many_candidate_lists(
                batch, max_candidates=args.max_candidates
            )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(submit_batch, pending))

        for batch_keys, batch, raw in results:
            for key, target in zip(batch_keys, batch):
                generated = unique_candidates(raw.get(target["unit_id"], []))
                all_generated[key].extend(generated)
                forbidden[key].update(str(row["new"]).casefold() for row in generated)
            validate_new_candidates(
                batch_keys,
                {key: unique_candidates(all_generated[key]) for key in batch_keys},
                occurrences,
                validated,
                nlp,
            )
            for key in batch_keys:
                if any(not validated[key].get(item["occurrence_id"]) for item in occurrences[key]):
                    next_unresolved.add(key)
                else:
                    stats["resolved_keys"] += 1
        unresolved = next_unresolved
        print(
            f"[global-batch] round={round_index + 1} keys={len(round_keys)} "
            f"tasks={len(pending)} workers={worker_count} "
            f"unresolved={len(unresolved)} calls={sum(editor.calls for editor in editors)}",
            flush=True,
        )

    query_rows = []
    inventory_rows = []
    for key in keys:
        occurrence_rows = {}
        for occurrence in occurrences[key]:
            candidates = unique_candidates(
                validated[key].get(occurrence["occurrence_id"], [])
            )
            if candidates:
                stats["valid_occurrences"] += 1
                occurrence_rows[occurrence["occurrence_id"]] = candidates
            else:
                stats["invalid_occurrences"] += 1
                occurrence_rows[occurrence["occurrence_id"]] = []
        inventory_rows.append({
            "key": key,
            "token": occurrences[key][0]["text"],
            "type": occurrences[key][0].get("type", ""),
            "pos": occurrences[key][0].get("pos", ""),
            "tag": occurrences[key][0].get("tag", ""),
            "occurrence_count": len(occurrences[key]),
            "candidates": unique_candidates(all_generated[key]),
            "valid_occurrences": sum(bool(values) for values in occurrence_rows.values()),
            "occurrences": occurrence_rows,
        })

    for record in records:
        query_id = record_id(record)
        replacements = {}
        for occurrence in query_units.get(query_id, []):
            values = unique_candidates(
                validated[occurrence["key"]].get(occurrence["occurrence_id"], [])
            )
            replacements[occurrence["unit_id"]] = {
                "ok": bool(values),
                "old": occurrence["text"],
                "new": values[0]["new"] if values else "",
                "candidates": values,
                "replacement_key": occurrence["key"],
                "policy": "global_batch_candidate_pool" if values else "global_batch_pool_exhausted",
                "validation": values[0].get("validation", {}) if values else {
                    "valid": False,
                    "reason": "no_valid_candidate_after_global_batch_rounds",
                },
            }
        query_rows.append({
            "id": query_id,
            "replacements": replacements,
            "valid_tokens": sum(value["ok"] for value in replacements.values()),
            "invalid_tokens": sum(not value["ok"] for value in replacements.values()),
            "generation_mode": "global_unique_token_batch",
            "llm_batch_calls": sum(editor.calls for editor in editors),
        })

    write_jsonl(args.out, query_rows)
    write_jsonl(args.inventory_out, inventory_rows)
    summary = {
        "queries": len(records),
        "unique_semantic_keys": len(keys),
        "semantic_occurrences": sum(len(values) for values in occurrences.values()),
        "resolved_keys": sum(
            all(validated[key].get(item["occurrence_id"])
                for item in occurrences[key])
            for key in keys
        ),
        "unresolved_keys": len(unresolved),
        **stats,
        "llm_batch_calls": sum(editor.calls for editor in editors),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pool": os.path.abspath(args.out),
        "inventory": os.path.abspath(args.inventory_out),
    }
    print("[global-batch summary] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
