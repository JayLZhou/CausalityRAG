"""Build a category-level replacement registry with very few LLM calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.replacement import GenericReplacementClient
from causalityrag.rules import ONTO_TYPES, TypedRuleLibrary
from causalityrag.token_units import units_from_cache_row


SEMANTIC_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
SKIP_TYPES = {"", "PUNCT", "STOPWORD"}


def read_rows(path: str, start: int, count: int) -> list[dict]:
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


def category(unit: dict) -> str:
    unit_type = str(unit.get("type", "")).upper()
    if unit_type in ONTO_TYPES:
        return unit_type
    return "POS_" + str(unit.get("pos", "")).upper()


def semantic(unit: dict) -> bool:
    return bool(
        str(unit.get("text", "")).strip()
        and any(char.isalnum() for char in str(unit.get("text", "")))
        and str(unit.get("type", "")).upper() not in SKIP_TYPES
        and str(unit.get("pos", "")).upper() in SEMANTIC_POS
    )


def valid_single_tokens(values: list[str], original: str) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = str(value).strip()
        if not value or value.casefold() == original.casefold():
            continue
        if any(char.isspace() for char in value):
            continue
        if value.casefold() in seen:
            continue
        seen.add(value.casefold())
        result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--cf-pools", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--spacy-base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    records = read_rows(args.input, args.start, args.n)
    unit_rows = read_rows(args.units_cache, args.start, args.n)
    if len(records) != len(unit_rows):
        raise ValueError("input and unit cache rows are misaligned")
    occurrences = defaultdict(list)
    for record, row in zip(records, unit_rows):
        query_id = record_id(record)
        contexts = {
            str(context["chunk_id"]): str(context["text"])
            for context in retrieved_contexts(record)[: args.k]
        }
        for unit in units_from_cache_row(record, row, k=args.k):
            if not semantic(unit):
                continue
            value = {
                **unit,
                "query_id": query_id,
                "context": contexts.get(str(unit["chunk_id"]), ""),
                "occurrence_id": f"{query_id}:{unit['unit_id']}",
            }
            occurrences[category(unit)].append(value)

    categories = sorted(occurrences)
    editor = GenericReplacementClient(base_url=args.llm_base_url)
    rules = editor.generate_category_rules(categories, max_candidates=32)
    library = TypedRuleLibrary.from_files(args.cf_pools)
    type_pools = library.type_pool

    category_candidates = {}
    for cat, values in occurrences.items():
        pool = list(rules.get(cat, {}).get("candidates", []))
        for unit_type in {
            str(unit.get("type", "")).upper() for unit in values
        }:
            pool.extend(type_pools.get(unit_type, []))
        category_candidates[cat] = valid_single_tokens(
            pool, str(values[0]["text"])
        )[:16]

    nlp = SpacyAnnotationClient(args.spacy_base_url)
    if not nlp.health().get("ok"):
        raise RuntimeError("spaCy annotation service is unhealthy")
    # Validate a small representative set per category, then apply the
    # validated category rule to every occurrence.  Context-specific failures
    # are left for the later fallback LLM path.
    category_valid = defaultdict(list)
    validation_items = []
    validation_keys = []
    for cat, values in occurrences.items():
        representative = values[0]
        for candidate in category_candidates[cat][:8]:
            validation_items.append({
                "unit": representative,
                "context": representative["context"],
                "replacement": {"new": candidate},
            })
            validation_keys.append((cat, candidate))
    for begin in range(0, len(validation_items), 256):
        batch = validation_items[begin : begin + 256]
        validations = nlp.validate_many(batch)
        for key, validation in zip(validation_keys[begin : begin + 256], validations):
            if validation.get("valid"):
                category_valid[key[0]].append({"new": key[1], "validation": validation})

    # Persist only the compact category registry.  Per-position replacements
    # are materialized after ReFlow selects tokens, which avoids expanding
    # hundreds of thousands of occurrences into a huge JSON file.
    output = {
        "categories": categories,
        "rules": rules,
        "validated_candidates": dict(category_valid),
        "llm_calls": editor.calls,
        "occurrences": sum(len(values) for values in occurrences.values()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as sink:
        json.dump(output, sink, ensure_ascii=False)
    print(json.dumps({
        "categories": categories,
        "llm_calls": editor.calls,
        "occurrences": output["occurrences"],
        "validated_categories": sum(bool(values) for values in category_valid.values()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
