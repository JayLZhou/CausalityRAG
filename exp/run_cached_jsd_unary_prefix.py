"""Experimental baseline: evaluate cached ARC-JSD unary prefixes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.arc_jsd import context_sentence_units
from causalityrag.io import iter_records, record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.reader import ReaderClient, answers_match
from causalityrag.replacement import GenericReplacementClient, generate_valid_replacement
from causalityrag.revision import apply_token_replacements
from causalityrag.rules import TypedRuleLibrary
def load_first_rows(paths: list[str], limit: int) -> list[dict]:
    """Load an ordered prefix from one or more JSONL shards."""

    rows = []
    for path in paths:
        for row in iter_records(path):
            rows.append(row)
            if len(rows) == limit:
                return rows
    return rows


def parse_budgets(value: str) -> list[int]:
    budgets = sorted({int(item) for item in value.split(",") if item.strip()})
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must be positive")
    return budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--jsd-results", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cf-pools", required=True)
    parser.add_argument("--type-rules", default="")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--budgets", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--use-stored-clean",
        action="store_true",
        help="Reuse the cached clean answer instead of issuing another reader call.",
    )
    parser.add_argument(
        "--spacy-base-url",
        default=os.environ.get("CAUSALITYRAG_SPACY_BASE_URL", "http://127.0.0.1:8021"),
    )
    args = parser.parse_args()
    budgets = parse_budgets(args.budgets)
    if args.n <= 0 or args.workers <= 0:
        raise ValueError("n and workers must be positive")

    records = load_first_rows([args.input], args.n)
    jsd_rows = load_first_rows(args.jsd_results, args.n)
    if len(records) != len(jsd_rows):
        raise ValueError("input and JSD result counts differ")
    for index, (record, jsd_row) in enumerate(zip(records, jsd_rows), 1):
        if record_id(record) != str(jsd_row.get("id")):
            raise ValueError(f"row ID mismatch at {index}")

    nlp = SpacyAnnotationClient(args.spacy_base_url)
    if not nlp.health().get("ok"):
        raise RuntimeError("spaCy annotation service is unhealthy")
    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None)
    generic_editor = GenericReplacementClient()
    reader = ReaderClient()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    started = time.monotonic()
    completed = 0
    with open(args.out, "w", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_query,
                    record,
                    jsd_row,
                    budgets,
                    args.k,
                    nlp,
                    library,
                    generic_editor,
                    reader,
                    args.use_stored_clean,
                ): index
                for index, (record, jsd_row) in enumerate(zip(records, jsd_rows))
            }
            for future in as_completed(futures):
                result = future.result()
                result["source_row"] = futures[future]
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                completed += 1
                if completed <= 10 or completed % 25 == 0 or completed == len(records):
                    flips = sum(
                        value["answer_changed"]
                        for value in result.get("prefixes", {}).values()
                    )
                    print(
                        f"[cached-unary-prefix] {completed}/{len(records)} "
                        f"id={result['id']} selected={result['n_selected']} "
                        f"flipping_prefixes={flips} seconds={result['elapsed_seconds']}",
                        flush=True,
                    )
    print(
        f"[cached-unary-prefix] complete rows={completed} "
        f"wall_seconds={round(time.monotonic() - started, 3)}",
        flush=True,
    )


def run_query(
    record,
    jsd_row,
    budgets,
    k,
    nlp,
    library,
    generic_editor,
    reader,
    use_stored_clean,
):
    started = time.monotonic()
    question = str(record.get("question", ""))
    contexts = retrieved_contexts(record)[:k]
    units, _sentences = context_sentence_units(record, k=k, nlp=nlp)
    score_map = (
        jsd_row.get("unary_scores")
        or jsd_row.get("arc_jsd_unary_token_scores")
        or {}
    )
    scores = {
        str(unit["unit_id"]): max(
            0.0,
            float(score_map.get(str(unit["unit_id"]), 0.0)),
        )
        for unit in units
    }
    replacement_cache = cached_replacements(jsd_row)
    selected, replacements, rejected = select_ranked_valid_tokens(
        units,
        scores,
        max(budgets),
        contexts,
        library,
        generic_editor,
        nlp,
        replacement_cache,
    )
    stored_clean = str(jsd_row.get("clean_answer", "")).strip()
    clean_answer = (
        stored_clean
        if use_stored_clean and stored_clean
        else answer_with_retry(reader, question, contexts)
    )
    prefixes = {}
    for budget in budgets:
        chosen = selected[:budget]
        chosen_replacements = {
            str(unit["unit_id"]): replacements[str(unit["unit_id"])]
            for unit in chosen
        }
        revision = apply_token_replacements(
            record, chosen, chosen_replacements, k=k
        )
        edited_answer = answer_with_retry(
            reader, question, revision["edited_contexts"]
        )
        prefixes[str(budget)] = {
            "budget": budget,
            "n_selected": len(chosen),
            "selected_ids": [str(unit["unit_id"]) for unit in chosen],
            "selected_scores": [scores[str(unit["unit_id"])] for unit in chosen],
            "edits": revision["edits"],
            "edited_answer": edited_answer,
            "answer_changed": not answers_match(clean_answer, edited_answer),
        }
    return {
        "id": record_id(record),
        "question": question,
        "gold_answer": str(record.get("answer", "")),
        "stored_clean_answer": str(jsd_row.get("clean_answer", "")),
        "clean_answer": clean_answer,
        "clean_source": "stored" if use_stored_clean and stored_clean else "reader",
        "clean_matches_stored": answers_match(clean_answer, str(jsd_row.get("clean_answer", ""))),
        "clean_correct": answers_match(clean_answer, str(record.get("answer", ""))),
        "method": "cached_arc_jsd_unary_prefix",
        "budgets": budgets,
        "n_context_tokens": len(units),
        "n_selected": len(selected),
        "rejected_uneditable_units": rejected,
        "prefixes": prefixes,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def cached_replacements(jsd_row: dict) -> dict[str, dict]:
    cache = {}
    for result in jsd_row.get("arc_jsd_unary_topk", {}).values():
        for edit in result.get("edits", []):
            if edit.get("ok") and edit.get("new") and edit.get("unit_id"):
                cache[str(edit["unit_id"])] = dict(edit)
    return cache


def select_ranked_valid_tokens(
    units: list[dict],
    token_scores: dict[str, float],
    budget: int,
    contexts: list[dict],
    library: TypedRuleLibrary,
    generic_editor: GenericReplacementClient,
    nlp,
    replacement_cache: dict[str, dict],
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    """Take the highest-scoring words with one shared replacement per token."""

    context_by_id = {
        str(context["chunk_id"]): str(context["text"])
        for context in contexts
    }
    ranked = sorted(
        (
            unit
            for unit in units
            if float(token_scores.get(str(unit["unit_id"]), 0.0)) > 0
        ),
        key=lambda unit: (
            -float(token_scores[str(unit["unit_id"])])
            / max(float(unit.get("cost", 1.0)), 1e-12),
            str(unit["unit_id"]),
        ),
    )
    selected = []
    replacements = {}
    rejected = []
    for unit in ranked:
        unit_id = str(unit["unit_id"])
        replacement = replacement_cache.get(unit_id)
        if replacement is None:
            replacement = generate_valid_replacement(
                unit,
                context_by_id[str(unit["chunk_id"])],
                library,
                generic_editor,
                nlp,
            )
            replacement_cache[unit_id] = replacement
        if not replacement.get("ok"):
            rejected.append({
                **unit,
                "replacement_failure": replacement,
            })
            continue
        selected.append(unit)
        replacements[unit_id] = replacement
        if len(selected) == budget:
            break
    return selected, replacements, rejected


def answer_with_retry(reader, question, contexts, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            return reader.answer(question, contexts)
        except Exception as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"reader failed after {retries} attempts: {last_error}")


if __name__ == "__main__":
    main()
