"""Fill missing typed keys in the method-independent replacement pool."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.counterfactual_pool_generation import (
    POLICY,
    SentenceCounterfactualClient,
)
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.semantic_filter import (
    ensure_wordnet_available,
    is_lexical_paraphrase,
)
from causalityrag.shared_replacement_pool import POOL_SCHEMA, file_sha256


_GENERIC_INVALID = {
    "different",
    "other",
    "replacement",
    "same",
    "unknown",
}
_LLM_GRAMMAR_OVERRIDE_REASONS = {
    "pos_mismatch",
    "tag_mismatch",
    "tokenization_mismatch",
}
_NUMERIC_SURFACE = re.compile(
    r"^[+-]?\d+(?:\.\d+)?(?:[-–—][+-]?\d+(?:\.\d+)?)?$"
)


def _numeric_shape(value: str) -> tuple[int, tuple[int, ...]] | None:
    if not _NUMERIC_SURFACE.fullmatch(value):
        return None
    parts = re.split(r"[-–—]", value.lstrip("+-"))
    decimal_places = tuple(
        len(part.rsplit(".", 1)[1]) if "." in part else 0
        for part in parts
    )
    return len(parts), decimal_places


def _is_format_preserving_numeric_counterfactual(
    row: dict,
    candidate: str,
) -> bool:
    original = str(row["surface"])
    if str(row.get("type", "")) != "CARDINAL":
        return False
    return (
        original != candidate
        and _numeric_shape(original) is not None
        and _numeric_shape(original) == _numeric_shape(candidate)
    )


def _allow_contextual_single_letter_override(
    row: dict,
    candidate: str,
    reason: str,
) -> bool:
    return (
        reason == "lexical_paraphrase"
        and str(row.get("type", "")) == "PROPER"
        and len(str(row["surface"])) == 1
        and len(candidate) == 1
        and str(row["surface"]).isalpha()
        and candidate.isalpha()
    )


def _target(row: dict, forbidden: list[str]) -> dict:
    examples = row.get("examples", [])
    example = examples[0] if examples else {}
    sentence = (
        str(example.get("marked_sentence", ""))
        if examples
        else f"[[{row['surface']}]]"
    )
    return {
        "unit_id": str(row["typed_key"]),
        "token": str(row["surface"]),
        "unit_type": str(row.get("type", "")),
        "pos": str(row.get("pos", "")),
        "tag": str(row.get("tag", "")),
        "entity_text": str(example.get("entity_text", "")),
        "entity_token_position": (
            int(example["entity_token_index"]) + 1
            if example.get("entity_token_index") is not None
            else None
        ),
        "sentence": sentence,
        "forbidden": tuple(forbidden),
    }


def _contextual_unit(row: dict) -> tuple[dict, str]:
    examples = row.get("examples", [])
    example = examples[0] if examples else {}
    sentence = str(example.get("sentence", ""))
    explicit_start = example.get("sentence_char_start")
    explicit_end = example.get("sentence_char_end")
    if (
        sentence
        and explicit_start is not None
        and explicit_end is not None
    ):
        context = sentence
        start = int(explicit_start)
        end = int(explicit_end)
    else:
        marked = str(example.get("marked_sentence", ""))
        left = marked.find("[[")
        right = marked.find("]]", left + 2)
        if left < 0 or right < 0:
            context = str(row["surface"])
            start = 0
            end = len(context)
        else:
            context = marked[:left] + marked[left + 2:right] + marked[right + 2:]
            start = left
            end = start + len(str(row["surface"]))
    if context[start:end] != str(row["surface"]):
        context = str(row["surface"])
        start = 0
        end = len(context)
    unit = {
        "text": str(row["surface"]),
        "chunk_char_start": start,
        "chunk_char_end": end,
        "pos": str(row.get("pos", "")),
        "tag": str(row.get("tag", "")),
        "morph": row.get("morph", ""),
        "type": str(row.get("type", "")),
    }
    return unit, context


def _filter_generated(
    row: dict,
    values: list[str],
    nlp: SpacyAnnotationClient,
) -> tuple[list[dict], list[dict]]:
    original = str(row["surface"])
    seen = {original.casefold()}
    proposals = []
    filtered = []
    pre_rejected = []
    unit, context = _contextual_unit(row)
    for value in values:
        candidate = str(value).strip()
        folded = candidate.casefold()
        if not candidate or folded in seen:
            continue
        seen.add(folded)
        reason = ""
        if folded in _GENERIC_INVALID:
            reason = "generic_placeholder"
        elif any(character.isspace() for character in candidate):
            reason = "multi_token_surface"
        elif is_lexical_paraphrase(
                original,
                candidate,
                str(row.get("pos", "")),
                str(row.get("type", "")),
        ):
            reason = "lexical_paraphrase"
        if reason:
            pre_rejected.append({
                "ok": False,
                "old": original,
                "new": candidate,
                "policy": POLICY,
                "validation": {"valid": False, "reason": reason},
            })
            continue
        proposals.append({
            "unit": unit,
            "context": context,
            "replacement": {
                "ok": True,
                "old": original,
                "new": candidate,
                "policy": POLICY,
            },
        })
        filtered.append(candidate)
    validations = nlp.validate_many(proposals) if proposals else []
    accepted = [
        {
            "ok": True,
            "old": original,
            "new": candidate,
            "policy": POLICY,
            "validation": validation,
        }
        for candidate, validation in zip(filtered, validations)
        if validation.get("valid") is True
    ]
    rejected = pre_rejected + [
        {
            "ok": False,
            "old": original,
            "new": candidate,
            "policy": POLICY,
            "validation": validation,
        }
        for candidate, validation in zip(filtered, validations)
        if validation.get("valid") is not True
    ]
    return accepted, rejected


def _generate_batch(
    rows: list[dict],
    editor: SentenceCounterfactualClient,
    nlp: SpacyAnnotationClient,
    *,
    max_candidates: int,
    generation_rounds: int,
    initial_forbidden: dict[str, list[str]] | None = None,
    attempt_offset: int = 0,
) -> tuple[list[dict], list[dict], int]:
    pending = {str(row["typed_key"]): row for row in rows}
    accumulated: dict[str, list[dict]] = {
        str(row["typed_key"]): [] for row in rows
    }
    forbidden: dict[str, list[str]] = {
        str(row["typed_key"]): list(
            (initial_forbidden or {}).get(str(row["typed_key"]), [])
        )
        for row in rows
    }
    calls = 0

    for attempt in range(generation_rounds):
        if not pending:
            break
        targets = [
            _target(row, forbidden[str(row["typed_key"])])
            for row in pending.values()
        ]
        generated = editor.generate_many(
            targets,
            max_candidates=max_candidates,
            attempt=attempt_offset + attempt,
        )
        calls += 1
        filtered = {
            key: _filter_generated(row, generated.get(key, []), nlp)
            for key, row in pending.items()
        }
        semantic_candidates = {
            key: [str(item["new"]) for item in accepted]
            for key, (accepted, _) in filtered.items()
            if accepted
        }
        semantic_judgments = (
            editor.classify_relations_many(targets, semantic_candidates)
            if semantic_candidates
            else {}
        )
        if semantic_candidates:
            calls += 1
        for key, (accepted, rejected) in filtered.items():
            semantically_valid = []
            for item in accepted:
                if _is_format_preserving_numeric_counterfactual(
                    pending[key],
                    str(item["new"]),
                ):
                    item["semantic_relation"] = {
                        "label": "COUNTERFACTUAL",
                        "classifier_value": "format_preserving_numeric_change",
                    }
                    semantically_valid.append(item)
                    continue
                judgment = semantic_judgments.get(
                    (key, str(item["new"]).casefold()),
                    {"label": "INVALID"},
                )
                if judgment.get("label") == "COUNTERFACTUAL":
                    item["semantic_relation"] = judgment
                    semantically_valid.append(item)
                else:
                    rejected.append({
                        **item,
                        "ok": False,
                        "semantic_relation": judgment,
                        "validation": {
                            "valid": False,
                            "reason": (
                                "llm_relation_"
                                + str(judgment.get("label", "INVALID")).lower()
                            ),
                        },
                    })
            filtered[key] = (semantically_valid, rejected)

        override_candidates = {}
        for key, (accepted, rejected) in filtered.items():
            if accepted:
                continue
            candidates = []
            for item in rejected:
                reason = str(item.get("validation", {}).get("reason", ""))
                if (
                    reason in _LLM_GRAMMAR_OVERRIDE_REASONS
                    or reason.startswith("morph_mismatch:")
                    or _allow_contextual_single_letter_override(
                        pending[key],
                        str(item["new"]),
                        reason,
                    )
                ):
                    candidates.append(str(item["new"]))
            if candidates:
                override_candidates[key] = candidates
        override_relations = (
            editor.classify_relations_many(targets, override_candidates)
            if override_candidates
            else {}
        )
        if override_candidates:
            calls += 1

        next_pending = {}
        for key, row in pending.items():
            current = accumulated[key]
            seen = {str(item["new"]).casefold() for item in current}
            accepted, rejected = filtered[key]
            for item in rejected:
                candidate = str(item.get("new", ""))
                relation = override_relations.get(
                    (key, candidate.casefold()),
                    {"label": "INVALID"},
                )
                if relation.get("label") == "COUNTERFACTUAL":
                    override = dict(item)
                    override["ok"] = True
                    override["semantic_relation"] = relation
                    override["llm_grammar_override"] = True
                    accepted.append(override)
                else:
                    forbidden[key].append(candidate)
            for candidate in accepted:
                folded = str(candidate["new"]).casefold()
                if folded in seen:
                    continue
                current.append(candidate)
                seen.add(folded)
            if not current:
                next_pending[key] = row
        pending = next_pending

    resolved = []
    unresolved = []
    by_key = {str(row["typed_key"]): row for row in rows}
    for key, row in by_key.items():
        candidates = accumulated[key][:max_candidates]
        if candidates:
            resolved.append({
                "schema": POOL_SCHEMA,
                "row_kind": "typed_candidates",
                "typed_key": key,
                "surface": row["surface"],
                "type": row.get("type", ""),
                "pos": row.get("pos", ""),
                "tag": row.get("tag", ""),
                "morph": row.get("morph", ""),
                "candidates": candidates,
                "source": POLICY,
            })
        else:
            unresolved.append({
                "typed_key": key,
                "surface": row["surface"],
                "type": row.get("type", ""),
                "pos": row.get("pos", ""),
                "tag": row.get("tag", ""),
                "attempts": generation_rounds,
                "forbidden": forbidden[key],
            })
    return resolved, unresolved, calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--typed-keys", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--unresolved-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--generation-rounds", type=int, default=8)
    parser.add_argument("--attempt-offset", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Generate at most this many missing typed keys; 0 means all.",
    )
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--spacy-base-url", default="")
    args = parser.parse_args()

    typed_rows = [
        json.loads(line)
        for line in open(args.typed_keys, encoding="utf-8")
        if line.strip()
    ]
    rows_by_key = {str(row["typed_key"]): row for row in typed_rows}
    resolved = {}
    for path in (args.seed, args.out):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("typed_key", ""))
                if key in rows_by_key and row.get("candidates"):
                    resolved[key] = row

    previous_forbidden = {}
    if os.path.exists(args.unresolved_out):
        with open(args.unresolved_out, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("typed_key", ""))
                if key in rows_by_key:
                    previous_forbidden[key] = list(dict.fromkeys(
                        str(value)
                        for value in row.get("forbidden", [])
                        if str(value).strip()
                    ))

    missing = [
        rows_by_key[key]
        for key in sorted(rows_by_key)
        if key not in resolved
    ]
    missing_before_limit = len(missing)
    if args.limit > 0:
        missing = missing[: args.limit]
    batches = [
        missing[start : start + args.batch_size]
        for start in range(0, len(missing), args.batch_size)
    ]
    editor = SentenceCounterfactualClient(
        base_url=args.llm_base_url or None,
        model=args.llm_model or None,
    )
    nlp = SpacyAnnotationClient(args.spacy_base_url or None)
    if not nlp.health().get("ok"):
        raise RuntimeError("spaCy service is unhealthy")
    ensure_wordnet_available()
    is_lexical_paraphrase("film", "movie", "NOUN", "CONTENT")
    write_lock = Lock()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    unresolved_rows = []
    calls = 0

    def run(batch: list[dict]):
        return _generate_batch(
            batch,
            editor,
            nlp,
            max_candidates=args.max_candidates,
            generation_rounds=args.generation_rounds,
            initial_forbidden=previous_forbidden,
            attempt_offset=args.attempt_offset,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, batch) for batch in batches]
        for completed, future in enumerate(as_completed(futures), start=1):
            generated_rows, failed_rows, batch_calls = future.result()
            calls += batch_calls
            with write_lock:
                with open(args.out, "a", encoding="utf-8") as output:
                    for row in generated_rows:
                        output.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
                        resolved[str(row["typed_key"])] = row
                    output.flush()
                unresolved_rows.extend(failed_rows)
            if completed % 10 == 0 or completed == len(batches):
                print(
                    f"[{completed}/{len(batches)} batches] "
                    f"covered={len(resolved)}/{len(rows_by_key)} "
                    f"calls={calls}",
                    flush=True,
                )

    temporary = args.out + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        for key in sorted(resolved):
            output.write(json.dumps(resolved[key], ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    with open(args.unresolved_out, "w", encoding="utf-8") as output:
        for row in sorted(unresolved_rows, key=lambda item: item["typed_key"]):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema": POOL_SCHEMA,
        "typed_keys_total": len(rows_by_key),
        "typed_keys_covered": len(resolved),
        "typed_key_coverage": len(resolved) / max(1, len(rows_by_key)),
        "unresolved": len(rows_by_key) - len(resolved),
        "missing_before_limit": missing_before_limit,
        "attempted_this_run": len(missing),
        "generator_calls": calls,
        "output": os.path.abspath(args.out),
        "sha256": file_sha256(args.out),
        "unresolved_output": os.path.abspath(args.unresolved_out),
    }
    with open(args.manifest_out, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
