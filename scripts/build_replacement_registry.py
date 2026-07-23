"""Build a shared, answer-blind registry of strict token replacements."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.replacement import (
    GenericReplacementClient,
    build_selected_replacements,
)
from causalityrag.rules import TypedRuleLibrary
from causalityrag.token_units import (
    context_sentence_units,
    units_from_context_row,
)


_LOCAL_NLP = None
_LOCAL_LIBRARY = None
_LOCAL_EDITOR = None


def _initialize_local_registry(
    spacy_model: str,
    cf_pools: str,
    type_rules: str,
) -> None:
    import spacy

    global _LOCAL_NLP, _LOCAL_LIBRARY, _LOCAL_EDITOR
    _LOCAL_NLP = spacy.load(spacy_model)
    _LOCAL_LIBRARY = TypedRuleLibrary.from_files(cf_pools, type_rules or None)
    _LOCAL_EDITOR = GenericReplacementClient()


def _process_local_registry(task: dict) -> dict:
    if _LOCAL_NLP is None or _LOCAL_LIBRARY is None or _LOCAL_EDITOR is None:
        raise RuntimeError("local replacement worker was not initialized")
    identifier = record_id(task["record"])
    before = _LOCAL_EDITOR.calls
    row = build_registry_row(
        offset=task["offset"],
        start=task["start"],
        record=task["record"],
        gates=[{identifier: gate_row} for gate_row in task["gate_rows"]],
        existing_by_id=(
            {identifier: task["existing"]} if task["existing"] else {}
        ),
        units_by_id=(
            {identifier: task["context_row"]}
            if task["context_row"]
            else {}
        ),
        k=task["k"],
        max_budget=task["max_budget"],
        max_native_tokens=task["max_native_tokens"],
        nlp=_LOCAL_NLP,
        library=_LOCAL_LIBRARY,
        editor=_LOCAL_EDITOR,
    )
    row["editor_llm_calls"] = _LOCAL_EDITOR.calls - before
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gates", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--existing-registry", default="")
    parser.add_argument(
        "--context-units",
        "--units-cache",
        dest="context_units",
        default="",
    )
    parser.add_argument("--cf-pools", required=True)
    parser.add_argument("--type-rules", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-budget", type=int, default=5)
    parser.add_argument("--max-native-tokens", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=("service", "local-process"),
        default="service",
    )
    parser.add_argument(
        "--spacy-model",
        default=os.environ.get("YVETTE_SPACY_MODEL", "en_core_web_lg"),
    )
    parser.add_argument(
        "--spacy-base-url",
        default=os.environ.get(
            "CAUSALITYRAG_SPACY_BASE_URL",
            "http://127.0.0.1:8021",
        ),
    )
    args = parser.parse_args()
    if args.max_budget <= 0:
        raise ValueError("max-budget must be positive")
    if args.max_native_tokens < 0:
        raise ValueError("max-native-tokens must be non-negative")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    records = load_records(args.input)[args.start:args.start + args.n]
    gates = [
        {str(row.get("id")): row for row in load_records(path)}
        for path in args.gates
    ]
    existing_by_id = (
        {
            str(row.get("id")): row
            for row in load_records(args.existing_registry)
        }
        if args.existing_registry
        else {}
    )
    units_by_id = (
        {
            str(row.get("id")): row
            for row in load_records(args.context_units)
        }
        if args.context_units
        else {}
    )
    if args.backend == "service":
        nlp = SpacyAnnotationClient(args.spacy_base_url)
        if not nlp.health().get("ok"):
            raise RuntimeError("spaCy annotation service is unhealthy")
        library = TypedRuleLibrary.from_files(
            args.cf_pools,
            args.type_rules or None,
        )
        editor = GenericReplacementClient()

        def process(item: tuple[int, dict]) -> dict:
            offset, record = item
            return build_registry_row(
                offset=offset,
                start=args.start,
                record=record,
                gates=gates,
                existing_by_id=existing_by_id,
                units_by_id=units_by_id,
                k=args.k,
                max_budget=args.max_budget,
                max_native_tokens=args.max_native_tokens,
                nlp=nlp,
                library=library,
                editor=editor,
            )

        executor = ThreadPoolExecutor(max_workers=args.workers)
        generated = executor.map(process, enumerate(records, 1))
    else:
        editor = None
        tasks = []
        for offset, record in enumerate(records, 1):
            identifier = record_id(record)
            gate_rows = []
            for gate_by_id in gates:
                gate_row = gate_by_id.get(identifier)
                if gate_row is None:
                    raise ValueError(f"missing gate row for {identifier}")
                gate_rows.append(gate_row)
            tasks.append({
                "offset": offset,
                "start": args.start,
                "record": record,
                "gate_rows": gate_rows,
                "existing": existing_by_id.get(identifier),
                "context_row": units_by_id.get(identifier),
                "k": args.k,
                "max_budget": args.max_budget,
                "max_native_tokens": args.max_native_tokens,
            })
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_local_registry,
            initargs=(args.spacy_model, args.cf_pools, args.type_rules),
        )
        generated = executor.map(_process_local_registry, tasks, chunksize=1)

    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        with executor:
            for completed, row in enumerate(generated, 1):
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"[replacement-registry] {completed}/{len(records)} "
                    f"candidates={row['candidate_tokens']} "
                    f"valid={row['valid_tokens']} invalid={row['invalid_tokens']}",
                    flush=True,
                )

    summary = {
        "queries": len(rows),
        "candidate_tokens": sum(row["candidate_tokens"] for row in rows),
        "valid_tokens": sum(row["valid_tokens"] for row in rows),
        "invalid_tokens": sum(row["invalid_tokens"] for row in rows),
        "editor_llm_calls": (
            editor.calls
            if editor is not None
            else sum(int(row.get("editor_llm_calls", 0)) for row in rows)
        ),
        "workers": args.workers,
        "backend": args.backend,
        "max_budget": args.max_budget,
        "max_native_tokens": args.max_native_tokens,
        "contract": "strict_contextual_pos_tag_morphology",
        "answer_blind": True,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[replacement-registry summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def build_registry_row(
    *,
    offset: int,
    start: int,
    record: dict,
    gates: list[dict[str, dict]],
    existing_by_id: dict[str, dict],
    units_by_id: dict[str, dict],
    k: int,
    max_budget: int,
    max_native_tokens: int,
    nlp,
    library,
    editor,
) -> dict:
    identifier = record_id(record)
    context_row = units_by_id.get(identifier)
    if context_row is not None:
        units = units_from_context_row(record, context_row, k=k)
    else:
        units, _ = context_sentence_units(record, k=k, nlp=nlp)
    by_id = {str(unit["unit_id"]): unit for unit in units}
    existing = existing_by_id.get(identifier, {})
    candidate_ids: set[str] = {
        str(unit_id) for unit_id in existing.get("candidate_ids", [])
    }
    for gate_by_id in gates:
        gate = gate_by_id.get(identifier)
        if gate is None:
            raise ValueError(f"missing gate row for {identifier}")
        candidate_ids.update(registry_candidate_ids(
            gate,
            max_budget=max_budget,
            max_native_tokens=max_native_tokens,
        ))

    contexts = retrieved_contexts(record)[:k]
    cache: dict[str, dict] = dict(existing.get("replacements", {}))
    invalid = dict(existing.get("invalid", {}))
    for unit_id in sorted(candidate_ids):
        if unit_id in cache or unit_id in invalid:
            continue
        unit = by_id.get(unit_id)
        if unit is None:
            invalid[unit_id] = {"reason": "unit_not_found"}
            continue
        replacements, rejected = build_selected_replacements(
            [unit],
            contexts,
            library,
            editor,
            nlp,
            cache,
            allow_relaxed_fallback=False,
        )
        if rejected:
            invalid[unit_id] = rejected[0]["replacement_failure"]
        elif unit_id not in replacements:
            invalid[unit_id] = {"reason": "replacement_not_returned"}

    valid = {
        unit_id: replacement
        for unit_id, replacement in cache.items()
        if replacement.get("ok")
    }
    return {
        "index": start + offset - 1,
        "id": identifier,
        "candidate_ids": sorted(candidate_ids),
        "replacements": valid,
        "invalid": invalid,
        "candidate_tokens": len(candidate_ids),
        "valid_tokens": len(valid),
        "invalid_tokens": len(candidate_ids) - len(valid),
        "contract": "strict_contextual_pos_tag_morphology",
        "answer_blind": True,
    }


def registry_candidate_ids(
    gate: dict,
    *,
    max_budget: int,
    max_native_tokens: int,
) -> set[str]:
    """Collect proposed and matched-baseline IDs under evaluation limits."""

    candidate_ids = {
        str(unit_id) for unit_id in gate.get("unary_order", [])[:max_budget]
    }

    def add_pair(candidate: dict) -> None:
        candidate_ids.update(
            str(unit_id) for unit_id in candidate.get("selected_ids", [])
        )
        candidate_ids.update(
            str(unit_id)
            for unit_id in candidate.get("unary_matched_ids", [])
        )

    for key in ("strict_candidate", "bicriteria_candidate"):
        candidate = gate.get(key)
        limit = (
            max(max_budget, max_native_tokens)
            if key == "bicriteria_candidate"
            else max_budget
        )
        if candidate and int(candidate.get("n_selected", 0)) <= limit:
            add_pair(candidate)
    for candidate in gate.get("candidates", []):
        if int(candidate.get("n_selected", 0)) <= max_budget:
            add_pair(candidate)
    for candidate in gate.get("budget_candidates", []):
        if int(candidate.get("budget", 0)) <= max_budget:
            add_pair(candidate)
    return candidate_ids


if __name__ == "__main__":
    main()
