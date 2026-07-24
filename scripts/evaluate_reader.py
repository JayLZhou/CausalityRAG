"""Evaluate saved token selections with matched token replacements."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.reader import (
    ReaderClient,
    answer_token_f1,
    answers_exact_match,
    answers_match,
)
from causalityrag.replacement import (
    GenericReplacementClient,
    build_selected_replacements,
)
from causalityrag.revision import apply_token_replacements
from causalityrag.rules import TypedRuleLibrary
from causalityrag.token_units import (
    context_sentence_units,
    units_from_context_row,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--clean-reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--cf-pools", required=True)
    parser.add_argument("--type-rules", default="")
    parser.add_argument("--replacement-registry", default="")
    parser.add_argument(
        "--context-units",
        "--units-cache",
        dest="context_units",
        default="",
    )
    parser.add_argument("--remaining-flow-threshold", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reader-workers", type=int, default=16)
    parser.add_argument("--reader-base-url", default="")
    parser.add_argument("--reader-model", default="")
    parser.add_argument(
        "--method-name",
        default="residual_flow",
        help="method key written under each row's methods object",
    )
    parser.add_argument("--include-clean-incorrect", action="store_true")
    parser.add_argument(
        "--fallback-to-minimum-flow-candidate",
        action="store_true",
        help=(
            "when no candidate meets the remaining-flow threshold, evaluate "
            "the nonempty contribution-flow candidate with minimum residual "
            "flow and mark it as an above-threshold fallback"
        ),
    )
    parser.add_argument(
        "--ignore-remaining-flow-threshold",
        action="store_true",
        help=(
            "evaluate the saved nonempty candidate without applying the "
            "contribution-flow threshold; intended for external baselines"
        ),
    )
    parser.add_argument(
        "--clean-correct-policy",
        choices=("exact", "lenient", "stored"),
        default="exact",
    )
    parser.add_argument("--strict-replacements", action="store_true")
    parser.add_argument(
        "--spacy-base-url",
        default=os.environ.get(
            "CAUSALITYRAG_SPACY_BASE_URL",
            "http://127.0.0.1:8021",
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.remaining_flow_threshold < 1:
        raise ValueError("remaining flow threshold must be in [0, 1)")
    if args.reader_workers <= 0:
        raise ValueError("reader workers must be positive")
    if (
        args.fallback_to_minimum_flow_candidate
        and args.ignore_remaining_flow_threshold
    ):
        raise ValueError(
            "flow fallback and threshold-free baseline evaluation "
            "are mutually exclusive"
        )

    records_by_id = {record_id(record): record for record in load_records(args.input)}
    reference_by_id = {
        str(row.get("id")): row for row in load_records(args.clean_reference)
    }
    gate_rows = load_records(args.gate)
    registry_by_id = (
        {str(row.get("id")): row for row in load_records(args.replacement_registry)}
        if args.replacement_registry
        else {}
    )
    units_by_id = (
        {str(row.get("id")): row for row in load_records(args.context_units)}
        if args.context_units
        else {}
    )
    nlp = SpacyAnnotationClient(args.spacy_base_url)
    if not nlp.health().get("ok"):
        raise RuntimeError("spaCy annotation service is unhealthy")
    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None)
    editor = GenericReplacementClient()
    reader = ReaderClient(
        base_url=args.reader_base_url or None,
        model=args.reader_model or None,
    )

    rows = []
    pending_requests: list[tuple[str, list[dict]]] = []
    for gate_row in gate_rows:
        identifier = str(gate_row.get("id", ""))
        reference = reference_by_id.get(identifier)
        if reference is None:
            raise ValueError(f"missing clean reference for {identifier}")
        record = records_by_id.get(identifier)
        if record is None:
            raise ValueError(f"missing input record for {identifier}")
        clean_answer = str(reference.get("clean_answer", ""))
        gold_answer = str(record.get("answer", ""))
        clean_correct_exact = answers_exact_match(clean_answer, gold_answer)
        clean_correct_lenient = answers_match(clean_answer, gold_answer)
        clean_correct_stored = bool(reference.get("clean_correct"))
        clean_correct = {
            "exact": clean_correct_exact,
            "lenient": clean_correct_lenient,
            "stored": clean_correct_stored,
        }[args.clean_correct_policy]
        if not args.include_clean_incorrect and not clean_correct:
            continue
        candidate, candidate_selection = evaluation_candidate(
            gate_row.get("candidates", []),
            args.remaining_flow_threshold,
            fallback_to_minimum_flow=args.fallback_to_minimum_flow_candidate,
            ignore_remaining_flow_threshold=(
                args.ignore_remaining_flow_threshold
            ),
        )
        if candidate is None:
            no_candidate = {
                "status": "no_candidate_under_selection_rule",
                "selected_ids": [],
                "selected_tokens": [],
                "n_selected": 0,
                "flip": False,
            }
            row = {
                "index": gate_row.get("index"),
                "id": identifier,
                "question": gate_row.get("question"),
                "gold_answer": gold_answer,
                "clean_answer": clean_answer,
                "clean_correct": clean_correct,
                "clean_correct_policy": args.clean_correct_policy,
                "clean_correct_exact": clean_correct_exact,
                "clean_correct_lenient": clean_correct_lenient,
                "clean_correct_stored": clean_correct_stored,
                "include_clean_incorrect": args.include_clean_incorrect,
                "selection_rule": (
                    "fixed_candidate_ignore_flow_threshold"
                    if args.ignore_remaining_flow_threshold
                    else (
                        "threshold_then_minimum_flow_fallback"
                        if args.fallback_to_minimum_flow_candidate
                        else "threshold_only"
                    )
                ),
                "candidate_selection": candidate_selection,
                "candidate_meets_remaining_flow_threshold": None,
                "remaining_flow_threshold": args.remaining_flow_threshold,
                "candidate_remaining_support_fraction": None,
                "replacement_contract": (
                    "strict_contextual_pos_morphology"
                    if args.strict_replacements
                    else "surface_valid_non_deleting_word"
                ),
                "reader_backend": "vllm_openai_compatible",
                "reader_calls": 0,
                "evaluated_method": args.method_name,
                "methods": {args.method_name: dict(no_candidate)},
            }
            rows.append(row)
            continue
        context_row = units_by_id.get(identifier)
        if context_row is not None:
            units = units_from_context_row(record, context_row, k=args.k)
        else:
            units, _ = context_sentence_units(record, k=args.k, nlp=nlp)
        by_id = {str(unit["unit_id"]): unit for unit in units}
        contexts = retrieved_contexts(record)[: args.k]
        method_rows = {}
        reader_job_by_signature: dict[
            tuple[tuple[str, str], ...],
            int,
        ] = {}
        replacement_cache = dict(
            registry_by_id.get(identifier, {}).get("replacements", {})
        )
        for method, selected_ids in (
            (args.method_name, candidate["selected_ids"]),
        ):
            selected = [by_id[unit_id] for unit_id in selected_ids]
            if registry_by_id:
                missing = [
                    unit_id
                    for unit_id in selected_ids
                    if unit_id not in replacement_cache
                ]
                if missing:
                    method_rows[method] = {
                        "status": "replacement_registry_missing",
                        "selected_ids": selected_ids,
                        "selected_tokens": [
                            str(unit.get("text", "")) for unit in selected
                        ],
                        "n_selected": len(selected),
                        "missing_registry_ids": missing,
                    }
                    continue
            replacements, rejected = build_selected_replacements(
                selected,
                contexts,
                library,
                editor,
                nlp,
                replacement_cache,
                allow_relaxed_fallback=(
                    not args.strict_replacements and not registry_by_id
                ),
            )
            result = {
                "selected_ids": selected_ids,
                "selected_tokens": [str(unit.get("text", "")) for unit in selected],
                "n_selected": len(selected),
                "rejected": rejected,
            }
            if rejected:
                result["status"] = "strict_replacement_failed"
                method_rows[method] = result
                continue
            revision = apply_token_replacements(
                record,
                selected,
                replacements,
                k=args.k,
            )
            signature = tuple(
                sorted(
                    (str(edit["unit_id"]), str(edit["new"]))
                    for edit in revision["edits"]
                    if edit.get("ok")
                )
            )
            cache_hit = signature in reader_job_by_signature
            if not cache_hit:
                reader_job_by_signature[signature] = len(pending_requests)
                pending_requests.append(
                    (
                        str(record.get("question", "")),
                        revision["edited_contexts"],
                    )
                )
            result.update(
                {
                    "status": "pending_reader",
                    "edits": revision["edits"],
                    "_reader_job": reader_job_by_signature[signature],
                    "reader_cache_hit": cache_hit,
                }
            )
            method_rows[method] = result

        row = {
            "index": gate_row.get("index"),
            "id": identifier,
            "question": gate_row.get("question"),
            "gold_answer": gold_answer,
            "clean_answer": clean_answer,
            "clean_correct": clean_correct,
            "clean_correct_policy": args.clean_correct_policy,
            "clean_correct_exact": clean_correct_exact,
            "clean_correct_lenient": clean_correct_lenient,
            "clean_correct_stored": clean_correct_stored,
            "include_clean_incorrect": args.include_clean_incorrect,
            "selection_rule": (
                "fixed_candidate_ignore_flow_threshold"
                if args.ignore_remaining_flow_threshold
                else (
                    "threshold_then_minimum_flow_fallback"
                    if args.fallback_to_minimum_flow_candidate
                    else "threshold_only"
                )
            ),
            "candidate_selection": candidate_selection,
            "candidate_meets_remaining_flow_threshold": (
                None
                if candidate_selection == "not_applicable"
                else candidate_selection == "within_threshold"
            ),
            "remaining_flow_threshold": args.remaining_flow_threshold,
            "candidate_remaining_support_fraction": candidate.get(
                "remaining_support_fraction"
            ),
            "replacement_contract": (
                "strict_contextual_pos_morphology"
                if args.strict_replacements
                else "surface_valid_non_deleting_word"
            ),
            "reader_backend": "vllm_openai_compatible",
            "reader_calls": len(reader_job_by_signature),
            "evaluated_method": args.method_name,
            "methods": method_rows,
        }
        rows.append(row)

    answers = run_reader_requests(
        reader,
        pending_requests,
        workers=args.reader_workers,
    )
    for row in rows:
        for result in row["methods"].values():
            if result.get("status") != "pending_reader":
                continue
            answer = answers[int(result.pop("_reader_job"))]
            clean_answer = str(row["clean_answer"])
            gold_answer = str(row["gold_answer"])
            result.update(
                {
                    "status": "ok",
                    "answer": answer,
                    "flip": not answers_exact_match(answer, clean_answer),
                    "lenient_containment_flip": not answers_match(
                        answer,
                        clean_answer,
                    ),
                    "answer_token_f1_to_clean": answer_token_f1(
                        answer,
                        clean_answer,
                    ),
                    "gold_exact_after": answers_exact_match(
                        answer,
                        gold_answer,
                    ),
                    "gold_token_f1_after": answer_token_f1(
                        answer,
                        gold_answer,
                    ),
                }
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            flow = row["methods"][args.method_name]
            print(
                f"[reader-evaluation] index={row['index']} "
                f"k={flow.get('n_selected', 0)} "
                f"flow={flow.get('flip')}",
                flush=True,
            )

    summary = summarize(rows)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[reader-evaluation summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def run_reader_requests(
    reader: ReaderClient,
    requests: list[tuple[str, list[dict]]],
    *,
    workers: int,
) -> list[str]:
    """Submit independent answer requests concurrently to the vLLM server."""

    answers = [""] * len(requests)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(reader.answer, question, contexts): index
            for index, (question, contexts) in enumerate(requests)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            answers[index] = future.result()
            completed += 1
            print(
                f"[vllm-reader] {completed}/{len(requests)}",
                flush=True,
            )
    return answers


def threshold_candidate(
    candidates: list[dict],
    remaining_flow_threshold: float,
) -> dict | None:
    eligible = [
        candidate
        for candidate in candidates
        if 0 < int(candidate.get("n_selected", 0))
        and float(candidate.get("remaining_support_fraction", 1.0))
        <= remaining_flow_threshold + 1e-12
    ]
    return min(
        eligible,
        key=lambda candidate: (
            int(candidate["n_selected"]),
            float(candidate["remaining_support_fraction"]),
            candidate["selected_ids"],
        ),
        default=None,
    )


def evaluation_candidate(
    candidates: list[dict],
    remaining_flow_threshold: float,
    *,
    fallback_to_minimum_flow: bool,
    ignore_remaining_flow_threshold: bool = False,
) -> tuple[dict | None, str]:
    """Select the threshold candidate, optionally falling back by residual flow."""

    if ignore_remaining_flow_threshold:
        nonempty = [
            candidate
            for candidate in candidates
            if int(candidate.get("n_selected", 0)) > 0
        ]
        candidate = min(
            nonempty,
            key=lambda row: (
                int(row["n_selected"]),
                row["selected_ids"],
            ),
            default=None,
        )
        return (
            (candidate, "not_applicable")
            if candidate is not None
            else (None, "none")
        )
    candidate = threshold_candidate(candidates, remaining_flow_threshold)
    if candidate is not None:
        return candidate, "within_threshold"
    if not fallback_to_minimum_flow:
        return None, "none"
    nonempty = [
        candidate
        for candidate in candidates
        if int(candidate.get("n_selected", 0)) > 0
    ]
    fallback = min(
        nonempty,
        key=lambda candidate: (
            float(candidate.get("remaining_support_fraction", 1.0)),
            int(candidate["n_selected"]),
            candidate["selected_ids"],
        ),
        default=None,
    )
    return (
        (fallback, "above_threshold_fallback")
        if fallback is not None
        else (None, "none")
    )


def summarize(rows: list[dict]) -> dict:
    summary = {
        "queries": len(rows),
        "query_scope": (
            "all_queries"
            if rows and rows[0].get("include_clean_incorrect")
            else "clean_correct_only"
        ),
        "clean_correct_queries": sum(
            bool(row.get("clean_correct"))
            for row in rows
        ),
        "clean_incorrect_queries": sum(
            not bool(row.get("clean_correct"))
            for row in rows
        ),
        "remaining_flow_threshold": (
            rows[0]["remaining_flow_threshold"] if rows else None
        ),
        "clean_correct_policy": (rows[0].get("clean_correct_policy") if rows else None),
        "reader_calls": sum(int(row.get("reader_calls", 0)) for row in rows),
        "reader_backend": "vllm_openai_compatible",
        "selection_rule": (
            rows[0].get("selection_rule", "threshold_only")
            if rows
            else None
        ),
        "within_threshold_candidate_queries": sum(
            row.get("candidate_selection") == "within_threshold"
            for row in rows
        ),
        "above_threshold_fallback_queries": sum(
            row.get("candidate_selection") == "above_threshold_fallback"
            for row in rows
        ),
    }
    method = (
        str(rows[0].get("evaluated_method", "residual_flow"))
        if rows
        else "residual_flow"
    )
    summary["evaluated_method"] = method
    valid = [
        row["methods"][method]
        for row in rows
        if row["methods"][method]["status"] == "ok"
    ]
    flips = sum(bool(result.get("flip")) for result in valid)
    candidate_queries = sum(
        row["methods"][method]["status"] != "no_candidate_under_selection_rule"
        for row in rows
    )
    summary[method] = {
        "total_queries": len(rows),
        "candidate_queries": candidate_queries,
        "valid_queries": len(valid),
        "no_candidate_queries": len(rows) - candidate_queries,
        "replacement_failures": candidate_queries - len(valid),
        "flips": flips,
        "flip_rate": flips / len(valid) if valid else None,
        "overall_flip_rate": flips / len(rows) if rows else None,
        "candidate_coverage": (candidate_queries / len(rows) if rows else None),
        "mean_selected_tokens": (
            sum(int(result["n_selected"]) for result in valid) / len(valid)
            if valid
            else None
        ),
    }
    summary["candidate_threshold_strata"] = {}
    for stratum in (
        "within_threshold",
        "above_threshold_fallback",
        "not_applicable",
    ):
        stratum_results = [
            row["methods"][method]
            for row in rows
            if row.get("candidate_selection") == stratum
            and row["methods"][method]["status"] == "ok"
        ]
        stratum_flips = sum(
            bool(result.get("flip"))
            for result in stratum_results
        )
        summary["candidate_threshold_strata"][stratum] = {
            "valid_queries": len(stratum_results),
            "flips": stratum_flips,
            "flip_rate": (
                stratum_flips / len(stratum_results)
                if stratum_results
                else None
            ),
            "mean_selected_tokens": (
                sum(int(result["n_selected"]) for result in stratum_results)
                / len(stratum_results)
                if stratum_results
                else None
            ),
        }
    return summary


if __name__ == "__main__":
    main()
