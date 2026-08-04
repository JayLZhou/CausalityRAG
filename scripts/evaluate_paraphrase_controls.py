"""Evaluate meaning-preserving controls on exactly the selected token positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.paraphrase_control import excluded_from_paraphrase_control
from causalityrag.reader import (
    ReaderClient,
    answer_token_f1,
    answers_exact_match,
    answers_match,
)
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import file_sha256
from causalityrag.token_units import units_from_cache_row


def stable_candidate(candidates: list[dict], unit_id: str, seed: int) -> dict:
    material = f"{seed}\0{unit_id}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return dict(candidates[index % len(candidates)])


def load_pool(path: str) -> dict[str, dict]:
    rows = load_records(path)
    pool = {}
    for row in rows:
        unit_id = str(row.get("unit_id", ""))
        candidates = row.get("candidates", [])
        if not unit_id or not candidates:
            raise ValueError("paraphrase pool contains an incomplete row")
        pool[unit_id] = row
    return pool


def evaluate_selected(
    *,
    record: dict,
    units: dict[str, dict],
    selected: list[str],
    clean_answer: str,
    gold_answer: str,
    pool: dict[str, dict],
    reader: ReaderClient,
    seed: int,
    k: int,
) -> dict:
    if not selected:
        return {"status": "no_selected_tokens", "reader_called": False}
    excluded = [
        unit_id
        for unit_id in selected
        if excluded_from_paraphrase_control(units[unit_id])
    ]
    excluded_set = set(excluded)
    control_selected = [
        unit_id for unit_id in selected if unit_id not in excluded_set
    ]
    if not control_selected:
        return {
            "status": "no_paraphrase_eligible_tokens",
            "reader_called": False,
            "selected_ids": selected,
            "control_selected_ids": [],
            "excluded_numeric_date_ids": excluded,
            "n_modified_tokens": 0,
            "edited_answer": clean_answer,
            "answer_flip": False,
            "f1_flip": False,
            "em_flip": False,
            "acc_flip": False,
        }
    missing = [unit_id for unit_id in control_selected if unit_id not in pool]
    if missing:
        return {
            "status": "missing_paraphrase",
            "reader_called": False,
            "selected_ids": selected,
            "control_selected_ids": control_selected,
            "excluded_numeric_date_ids": excluded,
            "missing_ids": missing,
        }
    replacements = {
        unit_id: stable_candidate(pool[unit_id]["candidates"], unit_id, seed)
        for unit_id in control_selected
    }
    revision = apply_token_replacements(
        record,
        [units[unit_id] for unit_id in control_selected],
        replacements,
        k=k,
        allow_whitespace=True,
        allow_case_only=True,
    )
    if (
        int(revision["n_failed_edits"])
        or int(revision["n_edits"]) != len(control_selected)
    ):
        return {
            "status": "protocol_violation_failed_edit",
            "reader_called": False,
            "selected_ids": selected,
            "control_selected_ids": control_selected,
            "excluded_numeric_date_ids": excluded,
            "edits": revision["edits"],
        }
    edited = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
    if not edited.strip():
        return {
            "status": "invalid_empty_answer",
            "reader_called": True,
            "selected_ids": selected,
            "control_selected_ids": control_selected,
            "excluded_numeric_date_ids": excluded,
            "n_modified_tokens": int(revision["n_edits"]),
            "edits": revision["edits"],
            "edited_answer": edited,
            "answer_flip": False,
            "f1_flip": False,
            "em_flip": False,
            "acc_flip": False,
        }
    clean_f1 = answer_token_f1(clean_answer, gold_answer)
    edited_f1 = answer_token_f1(edited, gold_answer)
    return {
        "status": "evaluated",
        "reader_called": True,
        "selected_ids": selected,
        "control_selected_ids": control_selected,
        "excluded_numeric_date_ids": excluded,
        "n_modified_tokens": int(revision["n_edits"]),
        "edits": revision["edits"],
        "edited_answer": edited,
        "answer_flip": not answers_exact_match(clean_answer, edited),
        "f1_flip": edited_f1 < clean_f1,
        "em_flip": (
            answers_exact_match(clean_answer, gold_answer)
            and not answers_exact_match(edited, gold_answer)
        ),
        "acc_flip": (
            answers_match(clean_answer, gold_answer)
            and not answers_match(edited, gold_answer)
        ),
    }


def summarize(rows: list[dict]) -> dict:
    method_names = sorted({
        name for row in rows for name in row.get("methods", {})
    })
    methods = {}
    for name in method_names:
        pairs = [
            (row, row.get("methods", {}).get(name, {}))
            for row in rows
        ]
        values = [method for _, method in pairs]
        executed = [row for row in values if row.get("reader_called")]
        valid = [
            row for row in executed
            if str(row.get("edited_answer", "")).strip()
        ]
        f1_clean_correct = [
            (parent, method)
            for parent, method in pairs
            if answer_token_f1(
                str(parent.get("clean_answer", "")),
                str(parent.get("gold_answer", "")),
            ) >= 1.0 - 1e-12
        ]
        em_clean_correct = [
            (parent, method)
            for parent, method in pairs
            if answers_exact_match(
                str(parent.get("clean_answer", "")),
                str(parent.get("gold_answer", "")),
            )
        ]
        acc_clean_correct = [
            (parent, method)
            for parent, method in pairs
            if answers_match(
                str(parent.get("clean_answer", "")),
                str(parent.get("gold_answer", "")),
            )
        ]

        def valid_correct(population: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
            return [
                (parent, method)
                for parent, method in population
                if method.get("reader_called")
                and str(method.get("edited_answer", "")).strip()
            ]

        valid_f1_correct = valid_correct(f1_clean_correct)
        valid_em_correct = valid_correct(em_clean_correct)
        valid_acc_correct = valid_correct(acc_clean_correct)
        f1_flips = sum(
            answer_token_f1(
                str(method.get("edited_answer", "")),
                str(parent.get("gold_answer", "")),
            ) < 1.0 - 1e-12
            for parent, method in valid_f1_correct
        )
        em_flips = sum(
            not answers_exact_match(
                str(method.get("edited_answer", "")),
                str(parent.get("gold_answer", "")),
            )
            for parent, method in valid_em_correct
        )
        acc_flips = sum(
            not answers_match(
                str(method.get("edited_answer", "")),
                str(parent.get("gold_answer", "")),
            )
            for parent, method in valid_acc_correct
        )
        methods[name] = {
            "queries": len(values),
            "executed_queries": len(executed),
            "valid_answer_queries": len(valid),
            "answer_flips": sum(bool(row.get("answer_flip")) for row in valid),
            "answer_flip_rate_itt": (
                sum(bool(row.get("answer_flip")) for row in valid)
                / max(1, len(values))
            ),
            "f1_clean_correct_queries": len(f1_clean_correct),
            "em_clean_correct_queries": len(em_clean_correct),
            "acc_clean_correct_queries": len(acc_clean_correct),
            "f1_flips": f1_flips,
            "f1_flip_rate_itt": f1_flips / max(1, len(f1_clean_correct)),
            "em_flips": em_flips,
            "em_flip_rate_itt": em_flips / max(1, len(em_clean_correct)),
            "acc_flips": acc_flips,
            "acc_flip_rate_itt": acc_flips / max(1, len(acc_clean_correct)),
            "excluded_numeric_date_selections": sum(
                len(row.get("excluded_numeric_date_ids", [])) for row in values
            ),
            "paraphrase_edits": sum(
                int(row.get("n_modified_tokens", 0)) for row in values
            ),
            "status_histogram": {
                status: sum(row.get("status") == status for row in values)
                for status in sorted({
                    str(row.get("status", ""))
                    for row in values
                    if row.get("status")
                })
            },
        }
    return {"queries": len(rows), "methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--reflow-results", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--paraphrase-pool", required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--replacement-seed", type=int, default=0)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    args = parser.parse_args()

    actual_sha = file_sha256(args.paraphrase_pool)
    if actual_sha != args.expected_pool_sha256:
        raise ValueError("paraphrase pool fingerprint mismatch")
    pool = load_pool(args.paraphrase_pool)
    records = load_records(args.input)[: args.n]
    unit_rows = load_records(args.units_cache)[: args.n]
    reflow_rows = load_records(args.reflow_results)[: args.n]
    baseline_rows = load_records(args.baseline_results)[: args.n]
    if len({len(records), len(unit_rows), len(reflow_rows), len(baseline_rows)}) != 1:
        raise ValueError("all inputs must have aligned row counts")
    reader = ReaderClient(
        base_url=args.llm_base_url or None,
        model=args.llm_model or None,
    )
    lock = Lock()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def run(index: int) -> dict:
        record = records[index]
        identifier = record_id(record)
        unit_row = unit_rows[index]
        reflow = reflow_rows[index]
        baselines = baseline_rows[index]
        aligned = {
            str(unit_row.get("id", "")),
            str(reflow.get("id", "")),
            str(baselines.get("id", "")),
        }
        if aligned != {identifier}:
            raise ValueError(f"misaligned query {identifier}: {aligned}")
        cache_k = int(unit_row.get("top_k", args.k))
        units = {
            str(unit["unit_id"]): unit
            for unit in units_from_cache_row(record, unit_row, k=cache_k)
        }
        clean_answer = str(reflow.get("clean_answer", ""))
        gold_answer = str(record.get("answer", ""))
        selections = {
            "reflow": [str(value) for value in reflow.get("selected_ids", [])],
            **{
                name: [str(value) for value in method.get("selected_ids", [])]
                for name, method in baselines.get("methods", {}).items()
            },
        }
        methods = {
            name: evaluate_selected(
                record=record,
                units=units,
                selected=selected,
                clean_answer=clean_answer,
                gold_answer=gold_answer,
                pool=pool,
                reader=reader,
                seed=args.replacement_seed,
                k=args.k,
            )
            for name, selected in selections.items()
        }
        row = {
            "index": index,
            "id": identifier,
            "clean_answer": clean_answer,
            "gold_answer": gold_answer,
            "methods": methods,
        }
        with lock:
            with open(args.out, "a", encoding="utf-8") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, index) for index in range(len(records))]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 100 == 0:
                print(f"[{completed}/{len(futures)}]", flush=True)
    rows.sort(key=lambda row: row["index"])
    temporary = args.out + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    summary = {
        **summarize(rows),
        "paraphrase_pool": os.path.abspath(args.paraphrase_pool),
        "paraphrase_pool_sha256": actual_sha,
        "replacement_seed": args.replacement_seed,
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
