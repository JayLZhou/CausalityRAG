"""Reverify grouped-flow attacks with one local HF eager reader backend."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.arc_jsd import ArcJsdModel
from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.reader import (
    answer_token_f1,
    answers_exact_match,
    answers_match,
    parse_json_object,
)
from causalityrag.revision import apply_token_replacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument(
        "--model-path",
        default="/data1/yujia/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    records_by_id = {
        record_id(record): record for record in load_records(args.input)
    }
    attacks_by_id = load_attack_methods(args.results)
    missing = sorted(set(attacks_by_id) - set(records_by_id))
    if missing:
        raise ValueError(f"attack rows have no input record: {missing[:5]}")

    model = ArcJsdModel(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for index, identifier in enumerate(attacks_by_id, 1):
            started = time.monotonic()
            record = records_by_id[identifier]
            question = str(record.get("question", ""))
            clean_contexts = retrieved_contexts(record)[:args.k]
            methods = attacks_by_id[identifier]
            variants = {(): clean_contexts}
            signature_by_method = {}
            for method_name, method in methods.items():
                edits = [edit for edit in method.get("edits", []) if edit.get("ok")]
                signature = tuple(sorted(
                    (str(edit["unit_id"]), str(edit["new"])) for edit in edits
                ))
                signature_by_method[method_name] = signature
                if signature in variants:
                    continue
                units = [
                    {
                        **edit,
                        "text": str(edit.get("old") or edit.get("token", "")),
                    }
                    for edit in edits
                ]
                replacements = {
                    str(edit["unit_id"]): edit for edit in edits
                }
                revision = apply_token_replacements(
                    record,
                    units,
                    replacements,
                    k=args.k,
                )
                if revision["n_failed_edits"]:
                    raise ValueError(
                        f"stored edit no longer applies for {identifier}/{method_name}"
                    )
                variants[signature] = revision["edited_contexts"]

            signatures = list(variants)
            responses = model.generate_responses_batch(
                question,
                [variants[signature] for signature in signatures],
                max_new_tokens=args.max_new_tokens,
            )
            response_by_signature = dict(zip(signatures, responses))
            clean_response = response_by_signature[()]
            clean_answer = answer_from_response(clean_response)
            gold_answer = str(record.get("answer", ""))
            verified = {}
            for method_name, method in methods.items():
                response = response_by_signature[signature_by_method[method_name]]
                answer = answer_from_response(response)
                verified[method_name] = {
                    "n_selected": int(method.get("n_selected", 0)),
                    "selected_ids": method.get("selected_ids", []),
                    "response": response,
                    "answer": answer,
                    "flip": not answers_exact_match(clean_answer, answer),
                    "lenient_containment_flip": not answers_match(
                        clean_answer,
                        answer,
                    ),
                    "answer_token_f1_to_clean": answer_token_f1(
                        clean_answer,
                        answer,
                    ),
                    "gold_exact_after": answers_exact_match(answer, gold_answer),
                    "vllm_answer": method.get("answer"),
                    "vllm_hf_exact_agreement": answers_exact_match(
                        answer,
                        str(method.get("answer", "")),
                    ),
                }
            row = {
                "id": identifier,
                "question": question,
                "gold_answer": gold_answer,
                "verified_clean_response": clean_response,
                "verified_clean_answer": clean_answer,
                "clean_exact": answers_exact_match(clean_answer, gold_answer),
                "clean_token_f1": answer_token_f1(clean_answer, gold_answer),
                "verification_mode": "batched_local_hf_eager",
                "n_unique_edited_variants": len(variants) - 1,
                "methods": verified,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[hf-reverify] {index}/{len(attacks_by_id)} "
                f"variants={len(variants)} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    summary = summarize(rows)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[hf-reverify summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def load_attack_methods(paths: list[str]) -> dict[str, dict[str, dict]]:
    attacks: dict[str, dict[str, dict]] = {}
    for path in paths:
        source_name = Path(path).stem
        for row in load_records(path):
            identifier = str(row.get("id", ""))
            if not identifier:
                continue
            mode = str(row.get("selection_mode", "selection"))
            setting = (
                f"b{row.get('token_budget')}"
                if mode == "budget"
                else f"t{row.get('remaining_flow_threshold')}"
            )
            methods = attacks.setdefault(identifier, {})
            for method_name, result in row.get("methods", {}).items():
                if result.get("status") != "ok":
                    continue
                key = f"{mode}_{setting}:{method_name}"
                if key in methods:
                    key = f"{source_name}:{key}"
                methods[key] = result
    return attacks


def summarize(rows: list[dict]) -> dict:
    method_rows: dict[str, list[tuple[bool, dict]]] = defaultdict(list)
    for row in rows:
        for method, result in row["methods"].items():
            method_rows[method].append((bool(row["clean_exact"]), result))
    methods = {
        method: {
            "queries": len(pairs),
            "flips": sum(result["flip"] for _, result in pairs),
            "flip_rate": (
                sum(result["flip"] for _, result in pairs) / len(pairs)
                if pairs
                else None
            ),
            "exact_clean_queries": sum(clean for clean, _ in pairs),
            "exact_clean_flips": sum(
                clean and result["flip"] for clean, result in pairs
            ),
            "exact_clean_flip_rate": (
                sum(clean and result["flip"] for clean, result in pairs)
                / sum(clean for clean, _ in pairs)
                if any(clean for clean, _ in pairs)
                else None
            ),
            "vllm_hf_exact_agreement": sum(
                result["vllm_hf_exact_agreement"] for _, result in pairs
            ) / len(pairs),
            "mean_selected_tokens": statistics.fmean(
                result["n_selected"] for _, result in pairs
            ),
        }
        for method, pairs in sorted(method_rows.items())
    }
    settings = sorted({method.rsplit(":", 1)[0] for method in method_rows})
    paired = {}
    for setting in settings:
        left = f"{setting}:residual_flow"
        right = f"{setting}:unary_matched"
        if left not in method_rows or right not in method_rows:
            continue
        paired[setting] = {
            "all": paired_method_summary(rows, left, right),
            "exact_clean": paired_method_summary(
                rows,
                left,
                right,
                require_exact_clean=True,
            ),
        }
    return {
        "queries": len(rows),
        "exact_clean_queries": sum(row["clean_exact"] for row in rows),
        "unique_edited_variants": sum(
            row["n_unique_edited_variants"] for row in rows
        ),
        "mean_seconds": (
            statistics.fmean(row["elapsed_seconds"] for row in rows)
            if rows
            else None
        ),
        "methods": methods,
        "paired": paired,
    }


def paired_method_summary(
    rows: list[dict],
    left: str,
    right: str,
    *,
    require_exact_clean: bool = False,
) -> dict:
    pairs = [
        (bool(row["methods"][left]["flip"]), bool(row["methods"][right]["flip"]))
        for row in rows
        if (not require_exact_clean or row["clean_exact"])
        and left in row["methods"]
        and right in row["methods"]
    ]
    differences = [int(left_flip) - int(right_flip) for left_flip, right_flip in pairs]
    left_only = sum(left_flip and not right_flip for left_flip, right_flip in pairs)
    right_only = sum(right_flip and not left_flip for left_flip, right_flip in pairs)
    left_flips = sum(left_flip for left_flip, _ in pairs)
    right_flips = sum(right_flip for _, right_flip in pairs)
    return {
        "queries": len(pairs),
        "left_flips": left_flips,
        "right_flips": right_flips,
        "left_flip_rate": left_flips / len(pairs) if pairs else None,
        "right_flip_rate": right_flips / len(pairs) if pairs else None,
        "left_only_flips": left_only,
        "right_only_flips": right_only,
        "both_flip": sum(left_flip and right_flip for left_flip, right_flip in pairs),
        "neither_flips": sum(not left_flip and not right_flip for left_flip, right_flip in pairs),
        "flip_rate_difference": (
            sum(differences) / len(differences) if differences else None
        ),
        "paired_bootstrap_95_ci": paired_bootstrap_ci(differences),
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(left_only, right_only),
    }


def mcnemar_exact_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    differences: list[int],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> list[float] | None:
    if not differences:
        return None
    rng = random.Random(seed)
    size = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def answer_from_response(response: str) -> str:
    parsed = parse_json_object(response)
    if isinstance(parsed, dict):
        return str(parsed.get("answer", "")).strip()
    return response.strip()


if __name__ == "__main__":
    main()
