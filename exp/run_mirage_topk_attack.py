"""Experimental baseline: rank editable context words with MIRAGE."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.arc_jsd import ArcJsdModel, all_context_word_units
from causalityrag.io import iter_records, record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from exp.mirage import MirageScorer
from causalityrag.reader import ReaderClient, answers_match, parse_json_object
from causalityrag.replacement import GenericReplacementClient, generate_valid_replacement
from causalityrag.revision import apply_token_replacements
from causalityrag.rules import TypedRuleLibrary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-path", default="/data1/yujia/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--cf-pools", required=True)
    parser.add_argument("--type-rules", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--budgets", default="1,3,5")
    parser.add_argument("--cti-std-threshold", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument(
        "--spacy-base-url",
        default=os.environ.get("CAUSALITYRAG_SPACY_BASE_URL", "http://127.0.0.1:8021"),
    )
    args = parser.parse_args()
    budgets = sorted({int(value) for value in args.budgets.split(",") if value.strip()})
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must be positive")

    records = list(iter_records(args.input, args.start + args.n))[args.start:]
    nlp = SpacyAnnotationClient(args.spacy_base_url)
    health = nlp.health()
    if not health.get("ok"):
        raise RuntimeError(f"spaCy annotation service is unhealthy: {health}")
    model = ArcJsdModel(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    scorer = MirageScorer(model, cti_std_threshold=args.cti_std_threshold)
    reader = ReaderClient()
    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None)
    generic_editor = GenericReplacementClient()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as output:
        for row_index, record in enumerate(records, 1):
            started = time.monotonic()
            question = str(record.get("question", ""))
            contexts = retrieved_contexts(record)[:args.k]
            units = all_context_word_units(record, k=args.k, nlp=nlp)
            trajectory = model.clean_trajectory(question, contexts)
            parsed = parse_json_object(trajectory.response_text)
            clean_answer = (
                str(parsed.get("answer", "")).strip()
                if isinstance(parsed, dict)
                else trajectory.response_text
            )
            token_scores, diagnostics = scorer.score(
                question, contexts, units, trajectory
            )
            selected, replacements, rejected = select_ranked_valid_tokens(
                units,
                token_scores,
                max(budgets),
                contexts,
                library,
                generic_editor,
                nlp,
            )
            budget_results = {}
            for budget in budgets:
                budget_selected = selected[:budget]
                budget_replacements = {
                    str(unit["unit_id"]): replacements[str(unit["unit_id"])]
                    for unit in budget_selected
                }
                revision = apply_token_replacements(
                    record, budget_selected, budget_replacements, k=args.k
                )
                edited_answer = (
                    reader.answer(question, revision["edited_contexts"])
                    if budget_selected else clean_answer
                )
                budget_results[str(budget)] = {
                    "budget": budget,
                    "selected_units": budget_selected,
                    "selected_scores": [
                        token_scores[str(unit["unit_id"])] for unit in budget_selected
                    ],
                    "n_selected": len(budget_selected),
                    "edits": revision["edits"],
                    "n_edits": revision["n_edits"],
                    "edited_answer": edited_answer,
                    "answer_changed": not answers_match(clean_answer, edited_answer),
                }
            result = {
                "id": record_id(record),
                "question": question,
                "gold_answer": str(record.get("answer", "")),
                "clean_response": trajectory.response_text,
                "clean_answer": clean_answer,
                "method": "mirage_saliency_unary_topk",
                "objective": "fixed_budget_topk_by_mirage_unary_weight",
                "n_context_tokens": len(units),
                "mirage": diagnostics.to_dict(),
                "token_scores": token_scores,
                "rejected_uneditable_units": rejected,
                "budgets": budget_results,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            flips = {
                budget: value["answer_changed"]
                for budget, value in budget_results.items()
            }
            print(
                f"[mirage-topk] {row_index}/{len(records)} tokens={len(units)} "
                f"cti={len(diagnostics.selected_response_indices)} flips={flips} "
                f"seconds={result['elapsed_seconds']}",
                flush=True,
            )


def select_ranked_valid_tokens(
    units: list[dict],
    token_scores: dict[str, float],
    budget: int,
    contexts: list[dict],
    library: TypedRuleLibrary,
    generic_editor: GenericReplacementClient,
    nlp,
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    context_by_id = {
        str(context["chunk_id"]): str(context["text"]) for context in contexts
    }
    ranked = sorted(
        (
            unit for unit in units
            if float(token_scores.get(str(unit["unit_id"]), 0.0)) > 0
        ),
        key=lambda unit: (
            -float(token_scores.get(str(unit["unit_id"]), 0.0))
            / max(float(unit.get("cost", 1.0)), 1e-12),
            str(unit["unit_id"]),
        ),
    )
    selected = []
    replacements = {}
    rejected = []
    for unit in ranked:
        unit_id = str(unit["unit_id"])
        replacement = generate_valid_replacement(
            unit,
            context_by_id[str(unit["chunk_id"])],
            library,
            generic_editor,
            nlp,
        )
        if not replacement.get("ok"):
            rejected.append({**unit, "replacement_failure": replacement})
            continue
        selected.append(unit)
        replacements[unit_id] = replacement
        if len(selected) == budget:
            break
    return selected, replacements, rejected


if __name__ == "__main__":
    main()
