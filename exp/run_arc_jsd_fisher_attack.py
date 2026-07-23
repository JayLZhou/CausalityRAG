"""Experimental baseline: evaluate an all-pair ARC-JSD Fisher graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.arc_jsd import (
    ArcJsdModel,
    all_context_word_units,
    fisher_pair_graph,
    replace_context_units,
)
from causalityrag.graph_cut import solve_source_target_supermodular_ratio
from causalityrag.io import iter_records, record_id, retrieved_contexts
from causalityrag.reader import ReaderClient, answers_match, parse_json_object
from causalityrag.replacement import GenericReplacementClient, deterministic_fallback
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fisher-top-k", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--replacement-mode",
        choices=["typed-or-generic", "deterministic"],
        default="typed-or-generic",
    )
    args = parser.parse_args()

    records = list(iter_records(args.input, args.start + args.n))[args.start:]
    model = ArcJsdModel(args.model_path, device=args.device, dtype=args.dtype)
    reader = ReaderClient()
    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None)
    generic_editor = GenericReplacementClient()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as output:
        for row_index, record in enumerate(records, 1):
            started = time.monotonic()
            question = str(record.get("question", ""))
            contexts = retrieved_contexts(record)[:args.k]
            units = all_context_word_units(record, k=args.k)
            replacements = build_replacements(
                units,
                contexts,
                library,
                generic_editor,
                deterministic=args.replacement_mode == "deterministic",
            )
            trajectory = model.clean_trajectory(question, contexts)
            parsed = parse_json_object(trajectory.response_text)
            clean_answer = (
                str(parsed.get("answer", "")).strip()
                if isinstance(parsed, dict)
                else trajectory.response_text
            )
            variants = (
                replace_context_units(contexts, [unit], replacements)
                for unit in units
            )
            singleton_scores, fisher_features = model.score_context_variants_with_fisher_features(
                question,
                variants,
                trajectory,
                batch_size=args.batch_size,
                top_k=args.fisher_top_k,
            )
            node_weights, interactions = fisher_pair_graph(
                units, singleton_scores, fisher_features
            )
            candidate = solve_source_target_supermodular_ratio(
                units, node_weights, interactions, {}
            )
            by_id = {str(unit["unit_id"]): unit for unit in units}
            selected = [by_id[unit_id] for unit_id in candidate["selected_ids"]]
            revision = apply_token_replacements(record, selected, replacements, k=args.k)
            edited_answer = (
                reader.answer(question, revision["edited_contexts"])
                if selected else clean_answer
            )
            result = {
                "id": record_id(record),
                "question": question,
                "gold_answer": str(record.get("answer", "")),
                "clean_response": trajectory.response_text,
                "clean_answer": clean_answer,
                "edited_answer": edited_answer,
                "answer_changed": not answers_match(clean_answer, edited_answer),
                "method": "all_pair_replacement_arc_jsd_fisher_ratio",
                "objective": "max_nonempty_S F(S)/|S|",
                "fisher_top_k": args.fisher_top_k,
                "n_context_tokens": len(units),
                "n_pairs": len(units) * (len(units) - 1) // 2,
                "n_positive_pair_edges": len(interactions),
                "candidate": candidate,
                "selected_units": selected,
                "n_selected": len(selected),
                "edits": revision["edits"],
                "n_edits": revision["n_edits"],
                "replacements": replacements,
                "singleton_scores": {
                    str(unit["unit_id"]): score
                    for unit, score in zip(units, singleton_scores)
                },
                "positive_pair_edges": [
                    {"left": left, "right": right, "weight": weight}
                    for (left, right), weight in interactions.items()
                ],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[arc-jsd-fisher] {row_index}/{len(records)} tokens={len(units)} "
                f"pairs={result['n_pairs']} positive={len(interactions)} "
                f"selected={len(selected)} flip={result['answer_changed']} "
                f"seconds={result['elapsed_seconds']}",
                flush=True,
            )


def build_replacements(
    units: list[dict],
    contexts: list[dict],
    library: TypedRuleLibrary,
    generic_editor: GenericReplacementClient,
    *,
    deterministic: bool,
) -> dict[str, dict]:
    context_by_id = {str(context["chunk_id"]): str(context["text"]) for context in contexts}
    replacements = {}
    cache: dict[tuple[str, str], dict] = {}
    for unit in units:
        key = (str(unit["text"]), str(unit.get("type", "")))
        replacement = cache.get(key)
        if replacement is None:
            replacement = library.replacement_for_token(
                key[0], key[1], context_by_id[str(unit["chunk_id"])]
            )
            if not replacement.get("ok"):
                replacement = (
                    deterministic_fallback(key[0], key[1])
                    if deterministic
                    else generic_editor.replace(
                        key[0], context_by_id[str(unit["chunk_id"])], key[1]
                    )
                )
            cache[key] = replacement
        replacements[str(unit["unit_id"])] = replacement
    return replacements


if __name__ == "__main__":
    main()
