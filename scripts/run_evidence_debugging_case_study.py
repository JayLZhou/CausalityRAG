#!/usr/bin/env python3
"""Prepare and summarize a blinded retrieved-evidence debugging study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import statistics
from pathlib import Path

from causalityrag.io import load_records, record_id
from causalityrag.reader import answers_exact_match
from causalityrag.shared_replacement_pool import POOL_SCHEMA, file_sha256


FACTUAL_TYPES = {
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def stable_order(values: list[object], material: str) -> list[object]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{material}\0{value}".encode("utf-8")
        ).digest(),
    )


def supporting_spans(raw: dict, record: dict, *, k: int) -> list[dict]:
    context = {str(title): sentences for title, sentences in raw.get("context", [])}
    spans = []
    for fact in raw.get("supporting_facts", []):
        if not isinstance(fact, (list, tuple)) or len(fact) < 2:
            continue
        title, sentence_index = str(fact[0]), int(fact[1])
        sentences = context.get(title, [])
        if not 0 <= sentence_index < len(sentences):
            continue
        sentence = str(sentences[sentence_index]).strip()
        if not sentence:
            continue
        for chunk in record.get("retrieved", [])[:k]:
            start = str(chunk.get("text", "")).find(sentence)
            if start >= 0:
                spans.append({
                    "chunk_id": str(chunk["chunk_id"]),
                    "start": start,
                    "end": start + len(sentence),
                    "title": title,
                    "sentence_index": sentence_index,
                    "sentence": sentence,
                })
    return spans


def valid_pool_candidates(row: dict, old: str) -> list[dict]:
    values = []
    seen = set()
    for candidate in row.get("candidates", []):
        new = str(candidate.get("new", "")).strip()
        normalized = new.casefold()
        if (
            not candidate.get("ok", True)
            or not new
            or any(character.isspace() for character in new)
            or normalized == old.casefold()
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        values.append(dict(candidate))
    return values


def token_candidates(
    unit_row: dict,
    spans: list[dict],
    pool: dict[str, dict],
    *,
    query_id: str,
) -> list[dict]:
    candidates = []
    for unit in unit_row.get("units", []):
        if str(unit.get("type", "")).upper() not in FACTUAL_TYPES:
            continue
        span = next(
            (
                item
                for item in spans
                if str(unit.get("chunk_id", "")) == item["chunk_id"]
                and int(unit.get("chunk_char_start", -1)) >= item["start"]
                and int(unit.get("chunk_char_end", -1)) <= item["end"]
            ),
            None,
        )
        if span is None:
            continue
        unit_id = str(unit.get("unit_id", ""))
        pool_row = pool.get(unit_id)
        if pool_row is None:
            continue
        alternatives = valid_pool_candidates(pool_row, str(unit.get("text", "")))
        same_length = [
            candidate
            for candidate in alternatives
            if len(str(candidate["new"])) == len(str(unit.get("text", "")))
        ]
        if not same_length:
            continue
        corruption = stable_order(
            same_length,
            f"{query_id}\0{unit_id}\0corruption",
        )[0]
        verification = [
            candidate
            for candidate in alternatives
            if str(candidate["new"]).casefold()
            != str(corruption["new"]).casefold()
        ]
        if not verification:
            continue
        candidates.append({
            "unit": unit,
            "pool_row": pool_row,
            "corruption": corruption,
            "verification_candidates": stable_order(
                verification,
                f"{query_id}\0{unit_id}\0verification",
            ),
            "support": span,
        })
    return stable_order(candidates, f"{query_id}\0planted-unit")


def replace_same_length(text: str, start: int, end: int, old: str, new: str) -> str:
    if len(old) != len(new):
        raise ValueError("planted corruption must preserve character length")
    if text[start:end] != old:
        raise ValueError(f"offset mismatch: expected {old!r}, found {text[start:end]!r}")
    return text[:start] + new + text[end:]


def prepare(args: argparse.Namespace) -> None:
    retrieval = load_records(args.retrieval)
    units = load_records(args.units)
    clean = load_records(args.clean_reference)
    if len({len(retrieval), len(units), len(clean)}) != 1:
        raise ValueError("retrieval, units, and clean reference must be aligned")
    raw_rows = json.loads(Path(args.raw_questions).read_text(encoding="utf-8"))
    raw_by_id = {record_id(row): row for row in raw_rows}

    pool = {}
    with open(args.shared_pool, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != POOL_SCHEMA:
                raise ValueError("unexpected replacement-pool schema")
            if row.get("row_kind") == "position_candidates":
                pool[str(row["unit_id"])] = row

    selected = []
    used_topk_chunks: set[str] = set()
    for original_index, (record, unit_row, clean_row) in enumerate(
        zip(retrieval, units, clean)
    ):
        query_id = record_id(record)
        if query_id != record_id(unit_row) or query_id != record_id(clean_row):
            raise ValueError(f"misaligned source row {original_index}")
        clean_answer = str(
            clean_row.get("clean_answer")
            or clean_row.get("target_answer")
            or clean_row.get("answer")
            or ""
        )
        if not answers_exact_match(clean_answer, str(record.get("answer", ""))):
            continue
        raw = raw_by_id.get(query_id)
        if raw is None:
            continue
        topk_chunks = {
            str(chunk["chunk_id"]) for chunk in record.get("retrieved", [])[: args.k]
        }
        if topk_chunks & used_topk_chunks:
            continue
        spans = supporting_spans(raw, record, k=args.k)
        candidates = token_candidates(unit_row, spans, pool, query_id=query_id)
        if not candidates:
            continue
        selected.append((original_index, record, unit_row, clean_answer, candidates[0]))
        used_topk_chunks.update(topk_chunks)
        if len(selected) == args.n:
            break
    if len(selected) != args.n:
        raise ValueError(f"found only {len(selected)}/{args.n} strict pilot instances")

    corrupted_records = []
    corrupted_units = []
    clean_targets = []
    manifest = []
    derived_pool = {}
    for pilot_index, (original_index, record, unit_row, clean_answer, choice) in enumerate(selected):
        unit = choice["unit"]
        unit_id = str(unit["unit_id"])
        chunk_id = str(unit["chunk_id"])
        start = int(unit["chunk_char_start"])
        end = int(unit["chunk_char_end"])
        old = str(unit["text"])
        new = str(choice["corruption"]["new"])

        corrupted_record = copy.deepcopy(record)
        target_chunk = next(
            chunk
            for chunk in corrupted_record["retrieved"][: args.k]
            if str(chunk["chunk_id"]) == chunk_id
        )
        target_chunk["text"] = replace_same_length(
            str(target_chunk["text"]), start, end, old, new
        )

        corrupted_unit_row = copy.deepcopy(unit_row)
        target_unit = next(
            item
            for item in corrupted_unit_row["units"]
            if str(item["unit_id"]) == unit_id
        )
        target_unit["text"] = new
        target_unit["lemma"] = new.casefold()
        if target_unit.get("entity_text") == old:
            target_unit["entity_text"] = new
        for sentence in corrupted_unit_row.get("sentences", []):
            if (
                str(sentence.get("chunk_id", "")) == chunk_id
                and int(sentence.get("chunk_char_start", -1)) <= start
                and int(sentence.get("chunk_char_end", -1)) >= end
            ):
                sentence_start = int(sentence["chunk_char_start"])
                relative_start = start - sentence_start
                relative_end = end - sentence_start
                sentence["text"] = replace_same_length(
                    str(sentence["text"]), relative_start, relative_end, old, new
                )
        corrupted_unit_row["context_sha256"] = {
            str(chunk["chunk_id"]): hashlib.sha256(
                str(chunk["text"]).encode("utf-8")
            ).hexdigest()
            for chunk in corrupted_record.get("retrieved", [])
        }

        topk_chunk_ids = {
            str(chunk["chunk_id"])
            for chunk in corrupted_record.get("retrieved", [])[: args.k]
        }
        corrupted_unit_row["top_k"] = args.k
        corrupted_unit_row["units"] = [
            item
            for item in corrupted_unit_row.get("units", [])
            if str(item.get("chunk_id", "")) in topk_chunk_ids
        ]
        corrupted_unit_row["sentences"] = [
            sentence
            for sentence in corrupted_unit_row.get("sentences", [])
            if str(sentence.get("chunk_id", "")) in topk_chunk_ids
        ]
        corrupted_unit_row["context_sha256"] = {
            str(chunk["chunk_id"]): hashlib.sha256(
                str(chunk["text"]).encode("utf-8")
            ).hexdigest()
            for chunk in corrupted_record.get("retrieved", [])[: args.k]
        }
        for item in corrupted_unit_row.get("units", []):
            item_id = str(item["unit_id"])
            if str(item.get("chunk_id", "")) not in topk_chunk_ids or item_id not in pool:
                continue
            row = copy.deepcopy(pool[item_id])
            if item_id == unit_id:
                row["surface"] = new
                row["candidates"] = []
                for candidate in choice["verification_candidates"]:
                    revised = copy.deepcopy(candidate)
                    revised["old"] = new
                    row["candidates"].append(revised)
            previous = derived_pool.get(item_id)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting derived pool row for {item_id}")
            derived_pool[item_id] = row

        corrupted_records.append(corrupted_record)
        corrupted_units.append(corrupted_unit_row)
        clean_targets.append({
            "index": pilot_index,
            "id": record_id(record),
            "clean_answer": clean_answer,
            "gold_answer": str(record.get("answer", "")),
            "clean_correct": True,
        })
        manifest.append({
            "index": pilot_index,
            "original_index": original_index,
            "id": record_id(record),
            "question": str(record.get("question", "")),
            "gold_answer": str(record.get("answer", "")),
            "clean_answer": clean_answer,
            "planted_unit_id": unit_id,
            "chunk_id": chunk_id,
            "chunk_rank": int(unit.get("chunk_rank", 0)),
            "chunk_char_start": start,
            "chunk_char_end": end,
            "original_token": old,
            "corrupted_token": new,
            "semantic_type": str(unit.get("type", "")),
            "support_title": choice["support"]["title"],
            "support_sentence_index": choice["support"]["sentence_index"],
            "support_sentence": choice["support"]["sentence"],
            "corruption_policy": choice["corruption"].get("policy", ""),
            "verification_alternatives": [
                str(candidate["new"])
                for candidate in choice["verification_candidates"]
            ],
        })

    root = Path(args.out_dir)
    write_jsonl(root / "corrupted_retrieval.jsonl", corrupted_records)
    write_jsonl(root / "corrupted_units.jsonl", corrupted_units)
    write_jsonl(root / "clean_targets.jsonl", clean_targets)
    write_jsonl(root / "corruption_manifest.jsonl", manifest)
    pool_rows = [derived_pool[key] for key in sorted(derived_pool)]
    write_jsonl(root / "shared_pool.jsonl", pool_rows)
    pool_sha = file_sha256(str(root / "shared_pool.jsonl"))
    summary = {
        "schema": "causalityrag.evidence_debugging_manifest.v1",
        "queries": len(manifest),
        "retrieved_chunks": args.k,
        "selection": (
            "clean-correct queries; gold supporting sentence retrieved; "
            "factual token and two non-clean counterfactual alternatives; "
            "disjoint top-k chunk domains"
        ),
        "shared_pool_sha256": pool_sha,
        "source_shared_pool_sha256": file_sha256(args.shared_pool),
        "type_histogram": {
            token_type: sum(row["semantic_type"] == token_type for row in manifest)
            for token_type in sorted({row["semantic_type"] for row in manifest})
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def ranked_ids(row: dict) -> list[str]:
    raw = row.get("ranked_ids")
    if not isinstance(raw, list):
        scores = row.get("token_scores", {})
        raw = sorted(scores, key=lambda key: (-float(scores[key]), str(key)))
    return list(dict.fromkeys(str(value) for value in raw if str(value)))


def summarize(args: argparse.Namespace) -> None:
    manifest = load_records(args.manifest)
    corrupted = {record_id(row): row for row in load_records(args.corrupted_targets)}
    reflow = {record_id(row): row for row in load_records(args.reflow_results)}
    units = {record_id(row): row for row in load_records(args.units)}
    rankers = {
        name: {record_id(row): row for row in load_records(path)}
        for name, path in (
            ("mirage", args.mirage),
            ("arc_jsd", args.arc_jsd),
        )
    }
    eligible = set()
    with open(args.shared_pool, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                eligible.add(str(json.loads(line).get("unit_id", "")))

    random_seeds = [0, 1, 2, 3, 4]
    rows = []
    for item in manifest:
        query_id = record_id(item)
        corrupted_row = corrupted[query_id]
        reflow_row = reflow[query_id]
        planted = str(item["planted_unit_id"])
        corrupted_answer = str(corrupted_row.get("clean_answer", ""))
        gold = str(item["gold_answer"])
        affected = not answers_exact_match(corrupted_answer, gold)
        reflow_selected = [
            str(value) for value in reflow_row.get("selected_ids", [])
        ]
        budget = len(reflow_selected)
        selections = {"reflow": reflow_selected}
        for name, table in rankers.items():
            ranking = [
                unit_id
                for unit_id in ranked_ids(table[query_id])
                if unit_id in eligible
            ]
            selections[name] = ranking[:budget]
        unit_ids = [
            str(unit["unit_id"])
            for unit in units[query_id].get("units", [])
            if str(unit["unit_id"]) in eligible
        ]
        for seed in random_seeds:
            rng = random.Random(f"{seed}\0{query_id}")
            ranking = list(unit_ids)
            rng.shuffle(ranking)
            selections[f"random_seed{seed}"] = ranking[:budget]
        methods = {}
        for name, selected_ids in selections.items():
            hit = planted in selected_ids
            methods[name] = {
                "selected_ids": selected_ids,
                "budget": len(selected_ids),
                "hit": hit,
                "repair_correct": (
                    True if hit else answers_exact_match(corrupted_answer, gold)
                ),
            }
        rows.append({
            **item,
            "corrupted_answer": corrupted_answer,
            "corruption_effective": affected,
            "reflow_reader_calls": int(reflow_row.get("reader_calls", 0)),
            "methods": methods,
        })

    affected_rows = [row for row in rows if row["corruption_effective"]]
    method_names = ["reflow", "mirage", "arc_jsd"] + [
        f"random_seed{seed}" for seed in random_seeds
    ]
    methods = {}
    for name in method_names:
        points = [row["methods"][name] for row in rows]
        affected_points = [row["methods"][name] for row in affected_rows]
        methods[name] = {
            "hit_at_matched_budget_all": statistics.fmean(
                float(point["hit"]) for point in points
            ),
            "hit_at_matched_budget_affected": statistics.fmean(
                float(point["hit"]) for point in affected_points
            ) if affected_points else None,
            "repair_accuracy_all": statistics.fmean(
                float(point["repair_correct"]) for point in points
            ),
            "repair_accuracy_affected": statistics.fmean(
                float(point["repair_correct"]) for point in affected_points
            ) if affected_points else None,
            "mean_inspected_tokens": statistics.fmean(
                point["budget"] for point in points
            ),
        }
    random_rows = [methods.pop(f"random_seed{seed}") for seed in random_seeds]
    methods["random_5seed_mean"] = {
        key: statistics.fmean(row[key] for row in random_rows)
        for key in random_rows[0]
    }
    summary = {
        "schema": "causalityrag.evidence_debugging_results.v1",
        "queries": len(rows),
        "corruption_effective_queries": len(affected_rows),
        "corruption_success_rate": len(affected_rows) / max(1, len(rows)),
        "mean_reflow_reader_calls": statistics.fmean(
            row["reflow_reader_calls"] for row in rows
        ),
        "methods": methods,
        "note": (
            "With one planted error and oracle restoration, affected-subset "
            "repair accuracy equals localization Hit@Budget."
        ),
    }
    write_jsonl(Path(args.out), rows)
    Path(args.summary_out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--raw-questions", required=True)
    prepare_parser.add_argument("--retrieval", required=True)
    prepare_parser.add_argument("--units", required=True)
    prepare_parser.add_argument("--clean-reference", required=True)
    prepare_parser.add_argument("--shared-pool", required=True)
    prepare_parser.add_argument("--out-dir", required=True)
    prepare_parser.add_argument("--n", type=int, default=100)
    prepare_parser.add_argument("--k", type=int, default=5)
    prepare_parser.set_defaults(function=prepare)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--manifest", required=True)
    summary_parser.add_argument("--corrupted-targets", required=True)
    summary_parser.add_argument("--reflow-results", required=True)
    summary_parser.add_argument("--units", required=True)
    summary_parser.add_argument("--shared-pool", required=True)
    summary_parser.add_argument("--mirage", required=True)
    summary_parser.add_argument("--arc-jsd", required=True)
    summary_parser.add_argument("--out", required=True)
    summary_parser.add_argument("--summary-out", required=True)
    summary_parser.set_defaults(function=summarize)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
