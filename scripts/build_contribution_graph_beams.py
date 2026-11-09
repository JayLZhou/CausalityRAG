"""Build several exact receiver-beam graphs from one model trace per query."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import ContributionGraphBuilder
from causalityrag.io import iter_records, record_id
from causalityrag.reader import answers_exact_match, parse_json_object
from causalityrag.token_units import all_context_word_units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--out-template",
        required=True,
        help="Output JSONL path containing a {beam} placeholder.",
    )
    parser.add_argument(
        "--summary-template",
        default="",
        help="Optional summary path containing a {beam} placeholder.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--target-results", nargs="+", required=True)
    parser.add_argument("--receiver-beams", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-unusable",
        action="store_true",
        help=(
            "Keep structurally unusable beam rows for downstream no-frontier "
            "targeting instead of failing the batch after all rows are written."
        ),
    )
    args = parser.parse_args()

    beams = list(dict.fromkeys(args.receiver_beams))
    if args.start < 0 or args.n <= 0 or args.k <= 0:
        parser.error("--start must be non-negative; --n and --k must be positive")
    if not beams or any(beam <= 0 for beam in beams):
        parser.error("--receiver-beams must contain positive integers")
    if "{beam}" not in args.out_template:
        parser.error("--out-template must contain {beam}")
    if args.summary_template and "{beam}" not in args.summary_template:
        parser.error("--summary-template must contain {beam}")

    records = list(
        islice(iter_records(args.input), args.start, args.start + args.n)
    )
    target_by_id = load_targets_by_id(args.target_results)
    missing = [
        record_id(record)
        for record in records
        if record_id(record) not in target_by_id
    ]
    if missing:
        raise ValueError(f"missing cached clean targets: {missing[:5]}")

    paths = {beam: args.out_template.format(beam=beam) for beam in beams}
    existing = {
        beam: list(iter_records(path))
        if args.resume and os.path.exists(path)
        else []
        for beam, path in paths.items()
    }
    completed = _validate_resume_prefix(records, existing)
    for path in paths.values():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    builder = ContributionGraphBuilder(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_receivers_per_layer=max(beams),
    )
    outputs = {
        beam: open(path, "a" if completed else "w", encoding="utf-8")
        for beam, path in paths.items()
    }
    try:
        for index, record in enumerate(records[completed:], completed + 1):
            started = time.monotonic()
            identifier = record_id(record)
            target = target_by_id[identifier]
            if target.strip() and any(character.isalnum() for character in target):
                rows = builder.build_many(
                    record,
                    target,
                    k=args.k,
                    receiver_beams=beams,
                )
            else:
                rows = {
                    beam: _abstention_row(record, target, args.k, beam)
                    for beam in beams
                }
            elapsed = round(time.monotonic() - started, 3)
            statuses = []
            for beam in beams:
                row = rows[beam]
                row["clean_answer"] = target
                row["target_source"] = "frozen_vllm_results"
                row["clean_correct"] = answers_exact_match(
                    target,
                    str(record.get("answer", "")),
                )
                row["elapsed_seconds"] = elapsed
                row["multi_beam_shared_trace"] = True
                row["joint_receiver_beams"] = beams
                outputs[beam].write(json.dumps(row, ensure_ascii=False) + "\n")
                outputs[beam].flush()
                statuses.append(f"d{beam}={row['status']}")
            print(
                f"[contribution-graph-beams] {index}/{len(records)} "
                f"{' '.join(statuses)} joint_seconds={elapsed}",
                flush=True,
            )
    finally:
        for output in outputs.values():
            output.close()

    failed = False
    for beam in beams:
        rows = list(iter_records(paths[beam]))
        summary = _summary(
            rows,
            paths[beam],
            beam,
            beams,
            allow_unusable=args.allow_unusable,
        )
        print(
            f"[contribution-graph-beams summary d={beam}] "
            + json.dumps(summary, ensure_ascii=False, indent=2)
        )
        if args.summary_template:
            summary_path = args.summary_template.format(beam=beam)
            os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as output:
                json.dump(summary, output, ensure_ascii=False, indent=2)
                output.write("\n")
        failed = failed or bool(summary["failed"])
    if failed and not args.allow_unusable:
        raise RuntimeError("one or more receiver-beam graphs are unusable")


def _validate_resume_prefix(
    records: list[dict], existing: dict[int, list[dict]]
) -> int:
    lengths = {len(rows) for rows in existing.values()}
    if len(lengths) != 1:
        raise ValueError(f"beam outputs have different resume lengths: {lengths}")
    completed = lengths.pop()
    if completed > len(records):
        raise ValueError("beam output is longer than the requested input slice")
    expected = [record_id(record) for record in records[:completed]]
    for beam, rows in existing.items():
        identifiers = [record_id(row) for row in rows]
        if identifiers != expected or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"beam {beam} resume IDs are not a unique input prefix")
        if any(
            int(row.get("message_flow_diagnostics", {}).get("receiver_beam", beam))
            != beam
            for row in rows
        ):
            raise ValueError(f"beam {beam} output contains a different receiver beam")
    return completed


def _answer_from_response(response: str) -> str:
    parsed = parse_json_object(response)
    if isinstance(parsed, dict):
        return str(parsed.get("answer", "")).strip()
    return response.strip()


def _answer_from_result_row(row: dict) -> str:
    response = str(row.get("clean_response", "")).strip()
    if response:
        return _answer_from_response(response)
    for key in ("clean_answer", "stored_clean_answer", "target_answer", "answer"):
        if key in row:
            return str(row[key]).strip()
    raise ValueError(f"target result row {row.get('id')} has no clean answer")


def load_targets_by_id(paths: list[str]) -> dict[str, str]:
    targets = {}
    for path in paths:
        for row in iter_records(path):
            identifier = record_id(row)
            if identifier:
                targets[identifier] = _answer_from_result_row(row)
    return targets


def _abstention_row(record: dict, target: str, k: int, beam: int) -> dict:
    status = (
        "reader_abstention_empty_answer"
        if not target.strip()
        else "reader_abstention_nonsemantic_answer"
    )
    token_units = all_context_word_units(record, k=k)
    return {
        "id": record_id(record),
        "question": str(record.get("question", "")),
        "gold_answer": str(record.get("answer", "")),
        "target_answer": target,
        "status": status,
        "method": "closed_flow_token_contribution_graph",
        "contribution_graph": {
            "source": "query_context_source",
            "target": "answer_target",
            "token_nodes": sorted(str(unit["unit_id"]) for unit in token_units),
            "source_edges": [],
            "interaction_edges": [],
            "target_edges": [],
            "diagnostics": {},
        },
        "message_flow_diagnostics": {"receiver_beam": beam},
    }


def _summary(
    rows: list[dict],
    path: str,
    beam: int,
    beams: list[int],
    *,
    allow_unusable: bool,
) -> dict:
    status_histogram = dict(
        sorted(Counter(str(row["status"]) for row in rows).items())
    )
    ok = sum(row["status"] == "ok" for row in rows)
    excluded = sum(
        str(row["status"]).startswith("reader_abstention_") for row in rows
    )
    return {
        "records": len(rows),
        "ok": ok,
        "excluded_reader_abstentions": excluded,
        "failed": len(rows) - ok - excluded,
        "status_histogram": status_histogram,
        "method": "closed_flow_token_contribution_graph",
        "target_objective": "mean-answer-logit",
        "target_source": "frozen_vllm_results",
        "graph_api": "ContributionGraphBuilder.build_many",
        "receiver_beam": beam,
        "joint_receiver_beams": beams,
        "multi_beam_shared_trace": True,
        "unusable_rows_retained_for_targeted_repair": allow_unusable,
        "out": path,
    }


if __name__ == "__main__":
    main()
