"""Canonical result accounting for ReFlow reader verification."""

from __future__ import annotations

import statistics


def canonicalize_reflow_row(row: dict) -> dict:
    """Record the intervention that was actually executed for each query.

    A verified failure requires a reader call on a nonempty edited token set.
    Candidates without executable counterfactual replacements remain diagnostic
    attempts and are never counted as reader failures.
    """

    result = dict(row)
    attempts = list(result.get("attempts", []))
    executed = [
        attempt
        for attempt in attempts
        if (
            attempt.get("reader_called")
            and int(
                attempt.get(
                    "n_edits",
                    len(attempt.get("selected_ids", [])),
                )
            )
            > 0
        )
    ]
    flipped = next(
        (attempt for attempt in executed if attempt.get("answer_changed")),
        None,
    )
    final_attempt = flipped or (executed[-1] if executed else None)

    if final_attempt is not None:
        selected_ids = list(final_attempt.get("selected_ids", []))
        selected_tokens = list(final_attempt.get("selected_tokens", []))
        n_modified = int(final_attempt.get("n_edits", len(selected_ids)))
        result.update(
            {
                "selected_ids": selected_ids,
                "selected_tokens": selected_tokens,
                "n_selected": len(selected_ids),
                "n_modified_tokens": n_modified,
                "edited_answer": final_attempt.get(
                    "edited_answer",
                    result.get("clean_answer", ""),
                ),
                "reader_executed": True,
                "verified_flip": flipped is not None,
                "verified_failure": flipped is None,
                "evaluation_status": (
                    "verified_flip" if flipped is not None else "verified_no_flip"
                ),
            }
        )
        return result

    has_frontier = bool(result.get("frontier_candidates"))
    result.update(
        {
            "selected_ids": [],
            "selected_tokens": [],
            "n_selected": 0,
            "n_modified_tokens": 0,
            "edited_answer": result.get("clean_answer", ""),
            "reader_executed": False,
            "verified_flip": False,
            "verified_failure": False,
            "evaluation_status": (
                "unsupported_no_executable_intervention"
                if has_frontier
                else "unsupported_no_frontier"
            ),
        }
    )
    return result


def summarize_reflow_rows(rows: list[dict]) -> dict:
    """Summarize actual reader executions, including unsuccessful edits."""

    normalized = [canonicalize_reflow_row(row) for row in rows]
    evaluated = [row for row in normalized if row["reader_executed"]]
    flips = [row for row in evaluated if row["verified_flip"]]
    failures = [row for row in evaluated if row["verified_failure"]]
    unsupported = [row for row in normalized if not row["reader_executed"]]
    cumulative_sizes = [
        sum(
            int(attempt.get("n_edits", len(attempt.get("selected_ids", []))))
            for attempt in row.get("attempts", [])
            if attempt.get("reader_called")
        )
        for row in normalized
    ]
    all_modified_sizes = [
        int(row["n_modified_tokens"]) for row in normalized
    ]
    evaluated_modified_sizes = [
        int(row["n_modified_tokens"]) for row in evaluated
    ]
    failure_modified_sizes = [
        int(row["n_modified_tokens"]) for row in failures
    ]
    return {
        "queries": len(normalized),
        "evaluated_queries": len(evaluated),
        "verified_failures": len(failures),
        "unsupported_queries": len(unsupported),
        "flips": len(flips),
        "overall_flip_rate": len(flips) / max(1, len(normalized)),
        "verified_flip_rate": len(flips) / max(1, len(evaluated)),
        "success_mean_tokens": (
            statistics.fmean(int(row["n_modified_tokens"]) for row in flips)
            if flips
            else 0.0
        ),
        "mean_modified_tokens": (
            statistics.fmean(all_modified_sizes) if all_modified_sizes else 0.0
        ),
        "evaluated_mean_modified_tokens": (
            statistics.fmean(evaluated_modified_sizes)
            if evaluated_modified_sizes
            else 0.0
        ),
        "failure_mean_tokens": (
            statistics.fmean(failure_modified_sizes)
            if failure_modified_sizes
            else 0.0
        ),
        "failure_min_tokens": (
            min(failure_modified_sizes) if failure_modified_sizes else 0
        ),
        "failure_max_tokens": (
            max(failure_modified_sizes) if failure_modified_sizes else 0
        ),
        "terminal_mean_tokens": (
            statistics.fmean(all_modified_sizes) if all_modified_sizes else 0.0
        ),
        "cumulative_attempted_tokens_mean": (
            statistics.fmean(cumulative_sizes) if cumulative_sizes else 0.0
        ),
    }
