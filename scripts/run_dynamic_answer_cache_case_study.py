#!/usr/bin/env python3
"""Replay answer-cache policies over frozen RAG requests and corpus updates."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from scripts.summarize_cache_invalidation_case_study import (
    index_rows,
    project_units_to_sentences,
    reflow_rank,
)


DEFAULT_CAPACITIES = (40, 100, 200)
DEFAULT_BUDGETS = (1, 3, 5)
DEFAULT_TTLS = (25, 50, 100, 250, 500, 1000)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    eviction: str = "lru"
    ttl: int | None = None
    invalidation: str = "none"
    budget: int | None = None


@dataclass
class CacheEntry:
    inserted_at: int
    last_access: int
    source_version: int
    frequency: int = 1
    stale_active: bool = False
    stale_failed: bool = False


def build_policy_specs(
    *, budgets: tuple[int, ...], ttls: tuple[int, ...]
) -> list[PolicySpec]:
    specs = [
        PolicySpec(name="lru"),
        PolicySpec(name="lfu", eviction="lfu"),
        PolicySpec(name="version_lru", invalidation="version"),
    ]
    specs.extend(
        PolicySpec(name=f"ttl_lru_{ttl}", ttl=ttl) for ttl in ttls
    )
    for budget in budgets:
        specs.append(
            PolicySpec(
                name=f"reflow_token_{budget}",
                invalidation="token",
                budget=budget,
            )
        )
        specs.append(
            PolicySpec(
                name=f"reflow_sentence_{budget}",
                invalidation="sentence",
                budget=budget,
            )
        )
    return specs


def make_request_trace(
    query_ids: list[str],
    *,
    n_requests: int,
    zipf_alpha: float,
    updates_per_query: int,
    seed: int,
) -> tuple[list[str], dict[int, str]]:
    rng = random.Random(seed)
    popularity_order = list(query_ids)
    rng.shuffle(popularity_order)
    weights = [1.0 / math.pow(rank + 1, zipf_alpha) for rank in range(len(query_ids))]
    requests = rng.choices(popularity_order, weights=weights, k=n_requests)

    positions: dict[str, list[int]] = {query_id: [] for query_id in query_ids}
    for index, query_id in enumerate(requests):
        positions[query_id].append(index)

    updates: dict[int, str] = {}
    for query_id in query_ids:
        occurrences = positions[query_id]
        if len(occurrences) < 2:
            continue
        update_count = min(updates_per_query, len(occurrences) - 1)
        for occurrence_index in rng.sample(
            range(1, len(occurrences)), update_count
        ):
            updates[occurrences[occurrence_index]] = query_id
    return requests, updates


def should_invalidate(
    spec: PolicySpec,
    *,
    query_id: str,
    update_tokens: set[str],
    update_sentences: set[str],
    token_signatures: dict[str, list[str]],
    sentence_signatures: dict[str, list[str]],
) -> bool:
    if spec.invalidation == "none":
        return False
    if spec.invalidation == "version":
        return True
    if spec.invalidation == "token":
        return bool(
            update_tokens & set(token_signatures[query_id][: spec.budget])
        )
    if spec.invalidation == "sentence":
        return bool(
            update_sentences & set(sentence_signatures[query_id][: spec.budget])
        )
    raise ValueError(f"unknown invalidation mode: {spec.invalidation}")


def select_victim(cache: dict[str, CacheEntry], eviction: str) -> str:
    if eviction == "lru":
        return min(cache, key=lambda query_id: cache[query_id].last_access)
    if eviction == "lfu":
        return min(
            cache,
            key=lambda query_id: (
                cache[query_id].frequency,
                cache[query_id].last_access,
            ),
        )
    raise ValueError(f"unknown eviction policy: {eviction}")


def replay_policy(
    *,
    spec: PolicySpec,
    capacity: int,
    requests: list[str],
    updates: dict[int, str],
    update_rows: dict[str, dict],
    sentence_by_query: dict[str, dict[str, str]],
    token_signatures: dict[str, list[str]],
    sentence_signatures: dict[str, list[str]],
) -> dict:
    cache: dict[str, CacheEntry] = {}
    source_versions: dict[str, int] = {}
    counters = {
        "requests": 0,
        "hits": 0,
        "fresh_hits": 0,
        "stale_hits": 0,
        "misses": 0,
        "capacity_evictions": 0,
        "ttl_expirations": 0,
        "update_invalidations": 0,
        "unnecessary_invalidations": 0,
        "resident_stale_updates": 0,
        "stale_episodes_prevented": 0,
        "stale_episodes_failed": 0,
    }

    def remove_entry(query_id: str, *, reason: str) -> None:
        entry = cache.pop(query_id)
        if entry.stale_active and not entry.stale_failed:
            counters["stale_episodes_prevented"] += 1
        counters[reason] += 1

    for step, query_id in enumerate(requests):
        update_query = updates.get(step)
        if update_query is not None:
            update = update_rows[update_query]
            source_versions[update_query] = 1 - source_versions.get(
                update_query, 0
            )
            entry = cache.get(update_query)
            if entry is not None:
                is_stale = bool(update.get("answer_changed")) and (
                    entry.source_version != source_versions[update_query]
                )
                if is_stale:
                    entry.stale_active = True
                    entry.stale_failed = False
                    counters["resident_stale_updates"] += 1
                else:
                    entry.stale_active = False
                    entry.stale_failed = False
                update_tokens = {
                    str(unit_id) for unit_id in update.get("selected_ids", [])
                }
                update_sentences = {
                    sentence_by_query[update_query].get(unit_id, "")
                    for unit_id in update_tokens
                } - {""}
                if should_invalidate(
                    spec,
                    query_id=update_query,
                    update_tokens=update_tokens,
                    update_sentences=update_sentences,
                    token_signatures=token_signatures,
                    sentence_signatures=sentence_signatures,
                ):
                    if not is_stale:
                        counters["unnecessary_invalidations"] += 1
                    remove_entry(update_query, reason="update_invalidations")

        counters["requests"] += 1
        entry = cache.get(query_id)
        if (
            entry is not None
            and spec.ttl is not None
            and step - entry.inserted_at >= spec.ttl
        ):
            remove_entry(query_id, reason="ttl_expirations")
            entry = None

        if entry is not None:
            counters["hits"] += 1
            entry.last_access = step
            entry.frequency += 1
            if entry.stale_active:
                counters["stale_hits"] += 1
                if not entry.stale_failed:
                    entry.stale_failed = True
                    counters["stale_episodes_failed"] += 1
            else:
                counters["fresh_hits"] += 1
            continue

        counters["misses"] += 1
        if len(cache) >= capacity:
            victim = select_victim(cache, spec.eviction)
            remove_entry(victim, reason="capacity_evictions")
        cache[query_id] = CacheEntry(
            inserted_at=step,
            last_access=step,
            source_version=source_versions.get(query_id, 0),
        )

    denominator = max(1, counters["requests"])
    stale_episode_denominator = max(1, counters["resident_stale_updates"])
    return {
        "policy": spec.name,
        "capacity": capacity,
        **counters,
        "cache_reuse_rate": counters["hits"] / denominator,
        "safe_reuse_rate": counters["fresh_hits"] / denominator,
        "unsafe_served_rate": counters["stale_hits"] / denominator,
        "recomputation_rate": counters["misses"] / denominator,
        "stale_episode_recall": counters["stale_episodes_prevented"]
        / stale_episode_denominator,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["policy"], row["capacity"]), []).append(row)
    metric_names = (
        "cache_reuse_rate",
        "safe_reuse_rate",
        "unsafe_served_rate",
        "recomputation_rate",
        "stale_episode_recall",
        "resident_stale_updates",
        "update_invalidations",
        "unnecessary_invalidations",
    )
    result = []
    for (policy, capacity), group in sorted(groups.items()):
        summary = {"policy": policy, "capacity": capacity, "traces": len(group)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-results", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--reflow-frontier", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--requests", type=int, default=20_000)
    parser.add_argument("--zipf-alpha", type=float, default=1.0)
    parser.add_argument("--updates-per-query", type=int, default=5)
    parser.add_argument("--capacities", type=int, nargs="+", default=DEFAULT_CAPACITIES)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--ttls", type=int, nargs="+", default=DEFAULT_TTLS)
    args = parser.parse_args()

    stress_rows = load_records(args.stress_results)
    units = index_rows(args.units)
    frontier = index_rows(args.reflow_frontier)
    query_ids = sorted({record_id(row) for row in stress_rows})
    seeds = sorted({int(row["seed"]) for row in stress_rows})
    stress_by_seed = {
        seed: {
            record_id(row): row
            for row in stress_rows
            if int(row["seed"]) == seed
        }
        for seed in seeds
    }
    for seed, rows in stress_by_seed.items():
        missing = set(query_ids) - set(rows)
        if missing:
            raise ValueError(f"seed {seed} misses {len(missing)} query updates")

    sentence_by_query = {
        query_id: {
            str(unit["unit_id"]): str(unit.get("sentence_id", ""))
            for unit in units[query_id].get("units", [])
            if unit.get("unit_id") is not None
        }
        for query_id in query_ids
    }
    token_signatures = {
        query_id: [
            unit_id
            for unit_id in reflow_rank(frontier[query_id])
            if unit_id in sentence_by_query[query_id]
        ]
        for query_id in query_ids
    }
    sentence_signatures = {
        query_id: project_units_to_sentences(
            token_signatures[query_id], sentence_by_query[query_id]
        )
        for query_id in query_ids
    }
    specs = build_policy_specs(
        budgets=tuple(args.budgets), ttls=tuple(args.ttls)
    )

    rows = []
    trace_metadata = []
    for seed in seeds:
        requests, updates = make_request_trace(
            query_ids,
            n_requests=args.requests,
            zipf_alpha=args.zipf_alpha,
            updates_per_query=args.updates_per_query,
            seed=seed,
        )
        trace_metadata.append(
            {
                "seed": seed,
                "requests": len(requests),
                "scheduled_updates": len(updates),
            }
        )
        for capacity in args.capacities:
            for spec in specs:
                row = replay_policy(
                    spec=spec,
                    capacity=capacity,
                    requests=requests,
                    updates=updates,
                    update_rows=stress_by_seed[seed],
                    sentence_by_query=sentence_by_query,
                    token_signatures=token_signatures,
                    sentence_signatures=sentence_signatures,
                )
                row["seed"] = seed
                rows.append(row)

    output = {
        "schema": "causalityrag.dynamic_answer_cache.v1",
        "queries": len(query_ids),
        "requests_per_trace": args.requests,
        "zipf_alpha": args.zipf_alpha,
        "updates_per_query": args.updates_per_query,
        "capacities": args.capacities,
        "budgets": args.budgets,
        "ttls": args.ttls,
        "stale_definition": "answer_changed",
        "trace_metadata": trace_metadata,
        "per_trace": rows,
        "summary": aggregate(rows),
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
