#!/usr/bin/env python3
"""Score files for the A1 ablations (Graph-Score = direct_score, Single-Flow =
individual_flow) in the schema read by evaluate_matched_budget_baselines.py.

Rankings come from the A1 runner's per-query caches; queries without a cache are
ranked with the same calls the runner makes. Queries whose frozen graph is invalid
get an empty ranking (a method failure that stays in the denominator).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_flow_ablation_pilot import keyed
from causalityrag.flow_ablation import METHODS, rank_graph_tokens
from causalityrag.flow_contract import require_complete_graph_domain
from causalityrag.shared_replacement_pool import FrozenSharedReplacementPool
from causalityrag.token_units import units_from_cache_row

R = Path("/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1")
for ds in sys.argv[1:]:
    config = json.loads((R / "out/flow_ablation_full8_20261024/configs" / f"{ds}.json").read_text())
    records, units, graphs = (keyed(config[n]) for n in ("input", "units", "graphs"))
    ids = list(records)
    caches = [R / "out/flow_ablation_full8_20261024" / ds] + ([R / "out/hotpotqa_pilot100"] if ds == "hotpotqa" else [])
    cached = {}
    for c in caches:
        for f in sorted(c.glob("ranking_*.json")):
            r = json.loads(f.read_text())
            cached.setdefault(str(r[METHODS[0]]["id"]), r)
    (R / "a1_official" / ds).mkdir(parents=True, exist_ok=True)
    pool = None
    out = {m: open(R / "a1_official" / ds / f"{m}.jsonl", "w") for m in METHODS}
    counts = {"cached": 0, "computed": 0, "failed": 0}
    for index, q in enumerate(ids):
        try:
            if q in cached:
                rankings = cached[q]
                counts["cached"] += 1
            else:
                pool = pool or FrozenSharedReplacementPool(config["pool"])
                all_units = units_from_cache_row(records[q], units[q], k=5)
                require_complete_graph_domain(graphs[q], all_units)
                eligible = {str(u["unit_id"]) for u in all_units if pool.is_eligible(str(u["unit_id"]))}
                rankings = {m: rank_graph_tokens(graphs[q], eligible, method=m, graph_domain="editable_projection") for m in METHODS}
                counts["computed"] += 1
            for m in METHODS:
                r = rankings[m]
                if r.get("id", q) != q:
                    raise ValueError(f"cache id mismatch at {index}")
                out[m].write(json.dumps({"index": index, "id": q, "method": m, "status": "ok",
                                         "ranked_ids": r["ranked_ids"], "token_scores": r.get("token_scores", {})}) + "\n")
        except Exception as exc:
            counts["failed"] += 1
            for m in METHODS:
                out[m].write(json.dumps({"index": index, "id": q, "method": m, "status": "error",
                                         "error": f"{type(exc).__name__}: {exc}", "ranked_ids": []}) + "\n")
    for f in out.values():
        f.close()
    print(ds, counts, flush=True)
