"""A2: per-query LLM verification calls and runtime from the existing Table 3 outputs."""
import glob, json, os, statistics as st
C = "/data1/yujia/CausalityRAG"
O = f"{C}/out_no_frontier_repair_20260810"
DS = ["hotpotqa", "musique", "popqa", "finqa", "2wiki", "timeqa", "triviaqa", "quartz"]

def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()] if path and os.path.exists(path) else []

def first(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None

def stats(vals):
    vals = [float(v) for v in vals if v is not None]
    return {"n": len(vals), "mean": st.mean(vals), "median": st.median(vals), "max": max(vals)} if vals else None

out = {}
for ds in DS:
    if ds == "popqa":
        base = f"{C}/out_table3_final8_20260811/popqa/methods/reflow"
        res, fr = first(f"{base}/results_top5_1000.jsonl", f"{base}/reflow_top5_1000.jsonl"), f"{base}/frontier_top5_1000.jsonl"
    else:
        res, fr = f"{O}/{ds}/results_1000.jsonl", f"{O}/{ds}/frontier_1000.jsonl"
    R = [r for r in rows(res) if r.get("eligible", True)]
    B = f"{C}/out/{ds}/methods/baselines"
    entry = {"results_file": res,
             "reader_calls": stats(r.get("reader_calls") for r in R),
             "verification_seconds": stats(r.get("elapsed_seconds") for r in R),
             "frontier_seconds": stats(r.get("elapsed_seconds") for r in rows(fr))}
    for name, pats in {"arc_jsd": [f"{B}/arc_jsd_projection_1000.jsonl", f"{B}/arc_jsd_top5_1000.jsonl"],
                       "mirage": [f"{B}/mirage_gamma1_setting_q1000_scores.jsonl", f"{B}/mirage_top5_1000.jsonl"]}.items():
        p = first(*pats)
        entry[name] = dict(file=p, seconds=stats(r.get("elapsed_seconds") for r in rows(p)))
    out[ds] = entry
    rc, vs, fs = entry["reader_calls"], entry["verification_seconds"], entry["frontier_seconds"]
    print(ds, f"calls mean {rc['mean']:.2f} max {rc['max']:.0f}" if rc else "calls -",
          f"| verify {vs['mean']:.2f}s" if vs else "", f"| frontier {fs['mean']:.3f}s max {fs['max']:.2f}" if fs else "| frontier -",
          *(f"| {n} {entry[n]['seconds']['mean']:.2f}s" for n in ("arc_jsd", "mirage") if entry[n]["seconds"]))
json.dump(out, open("/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a2_runtime/existing_runs.json", "w"), indent=1)
