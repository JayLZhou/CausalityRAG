"""A8 summary: agreement between the first-order prediction -r and the exact effect dS of removing one message."""
import glob, json, os, statistics as st
from scipy.stats import spearmanr
D = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a8_first_order"
DS = ["hotpotqa", "musique", "popqa", "finqa", "2wiki", "timeqa", "triviaqa", "quartz"]
out = {}
for ds in DS:
    p = f"{D}/a8_{ds}.jsonl"
    if not os.path.exists(p):
        continue
    rows = [r for r in map(json.loads, open(p)) if r.get("status") == "ok"]
    if not rows:  # still running
        continue
    entry = {"queries": len(rows)}
    for kind in ("top", "context"):
        rho, sign, rel, n = [], [], [], 0
        for r in rows:
            ms = [m for m in r["messages"] if m["set"] == kind]
            pred, obs = [-m["r"] for m in ms], [m["dS"] for m in ms]
            if len(ms) > 2:
                rho.append(spearmanr(pred, obs).statistic)
            sign += [(p > 0) == (o > 0) for p, o in zip(pred, obs) if p != 0]
            rel += [abs(o - p) / abs(p) for p, o in zip(pred, obs) if p != 0]
            n += len(ms)
        entry[kind] = {"messages": n, "spearman_mean": st.mean(rho), "spearman_median": st.median(rho),
                       "sign_agreement": sum(sign) / len(sign), "median_rel_error": st.median(rel)}
    out[ds] = entry
    t, c = entry["top"], entry["context"]
    print(f"{ds:9s} q={len(rows):3d} | top-200: rho {t['spearman_mean']:.3f} (median {t['spearman_median']:.3f}), "
          f"sign {100*t['sign_agreement']:.1f}%, rel.err {t['median_rel_error']:.2f} | context top-100: rho "
          f"{c['spearman_mean']:.3f}, sign {100*c['sign_agreement']:.1f}%, rel.err {c['median_rel_error']:.2f}")
json.dump(out, open(f"{D}/a8_summary.json", "w"), indent=1)
